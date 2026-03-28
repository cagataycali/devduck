"""🎤 Background speech listener with Whisper transcription for DevDuck.

Features:
- Background audio capture from any input device
- Voice activity detection (WebRTC VAD or energy-based)
- Whisper transcription of speech segments
- Trigger keyword activation (e.g., "hey duck")
- Auto stealth mode for passive monitoring
- Transcript history and JSONL logging

Dependencies (optional - graceful fallback):
- openai-whisper, sounddevice, webrtcvad
"""

import os
import io
import time
import json
import wave
import queue
import threading
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import webrtcvad
except Exception:
    webrtcvad = None

try:
    import whisper
except Exception:
    whisper = None

from strands import tool

logger = logging.getLogger(__name__)

# Module-level state
STATE: Dict[str, Any] = {
    "running": False,
    "start_time": None,
    "threads": {},
    "stop_event": None,
    "audio_queue": None,
    "segment_queue": None,
    "transcript_log": [],
    "transcript_count": 0,
    "save_dir": None,
    "log_path": None,
    "model_name": None,
    "device_name": None,
    "sample_rate": 16000,
    "channels": 1,
    "energy_threshold": 0.01,
    "pause_duration": 0.8,
    "use_vad": True,
    "trigger_keyword": None,
    "agent": None,
    "auto_mode": False,
    "length_threshold": 50,
    "transcript_callback": None,  # Callable for live TUI transcript push
}

MAX_TRANSCRIPTS = 50

# Short/noise transcripts to filter out (Whisper hallucinations on silence/noise)
NOISE_PHRASES = {
    "thank you", "thank you.", "thanks.", "thanks",
    "you", "you.", "bye.", "bye",
    "the end.", "the end",
    "okay.", "okay",
    "so.", "so",
    "hmm.", "hmm",
    "uh.", "uh",
    "um.", "um",
    "", ".",
}

# Minimum audio duration in seconds to keep a segment
MIN_SEGMENT_DURATION_SEC = float(os.environ.get("DEVDUCK_LISTEN_MIN_DURATION", "1.5"))


def _now_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S.%fZ")


def _push_transcript_to_context(text: str) -> None:
    """Push a new transcript into shared agent messages and ring context.

    This makes transcriptions immediately available to:
    1. The TUI shared message history (agents see it as context)
    2. The unified mesh ring (visible in sidebar network feed)
    3. Any registered callback (e.g., TUI live transcript panel)
    4. The unified event bus (for sidebar event stream)
    """
    timestamp = datetime.utcnow().strftime("%H:%M:%S")

    # 1. Push to unified event bus (for TUI sidebar + agent context)
    try:
        from devduck.tools.event_bus import emit
        emit("listen.transcript", "listen", text[:80], text, {"timestamp": timestamp})
    except ImportError:
        pass

    # 2. Push to unified mesh ring (visible across all agents)
    try:
        from devduck.tools.unified_mesh import add_to_ring
        add_to_ring(
            "listen:whisper",
            "local",
            f"🎤 [{timestamp}] {text}",
            {"source": "listen", "type": "transcript"},
        )
    except (ImportError, Exception):
        pass

    # 3. Call registered transcript callback (for TUI live updates)
    cb = STATE.get("transcript_callback")
    if cb:
        try:
            cb(text, timestamp)
        except Exception:
            pass


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a), dtype=np.float64))) if a.size else 0.0


def _write_wav(path: str, data: np.ndarray, sr: int) -> None:
    if data.dtype != np.int16:
        data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def _find_input_device(name_substr: Optional[str]) -> Optional[int]:
    if sd is None:
        return None
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    if name_substr:
        for i, d in enumerate(devices):
            if (
                d.get("max_input_channels", d.get("maxInputChannels", 0)) > 0
                and name_substr.lower() in str(d.get("name", "")).lower()
            ):
                return i

    # Default input
    default = sd.default.device
    in_idx = default[0] if isinstance(default, (list, tuple)) else default
    try:
        info = sd.query_devices(in_idx)
        if info.get("max_input_channels", info.get("maxInputChannels", 0)) > 0:
            return in_idx
    except Exception:
        pass

    # Fallback: any input device
    for i, d in enumerate(devices):
        if d.get("max_input_channels", d.get("maxInputChannels", 0)) > 0:
            return i
    return None


def _audio_callback(indata, frames, time_info, status):
    data = indata.reshape(-1) if indata.ndim > 1 else indata.reshape(-1)
    try:
        STATE["audio_queue"].put_nowait(data.astype(np.float32))
    except queue.Full:
        pass


def _segmenter_worker(stop_event: threading.Event) -> None:
    sr = STATE["sample_rate"]
    energy_th = float(STATE["energy_threshold"])
    pause_dur = float(STATE["pause_duration"])

    vad = None
    if STATE["use_vad"] and webrtcvad is not None:
        vad = webrtcvad.Vad(2)

    current: List[np.ndarray] = []
    speaking = False
    last_voice_time = 0.0
    seg_start_ts = None
    frame_len = int(0.02 * sr)
    frame_buf = np.empty((0,), dtype=np.float32)

    while not stop_event.is_set():
        try:
            chunk = STATE["audio_queue"].get(timeout=0.2)
        except queue.Empty:
            if speaking and (time.time() - last_voice_time) >= pause_dur and current:
                seg = (
                    np.concatenate(current)
                    if current
                    else np.empty((0,), dtype=np.float32)
                )
                STATE["segment_queue"].put({"audio": seg, "started": seg_start_ts})
                current.clear()
                speaking = False
            continue

        if chunk is None:
            break

        frame_buf = np.concatenate([frame_buf, chunk])

        while frame_buf.shape[0] >= frame_len:
            frame = frame_buf[:frame_len]
            frame_buf = frame_buf[frame_len:]

            is_voiced = False
            if vad is not None:
                try:
                    fbytes = (
                        (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    )
                    is_voiced = vad.is_speech(fbytes, sr)
                except Exception:
                    is_voiced = _rms(frame) >= energy_th
            else:
                is_voiced = _rms(frame) >= energy_th

            if is_voiced:
                if not speaking:
                    seg_start_ts = _now_ts()
                speaking = True
                last_voice_time = time.time()
            current.append(frame) if speaking else None

        if speaking and (time.time() - last_voice_time) >= pause_dur and current:
            seg = (
                np.concatenate(current) if current else np.empty((0,), dtype=np.float32)
            )
            STATE["segment_queue"].put({"audio": seg, "started": seg_start_ts})
            current.clear()
            speaking = False
            seg_start_ts = None



def get_recent_transcripts_context(max_entries: int = 10) -> str:
    """Get recent transcripts formatted for system prompt / dynamic context injection.

    Returns empty string if listener not running or no transcripts.
    """
    if not STATE.get("running"):
        return ""

    items = list(STATE.get("transcript_log", []))[-max_entries:]
    if not items:
        return ""

    lines = ["\n\n## 🎤 Recent Voice Transcriptions (Whisper):"]
    for t in items:
        ts = t.get("timestamp", "?")[:19]
        txt = t.get("text", "")
        if txt and not txt.startswith("[transcription_error]"):
            lines.append(f"- [{ts}] {txt[:300]}")

    if len(lines) <= 1:
        return ""

    lines.append("*Listen tool active — transcripts are live.*")
    return "\n".join(lines)



def _transcribe_numpy(model, audio_f32: np.ndarray) -> str:
    """Transcribe a numpy float32 audio array using whisper, fully subprocess-safe.

    In Python 3.13 + TUI/PTY environments, subprocess.run() fails with
    "bad value(s) in fds_to_keep" due to PTY FD races. Whisper's transcribe()
    *should* accept numpy directly, but torch internals or whisper's audio
    pipeline can still trigger subprocess calls (e.g. ffmpeg, torch compile).

    This function:
    1. Converts audio to torch tensor ourselves (bypass load_audio/ffmpeg)
    2. Computes log-mel spectrogram directly via whisper.audio
    3. Calls whisper's decode pipeline on the mel features
    4. Falls back to monkey-patching subprocess if needed
    """
    import torch

    try:
        # Direct path: pass numpy array — whisper.transcribe handles it
        result = model.transcribe(audio_f32, fp16=False)
        return (result or {}).get("text", "").strip()
    except (ValueError, OSError) as e:
        if "fds_to_keep" not in str(e):
            raise
        # Fall through to manual mel spectrogram path
        logger.debug(f"fds_to_keep error, using manual mel path: {e}")

    # Manual path: compute mel spectrogram ourselves, then decode
    try:
        from whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES, SAMPLE_RATE
        from whisper import DecodingOptions, decode

        # Convert numpy → torch tensor (this is what log_mel_spectrogram does internally)
        audio_tensor = torch.from_numpy(audio_f32).float()

        # Compute mel spectrogram — pure torch, no subprocess
        mel = log_mel_spectrogram(audio_tensor, model.dims.n_mels, padding=16000 * 30)
        mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(torch.float32)

        # Decode
        options = DecodingOptions(language="en", fp16=False)
        result = decode(model, mel_segment, options)

        if isinstance(result, list):
            return " ".join(r.text.strip() for r in result if r.text)
        return result.text.strip() if hasattr(result, "text") else ""

    except Exception as e2:
        logger.error(f"Manual mel transcription also failed: {e2}")
        return f"[transcription_error] {e2}"


def _transcriber_worker(stop_event: threading.Event) -> None:
    if whisper is None:
        return

    try:
        model = whisper.load_model(STATE["model_name"] or "base")
    except Exception:
        model = whisper.load_model("base")

    while not stop_event.is_set():
        try:
            item = STATE["segment_queue"].get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            break

        seg_audio = item["audio"]
        seg_started = item["started"]
        sr = STATE["sample_rate"]

        # ── Filter 1: Skip segments shorter than MIN_SEGMENT_DURATION_SEC ──
        duration_sec = len(seg_audio) / sr if sr > 0 else 0
        if duration_sec < MIN_SEGMENT_DURATION_SEC:
            logger.debug(f"Skipping short segment ({duration_sec:.2f}s < {MIN_SEGMENT_DURATION_SEC}s)")
            STATE["segment_queue"].task_done()
            continue

        # Convert to float32 for whisper
        try:
            if seg_audio.dtype == np.int16:
                audio_f32 = seg_audio.astype(np.float32) / 32768.0
            elif seg_audio.dtype == np.float32:
                audio_f32 = seg_audio
            else:
                audio_f32 = seg_audio.astype(np.float32)

            text = _transcribe_numpy(model, audio_f32)
        except Exception as e:
            text = f"[transcription_error] {e}"

        # ── Filter 2: Skip known noise/hallucination phrases ──
        text_clean = (text or "").strip().lower()
        if text_clean in NOISE_PHRASES:
            logger.debug(f"Filtered noise transcript: '{text}' ({duration_sec:.2f}s)")
            STATE["segment_queue"].task_done()
            continue

        # Only save WAV + log for segments that pass both filters
        fname = f"segment_{seg_started}_{_now_ts()}.wav"
        wav_path = os.path.join(STATE["save_dir"], fname)
        try:
            _write_wav(wav_path, seg_audio, sr)
        except Exception as e:
            logger.warning(f"Failed to write WAV {wav_path}: {e}")

        record = {"timestamp": _now_ts(), "wav_path": wav_path, "text": text}
        STATE["transcript_log"].append(record)
        STATE["transcript_log"] = STATE["transcript_log"][-MAX_TRANSCRIPTS:]
        STATE["transcript_count"] += 1

        try:
            log_path = STATE.get("log_path")
            if log_path:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write transcript log: {e}")

        # 🔗 Push transcript to shared messages immediately (for TUI/agent awareness)
        if text and not text.startswith("[transcription_error]"):
            _push_transcript_to_context(text)

        # Trigger keyword or auto mode
        agent = STATE.get("agent")
        trigger_kw = STATE.get("trigger_keyword")

        if not text or text.startswith("[transcription_error]") or not agent:
            STATE["segment_queue"].task_done()
            continue

        trigger_hit = trigger_kw and trigger_kw.lower() in text.lower()
        auto_hit = STATE.get("auto_mode") and len(text.strip()) >= STATE.get(
            "length_threshold", 50
        )

        if trigger_hit or auto_hit:
            try:
                if trigger_hit and trigger_kw:
                    idx = text.lower().find(trigger_kw.lower())
                    prompt_text = (
                        text[idx + len(trigger_kw) :].strip()
                        or "I'm listening, how can I help?"
                    )
                    mode = "trigger"
                else:
                    prompt_text = f"Process this overheard speech silently: {text}"
                    mode = "auto"

                logger.info(f"🎤 [{mode}] Processing: {prompt_text[:100]}")

                response = agent.tool.use_agent(
                    prompt=prompt_text,
                    system_prompt=f"You are responding to voice input. User said: '{text}'. Be concise and conversational.",
                    record_direct_tool_call=False,
                    agent=agent,
                )

                agent_record = {
                    "timestamp": _now_ts(),
                    "wav_path": None,
                    "text": f"[AGENT_{mode.upper()}] {response.get('content', [{}])[0].get('text', '')}",
                    "original": text,
                    "mode": mode,
                }
                STATE["transcript_log"].append(agent_record)
                STATE["transcript_log"] = STATE["transcript_log"][-MAX_TRANSCRIPTS:]
            except Exception as e:
                logger.error(f"Listen agent trigger error: {e}")

        STATE["segment_queue"].task_done()


@tool
def listen(
    action: str = "status",
    model_name: str = "base",
    device_name: Optional[str] = None,
    save_dir: str = "/tmp/devduck/listen",
    sample_rate: int = 16000,
    channels: int = 1,
    energy_threshold: float = 0.01,
    pause_duration: float = 0.8,
    use_vad: bool = True,
    limit: int = 10,
    trigger_keyword: Optional[str] = None,
    auto_mode: bool = False,
    length_threshold: int = 50,
    agent: Any = None,
) -> Dict[str, Any]:
    """🎤 Background speech listener with Whisper transcription.

    Args:
        action: One of: start, stop, status, list_devices, get_transcripts
        model_name: Whisper model (tiny, base, small, medium, large)
        device_name: Input device name substring (e.g., "BlackHole", "MacBook")
        save_dir: Directory for WAV segments and transcripts
        sample_rate: Audio sample rate (16000 recommended for Whisper)
        channels: Audio channels (1 = mono recommended)
        energy_threshold: Energy threshold for voice detection
        pause_duration: Seconds of silence to end a segment
        use_vad: Use WebRTC VAD if available
        limit: Max transcripts to return (for get_transcripts)
        trigger_keyword: Keyword to activate agent (e.g., "hey duck")
        auto_mode: Enable stealth mode - auto-triggers on long speech
        length_threshold: Character threshold for auto mode (default: 50)
        agent: Parent agent for trigger/auto mode

    Returns:
        Dict with status and content
    """
    action = (action or "status").lower()

    if action == "start":
        if STATE.get("running"):
            return {
                "status": "success",
                "content": [{"text": "Listener already running."}],
            }

        if sd is None:
            return {
                "status": "error",
                "content": [
                    {"text": "sounddevice not installed. pip install sounddevice"}
                ],
            }
        if whisper is None:
            return {
                "status": "error",
                "content": [
                    {"text": "openai-whisper not installed. pip install openai-whisper"}
                ],
            }

        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "transcripts.jsonl")

        device_index = _find_input_device(device_name)
        if device_index is None:
            return {
                "status": "error",
                "content": [
                    {"text": f"No input device found for: {device_name or 'default'}"}
                ],
            }

        stop_event = threading.Event()
        STATE.update(
            {
                "running": True,
                "start_time": time.time(),
                "stop_event": stop_event,
                "audio_queue": queue.Queue(maxsize=100),
                "segment_queue": queue.Queue(maxsize=50),
                "save_dir": save_dir,
                "log_path": log_path,
                "model_name": model_name,
                "device_name": device_name,
                "sample_rate": sample_rate,
                "channels": channels,
                "energy_threshold": energy_threshold,
                "pause_duration": pause_duration,
                "use_vad": use_vad,
                "trigger_keyword": trigger_keyword,
                "agent": agent,
                "auto_mode": auto_mode,
                "length_threshold": length_threshold,
            }
        )

        seg_t = threading.Thread(
            target=_segmenter_worker, args=(stop_event,), daemon=True
        )
        tx_t = threading.Thread(
            target=_transcriber_worker, args=(stop_event,), daemon=True
        )
        seg_t.start()
        tx_t.start()

        def stream_thread():
            try:
                with sd.InputStream(
                    device=device_index,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="float32",
                    callback=_audio_callback,
                ):
                    while not stop_event.is_set():
                        time.sleep(0.1)
            except Exception:
                stop_event.set()

        a_t = threading.Thread(target=stream_thread, daemon=True)
        a_t.start()
        STATE["threads"] = {"audio": a_t, "segmenter": seg_t, "transcriber": tx_t}

        features = []
        if trigger_keyword:
            features.append(f"trigger='{trigger_keyword}'")
        if auto_mode:
            features.append(f"auto_mode(>{length_threshold} chars)")

        return {
            "status": "success",
            "content": [
                {
                    "text": f"🎤 Listening started (model={model_name}, device={device_name or 'default'}{', ' + ', '.join(features) if features else ''}). Saving to: {save_dir}"
                }
            ],
        }

    elif action == "stop":
        if not STATE.get("running"):
            return {
                "status": "success",
                "content": [{"text": "Listener already stopped."}],
            }

        stop_event = STATE.get("stop_event")
        if stop_event:
            stop_event.set()

        for q_name in ["audio_queue", "segment_queue"]:
            q = STATE.get(q_name)
            if q:
                try:
                    q.put_nowait(None)
                except Exception:
                    pass

        for t in (STATE.get("threads") or {}).values():
            if isinstance(t, threading.Thread):
                t.join(timeout=2.0)

        count = STATE.get("transcript_count", 0)
        STATE.update(
            {
                "running": False,
                "threads": {},
                "stop_event": None,
                "audio_queue": None,
                "segment_queue": None,
            }
        )
        return {
            "status": "success",
            "content": [
                {"text": f"🎤 Listening stopped. {count} segments transcribed."}
            ],
        }

    elif action == "status":
        running = STATE.get("running", False)
        info = {
            "running": running,
            "model": STATE.get("model_name"),
            "device": STATE.get("device_name"),
            "transcripts": STATE.get("transcript_count", 0),
            "trigger_keyword": STATE.get("trigger_keyword"),
            "auto_mode": STATE.get("auto_mode", False),
            "uptime": (
                f"{time.time() - STATE.get('start_time', time.time()):.0f}s"
                if running
                else "0s"
            ),
        }
        return {"status": "success", "content": [{"text": json.dumps(info, indent=2)}]}

    elif action == "list_devices":
        if sd is None:
            return {
                "status": "error",
                "content": [{"text": "sounddevice not installed."}],
            }
        try:
            devices = sd.query_devices()
            inputs = []
            for i, d in enumerate(devices):
                max_in = d.get("max_input_channels", d.get("maxInputChannels", 0))
                if max_in and max_in > 0:
                    inputs.append(
                        f"  [{i}] {d.get('name', '')} ({max_in}ch, {d.get('default_samplerate', d.get('defaultSampleRate', '?'))}Hz)"
                    )
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "🎤 Input Devices:\n" + "\n".join(inputs)
                            if inputs
                            else "No input devices found."
                        )
                    }
                ],
            }
        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"Error querying devices: {e}"}],
            }

    elif action == "get_transcripts":
        items = list(STATE.get("transcript_log", []))[-limit:]
        if not items:
            return {"status": "success", "content": [{"text": "No transcripts yet."}]}
        lines = []
        for t in items:
            ts = t.get("timestamp", "?")[:19]
            txt = t.get("text", "")[:200]
            lines.append(f"[{ts}] {txt}")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Unknown action: {action}. Valid: start, stop, status, list_devices, get_transcripts"
                }
            ],
        }
