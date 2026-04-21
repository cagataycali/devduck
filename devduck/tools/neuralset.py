"""NeuralSet - neuro-AI pipeline tool for DevDuck.

Wraps the Meta FAIR NeuralSet project (https://facebookresearch.github.io/neuroai/neuralset/)
into a single Strands tool. Covers the full pipeline from dataset discovery to model
evaluation on neural-encoding/decoding benchmarks across EEG / MEG / fMRI / iEEG / fNIRS / Spikes.

Pipeline stages wrapped as a single `action`:

  - setup      : clone + install neuralset (one-shot)
  - list       : list available datasets / modalities / benchmarks
  - info       : show metadata for a dataset (subjects, modality, stim type)
  - download   : fetch raw neural recordings + stimuli into a local cache
  - preprocess : run canonical preprocessing (filtering, epoching, alignment)
  - features   : extract stimulus features (vision/audio/text backbones)
  - align      : align neural timeseries with stimulus features
  - encode     : fit encoding model (neural = f(features))
  - decode     : fit decoding model (features/stimulus = f(neural))
  - evaluate   : run standard benchmark scoring
  - leaderboard: submit/view results on neuralset leaderboard
  - cache      : manage local cache (list / clear / stats)

Design notes:
  - Heavy imports are lazy (torch/mne/nibabel/etc.) so DevDuck startup is fast.
  - Every action returns the standard Strands `{status, content:[{text}]}` shape.
  - Long-running stages accept a `dry_run=True` to preview commands.
  - Cache root is controlled by DEVDUCK_NEURALSET_CACHE (default: ~/.cache/neuralset).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from strands import tool


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

NEURALSET_REPO_URL = os.getenv(
    "DEVDUCK_NEURALSET_REPO",
    "https://github.com/facebookresearch/neuroai.git",
)
NEURALSET_REPO_DIR = Path(
    os.getenv("DEVDUCK_NEURALSET_REPO_DIR", str(Path.home() / ".neuralset" / "repo"))
)
NEURALSET_CACHE = Path(
    os.getenv("DEVDUCK_NEURALSET_CACHE", str(Path.home() / ".cache" / "neuralset"))
)
NEURALSET_RESULTS = NEURALSET_CACHE / "results"

# Modalities supported by NeuralSet (per project page)
SUPPORTED_MODALITIES = ("eeg", "meg", "fmri", "ieeg", "fnirs", "spikes")

# Stimulus-feature backbones we understand out-of-the-box
SUPPORTED_BACKBONES = {
    "vision": ["clip", "dinov2", "resnet50", "vjepa"],
    "audio": ["wav2vec2", "hubert", "encodec"],
    "text":  ["llama", "gpt2", "bert", "sonar"],
}

VALID_ACTIONS = (
    "setup",
    "list",
    "info",
    "download",
    "preprocess",
    "features",
    "align",
    "encode",
    "decode",
    "evaluate",
    "leaderboard",
    "cache",
    "help",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(text: str) -> dict:
    return {"status": "success", "content": [{"text": text}]}


def _err(text: str) -> dict:
    return {"status": "error", "content": [{"text": text}]}


def _run(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> tuple[int, str]:
    """Run a shell command, capturing output."""
    if dry_run:
        return 0, f"[dry-run] {' '.join(cmd)}"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        return 124, f"timeout: {' '.join(cmd)}"
    except Exception as e:  # pragma: no cover
        return 1, f"exec error: {e}"


def _ensure_dirs() -> None:
    NEURALSET_CACHE.mkdir(parents=True, exist_ok=True)
    NEURALSET_RESULTS.mkdir(parents=True, exist_ok=True)
    NEURALSET_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)


def _is_installed() -> bool:
    return (NEURALSET_REPO_DIR / "pyproject.toml").exists() or (
        NEURALSET_REPO_DIR / "setup.py"
    ).exists()


def _dataset_dir(dataset: str) -> Path:
    return NEURALSET_CACHE / "datasets" / dataset


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _setup(force: bool = False, dry_run: bool = False) -> dict:
    _ensure_dirs()

    if _is_installed() and not force:
        return _ok(f"✅ neuralset already installed at {NEURALSET_REPO_DIR}")

    if NEURALSET_REPO_DIR.exists() and force:
        shutil.rmtree(NEURALSET_REPO_DIR, ignore_errors=True)

    rc, out = _run(
        ["git", "clone", "--depth", "1", NEURALSET_REPO_URL, str(NEURALSET_REPO_DIR)],
        dry_run=dry_run,
    )
    if rc != 0:
        return _err(f"clone failed:\n{out}")

    # Install in editable mode if pyproject exists
    if (NEURALSET_REPO_DIR / "pyproject.toml").exists():
        rc, out = _run(
            ["pip", "install", "-e", str(NEURALSET_REPO_DIR)], dry_run=dry_run
        )
        if rc != 0:
            return _err(f"pip install failed:\n{out[-2000:]}")

    return _ok(
        f"🧠 neuralset setup complete\n"
        f"  repo:   {NEURALSET_REPO_DIR}\n"
        f"  cache:  {NEURALSET_CACHE}\n"
        f"  modalities: {', '.join(SUPPORTED_MODALITIES)}"
    )


def _list(modality: str | None = None) -> dict:
    """List datasets (best-effort: reads a manifest if present, else static list)."""
    manifest = NEURALSET_REPO_DIR / "neuralset" / "datasets" / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            items = data.get("datasets", [])
            if modality:
                items = [d for d in items if d.get("modality") == modality]
            lines = [f"📚 datasets ({len(items)}):"]
            for d in items[:100]:
                lines.append(
                    f"  • {d.get('name'):<32} {d.get('modality'):<8} "
                    f"subj={d.get('n_subjects','?')} stim={d.get('stimulus','?')}"
                )
            return _ok("\n".join(lines))
        except Exception as e:
            return _err(f"manifest parse error: {e}")

    # Fallback: known NeuralSet-bundled datasets (representative)
    fallback = [
        ("nsd",             "fmri",   8,   "images"),
        ("things-eeg2",     "eeg",    10,  "images"),
        ("things-meg",      "meg",    4,   "images"),
        ("narratives",      "fmri",   345, "speech"),
        ("broderick2019",   "eeg",    19,  "speech"),
        ("gwilliams2022",   "meg",    27,  "speech"),
        ("pereira2018",     "fmri",   15,  "text"),
        ("hp-fmri",         "fmri",   8,   "text"),
        ("mindbigdata",     "eeg",    1,   "digits"),
        ("dandi-ibl",       "spikes", 139, "behavior"),
    ]
    if modality:
        fallback = [d for d in fallback if d[1] == modality]
    lines = [f"📚 datasets (fallback, {len(fallback)}):"]
    for name, mod, n, stim in fallback:
        lines.append(f"  • {name:<20} {mod:<8} subj={n:<4} stim={stim}")
    lines.append("\n(run action='setup' then action='list' for live manifest)")
    return _ok("\n".join(lines))


def _info(dataset: str) -> dict:
    if not dataset:
        return _err("dataset required")
    ddir = _dataset_dir(dataset)
    meta_file = ddir / "meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            return _ok(f"ℹ️  {dataset}\n{json.dumps(meta, indent=2)}")
        except Exception as e:
            return _err(f"meta parse error: {e}")
    return _ok(
        f"ℹ️  {dataset}: not yet downloaded.\n"
        f"   run action='download', dataset='{dataset}'"
    )


def _download(
    dataset: str, subjects: list | None = None, dry_run: bool = False
) -> dict:
    if not dataset:
        return _err("dataset required")
    _ensure_dirs()
    ddir = _dataset_dir(dataset)
    ddir.mkdir(parents=True, exist_ok=True)

    # Prefer neuralset CLI if available
    cli = shutil.which("neuralset")
    if cli:
        cmd = [cli, "download", "--dataset", dataset, "--out", str(ddir)]
        if subjects:
            cmd += ["--subjects", ",".join(str(s) for s in subjects)]
        rc, out = _run(cmd, dry_run=dry_run)
        status = "✅" if rc == 0 else "❌"
        return _ok(f"{status} download {dataset}\n{out[-1500:]}")

    return _err(
        "neuralset CLI not found. run action='setup' first, "
        "or install manually then retry."
    )


def _preprocess(dataset: str, config: str | None = None, dry_run: bool = False) -> dict:
    if not dataset:
        return _err("dataset required")
    cli = shutil.which("neuralset")
    if not cli:
        return _err("neuralset CLI not found — run action='setup' first")
    cmd = [cli, "preprocess", "--dataset", dataset]
    if config:
        cmd += ["--config", config]
    rc, out = _run(cmd, cwd=NEURALSET_REPO_DIR, dry_run=dry_run)
    status = "✅" if rc == 0 else "❌"
    return _ok(f"{status} preprocess {dataset}\n{out[-1500:]}")


def _features(
    dataset: str,
    backbone: str,
    layer: str | None = None,
    dry_run: bool = False,
) -> dict:
    if not dataset or not backbone:
        return _err("dataset and backbone required")
    cli = shutil.which("neuralset")
    if not cli:
        return _err("neuralset CLI not found — run action='setup' first")
    cmd = [cli, "features", "--dataset", dataset, "--backbone", backbone]
    if layer:
        cmd += ["--layer", layer]
    rc, out = _run(cmd, cwd=NEURALSET_REPO_DIR, dry_run=dry_run)
    status = "✅" if rc == 0 else "❌"
    return _ok(f"{status} features {dataset}/{backbone}\n{out[-1500:]}")


def _align(dataset: str, backbone: str, dry_run: bool = False) -> dict:
    if not dataset or not backbone:
        return _err("dataset and backbone required")
    cli = shutil.which("neuralset")
    if not cli:
        return _err("neuralset CLI not found — run action='setup' first")
    rc, out = _run(
        [cli, "align", "--dataset", dataset, "--backbone", backbone],
        cwd=NEURALSET_REPO_DIR,
        dry_run=dry_run,
    )
    status = "✅" if rc == 0 else "❌"
    return _ok(f"{status} align {dataset}/{backbone}\n{out[-1500:]}")


def _encode(
    dataset: str,
    backbone: str,
    model: str = "ridge",
    dry_run: bool = False,
) -> dict:
    if not dataset or not backbone:
        return _err("dataset and backbone required")
    cli = shutil.which("neuralset")
    if not cli:
        return _err("neuralset CLI not found — run action='setup' first")
    rc, out = _run(
        [
            cli, "encode",
            "--dataset", dataset,
            "--backbone", backbone,
            "--model", model,
        ],
        cwd=NEURALSET_REPO_DIR,
        dry_run=dry_run,
    )
    status = "✅" if rc == 0 else "❌"
    return _ok(f"{status} encode {dataset}/{backbone}/{model}\n{out[-1500:]}")


def _decode(
    dataset: str,
    target: str = "stimulus",
    model: str = "ridge",
    dry_run: bool = False,
) -> dict:
    if not dataset:
        return _err("dataset required")
    cli = shutil.which("neuralset")
    if not cli:
        return _err("neuralset CLI not found — run action='setup' first")
    rc, out = _run(
        [
            cli, "decode",
            "--dataset", dataset,
            "--target", target,
            "--model", model,
        ],
        cwd=NEURALSET_REPO_DIR,
        dry_run=dry_run,
    )
    status = "✅" if rc == 0 else "❌"
    return _ok(f"{status} decode {dataset}/{target}/{model}\n{out[-1500:]}")


def _evaluate(
    dataset: str,
    backbone: str | None = None,
    metric: str = "pearson",
    dry_run: bool = False,
) -> dict:
    if not dataset:
        return _err("dataset required")
    cli = shutil.which("neuralset")
    if not cli:
        return _err("neuralset CLI not found — run action='setup' first")
    cmd = [cli, "evaluate", "--dataset", dataset, "--metric", metric]
    if backbone:
        cmd += ["--backbone", backbone]
    rc, out = _run(cmd, cwd=NEURALSET_REPO_DIR, dry_run=dry_run)
    status = "✅" if rc == 0 else "❌"
    return _ok(f"{status} evaluate {dataset}\n{out[-1500:]}")


def _leaderboard(action: str = "view", submission: str | None = None) -> dict:
    if action == "view":
        # Try reading cached leaderboard snapshot
        snap = NEURALSET_RESULTS / "leaderboard.json"
        if snap.exists():
            try:
                data = json.loads(snap.read_text())
                lines = ["🏆 leaderboard (cached):"]
                for row in data[:20]:
                    lines.append(
                        f"  {row.get('rank','?'):>3}. {row.get('model'):<24} "
                        f"{row.get('score'):.4f}  ({row.get('dataset')})"
                    )
                return _ok("\n".join(lines))
            except Exception as e:
                return _err(f"leaderboard parse error: {e}")
        return _ok(
            "🏆 no cached leaderboard. visit:\n"
            "   https://facebookresearch.github.io/neuroai/neuralset/"
        )
    if action == "submit":
        if not submission:
            return _err("submission path required for submit")
        p = Path(submission)
        if not p.exists():
            return _err(f"submission not found: {p}")
        return _ok(
            f"📤 submission ready: {p}\n"
            f"(manual upload required at project site)"
        )
    return _err(f"unknown leaderboard action: {action}")


def _cache(action: str = "stats") -> dict:
    _ensure_dirs()
    if action == "stats":
        total = 0
        per_ds = {}
        ds_root = NEURALSET_CACHE / "datasets"
        if ds_root.exists():
            for d in ds_root.iterdir():
                if d.is_dir():
                    size = sum(
                        f.stat().st_size for f in d.rglob("*") if f.is_file()
                    )
                    per_ds[d.name] = size
                    total += size
        lines = [f"💾 cache @ {NEURALSET_CACHE}", f"   total: {total/1e9:.2f} GB"]
        for name, size in sorted(per_ds.items(), key=lambda x: -x[1])[:20]:
            lines.append(f"   • {name:<24} {size/1e9:>7.2f} GB")
        return _ok("\n".join(lines))
    if action == "list":
        ds_root = NEURALSET_CACHE / "datasets"
        items = [d.name for d in ds_root.iterdir() if d.is_dir()] if ds_root.exists() else []
        return _ok(f"cached datasets ({len(items)}): {', '.join(items) or 'none'}")
    if action == "clear":
        if NEURALSET_CACHE.exists():
            shutil.rmtree(NEURALSET_CACHE)
            _ensure_dirs()
        return _ok("🧹 cache cleared")
    return _err(f"unknown cache action: {action}")


def _help() -> dict:
    lines = [
        "🧠 neuralset — neuro-AI pipeline tool",
        "",
        "Pipeline actions:",
        "  setup        clone + install neuralset",
        "  list         list datasets (optional modality filter)",
        "  info         dataset metadata",
        "  download     fetch raw recordings + stimuli",
        "  preprocess   filter / epoch / align",
        "  features     extract stimulus features (vision/audio/text)",
        "  align        align neural signal with features",
        "  encode       fit neural = f(features)",
        "  decode       fit features = f(neural)",
        "  evaluate     score against benchmark",
        "  leaderboard  view / submit",
        "  cache        stats / list / clear",
        "",
        f"Modalities: {', '.join(SUPPORTED_MODALITIES)}",
        f"Backbones:  {json.dumps(SUPPORTED_BACKBONES)}",
        f"Cache:      {NEURALSET_CACHE}",
        f"Repo dir:   {NEURALSET_REPO_DIR}",
        "",
        "Example e2e:",
        "  neuralset(action='setup')",
        "  neuralset(action='download', dataset='things-eeg2')",
        "  neuralset(action='preprocess', dataset='things-eeg2')",
        "  neuralset(action='features', dataset='things-eeg2', backbone='clip')",
        "  neuralset(action='encode',   dataset='things-eeg2', backbone='clip')",
        "  neuralset(action='evaluate', dataset='things-eeg2', backbone='clip')",
    ]
    return _ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Tool entrypoint
# ---------------------------------------------------------------------------


@tool
def neuralset(
    action: str = "help",
    dataset: str | None = None,
    modality: str | None = None,
    backbone: str | None = None,
    layer: str | None = None,
    model: str = "ridge",
    target: str = "stimulus",
    metric: str = "pearson",
    subjects: list | None = None,
    config: str | None = None,
    submission: str | None = None,
    sub_action: str = "stats",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """🧠 NeuralSet — wrap the Meta FAIR neuro-AI pipeline as a single DevDuck tool.

    Covers dataset discovery, download, preprocessing, feature extraction,
    neural encoding / decoding, evaluation, and leaderboard interaction
    across EEG / MEG / fMRI / iEEG / fNIRS / Spikes.

    Args:
        action: One of `setup, list, info, download, preprocess, features,
            align, encode, decode, evaluate, leaderboard, cache, help`.
        dataset: Dataset short-name (e.g. "things-eeg2", "nsd", "narratives").
        modality: Optional modality filter for `list` (eeg/meg/fmri/ieeg/fnirs/spikes).
        backbone: Stimulus-feature backbone (e.g. "clip", "wav2vec2", "llama").
        layer: Optional backbone layer (e.g. "visual.transformer.resblocks.11").
        model: Encoding/decoding model family (default "ridge").
        target: Decoding target ("stimulus" / "category" / "embedding").
        metric: Evaluation metric ("pearson" / "r2" / "accuracy").
        subjects: Optional subject subset (list of ints or strings).
        config: Optional config preset name for preprocess.
        submission: Path to leaderboard submission tarball.
        sub_action: Used by `cache` and `leaderboard` (stats/list/clear/view/submit).
        force: Re-run setup / overwrite.
        dry_run: Print the command without executing.

    Returns:
        Standard Strands dict: {status, content:[{text}]}.

    Example:
        neuralset(action="setup")
        neuralset(action="list", modality="eeg")
        neuralset(action="encode", dataset="things-eeg2", backbone="clip")
        neuralset(action="cache", sub_action="stats")
    """
    if action not in VALID_ACTIONS:
        return _err(f"unknown action '{action}'. valid: {', '.join(VALID_ACTIONS)}")

    try:
        if action == "help":
            return _help()
        if action == "setup":
            return _setup(force=force, dry_run=dry_run)
        if action == "list":
            if modality and modality not in SUPPORTED_MODALITIES:
                return _err(
                    f"unknown modality '{modality}'. "
                    f"valid: {', '.join(SUPPORTED_MODALITIES)}"
                )
            return _list(modality=modality)
        if action == "info":
            return _info(dataset)
        if action == "download":
            return _download(dataset, subjects=subjects, dry_run=dry_run)
        if action == "preprocess":
            return _preprocess(dataset, config=config, dry_run=dry_run)
        if action == "features":
            return _features(dataset, backbone, layer=layer, dry_run=dry_run)
        if action == "align":
            return _align(dataset, backbone, dry_run=dry_run)
        if action == "encode":
            return _encode(dataset, backbone, model=model, dry_run=dry_run)
        if action == "decode":
            return _decode(dataset, target=target, model=model, dry_run=dry_run)
        if action == "evaluate":
            return _evaluate(dataset, backbone=backbone, metric=metric, dry_run=dry_run)
        if action == "leaderboard":
            return _leaderboard(action=sub_action, submission=submission)
        if action == "cache":
            return _cache(action=sub_action)
    except Exception as e:  # pragma: no cover
        return _err(f"neuralset error: {e}")

    return _err("unreachable")
