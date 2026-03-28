"""
🎮 RL & ML Tools for DevDuck — Learn, Train, Fine-tune ANYTHING.

Three domains in one tool:

1. REINFORCEMENT LEARNING (Stable-Baselines3 + Gymnasium)
   - Train RL agents on any env (PPO, A2C, DQN, SAC, TD3, DDPG)
   - Create custom envs from reward functions
   - Visual debugging: renders frames as native images back to the agent
   - CNN/Multi-input policies for image-based envs
   - Sweep, compare, curriculum learning

2. LLM FINE-TUNING (Transformers + PEFT/LoRA + TRL)
   - Fine-tune any HuggingFace model with LoRA/QLoRA
   - SFT (supervised fine-tuning) with custom datasets
   - DPO/RLHF preference tuning
   - Push to HuggingFace Hub

3. CLASSICAL ML (scikit-learn patterns)
   - Train sklearn models from CSV/JSON data
   - Auto model selection
   - Evaluation with metrics

Examples:
    # RL: Train CartPole
    rl(action="train", env_id="CartPole-v1", algorithm="PPO", total_timesteps=50000)

    # RL: Create custom env + train
    rl(action="create_env", env_name="snake", reward_code="...", obs_dim=(10,10,3), act_dim=4, act_type="discrete")
    rl(action="train", env_id="custom:snake", algorithm="DQN", policy_type="CnnPolicy")

    # RL: Visual debug — get rendered frame as image
    rl(action="render_frame", env_id="CartPole-v1", model_path="rl_models/.../best_model")

    # LLM: Fine-tune with LoRA
    rl(action="finetune", model_id="meta-llama/Llama-3.2-1B", dataset_id="tatsu-lab/alpaca", method="lora")

    # LLM: SFT with custom data
    rl(action="sft", model_id="Qwen/Qwen2.5-0.5B", dataset_path="./my_data.jsonl")
"""

import io
import json
import os
import sys
import time
import threading
import tempfile
import importlib
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from strands import tool

# ─── Constants ───────────────────────────────────────────────────────────────

# CRITICAL: Prevent pygame/SDL2 from initializing display on non-main threads.
# On macOS, pygame.display.init() MUST run on the main thread (AppKit requirement).
# Since SB3 callbacks and our tool calls run on background threads, we force SDL
# to use a dummy video driver. This avoids the NSInternalInconsistencyException
# crash: "API misuse: setting the main menu on a non-main thread."
# We exclusively use rgb_array + PIL for rendering — no pygame window needed.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

RL_MODELS_DIR = Path(os.getenv("DEVDUCK_RL_MODELS_DIR", "./rl_models"))
RL_CUSTOM_ENVS_DIR = Path(os.getenv("DEVDUCK_RL_ENVS_DIR", "./rl_envs"))
RL_LOGS_DIR = Path(os.getenv("DEVDUCK_RL_LOGS_DIR", "./rl_logs"))
ML_MODELS_DIR = Path(os.getenv("DEVDUCK_ML_MODELS_DIR", "./ml_models"))

# Supported RL algorithms
ALGORITHMS = {
    "PPO": "stable_baselines3.PPO",
    "A2C": "stable_baselines3.A2C",
    "DQN": "stable_baselines3.DQN",
    "SAC": "stable_baselines3.SAC",
    "TD3": "stable_baselines3.TD3",
    "DDPG": "stable_baselines3.DDPG",
}

# SB3 policy types
POLICY_TYPES = {
    "mlp": "MlpPolicy",
    "cnn": "CnnPolicy",
    "multi": "MultiInputPolicy",
    "MlpPolicy": "MlpPolicy",
    "CnnPolicy": "CnnPolicy",
    "MultiInputPolicy": "MultiInputPolicy",
}

# ─── Custom Environment Registry ────────────────────────────────────────────

_custom_envs: Dict[str, Any] = {}


def _get_algorithm_class(name: str):
    """Dynamically import an SB3 algorithm class."""
    name = name.upper()
    if name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {name}. Available: {list(ALGORITHMS.keys())}"
        )
    module_path, class_name = ALGORITHMS[name].rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _resolve_policy_type(policy_type: str, env) -> str:
    """Resolve policy type string to SB3 policy class name.
    Auto-detects CNN if observation space is image-like."""
    import gymnasium as gym

    if policy_type:
        resolved = POLICY_TYPES.get(policy_type, policy_type)
        return resolved

    # Auto-detect from observation space
    obs_space = env.observation_space if hasattr(env, "observation_space") else None
    if obs_space is None:
        return "MlpPolicy"

    if isinstance(obs_space, gym.spaces.Box):
        if len(obs_space.shape) == 3:
            # Image-like: (H, W, C) or (C, H, W)
            return "CnnPolicy"
        elif len(obs_space.shape) >= 2:
            return "CnnPolicy"
    elif isinstance(obs_space, gym.spaces.Dict):
        return "MultiInputPolicy"

    return "MlpPolicy"


def _make_env(env_id: str, seed: int = None, render_mode: str = None):
    """Create a Gymnasium environment by ID, supporting custom envs."""
    import gymnasium as gym

    if env_id.startswith("custom:"):
        custom_name = env_id.split(":", 1)[1]
        if custom_name not in _custom_envs:
            env_file = RL_CUSTOM_ENVS_DIR / f"{custom_name}.py"
            if env_file.exists():
                _load_custom_env(custom_name, env_file)
            else:
                raise ValueError(
                    f"Custom env '{custom_name}' not found. "
                    f"Create it first with action='create_env'"
                )
        env_cls = _custom_envs[custom_name]
        env = env_cls(render_mode=render_mode)
    else:
        env = gym.make(env_id, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)
    return env


def _load_custom_env(name: str, path: Path):
    """Load a custom environment class from a .py file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"rl_env_{name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import gymnasium as gym

    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, gym.Env) and attr is not gym.Env:
            _custom_envs[name] = attr
            return

    raise ValueError(f"No gymnasium.Env subclass found in {path}")


def _render_env_to_image(env, model=None, seed=42, steps=0):
    """Render an environment frame and return as native image content block.

    Returns a content block with {"image": {"format": "png", "source": {"bytes": ...}}}
    that the model can see directly.
    """
    import numpy as np

    # If model provided, step through some actions first
    if model and steps > 0:
        obs, _ = env.reset(seed=seed)
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=seed)

    # Render the frame
    frame = env.render()

    if frame is None:
        return None

    # Convert numpy array to PNG bytes
    from PIL import Image

    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        return None

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    return {"image": {"format": "png", "source": {"bytes": png_bytes}}}


def _render_multiple_frames(env, model, seed=42, n_frames=4, frame_interval=10):
    """Render multiple frames as a grid image for visual debugging.

    Returns a single image showing N frames side by side.
    """
    import numpy as np
    from PIL import Image

    frames = []
    obs, _ = env.reset(seed=seed)
    step_count = 0

    for frame_idx in range(n_frames):
        # Step forward
        for _ in range(frame_interval):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if terminated or truncated:
                obs, _ = env.reset(seed=seed)

        frame = env.render()
        if frame is not None and isinstance(frame, np.ndarray):
            frames.append(frame)

    if not frames:
        return None

    # Create grid: all frames side by side
    max_h = max(f.shape[0] for f in frames)
    total_w = sum(f.shape[1] for f in frames)

    grid = np.zeros((max_h, total_w, 3), dtype=np.uint8)
    x_offset = 0
    for f in frames:
        h, w = f.shape[:2]
        if f.ndim == 2:  # Grayscale
            f = np.stack([f, f, f], axis=-1)
        grid[:h, x_offset : x_offset + w, :] = f[:, :, :3]
        x_offset += w

    img = Image.fromarray(grid)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    return {"image": {"format": "png", "source": {"bytes": png_bytes}}}


# ─── Training Progress Callback ─────────────────────────────────────────────


class _ProgressCallback:
    """Simple training progress tracker."""

    def __init__(self, total_timesteps: int, print_freq: int = 10000):
        from stable_baselines3.common.callbacks import BaseCallback

        self.total = total_timesteps
        self.print_freq = print_freq
        self.start_time = time.time()

        class Inner(BaseCallback):
            def __init__(inner_self, outer=self, verbose=0):
                super().__init__(verbose)
                inner_self.outer = outer

            def _on_step(inner_self) -> bool:
                if inner_self.num_timesteps % inner_self.outer.print_freq == 0:
                    elapsed = time.time() - inner_self.outer.start_time
                    fps = inner_self.num_timesteps / max(elapsed, 1)
                    pct = (inner_self.num_timesteps / inner_self.outer.total) * 100
                    print(
                        f"  ⏳ {inner_self.num_timesteps:,}/{inner_self.outer.total:,} "
                        f"({pct:.0f}%) | {fps:.0f} fps | {elapsed:.0f}s",
                        flush=True,
                    )
                return True

        self.callback_cls = Inner

    def get(self):
        return self.callback_cls()


# ─── RL Actions ──────────────────────────────────────────────────────────────


def _action_train(
    env_id: str,
    algorithm: str = "PPO",
    total_timesteps: int = 100000,
    seed: int = 42,
    hyperparams: Dict[str, Any] = None,
    save_name: str = None,
    n_envs: int = 1,
    eval_freq: int = 10000,
    device: str = "auto",
    policy_type: str = None,
) -> Dict[str, Any]:
    """Train an RL agent."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    import gymnasium as gym

    algo_cls = _get_algorithm_class(algorithm)

    # Create vectorized training env
    def make_train_env(i):
        def _init():
            env = _make_env(env_id, seed=seed + i)
            return Monitor(env)

        return _init

    if n_envs > 1:
        try:
            import multiprocessing

            multiprocessing.set_start_method("forkserver", force=True)
            train_env = SubprocVecEnv([make_train_env(i) for i in range(n_envs)])
        except Exception:
            print(f"  ⚠ SubprocVecEnv failed, using DummyVecEnv with {n_envs} envs")
            train_env = DummyVecEnv([make_train_env(i) for i in range(n_envs)])
    else:
        train_env = DummyVecEnv([make_train_env(0)])

    # Create eval env
    eval_env = DummyVecEnv([lambda: Monitor(_make_env(env_id, seed=seed + 1000))])

    # Model save path
    name = save_name or f"{env_id.replace(':', '_').replace('/', '_')}_{algorithm}"
    model_dir = RL_MODELS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = RL_LOGS_DIR / name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve policy type (auto-detect CNN for image observations)
    test_env = _make_env(env_id, seed=seed)
    action_space = test_env.action_space
    obs_space = test_env.observation_space
    resolved_policy = _resolve_policy_type(policy_type, test_env)
    test_env.close()

    # Validate algorithm vs action space compatibility
    if algorithm.upper() == "DQN" and not isinstance(action_space, gym.spaces.Discrete):
        return {
            "status": "error",
            "content": [
                {
                    "text": f"❌ DQN requires discrete actions, but {env_id} has {type(action_space).__name__}. Use PPO, SAC, TD3, or DDPG instead."
                }
            ],
        }

    if algorithm.upper() in ("SAC", "TD3", "DDPG") and isinstance(
        action_space, gym.spaces.Discrete
    ):
        return {
            "status": "error",
            "content": [
                {
                    "text": f"❌ {algorithm} requires continuous actions, but {env_id} has Discrete. Use PPO, A2C, or DQN instead."
                }
            ],
        }

    # Hyperparameters
    hp = hyperparams or {}
    hp.setdefault("verbose", 1)
    hp.setdefault("device", device)
    hp.setdefault("seed", seed)

    print(f"\n🎮 Training {algorithm} on {env_id}")
    print(f"   Policy: {resolved_policy}")
    print(f"   Steps: {total_timesteps:,} | Envs: {n_envs} | Seed: {seed}")
    print(f"   Save: {model_dir}")
    print(f"   Action space: {action_space}")
    print(f"   Obs space: {obs_space}")
    if hp:
        filtered_hp = {
            k: str(v) for k, v in hp.items() if k not in ("verbose", "device", "seed")
        }
        if filtered_hp:
            print(f"   Hyperparams: {json.dumps(filtered_hp, indent=2)}")
    print()

    # Create model
    start_time = time.time()
    # Skip tensorboard — it deadlocks on macOS Python 3.13 in subprocess threads
    tb_log = None
    if os.getenv("DEVDUCK_RL_TENSORBOARD", "false").lower() == "true":
        tb_log = str(log_dir)
    model = algo_cls(resolved_policy, train_env, tensorboard_log=tb_log, **hp)

    # Callbacks
    progress = _ProgressCallback(total_timesteps)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress.get(), eval_callback],
        progress_bar=False,
    )

    # Save final model
    final_path = model_dir / "final_model"
    model.save(str(final_path))

    elapsed = time.time() - start_time
    fps = total_timesteps / max(elapsed, 1)

    # Load eval results
    eval_results = {}
    eval_log = log_dir / "evaluations.npz"
    if eval_log.exists():
        import numpy as np

        data = np.load(str(eval_log))
        if "results" in data:
            results = data["results"]
            eval_results = {
                "best_mean_reward": float(results.mean(axis=1).max()),
                "final_mean_reward": (
                    float(results[-1].mean()) if len(results) > 0 else 0
                ),
                "best_std": float(results[results.mean(axis=1).argmax()].std()),
            }

    # Render a frame from the best model for visual feedback
    visual_content = []
    try:
        best_model_path = model_dir / "best" / "best_model"
        if best_model_path.with_suffix(".zip").exists():
            vis_env = _make_env(env_id, seed=seed, render_mode="rgb_array")
            best_model = algo_cls.load(str(best_model_path))
            img_block = _render_multiple_frames(
                vis_env, best_model, seed=seed, n_frames=4, frame_interval=20
            )
            vis_env.close()
            if img_block:
                visual_content.append(img_block)
    except Exception as e:
        print(f"  ⚠ Could not render visual: {e}")

    train_env.close()
    eval_env.close()

    summary = (
        f"✅ Training complete!\n"
        f"   Algorithm: {algorithm} | Policy: {resolved_policy}\n"
        f"   Environment: {env_id}\n"
        f"   Total steps: {total_timesteps:,}\n"
        f"   Time: {elapsed:.1f}s ({fps:.0f} fps)\n"
        f"   Best mean reward: {eval_results.get('best_mean_reward', 'N/A')}\n"
        f"   Final mean reward: {eval_results.get('final_mean_reward', 'N/A')}\n"
        f"   Model saved: {final_path}\n"
        f"   Best model: {model_dir / 'best' / 'best_model'}\n"
        f"   TensorBoard: tensorboard --logdir {log_dir}"
    )

    content = [
        {"text": summary},
        {
            "json": {
                "model_path": str(final_path),
                "best_model_path": str(model_dir / "best" / "best_model"),
                "log_dir": str(log_dir),
                "elapsed_seconds": elapsed,
                "fps": fps,
                "policy_type": resolved_policy,
                **eval_results,
            }
        },
    ]
    content.extend(visual_content)

    return {"status": "success", "content": content}


def _action_eval(
    model_path: str,
    env_id: str = None,
    algorithm: str = None,
    n_episodes: int = 20,
    deterministic: bool = True,
    seed: int = 42,
    render: bool = False,
) -> Dict[str, Any]:
    """Evaluate a trained model."""
    import numpy as np

    if algorithm is None:
        for algo_name in ALGORITHMS:
            if algo_name in str(model_path).upper():
                algorithm = algo_name
                break
        if algorithm is None:
            algorithm = "PPO"

    if env_id is None:
        model_dir = Path(model_path).parent
        for part in [model_dir.name, model_dir.parent.name]:
            for algo_name in ALGORITHMS:
                if part.endswith(f"_{algo_name}"):
                    env_id = part[: -(len(algo_name) + 1)]
                    break
            if env_id:
                break

    if env_id is None:
        return {
            "status": "error",
            "content": [
                {"text": "❌ Could not auto-detect env_id. Please specify it."}
            ],
        }

    algo_cls = _get_algorithm_class(algorithm)
    model = algo_cls.load(model_path)

    # Always use rgb_array — never "human" mode (pygame crashes on macOS non-main threads)
    render_mode = "rgb_array" if render else None
    env = _make_env(env_id, seed=seed, render_mode=render_mode)

    print(f"\n🎯 Evaluating {algorithm} on {env_id} ({n_episodes} episodes)")

    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(
                f"  Episode {ep + 1}/{n_episodes}: reward={total_reward:.2f}, steps={steps}"
            )

    env.close()

    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)

    summary = (
        f"✅ Evaluation complete!\n"
        f"   Episodes: {n_episodes}\n"
        f"   Mean reward: {rewards.mean():.2f} ± {rewards.std():.2f}\n"
        f"   Min/Max reward: {rewards.min():.2f} / {rewards.max():.2f}\n"
        f"   Mean episode length: {lengths.mean():.1f} ± {lengths.std():.1f}\n"
        f"   Success rate (reward > 0): {(rewards > 0).mean() * 100:.1f}%"
    )

    content = [
        {"text": summary},
        {
            "json": {
                "mean_reward": float(rewards.mean()),
                "std_reward": float(rewards.std()),
                "min_reward": float(rewards.min()),
                "max_reward": float(rewards.max()),
                "mean_length": float(lengths.mean()),
                "success_rate": float((rewards > 0).mean()),
                "all_rewards": [float(r) for r in rewards],
            }
        },
    ]

    # If render requested, return visual frames as images
    if render and render_mode == "rgb_array":
        try:
            vis_env = _make_env(env_id, seed=seed, render_mode="rgb_array")
            grid_img = _render_multiple_frames(
                vis_env, model, seed=seed, n_frames=4, frame_interval=20
            )
            vis_env.close()
            if grid_img:
                content.append(grid_img)
        except Exception:
            pass

    return {
        "status": "success",
        "content": content,
    }


def _action_play(
    model_path: str,
    env_id: str = None,
    algorithm: str = None,
    n_episodes: int = 3,
    seed: int = 42,
    record_video: bool = False,
    video_path: str = None,
    render: bool = False,
) -> Dict[str, Any]:
    """Watch a trained agent play. Always returns frames as images (rgb_array).

    NOTE: render=True is ignored on macOS because pygame.display.init() crashes
    when called from non-main threads (AppKit requires main thread for UI).
    We always use rgb_array + PIL to avoid SDL2/pygame entirely.
    """

    if algorithm is None:
        for algo_name in ALGORITHMS:
            if algo_name in str(model_path).upper():
                algorithm = algo_name
                break
        if algorithm is None:
            algorithm = "PPO"

    if env_id is None:
        model_dir = Path(model_path).parent
        for part in [model_dir.name, model_dir.parent.name]:
            for algo_name in ALGORITHMS:
                if part.endswith(f"_{algo_name}"):
                    env_id = part[: -(len(algo_name) + 1)]
                    break
            if env_id:
                break

    if env_id is None:
        return {
            "status": "error",
            "content": [
                {"text": "❌ Could not auto-detect env_id. Please specify it."}
            ],
        }

    algo_cls = _get_algorithm_class(algorithm)
    model = algo_cls.load(model_path)

    if record_video:
        import gymnasium as gym
        from gymnasium.wrappers import RecordVideo

        vpath = video_path or f"rl_videos/{env_id}_{algorithm}"
        Path(vpath).mkdir(parents=True, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=vpath, episode_trigger=lambda x: True)
    else:
        # Always use rgb_array — never pygame "human" mode (crashes on macOS threads)
        env = _make_env(env_id, seed=seed, render_mode="rgb_array")

    print(f"\n🎬 Playing {algorithm} on {env_id} ({n_episodes} episodes)")

    content = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"  Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    # Render frames as images for the agent to see
    if not record_video:
        grid_img = _render_multiple_frames(
            env, model, seed=seed, n_frames=4, frame_interval=15
        )
        if grid_img:
            content.append(grid_img)

    env.close()

    text = f"✅ Played {n_episodes} episodes of {env_id}"
    if record_video:
        text += f"\n   Videos saved to: {vpath}"
    if render:
        text += f"\n   ℹ️ Live rendering skipped (macOS thread safety). Returning image frames instead."

    content.insert(0, {"text": text})
    return {"status": "success", "content": content}


def _action_render_frame(
    env_id: str,
    model_path: str = None,
    algorithm: str = None,
    seed: int = 42,
    steps: int = 50,
    n_frames: int = 4,
    frame_interval: int = 20,
) -> Dict[str, Any]:
    """Render frames from an environment and return as native images.

    This is the 'eyes' for the agent — it can see what the RL agent is doing.
    """
    if algorithm is None and model_path:
        for algo_name in ALGORITHMS:
            if algo_name in str(model_path).upper():
                algorithm = algo_name
                break
    if algorithm is None:
        algorithm = "PPO"

    env = _make_env(env_id, seed=seed, render_mode="rgb_array")

    content = []

    if model_path:
        # Render with trained model
        algo_cls = _get_algorithm_class(algorithm)
        model = algo_cls.load(model_path)

        grid_img = _render_multiple_frames(
            env, model, seed=seed, n_frames=n_frames, frame_interval=frame_interval
        )
        if grid_img:
            content.append(
                {
                    "text": f"📸 {n_frames} frames from {env_id} with trained {algorithm} agent (every {frame_interval} steps):"
                }
            )
            content.append(grid_img)
        else:
            content.append(
                {"text": "⚠ Could not render frames (env may not support rgb_array)"}
            )
    else:
        # Render with random actions
        import numpy as np

        obs, _ = env.reset(seed=seed)

        frame = env.render()
        if frame is not None and isinstance(frame, np.ndarray):
            from PIL import Image

            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            content.append(
                {"text": f"📸 Initial frame from {env_id} (no model, random state):"}
            )
            content.append(
                {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}}
            )
        else:
            content.append(
                {"text": "⚠ Could not render (env may not support rgb_array)"}
            )

    env.close()
    return {"status": "success", "content": content}


def _action_create_env(
    env_name: str,
    reward_code: str,
    obs_dim: Any = 4,
    act_dim: int = 1,
    act_type: str = "continuous",
    max_steps: int = 1000,
    description: str = "",
    reset_code: str = None,
    step_code: str = None,
) -> Dict[str, Any]:
    """Create a custom Gymnasium environment from Python code.

    obs_dim can be:
    - int: flat vector (e.g., 4)
    - tuple/list: image shape (e.g., (84, 84, 3) for CNN)
    """
    RL_CUSTOM_ENVS_DIR.mkdir(parents=True, exist_ok=True)

    # Handle obs_dim as tuple for image observations
    if isinstance(obs_dim, (list, tuple)):
        obs_shape = tuple(obs_dim)
        obs_shape_str = str(obs_shape)
        is_image = len(obs_shape) >= 2
    else:
        obs_shape = (int(obs_dim),)
        obs_shape_str = f"({int(obs_dim)},)"
        is_image = False

    if act_type == "continuous":
        action_space_code = (
            f"spaces.Box(low=-1.0, high=1.0, shape=({act_dim},), dtype=np.float32)"
        )
    else:
        action_space_code = f"spaces.Discrete({act_dim})"

    # Observation space
    if is_image:
        obs_space_code = (
            f"spaces.Box(low=0, high=255, shape={obs_shape_str}, dtype=np.uint8)"
        )
    else:
        obs_space_code = f"spaces.Box(low=-np.inf, high=np.inf, shape={obs_shape_str}, dtype=np.float32)"

    # Build the environment class
    if step_code and reset_code:
        env_code = f'''"""Custom RL Environment: {env_name}
{description}
Generated by DevDuck on {datetime.now().isoformat()}
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class {env_name.title().replace("_", "")}Env(gym.Env):
    """Custom environment: {env_name}"""

    metadata = {{"render_modes": ["human", "rgb_array"], "render_fps": 30}}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = {obs_space_code}
        self.action_space = {action_space_code}
        self.max_steps = {max_steps}
        self.current_step = 0
        self.state = np.zeros({obs_shape_str}, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
{_indent(reset_code, 8)}

    def step(self, action):
        self.current_step += 1
{_indent(step_code, 8)}

    def render(self):
        if self.render_mode == "rgb_array":
            state = self.state
            if state.ndim >= 2 and state.shape[-1] in (1, 3):
                return np.clip(state, 0, 255).astype(np.uint8)
            # Fallback: simple bar visualization
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            flat = state.flatten()
            for i, val in enumerate(flat[:64]):
                col = int(np.clip(abs(val) * 255, 0, 255))
                h = 64 // max(len(flat), 1)
                img[i * h:(i + 1) * h, :, :] = col
            return img
        return None
'''
    else:
        env_code = f'''"""Custom RL Environment: {env_name}
{description}
Generated by DevDuck on {datetime.now().isoformat()}
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# User-defined reward function
{reward_code}


class {env_name.title().replace("_", "")}Env(gym.Env):
    """Custom environment: {env_name} — auto-managed state with user reward function."""

    metadata = {{"render_modes": ["human", "rgb_array"], "render_fps": 30}}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = {obs_space_code}
        self.action_space = {action_space_code}
        self.max_steps = {max_steps}
        self.current_step = 0
        self.state = np.zeros({obs_shape_str}, dtype={"np.uint8" if is_image else "np.float32"})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self.np_random.{"integers(0, 256, size=" + obs_shape_str + ").astype(np.uint8)" if is_image else "uniform(low=-0.5, high=0.5, size=" + obs_shape_str + ").astype(np.float32)"}
        return self.state.copy(), {{}}

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action).flatten() if not isinstance(action, (int, np.integer)) else action

        next_state, rew, terminated = reward(self.state, action)
        self.state = np.asarray(next_state, dtype={"np.uint8" if is_image else "np.float32"})

        truncated = self.current_step >= self.max_steps
        return self.state.copy(), float(rew), bool(terminated), truncated, {{}}

    def render(self):
        if self.render_mode == "rgb_array":
            state = self.state
            if state.ndim >= 2 and state.shape[-1] in (1, 3):
                return np.clip(state, 0, 255).astype(np.uint8)
            if state.ndim >= 2:
                # 2D grid: normalize to grayscale image
                norm = ((state - state.min()) / max(state.max() - state.min(), 1e-8) * 255).astype(np.uint8)
                if norm.ndim == 2:
                    return np.stack([norm, norm, norm], axis=-1)
                return norm
            # 1D: bar visualization
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            flat = state.flatten()
            for i, val in enumerate(flat[:64]):
                col = int(np.clip(abs(val) * 255, 0, 255))
                h = 64 // max(len(flat), 1)
                img[i * h:(i + 1) * h, :, :] = col
            return img
        return None
'''

    # Save the env file
    env_file = RL_CUSTOM_ENVS_DIR / f"{env_name}.py"
    with open(env_file, "w") as f:
        f.write(env_code)

    # Load and validate
    try:
        _load_custom_env(env_name, env_file)

        test_env = _custom_envs[env_name](render_mode="rgb_array")
        obs, info = test_env.reset(seed=42)
        action = test_env.action_space.sample()
        obs2, rew, term, trunc, info = test_env.step(action)

        # Render initial frame
        content = []
        frame = test_env.render()
        test_env.close()

        policy_hint = "CnnPolicy" if is_image else "MlpPolicy"
        algo_hint = "DQN" if act_type == "discrete" else "PPO"

        summary = (
            f"✅ Custom environment '{env_name}' created and validated!\n"
            f"   File: {env_file}\n"
            f"   Obs space: {test_env.observation_space}\n"
            f"   Action space: {test_env.action_space}\n"
            f"   Max steps: {max_steps}\n"
            f"   Test step: obs_shape={obs2.shape}, reward={rew:.4f}\n"
            f"\n   Train with: rl(action='train', env_id='custom:{env_name}', algorithm='{algo_hint}', policy_type='{policy_hint}')"
        )
        content.append({"text": summary})

        # Add rendered frame
        if frame is not None:
            import numpy as np
            from PIL import Image

            if isinstance(frame, np.ndarray):
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                content.append({"text": "📸 Initial render:"})
                content.append(
                    {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}}
                )

        return {"status": "success", "content": content}

    except Exception as e:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"❌ Environment validation failed:\n{traceback.format_exc()}\n\nFile saved at {env_file} — fix and retry."
                }
            ],
        }


def _action_list_envs(category: str = None) -> Dict[str, Any]:
    """List available environments."""
    import gymnasium as gym

    all_envs = gym.envs.registry.keys()

    categories = {
        "classic": ["CartPole", "MountainCar", "Pendulum", "Acrobot", "LunarLander"],
        "box2d": ["BipedalWalker", "CarRacing", "LunarLander"],
        "mujoco": [
            "Ant",
            "HalfCheetah",
            "Hopper",
            "Humanoid",
            "Reacher",
            "Swimmer",
            "Walker2d",
            "InvertedPendulum",
        ],
        "atari": ["Breakout", "Pong", "SpaceInvaders"],
    }

    lines = ["🎮 Available Environments:\n"]

    if category and category in categories:
        keywords = categories[category]
        envs = [e for e in sorted(all_envs) if any(k in e for k in keywords)]
        lines.append(f"**{category.upper()}** ({len(envs)}):")
        for e in envs:
            lines.append(f"  • {e}")
    else:
        for cat, keywords in categories.items():
            cat_envs = [e for e in all_envs if any(k in e for k in keywords)]
            lines.append(
                f"**{cat.upper()}** ({len(cat_envs)}): {', '.join(sorted(cat_envs)[:5])}..."
            )

        lines.append(f"\n**Total registered**: {len(list(all_envs))}")

    # Custom envs
    custom_files = (
        list(RL_CUSTOM_ENVS_DIR.glob("*.py")) if RL_CUSTOM_ENVS_DIR.exists() else []
    )
    if custom_files or _custom_envs:
        lines.append(f"\n**CUSTOM** ({max(len(custom_files), len(_custom_envs))}):")
        names = set(list(_custom_envs.keys()) + [f.stem for f in custom_files])
        for name in sorted(names):
            loaded = "✅" if name in _custom_envs else "💾"
            lines.append(f"  {loaded} custom:{name}")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _action_list_models() -> Dict[str, Any]:
    """List saved RL models."""
    lines = ["📦 Saved Models:\n"]

    # RL models
    if RL_MODELS_DIR.exists():
        lines.append("**RL Models:**")
        for model_dir in sorted(RL_MODELS_DIR.iterdir()):
            if not model_dir.is_dir():
                continue

            final = model_dir / "final_model.zip"
            best = model_dir / "best" / "best_model.zip"

            size_mb = 0
            if final.exists():
                size_mb = final.stat().st_size / (1024 * 1024)
            elif best.exists():
                size_mb = best.stat().st_size / (1024 * 1024)

            has_final = "✅" if final.exists() else "❌"
            has_best = "✅" if best.exists() else "❌"

            lines.append(f"  📁 **{model_dir.name}** ({size_mb:.1f} MB)")
            lines.append(f"     Final: {has_final} | Best: {has_best}")
    else:
        lines.append("No RL models yet.")

    # ML/LLM models
    if ML_MODELS_DIR.exists():
        lines.append("\n**ML/LLM Models:**")
        for model_dir in sorted(ML_MODELS_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            size_mb = sum(
                f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
            ) / (1024 * 1024)
            lines.append(f"  📁 **{model_dir.name}** ({size_mb:.1f} MB)")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _action_sweep(
    env_id: str,
    algorithm: str = "PPO",
    n_trials: int = 5,
    total_timesteps: int = 50000,
    seed: int = 42,
    policy_type: str = None,
) -> Dict[str, Any]:
    """Hyperparameter sweep."""
    import numpy as np

    search_spaces = {
        "PPO": [
            {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
            },
            {
                "learning_rate": 1e-3,
                "n_steps": 1024,
                "batch_size": 32,
                "n_epochs": 5,
                "gamma": 0.99,
            },
            {
                "learning_rate": 1e-4,
                "n_steps": 4096,
                "batch_size": 128,
                "n_epochs": 20,
                "gamma": 0.999,
            },
            {
                "learning_rate": 5e-4,
                "n_steps": 512,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.98,
            },
            {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 256,
                "n_epochs": 10,
                "gamma": 0.995,
            },
            {
                "learning_rate": 7e-4,
                "n_steps": 1024,
                "batch_size": 64,
                "n_epochs": 15,
                "gamma": 0.99,
            },
            {
                "learning_rate": 2e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.999,
                "ent_coef": 0.01,
            },
            {
                "learning_rate": 5e-4,
                "n_steps": 2048,
                "batch_size": 128,
                "n_epochs": 5,
                "gamma": 0.99,
                "clip_range": 0.1,
            },
        ],
        "A2C": [
            {"learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99},
            {"learning_rate": 1e-3, "n_steps": 8, "gamma": 0.995},
            {"learning_rate": 3e-4, "n_steps": 16, "gamma": 0.99},
            {"learning_rate": 5e-4, "n_steps": 5, "gamma": 0.999},
            {"learning_rate": 1e-4, "n_steps": 32, "gamma": 0.99},
        ],
        "DQN": [
            {
                "learning_rate": 1e-3,
                "buffer_size": 100000,
                "batch_size": 32,
                "gamma": 0.99,
            },
            {
                "learning_rate": 5e-4,
                "buffer_size": 50000,
                "batch_size": 64,
                "gamma": 0.99,
            },
            {
                "learning_rate": 1e-4,
                "buffer_size": 200000,
                "batch_size": 128,
                "gamma": 0.999,
            },
            {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "batch_size": 32,
                "gamma": 0.98,
            },
            {
                "learning_rate": 5e-4,
                "buffer_size": 100000,
                "batch_size": 64,
                "gamma": 0.99,
                "exploration_fraction": 0.2,
            },
        ],
        "SAC": [
            {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "batch_size": 256,
                "gamma": 0.99,
            },
            {
                "learning_rate": 1e-3,
                "buffer_size": 50000,
                "batch_size": 128,
                "gamma": 0.99,
            },
            {
                "learning_rate": 1e-4,
                "buffer_size": 300000,
                "batch_size": 256,
                "gamma": 0.999,
            },
            {
                "learning_rate": 5e-4,
                "buffer_size": 100000,
                "batch_size": 64,
                "gamma": 0.98,
            },
            {
                "learning_rate": 3e-4,
                "buffer_size": 200000,
                "batch_size": 256,
                "gamma": 0.99,
                "tau": 0.01,
            },
        ],
    }

    configs = search_spaces.get(algorithm.upper(), search_spaces["PPO"])[:n_trials]
    while len(configs) < n_trials:
        base = configs[np.random.randint(0, len(configs))]
        variant = {
            k: v * np.random.uniform(0.5, 2.0) if isinstance(v, float) else v
            for k, v in base.items()
        }
        configs.append(variant)

    print(f"\n🔍 Hyperparameter sweep: {algorithm} on {env_id}")
    print(f"   Trials: {n_trials} | Steps per trial: {total_timesteps:,}\n")

    results = []
    for i, hp in enumerate(configs):
        print(f"\n--- Trial {i + 1}/{n_trials} ---")
        print(
            f"   Config: {json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in hp.items()})}"
        )

        try:
            result = _action_train(
                env_id=env_id,
                algorithm=algorithm,
                total_timesteps=total_timesteps,
                seed=seed + i * 100,
                hyperparams=hp,
                save_name=f"sweep_{env_id.replace('/', '_')}_{algorithm}_trial{i}",
                policy_type=policy_type,
            )

            json_data = None
            for content in result.get("content", []):
                if "json" in content:
                    json_data = content["json"]
                    break

            mean_reward = json_data.get("best_mean_reward", 0) if json_data else 0
            results.append(
                {
                    "trial": i,
                    "config": hp,
                    "mean_reward": mean_reward,
                    "model_path": (
                        json_data.get("best_model_path", "") if json_data else ""
                    ),
                }
            )
            print(f"   → Mean reward: {mean_reward:.2f}")

        except Exception as e:
            print(f"   → Failed: {e}")
            results.append(
                {
                    "trial": i,
                    "config": hp,
                    "mean_reward": -float("inf"),
                    "error": str(e),
                }
            )

    best = max(results, key=lambda x: x["mean_reward"])

    lines = [
        f"\n🏆 Sweep Complete!\n",
        f"   Best trial: #{best['trial']} with mean reward {best['mean_reward']:.2f}",
        f"   Best config: {json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in best['config'].items()}, indent=4)}",
        f"   Best model: {best.get('model_path', 'N/A')}",
        f"\n   All results:",
    ]
    for r in sorted(results, key=lambda x: x["mean_reward"], reverse=True):
        status = "🏆" if r["trial"] == best["trial"] else "  "
        lines.append(f"   {status} Trial {r['trial']}: reward={r['mean_reward']:.2f}")

    return {
        "status": "success",
        "content": [
            {"text": "\n".join(lines)},
            {"json": {"best_trial": best, "all_results": results}},
        ],
    }


def _action_continue_training(
    model_path: str,
    env_id: str = None,
    algorithm: str = None,
    additional_timesteps: int = 100000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Continue training a saved model."""

    if algorithm is None:
        for algo_name in ALGORITHMS:
            if algo_name in str(model_path).upper():
                algorithm = algo_name
                break
        if algorithm is None:
            algorithm = "PPO"

    if env_id is None:
        model_dir = Path(model_path).parent
        for part in [model_dir.name, model_dir.parent.name]:
            for algo_name in ALGORITHMS:
                if part.endswith(f"_{algo_name}"):
                    env_id = part[: -(len(algo_name) + 1)]
                    break
            if env_id:
                break

    if env_id is None:
        return {
            "status": "error",
            "content": [
                {"text": "❌ Could not auto-detect env_id. Please specify it."}
            ],
        }

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    algo_cls = _get_algorithm_class(algorithm)
    env = DummyVecEnv([lambda: Monitor(_make_env(env_id, seed=seed))])

    print(f"\n🔄 Continuing training: {algorithm} on {env_id}")
    print(f"   Loading: {model_path}")
    print(f"   Additional steps: {additional_timesteps:,}\n")

    model = algo_cls.load(model_path, env=env)

    progress = _ProgressCallback(additional_timesteps)
    start_time = time.time()

    model.learn(
        total_timesteps=additional_timesteps,
        callback=[progress.get()],
        reset_num_timesteps=False,
    )

    model.save(model_path)
    elapsed = time.time() - start_time
    env.close()

    return {
        "status": "success",
        "content": [
            {
                "text": f"✅ Continued training for {additional_timesteps:,} more steps ({elapsed:.1f}s). Model saved to {model_path}"
            }
        ],
    }


def _action_compare(
    model_paths: List[str],
    env_id: str,
    n_episodes: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare multiple models on the same environment."""
    import numpy as np

    results = []

    for path in model_paths:
        algorithm = "PPO"
        for algo_name in ALGORITHMS:
            if algo_name in str(path).upper():
                algorithm = algo_name
                break

        print(f"\n📊 Evaluating: {Path(path).parent.name} ({algorithm})")

        eval_result = _action_eval(
            model_path=path,
            env_id=env_id,
            algorithm=algorithm,
            n_episodes=n_episodes,
            seed=seed,
        )

        json_data = None
        for content in eval_result.get("content", []):
            if "json" in content:
                json_data = content["json"]
                break

        results.append(
            {
                "path": path,
                "name": Path(path).parent.name,
                "algorithm": algorithm,
                **(json_data or {"mean_reward": 0, "std_reward": 0}),
            }
        )

    results.sort(key=lambda x: x["mean_reward"], reverse=True)

    lines = [f"\n🏆 Model Comparison on {env_id} ({n_episodes} episodes each):\n"]
    for i, r in enumerate(results):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        lines.append(
            f"  {medal} {r['name']} ({r['algorithm']}): {r['mean_reward']:.2f} ± {r.get('std_reward', 0):.2f}"
        )

    return {
        "status": "success",
        "content": [{"text": "\n".join(lines)}, {"json": {"rankings": results}}],
    }


# ─── LLM Fine-tuning Actions ────────────────────────────────────────────────


def _action_finetune(
    model_id: str,
    dataset_id: str = None,
    dataset_path: str = None,
    method: str = "lora",
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = None,
    push_to_hub: str = None,
    device: str = "auto",
    fp16: bool = False,
    bf16: bool = False,
    gradient_accumulation_steps: int = 4,
    text_field: str = "text",
    max_samples: int = None,
) -> Dict[str, Any]:
    """Fine-tune a HuggingFace model with LoRA/QLoRA.

    Supports:
    - LoRA (PEFT) fine-tuning
    - Full fine-tuning
    - SFT with TRL
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from datasets import load_dataset

    except ImportError as e:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"❌ Missing dependency: {e}\n\nInstall: pip install transformers datasets torch peft trl accelerate"
                }
            ],
        }

    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = output_dir or str(ML_MODELS_DIR / f"{model_id.split('/')[-1]}_{method}")

    print(f"\n🧠 Fine-tuning {model_id}")
    print(f"   Method: {method}")
    print(f"   Output: {out_dir}")

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = device

    print(f"   Device: {device_str}")

    # Load dataset
    if dataset_id:
        print(f"   Dataset: {dataset_id}")
        dataset = load_dataset(dataset_id, split="train")
    elif dataset_path:
        print(f"   Dataset: {dataset_path}")
        if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_path.endswith(".csv"):
            dataset = load_dataset("csv", data_files=dataset_path, split="train")
        else:
            dataset = load_dataset(dataset_path, split="train")
    else:
        return {
            "status": "error",
            "content": [{"text": "❌ Either dataset_id or dataset_path required"}],
        }

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"   Samples: {max_samples} (truncated from {len(dataset)})")
    else:
        print(f"   Samples: {len(dataset)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_fn(examples):
        texts = (
            examples[text_field]
            if text_field in examples
            else examples[list(examples.keys())[0]]
        )
        return tokenizer(
            texts, truncation=True, max_length=max_seq_length, padding="max_length"
        )

    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names
    )

    # Load model
    model_kwargs = {"trust_remote_code": True}

    # Handle precision
    if device_str == "cpu":
        model_kwargs["torch_dtype"] = torch.float32
    elif bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif fp16:
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    print(f"   Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    # Apply LoRA if requested
    if method in ("lora", "qlora"):
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            targets = target_modules.split(",") if target_modules else None
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=targets,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"   LoRA: r={lora_r}, alpha={lora_alpha}")
            print(
                f"   Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)"
            )

        except ImportError:
            return {
                "status": "error",
                "content": [{"text": "❌ pip install peft required for LoRA"}],
            }

    # Move to device
    if device_str != "cpu":
        model = model.to(device_str)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=(fp16 and device_str == "cuda"),
        bf16=(bf16 and device_str in ("cuda", "mps")),
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=(device_str == "cpu"),
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train
    start_time = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print(f"\n   Training...")
    train_result = trainer.train()
    elapsed = time.time() - start_time

    # Save
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Push to hub
    if push_to_hub:
        print(f"   Pushing to HuggingFace Hub: {push_to_hub}")
        try:
            model.push_to_hub(push_to_hub)
            tokenizer.push_to_hub(push_to_hub)
        except Exception as e:
            print(f"   ⚠ Push failed: {e}")

    metrics = train_result.metrics
    summary = (
        f"✅ Fine-tuning complete!\n"
        f"   Model: {model_id}\n"
        f"   Method: {method}\n"
        f"   Epochs: {epochs}\n"
        f"   Time: {elapsed:.1f}s\n"
        f"   Train loss: {metrics.get('train_loss', 'N/A')}\n"
        f"   Saved to: {out_dir}\n"
    )
    if push_to_hub:
        summary += f"   Hub: https://huggingface.co/{push_to_hub}\n"

    return {
        "status": "success",
        "content": [
            {"text": summary},
            {"json": {"output_dir": out_dir, "elapsed": elapsed, "metrics": metrics}},
        ],
    }


def _action_sft(
    model_id: str,
    dataset_id: str = None,
    dataset_path: str = None,
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    push_to_hub: str = None,
    device: str = "auto",
    max_samples: int = None,
) -> Dict[str, Any]:
    """Supervised Fine-Tuning using TRL's SFTTrainer.

    Expects dataset with 'messages' format (chat) or 'text' field.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
        from datasets import load_dataset
    except ImportError as e:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"❌ Missing dependency: {e}\n\nInstall: pip install trl transformers datasets peft"
                }
            ],
        }

    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = output_dir or str(ML_MODELS_DIR / f"{model_id.split('/')[-1]}_sft")

    print(f"\n🎓 SFT Training {model_id}")

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = device

    # Load dataset
    if dataset_id:
        dataset = load_dataset(dataset_id, split="train")
    elif dataset_path:
        if dataset_path.endswith((".jsonl", ".json")):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        else:
            dataset = load_dataset(dataset_path, split="train")
    else:
        return {
            "status": "error",
            "content": [{"text": "❌ Either dataset_id or dataset_path required"}],
        }

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    print(f"   Dataset: {len(dataset)} samples")
    print(f"   Device: {device_str}")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32 if device_str == "cpu" else torch.bfloat16,
    )

    # LoRA config
    peft_config = None
    try:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        print(f"   LoRA: r={lora_r}, alpha={lora_alpha}")
    except ImportError:
        print("   ⚠ peft not installed — full fine-tuning")

    # SFT Config
    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        use_cpu=(device_str == "cpu"),
    )

    start_time = time.time()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"   Training...")
    train_result = trainer.train()
    elapsed = time.time() - start_time

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    if push_to_hub:
        try:
            trainer.push_to_hub(push_to_hub)
        except Exception as e:
            print(f"   ⚠ Push failed: {e}")

    metrics = train_result.metrics
    summary = (
        f"✅ SFT complete!\n"
        f"   Model: {model_id}\n"
        f"   Epochs: {epochs} | Samples: {len(dataset)}\n"
        f"   Time: {elapsed:.1f}s\n"
        f"   Train loss: {metrics.get('train_loss', 'N/A')}\n"
        f"   Saved to: {out_dir}"
    )

    return {
        "status": "success",
        "content": [
            {"text": summary},
            {"json": {"output_dir": out_dir, "elapsed": elapsed, "metrics": metrics}},
        ],
    }


def _action_inference(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "auto",
) -> Dict[str, Any]:
    """Run inference on a fine-tuned model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as e:
        return {"status": "error", "content": [{"text": f"❌ Missing: {e}"}]}

    if device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = device

    print(f"🤖 Running inference from {model_path} on {device_str}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32 if device_str == "cpu" else torch.bfloat16,
    )

    if device_str != "cpu":
        model = model.to(device_str)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        device=device_str if device_str != "cpu" else -1,
    )

    start_time = time.time()
    output = pipe(prompt)
    elapsed = time.time() - start_time

    generated_text = output[0]["generated_text"]

    return {
        "status": "success",
        "content": [
            {"text": f"🤖 Generated ({elapsed:.2f}s):\n\n{generated_text}"},
            {"json": {"text": generated_text, "elapsed": elapsed}},
        ],
    }


# ─── Helper ──────────────────────────────────────────────────────────────────


def _indent(code: str, spaces: int) -> str:
    """Indent code block by N spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in code.strip().split("\n"))


# ─── Main Tool Entry Point ──────────────────────────────────────────────────


@tool
def rl(
    action: str,
    # RL params
    env_id: str = None,
    algorithm: str = "PPO",
    total_timesteps: int = 100000,
    seed: int = 42,
    hyperparams: str = None,
    save_name: str = None,
    n_envs: int = 1,
    eval_freq: int = 10000,
    device: str = "auto",
    policy_type: str = None,
    model_path: str = None,
    n_episodes: int = 20,
    deterministic: bool = True,
    render: bool = False,
    record_video: bool = False,
    video_path: str = None,
    env_name: str = None,
    reward_code: str = None,
    obs_dim: int = 4,
    act_dim: int = 1,
    act_type: str = "continuous",
    max_steps: int = 1000,
    description: str = "",
    reset_code: str = None,
    step_code: str = None,
    category: str = None,
    n_trials: int = 5,
    additional_timesteps: int = 100000,
    model_paths: str = None,
    # Render frame params
    steps: int = 50,
    n_frames: int = 4,
    frame_interval: int = 20,
    # LLM fine-tune params
    model_id: str = None,
    dataset_id: str = None,
    dataset_path: str = None,
    output_dir: str = None,
    method: str = "lora",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = None,
    push_to_hub: str = None,
    fp16: bool = False,
    bf16: bool = False,
    gradient_accumulation_steps: int = 4,
    text_field: str = "text",
    max_samples: int = None,
    # Inference params
    prompt: str = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    🎮 Reinforcement Learning & ML toolkit for DevDuck.

    Train, evaluate, and deploy RL agents using Stable-Baselines3 + Gymnasium.
    Fine-tune LLMs with LoRA/SFT. Create custom environments from reward functions.
    Visual debugging: rendered frames returned as native images.

    RL Actions:
        train         - Train an RL agent (PPO, A2C, DQN, SAC, TD3, DDPG)
        eval          - Evaluate a trained model
        play          - Watch a trained agent play (render or record video)
        create_env    - Create a custom Gymnasium env from Python code
        list_envs     - List available environments (built-in + custom)
        list_models   - List saved RL models
        sweep         - Hyperparameter sweep (tries N configs, picks best)
        continue      - Continue training a saved model for more steps
        compare       - Compare multiple models on the same environment
        render_frame  - Render env frames as images (visual debugging)

    LLM Actions:
        finetune      - Fine-tune with LoRA/QLoRA (transformers + PEFT)
        sft           - Supervised Fine-Tuning with TRL
        inference     - Run inference on a fine-tuned model

    Common:
        list_models   - List all saved models (RL + ML)

    Examples:
        # RL: Train CartPole
        rl(action="train", env_id="CartPole-v1", algorithm="PPO", total_timesteps=50000)

        # RL: Create custom env with image obs for CNN
        rl(action="create_env", env_name="snake", reward_code="...", obs_dim=4, act_dim=4, act_type="discrete")
        rl(action="train", env_id="custom:snake", algorithm="DQN", policy_type="CnnPolicy")

        # RL: Visual debug — see what the agent sees
        rl(action="render_frame", env_id="CartPole-v1", model_path="rl_models/.../best_model")

        # LLM: LoRA fine-tune
        rl(action="finetune", model_id="Qwen/Qwen2.5-0.5B", dataset_id="tatsu-lab/alpaca", method="lora")

        # LLM: SFT with TRL
        rl(action="sft", model_id="meta-llama/Llama-3.2-1B", dataset_path="./data.jsonl")

        # LLM: Run inference
        rl(action="inference", model_path="./ml_models/my_model", prompt="Hello world")

        # Hyperparameter sweep
        rl(action="sweep", env_id="LunarLander-v3", n_trials=8)
    """
    try:
        hp = json.loads(hyperparams) if hyperparams else None

        # ── RL Actions ──
        if action == "train":
            if not env_id:
                return {
                    "status": "error",
                    "content": [{"text": "❌ env_id required for training"}],
                }
            return _action_train(
                env_id=env_id,
                algorithm=algorithm,
                total_timesteps=total_timesteps,
                seed=seed,
                hyperparams=hp,
                save_name=save_name,
                n_envs=n_envs,
                eval_freq=eval_freq,
                device=device,
                policy_type=policy_type,
            )

        elif action == "eval":
            if not model_path:
                return {
                    "status": "error",
                    "content": [{"text": "❌ model_path required for evaluation"}],
                }
            return _action_eval(
                model_path=model_path,
                env_id=env_id,
                algorithm=algorithm,
                n_episodes=n_episodes,
                deterministic=deterministic,
                seed=seed,
                render=render,
            )

        elif action == "play":
            if not model_path:
                return {
                    "status": "error",
                    "content": [{"text": "❌ model_path required for play"}],
                }
            return _action_play(
                model_path=model_path,
                env_id=env_id,
                algorithm=algorithm,
                n_episodes=n_episodes,
                seed=seed,
                record_video=record_video,
                video_path=video_path,
                render=render,
            )

        elif action == "render_frame":
            if not env_id:
                return {"status": "error", "content": [{"text": "❌ env_id required"}]}
            return _action_render_frame(
                env_id=env_id,
                model_path=model_path,
                algorithm=algorithm,
                seed=seed,
                steps=steps,
                n_frames=n_frames,
                frame_interval=frame_interval,
            )

        elif action == "create_env":
            if not env_name:
                return {
                    "status": "error",
                    "content": [{"text": "❌ env_name required"}],
                }
            if not reward_code and not (step_code and reset_code):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "❌ Either reward_code OR (step_code + reset_code) required"
                        }
                    ],
                }
            return _action_create_env(
                env_name=env_name,
                reward_code=reward_code or "",
                obs_dim=obs_dim,
                act_dim=act_dim,
                act_type=act_type,
                max_steps=max_steps,
                description=description,
                reset_code=reset_code,
                step_code=step_code,
            )

        elif action == "list_envs":
            return _action_list_envs(category=category)

        elif action == "list_models":
            return _action_list_models()

        elif action == "sweep":
            if not env_id:
                return {
                    "status": "error",
                    "content": [{"text": "❌ env_id required for sweep"}],
                }
            return _action_sweep(
                env_id=env_id,
                algorithm=algorithm,
                n_trials=n_trials,
                total_timesteps=total_timesteps,
                seed=seed,
                policy_type=policy_type,
            )

        elif action == "continue":
            if not model_path:
                return {
                    "status": "error",
                    "content": [{"text": "❌ model_path required for continue"}],
                }
            return _action_continue_training(
                model_path=model_path,
                env_id=env_id,
                algorithm=algorithm,
                additional_timesteps=additional_timesteps,
                seed=seed,
            )

        elif action == "compare":
            if not model_paths:
                return {
                    "status": "error",
                    "content": [{"text": "❌ model_paths required (comma-separated)"}],
                }
            if not env_id:
                return {
                    "status": "error",
                    "content": [{"text": "❌ env_id required for compare"}],
                }
            paths = [p.strip() for p in model_paths.split(",")]
            return _action_compare(
                model_paths=paths, env_id=env_id, n_episodes=n_episodes, seed=seed
            )

        # ── LLM Actions ──
        elif action == "finetune":
            if not model_id:
                return {
                    "status": "error",
                    "content": [
                        {"text": "❌ model_id required (e.g., 'Qwen/Qwen2.5-0.5B')"}
                    ],
                }
            return _action_finetune(
                model_id=model_id,
                dataset_id=dataset_id,
                dataset_path=dataset_path,
                method=method,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                push_to_hub=push_to_hub,
                device=device,
                fp16=fp16,
                bf16=bf16,
                gradient_accumulation_steps=gradient_accumulation_steps,
                text_field=text_field,
                max_samples=max_samples,
            )

        elif action == "sft":
            if not model_id:
                return {
                    "status": "error",
                    "content": [{"text": "❌ model_id required"}],
                }
            return _action_sft(
                model_id=model_id,
                dataset_id=dataset_id,
                dataset_path=dataset_path,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                push_to_hub=push_to_hub,
                device=device,
                max_samples=max_samples,
            )

        elif action == "inference":
            if not model_path:
                return {
                    "status": "error",
                    "content": [{"text": "❌ model_path required for inference"}],
                }
            if not prompt:
                return {
                    "status": "error",
                    "content": [{"text": "❌ prompt required for inference"}],
                }
            return _action_inference(
                model_path=model_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"❌ Unknown action: {action}.\n\nRL: train, eval, play, render_frame, create_env, list_envs, list_models, sweep, continue, compare\nLLM: finetune, sft, inference"
                    }
                ],
            }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Error:\n{traceback.format_exc()}"}],
        }
