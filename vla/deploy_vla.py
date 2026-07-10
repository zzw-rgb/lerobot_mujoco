"""统一部署 π0 / SmolVLA，支持窗口、跨平台无头视频和多随机种子评估。

运行命令：

    # π0：窗口部署
    python vla/deploy_vla.py --config_path=config/vla/pi0_franka.yaml

    # π0：无头部署，视频默认输出到 output/pi0/
    CUDA_VISIBLE_DEVICES=7 python vla/deploy_vla.py --config_path=config/vla/pi0_franka.yaml --checkpoint=./ckpt/pi0_franka_v2/checkpoints/040000/pretrained_model --device=cuda --seed=0 --max_steps=2000 --headless

    # SmolVLA：窗口部署
    python vla/deploy_vla.py --config_path=config/vla/smolvla_franka.yaml

    # SmolVLA：无头部署，视频默认输出到 output/smolvla/
    CUDA_VISIBLE_DEVICES=7 python vla/deploy_vla.py --config_path=config/vla/smolvla_franka.yaml --checkpoint=./ckpt/smolvla_franka_v2/checkpoints/030000/pretrained_model --device=cuda --seed=0 --max_steps=2000 --headless
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from headless_gl import configure_headless_gl, option_from_argv, render_diagnostics

# 必须在导入 mujoco/SimpleEnv2 之前选择跨平台离屏后端。
if "--headless" in sys.argv:
    configure_headless_gl(option_from_argv(sys.argv, "--gl_backend", "auto"))

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor


os.chdir(PROJECT_ROOT)

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig
from mujoco_env.eval_utils import (
    EpisodeResult,
    open_video,
    sample_random_seeds,
    save_summary,
    video_frame,
    video_path_for_seed,
)


SUPPORTED_POLICIES = {"pi0", "smolvla"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a π0 or SmolVLA checkpoint in MuJoCo.")
    parser.add_argument("--config_path", required=True, help="Training YAML used to select policy/dataset/output_dir.")
    parser.add_argument("--checkpoint", default=None, help="Optional pretrained_model directory override.")
    parser.add_argument("--device", default=None, help="cuda or cpu. Defaults to YAML/checkpoint setting.")
    parser.add_argument("--seed", type=int, default=0, help="First MuJoCo scene seed.")
    parser.add_argument("--instruction", default=None, help="Optional fixed red/blue instruction for every episode.")
    parser.add_argument("--control_hz", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--gl_backend",
        choices=["auto", "egl", "osmesa", "wgl", "cgl", "glfw"],
        default="auto",
        help="Headless GL backend; auto selects WGL/EGL/CGL by platform.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Base MP4 path; defaults to output/<policy>/<policy>_seed<N>.mp4.",
    )
    parser.add_argument("--render_width", type=int, default=400)
    parser.add_argument("--render_height", type=int, default=300)
    parser.add_argument(
        "--random_seeds",
        type=int,
        default=10,
        help="After the first headless success, evaluate this many additional random seeds; 0 disables it.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_checkpoint(config: dict, override: str | None) -> Path:
    if override:
        candidates = [Path(override)]
    else:
        output_dir = Path(config.get("output_dir", ""))
        candidates = [
            output_dir / "checkpoints" / "last" / "pretrained_model",
            output_dir / "pretrained_model",
            output_dir,
        ]
        checkpoints_dir = output_dir / "checkpoints"
        if checkpoints_dir.is_dir():
            candidates.extend(
                sorted(
                    (path / "pretrained_model" for path in checkpoints_dir.iterdir() if path.name.isdigit()),
                    reverse=True,
                )
            )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    rendered = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No deployable checkpoint found. Checked:\n  - {rendered}")


def image_tensor(image: np.ndarray, feature) -> torch.Tensor:
    channels, height, width = feature.shape
    if channels != 3:
        raise ValueError(f"Expected RGB input, got feature shape {feature.shape}")
    return to_tensor(Image.fromarray(image).resize((width, height))).unsqueeze(0)


def build_observation(policy, state: np.ndarray, agent_image: np.ndarray, wrist_image: np.ndarray, task: str) -> dict:
    observation = {}
    for key, feature in policy.config.input_features.items():
        if key == "observation.state":
            observation[key] = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        elif key == "observation.image":
            observation[key] = image_tensor(agent_image, feature)
        elif key == "observation.wrist_image":
            observation[key] = image_tensor(wrist_image, feature)
        else:
            raise KeyError(f"VLA deployment does not know how to produce required feature: {key}")
    observation = {key: value.to(policy.config.device) for key, value in observation.items()}
    observation["task"] = [task]
    return observation


def get_policy_state(policy, env) -> np.ndarray:
    """按检查点期望维度生成状态，兼容旧 7 维和新 8 维 VLA。"""
    feature = policy.config.input_features.get("observation.state")
    if feature is None:
        raise KeyError("VLA checkpoint does not define observation.state")
    expected_dim = int(feature.shape[0])
    full_state = env.get_joint_state().astype(np.float32)
    if expected_dim == 7:  # 旧检查点：仅 7 个关节角
        return full_state[:7]
    if expected_dim == 8:  # 新检查点：7 个关节角 + 当前夹爪状态
        return full_state
    raise ValueError(f"Unsupported VLA observation.state shape: {feature.shape}")


def run_episode(policy, env, args: argparse.Namespace, seed: int, video_path: Path | None) -> EpisodeResult:
    env.reset(seed)
    if args.instruction:
        env.set_instruction(args.instruction)
    instruction = env.instruction
    policy.reset()
    writer = None
    if video_path is not None:
        writer = open_video(video_path, args.render_width, args.render_height, args.control_hz)
        print(f"Seed {seed}: {instruction} | video -> {video_path}")

    step = 0
    success = False
    try:
        while args.headless or env.env.is_viewer_alive():
            env.step_env()
            if not env.env.loop_every(HZ=args.control_hz):
                continue

            state = get_policy_state(policy, env)
            agent_image, wrist_image = env.grab_image()
            observation = build_observation(policy, state, agent_image, wrist_image, instruction)
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = action.squeeze(0).detach().cpu().numpy()[:8]
            env.step(action)

            if writer is not None:
                writer.write(video_frame(agent_image, wrist_image, step, instruction))
            else:
                env.render()
            step += 1

            success = env.check_success()
            if success:
                print(f"Seed {seed}: 成功，steps={step}，instruction={instruction!r}")
                break
            if args.max_steps > 0 and step >= args.max_steps:
                print(f"Seed {seed}: 达到 max_steps={args.max_steps}，未成功。")
                break
    finally:
        if writer is not None:
            writer.release()

    return EpisodeResult(
        seed=seed,
        success=success,
        steps=step,
        video=str(video_path) if video_path else None,
        instruction=instruction,
    )


def main() -> None:
    args = parse_args()
    if args.headless and args.max_steps <= 0:
        raise ValueError("无头模式必须设置 --max_steps。")
    if args.render_width <= 0 or args.render_height <= 0:
        raise ValueError("render_width 和 render_height 必须大于 0。")
    if args.random_seeds < 0:
        raise ValueError("random_seeds 不能小于 0。")
    if args.headless:
        print(f"Headless GL: {render_diagnostics()}")

    # 延迟导入，确保离屏 GL 后端在 mujoco 第一次加载前生效。
    from mujoco_env.SimpleEnv2 import SimpleEnv2

    train_config = load_yaml(args.config_path)
    requested_type = train_config.get("policy", {}).get("type")
    if requested_type not in SUPPORTED_POLICIES:
        raise ValueError(f"policy.type must be one of {sorted(SUPPORTED_POLICIES)}, got {requested_type!r}")

    checkpoint = resolve_checkpoint(train_config, args.checkpoint)
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    if policy_config.type != requested_type:
        raise ValueError(
            f"YAML requests {requested_type!r}, but checkpoint contains {policy_config.type!r}: {checkpoint}"
        )

    device = args.device or train_config.get("policy", {}).get("device") or policy_config.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    policy_config.device = device

    dataset_config = train_config.get("dataset", {})
    dataset_root = Path(dataset_config.get("root", "./demo_data_language")).expanduser().resolve()
    dataset_metadata = LeRobotDatasetMetadata(dataset_config.get("repo_id", "franka_pnp_language"), root=dataset_root)
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(
        checkpoint,
        config=policy_config,
        dataset_stats=dataset_metadata.stats,
    )
    policy.eval()

    print(f"Policy: {policy_config.type}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")

    env = SimpleEnv2(
        "./asset/example_scene_y2.xml",
        seed=args.seed,
        action_type="joint_angle",
        state_type="joint_angle",
        headless=args.headless,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    base_video = None
    if args.headless:
        default_video = Path("output") / requested_type / f"{requested_type}_seed{args.seed}.mp4"
        base_video = Path(args.video or default_video).expanduser().resolve()
        print(f"Headless EGL mode: first video will be saved to {base_video}")

    results: list[EpisodeResult] = []
    try:
        first_video = video_path_for_seed(base_video, args.seed) if base_video else None
        first_result = run_episode(policy, env, args, args.seed, first_video)
        results.append(first_result)
        if args.headless and first_result.success and args.random_seeds > 0:
            extra_seeds = sample_random_seeds(args.seed, args.random_seeds)
            print(f"首轮成功，开始评估 {len(extra_seeds)} 个随机种子：{extra_seeds}")
            for seed in extra_seeds:
                results.append(run_episode(policy, env, args, seed, video_path_for_seed(base_video, seed)))
    finally:
        env.close()

    if base_video:
        summary_path = save_summary(base_video, requested_type, results)
        successes = sum(result.success for result in results)
        print(f"评估完成：{successes}/{len(results)} 成功，汇总保存至 {summary_path}")


if __name__ == "__main__":
    main()
