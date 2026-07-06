"""
统一部署 ACT / Diffusion Policy。

训练和部署使用同一份 YAML：

    python il/deploy_il.py --config_path=config/il/act_franka.yaml
    python il/deploy_il.py --config_path=config/il/diffusion_franka.yaml
    CUDA_VISIBLE_DEVICES=7 python il/deploy_il.py --config_path=config/il/act_franka.yaml --headless --max_steps=2000 --video=./outputs/act.mp4

脚本会从 ``output_dir/checkpoints/last/pretrained_model`` 加载最新检查点，
并根据检查点中的 ``input_features`` 自动决定使用主相机、腕部相机
和末端位姿，不再在部署脚本里重复手写模型配置。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 必须在导入 mujoco/SimpleEnv 之前选择 EGL，否则无桌面服务器会退回 GLFW/llvmpipe。
if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from lerobot.common.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig


SUPPORTED_POLICIES = {"act", "diffusion"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy an ACT or Diffusion Policy checkpoint in MuJoCo.")
    parser.add_argument("--config_path", required=True, help="Training YAML used to select policy/output_dir.")
    parser.add_argument("--checkpoint", default=None, help="Optional pretrained_model directory override.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or mps. Defaults to YAML/checkpoint setting.")
    parser.add_argument("--seed", type=int, default=0, help="MuJoCo scene seed.")
    parser.add_argument("--control_hz", type=int, default=20, help="Policy control frequency.")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after N policy steps; 0 means unlimited.")
    parser.add_argument("--headless", action="store_true", help="Run without a GLFW window using EGL rendering.")
    parser.add_argument("--video", default=None, help="MP4 output path; defaults to outputs/<policy>_seed<N>.mp4.")
    parser.add_argument("--render_width", type=int, default=400, help="Headless camera width.")
    parser.add_argument("--render_height", type=int, default=300, help="Headless camera height.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def resolve_checkpoint(config: dict, override: str | None) -> Path:
    if override:
        candidates = [Path(override)]
    else:
        output_dir = Path(config.get("output_dir", ""))
        candidates = [
            output_dir / "checkpoints" / "last" / "pretrained_model",
            output_dir / "pretrained_model",
            output_dir,  # 兼容旧版直接 save_pretrained(output_dir) 的 ACT 检查点
        ]
        checkpoints_dir = output_dir / "checkpoints"
        if checkpoints_dir.is_dir():
            numbered = sorted(
                (path / "pretrained_model" for path in checkpoints_dir.iterdir() if path.name.isdigit()),
                reverse=True,
            )
            candidates.extend(numbered)

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
    pil_image = Image.fromarray(image).resize((width, height))
    return to_tensor(pil_image).unsqueeze(0)


def build_observation(policy, state: np.ndarray, agent_image: np.ndarray, wrist_image: np.ndarray) -> dict:
    observation = {}
    for key, feature in policy.config.input_features.items():
        if key == "observation.state":
            observation[key] = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        elif key == "observation.image":
            observation[key] = image_tensor(agent_image, feature)
        elif key == "observation.wrist_image":
            observation[key] = image_tensor(wrist_image, feature)
        else:
            raise KeyError(f"Deployment does not know how to produce required feature: {key}")
    return {key: value.to(policy.config.device) for key, value in observation.items()}


def video_frame(agent_image: np.ndarray, wrist_image: np.ndarray, step: int) -> np.ndarray:
    """把策略使用的两路 RGB 观测并排合成为一帧 MP4（OpenCV 使用 BGR）。"""
    height = min(agent_image.shape[0], wrist_image.shape[0])
    width = min(agent_image.shape[1], wrist_image.shape[1])
    agent = cv2.resize(agent_image, (width, height), interpolation=cv2.INTER_AREA)
    wrist = cv2.resize(wrist_image, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.concatenate([agent, wrist], axis=1)
    cv2.putText(frame, "Agent View", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Wrist View", (width + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Step {step}", (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def open_video(path: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建 MP4 视频：{path}。请检查 OpenCV/FFmpeg 编码支持。")
    return writer


def main() -> None:
    args = parse_args()
    if args.headless and args.max_steps <= 0:
        raise ValueError("无头模式必须设置 --max_steps，避免没有窗口关闭按钮时无限运行。")
    if args.render_width <= 0 or args.render_height <= 0:
        raise ValueError("render_width 和 render_height 必须大于 0。")

    # 延迟导入，确保上面的 MUJOCO_GL=egl 在 mujoco 第一次加载之前生效。
    from mujoco_env.SimpleEnv1 import SimpleEnv

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
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False. Use --device cpu if needed.")
    policy_config.device = device
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(checkpoint, config=policy_config)
    policy.eval()
    policy.reset()

    print(f"Policy: {policy_config.type}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Inputs: {list(policy.config.input_features)}")

    # 数据集中的 action 是“7 个绝对关节角 + 夹爪”，部署时必须用 joint_angle 执行。
    # 若沿用 SimpleEnv 默认的 eef_pose，模型输出会被误当成末端位姿增量，导致机械臂乱动。
    env = SimpleEnv(
        "./asset/example_scene_y.xml",
        seed=args.seed,
        action_type="joint_angle",
        state_type="joint_angle",
        headless=args.headless,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    video_path = None
    writer = None
    if args.headless:
        video_path = Path(args.video or f"./outputs/{requested_type}_seed{args.seed}.mp4").expanduser().resolve()
        writer = open_video(video_path, args.render_width, args.render_height, args.control_hz)
        print(f"Headless EGL mode: video will be saved to {video_path}")

    step = 0
    try:
        while args.headless or env.env.is_viewer_alive():
            env.step_env()
            if not env.env.loop_every(HZ=args.control_hz):
                continue

            state = env.get_ee_pose()
            agent_image, wrist_image = env.grab_image()
            observation = build_observation(policy, state, agent_image, wrist_image)

            with torch.inference_mode():
                action = policy.select_action(observation)
            action = action.squeeze(0).detach().cpu().numpy()
            env.step(action)
            if args.headless:
                writer.write(video_frame(agent_image, wrist_image, step))
            else:
                env.render()
            step += 1

            if env.check_success():
                print("成功：杯子已稳定放到盘子上。")
                break
            if args.max_steps > 0 and step >= args.max_steps:
                print(f"Reached max_steps={args.max_steps}.")
                break
    finally:
        if writer is not None:
            writer.release()
            print(f"MP4 已保存：{video_path}")
        env.close()


if __name__ == "__main__":
    main()
