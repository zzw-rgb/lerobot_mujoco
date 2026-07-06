"""
统一部署 ACT / Diffusion Policy。

训练和部署使用同一份 YAML：

    python il/deploy_il.py --config_path=config/il/act_franka.yaml
    python il/deploy_il.py --config_path=config/il/diffusion_franka.yaml

脚本会从 ``output_dir/checkpoints/last/pretrained_model`` 加载最新检查点，
并根据检查点中的 ``input_features`` 自动决定使用主相机、腕部相机
和末端位姿，不再在部署脚本里重复手写模型配置。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
from mujoco_env.SimpleEnv1 import SimpleEnv


SUPPORTED_POLICIES = {"act", "diffusion"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy an ACT or Diffusion Policy checkpoint in MuJoCo.")
    parser.add_argument("--config_path", required=True, help="Training YAML used to select policy/output_dir.")
    parser.add_argument("--checkpoint", default=None, help="Optional pretrained_model directory override.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or mps. Defaults to YAML/checkpoint setting.")
    parser.add_argument("--seed", type=int, default=0, help="MuJoCo scene seed.")
    parser.add_argument("--control_hz", type=int, default=20, help="Policy control frequency.")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after N policy steps; 0 means unlimited.")
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


def main() -> None:
    args = parse_args()
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
    )
    step = 0
    try:
        while env.env.is_viewer_alive():
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
            env.render()
            step += 1

            if env.check_success():
                print("成功：杯子已稳定放到盘子上。")
                break
            if args.max_steps > 0 and step >= args.max_steps:
                print(f"Reached max_steps={args.max_steps}.")
                break
    finally:
        env.env.close_viewer()


if __name__ == "__main__":
    main()
