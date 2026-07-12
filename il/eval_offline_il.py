"""离线开环评估 ACT / Diffusion Policy 检查点。

把数据集里已录制的观测逐帧喂给策略（不接 MuJoCo、不渲染），对比
模型预测动作与数据集里的真实动作，输出：

- 7 个关节的平均/最大误差（弧度）；
- 夹爪命令的准确率，以及首次闭合/张开时机相对示教数据的偏移帧数
  （SimpleEnv 部署时按 0.75/0.25 的迟滞阈值锁存夹爪）。

用途：区分“模型没学好”和“部署管线错位”。

- 离线误差大（关节 MAE 明显超过数据本身相邻帧增量 ~0.03 rad）
  → 训练不足或数据/配置有问题，先解决训练；
- 离线误差小但 deploy_il.py 闭环成功率低
  → 问题出在闭环漂移（策略自己执行时偏离数据分布）或环境交互，
    优先增加数据覆盖/训练步数，再检查部署侧。

运行命令：

    # 评估 last 检查点（与训练同一份 YAML）
    python il/eval_offline_il.py --config_path=config/il/act_franka.yaml

    # 指定检查点与回合数
    python il/eval_offline_il.py --config_path=config/il/diffusion_franka.yaml \
        --checkpoint=./ckpt/diffusion_franka/checkpoints/040000/pretrained_model --num_episodes=5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
IL_DIR = PROJECT_ROOT / "il"
if str(IL_DIR) not in sys.path:
    sys.path.insert(0, str(IL_DIR))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

os.chdir(PROJECT_ROOT)

import numpy as np
import torch

from deploy_il import SUPPORTED_POLICIES, load_yaml, resolve_checkpoint
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig

# 与 SimpleEnv.step 的夹爪迟滞锁存阈值一致：>=0.75 触发闭合，<=0.25 触发张开。
GRIPPER_CLOSE_THRESHOLD = 0.75
GRIPPER_OPEN_THRESHOLD = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-loop evaluation of an IL checkpoint on recorded episodes.")
    parser.add_argument("--config_path", required=True, help="Training YAML used to select policy/dataset/output_dir.")
    parser.add_argument("--checkpoint", default=None, help="Optional pretrained_model directory override.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or mps. Defaults to YAML/checkpoint setting.")
    parser.add_argument("--num_episodes", type=int, default=5, help="How many episodes to evaluate (evenly spaced).")
    return parser.parse_args()


def pick_episode_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    count = min(count, total)
    return sorted({round(i * (total - 1) / max(count - 1, 1)) for i in range(count)})


def first_transition(values: np.ndarray, close: bool) -> int | None:
    """返回夹爪命令序列中首次闭合（或闭合后首次张开）的帧号。"""
    if close:
        hits = np.flatnonzero(values >= GRIPPER_CLOSE_THRESHOLD)
        return int(hits[0]) if hits.size else None
    close_idx = first_transition(values, close=True)
    if close_idx is None:
        return None
    hits = np.flatnonzero(values[close_idx:] <= GRIPPER_OPEN_THRESHOLD)
    return int(close_idx + hits[0]) if hits.size else None


def evaluate_episode(policy, dataset: LeRobotDataset, episode_index: int, device: str) -> dict:
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()
    policy.reset()

    predictions = []
    targets = []
    for frame_id in range(from_idx, to_idx):
        item = dataset[frame_id]
        observation = {}
        for key in policy.config.input_features:
            if key not in item:
                raise KeyError(f"Dataset frame is missing required policy input: {key}")
            observation[key] = item[key].unsqueeze(0).to(device)
        with torch.inference_mode():
            action = policy.select_action(observation)
        predictions.append(action.squeeze(0).float().cpu().numpy())
        targets.append(item["action"].float().cpu().numpy())

    predictions = np.stack(predictions)
    targets = np.stack(targets)

    joint_err = np.abs(predictions[:, :7] - targets[:, :7])
    grip_pred = predictions[:, 7]
    grip_true = targets[:, 7]
    grip_acc = float(((grip_pred >= 0.5) == (grip_true >= 0.5)).mean())

    def lag(close: bool) -> int | None:
        true_idx = first_transition(grip_true, close)
        pred_idx = first_transition(grip_pred, close)
        if true_idx is None or pred_idx is None:
            return None
        return pred_idx - true_idx

    return {
        "episode": episode_index,
        "frames": len(predictions),
        "joint_mae": float(joint_err.mean()),
        "joint_max": float(joint_err.max()),
        "per_joint_mae": joint_err.mean(axis=0),
        "gripper_acc": grip_acc,
        "close_lag": lag(close=True),
        "open_lag": lag(close=False),
    }


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
        print("CUDA 不可用，改用 CPU 进行离线评估。")
        device = "cpu"
    policy_config.device = device
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(checkpoint, config=policy_config)
    policy.eval()

    dataset_cfg = train_config.get("dataset", {})
    dataset = LeRobotDataset(dataset_cfg.get("repo_id", "franka_pnp"), root=dataset_cfg.get("root", "./demo_data"))

    print(f"Policy: {policy_config.type}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    episode_indices = pick_episode_indices(dataset.num_episodes, args.num_episodes)
    results = [evaluate_episode(policy, dataset, index, device) for index in episode_indices]

    print("\nepisode | frames | joint MAE (rad) | joint max | gripper acc | close lag | open lag")
    for result in results:
        close_lag = "n/a" if result["close_lag"] is None else f"{result['close_lag']:+d}"
        open_lag = "n/a" if result["open_lag"] is None else f"{result['open_lag']:+d}"
        print(
            f"{result['episode']:7d} | {result['frames']:6d} | {result['joint_mae']:15.4f} | "
            f"{result['joint_max']:9.4f} | {result['gripper_acc']:11.3f} | {close_lag:>9} | {open_lag:>8}"
        )

    joint_mae = float(np.mean([result["joint_mae"] for result in results]))
    per_joint = np.mean([result["per_joint_mae"] for result in results], axis=0)
    gripper_acc = float(np.mean([result["gripper_acc"] for result in results]))
    print(f"\n总体：joint MAE={joint_mae:.4f} rad，gripper acc={gripper_acc:.3f}")
    print("分关节 MAE：", np.round(per_joint, 4))
    print(
        "参考：示教数据相邻帧的关节目标增量约 0.03 rad。"
        "joint MAE 远大于该值说明拟合不足；"
        "close/open lag 为正表示夹爪动作比示教晚 N 帧（20 帧 = 1 秒）。"
    )


if __name__ == "__main__":
    main()
