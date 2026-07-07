"""IL/VLA 无头评估共用的视频、随机种子与汇总工具。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    steps: int
    video: str | None
    instruction: str | None = None


def video_frame(
    agent_image: np.ndarray,
    wrist_image: np.ndarray,
    step: int,
    caption: str | None = None,
) -> np.ndarray:
    """把两路 RGB 观测并排合成为一帧 MP4（OpenCV 写入时转换为 BGR）。"""
    height = min(agent_image.shape[0], wrist_image.shape[0])
    width = min(agent_image.shape[1], wrist_image.shape[1])
    agent = cv2.resize(agent_image, (width, height), interpolation=cv2.INTER_AREA)
    wrist = cv2.resize(wrist_image, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.concatenate([agent, wrist], axis=1)
    cv2.putText(frame, "Agent View", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Wrist View", (width + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Step {step}", (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    if caption:
        cv2.putText(frame, caption, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def open_video(path: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建 MP4 视频：{path}。请检查 OpenCV/FFmpeg 编码支持。")
    return writer


def video_path_for_seed(base_path: Path, seed: int) -> Path:
    """把 act_seed0.mp4 这样的基础名称替换为当前 seed，避免出现双重 seed 后缀。"""
    stem = re.sub(r"_seed-?\d+$", "", base_path.stem)
    return base_path.with_name(f"{stem}_seed{seed}{base_path.suffix}")


def sample_random_seeds(first_seed: int, count: int) -> list[int]:
    """生成可复现且不包含首轮 seed 的不重复随机种子。"""
    if count <= 0:
        return []
    rng = np.random.default_rng(first_seed)
    seeds: list[int] = []
    while len(seeds) < count:
        candidate = int(rng.integers(0, 1_000_000))
        if candidate != first_seed and candidate not in seeds:
            seeds.append(candidate)
    return seeds


def save_summary(base_video: Path, policy_type: str, results: list[EpisodeResult]) -> Path:
    stem = re.sub(r"_seed-?\d+$", "", base_video.stem)
    summary_path = base_video.with_name(f"{stem}_summary.json")
    successes = sum(result.success for result in results)
    payload = {
        "policy": policy_type,
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "results": [asdict(result) for result in results],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path
