"""
统一的模仿学习训练入口。

策略由 YAML 中的 ``policy.type`` 选择，目前支持：

- ``act``：Action Chunking Transformer
- ``diffusion``：Diffusion Policy

运行命令：

    # ACT（20k steps，输出到 ckpt/act_franka）
    CUDA_VISIBLE_DEVICES=0 python il/train_il.py --config_path=config/il/act_franka.yaml

    # Diffusion Policy（20k steps，输出到 ckpt/diffusion_franka）
    CUDA_VISIBLE_DEVICES=0 python il/train_il.py --config_path=config/il/diffusion_franka.yaml

本文件直接复用当前 LeRobot 版本的通用训练器，因此优化器、
学习率调度、断点续训和检查点格式与 VLA 训练流程保持一致。
"""

import json
import os
import re
import sys
from pathlib import Path

import yaml


# Windows 中文终端默认 GBK，LeRobot 帮助/日志中包含数学符号时会编码失败。
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def _config_path_from_argv() -> Path | None:
    """读取 LeRobot 的 --config_path，同时兼容等号和空格写法。"""
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument.startswith("--config_path="):
            return Path(argument.split("=", 1)[1]).expanduser().resolve()
        if argument == "--config_path" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).expanduser().resolve()
    return None


def validate_il_dataset_before_training() -> None:
    """在进入 LeRobot 训练器前给出可读的数据结构错误，而不是 Arrow 深层报错。"""
    config_path = _config_path_from_argv()
    if config_path is None or not config_path.is_file():
        return

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    root_value = config.get("dataset", {}).get("root")
    if not root_value:
        return
    dataset_root = Path(root_value).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = Path(PROJECT_ROOT) / dataset_root
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return  # 交给 LeRobot 处理不存在/需要下载的数据集。

    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)
    features = info.get("features", {})
    state_shape = tuple(features.get("observation.state", {}).get("shape", ()))
    action_shape = tuple(features.get("action", {}).get("shape", ()))
    if state_shape != (8,) or action_shape != (8,):
        raise ValueError(
            "IL 数据集结构不兼容：当前 observation.state/action 分别为 "
            f"{state_shape}/{action_shape}，新版 ACT/DP 需要 (8,)/(8,)（7 个实际关节角 + 夹爪）。\n"
            "请使用新版采集代码重新采集；旧末端位姿状态不能可靠转换成实际关节状态。"
        )

    expected_episodes = int(info.get("total_episodes", 0))
    parquet_files = sorted(dataset_root.rglob("episode_*.parquet"))
    episode_indices = []
    for path in parquet_files:
        match = re.fullmatch(r"episode_(\d+)", path.stem)
        if match:
            episode_indices.append(int(match.group(1)))
    expected_indices = list(range(expected_episodes))
    if sorted(episode_indices) != expected_indices:
        extras = sorted(set(episode_indices) - set(expected_indices))
        missing = sorted(set(expected_indices) - set(episode_indices))
        raise ValueError(
            "IL 数据集文件与 meta/info.json 不一致："
            f"元数据声明 {expected_episodes} 条，但发现 {len(episode_indices)} 个 Parquet。"
            f" extra={extras[:10]}, missing={missing[:10]}。\n"
            "这通常是 Hugging Face 重复上传没有删除旧 episode；请清理远端/本地残留文件后重试。"
        )

    # 检查每个文件的真实 Arrow 维度，避免元数据已更新但旧 6/7 维 Parquet 仍混在目录里。
    import pyarrow.parquet as pq

    for path in parquet_files:
        schema = pq.read_schema(path)
        state_size = getattr(schema.field("observation.state").type, "list_size", None)
        action_size = getattr(schema.field("action").type, "list_size", None)
        if state_size != 8 or action_size != 8:
            raise ValueError(
                f"IL 数据集含旧结构文件：{path} 的 state/action 维度为 "
                f"{state_size}/{action_size}，需要 8/8。请删除旧文件并重新采集或重新上传。"
            )

    print(f"IL dataset validation OK: {expected_episodes} episodes, state/action=(8,)/(8,)")


def run_lerobot_train() -> None:
    """数据检查通过后再导入较重的 Torch/LeRobot 训练模块。"""
    from lerobot.common.utils.utils import init_logging
    from lerobot.scripts import train as train_script

    # LeRobot 用符号链接维护 checkpoints/last。Windows 未开启开发者模式时可能无权创建，
    # 但数字编号的检查点已经完整保存；这时只警告而不让训练在收尾阶段失败。
    update_last_checkpoint = train_script.update_last_checkpoint

    def safe_update_last_checkpoint(checkpoint_dir):
        try:
            return update_last_checkpoint(checkpoint_dir)
        except OSError as error:
            print(f"Warning: could not create checkpoints/last symlink: {error}")
            print(f"The numbered checkpoint is still valid: {checkpoint_dir}")
            return checkpoint_dir

    train_script.update_last_checkpoint = safe_update_last_checkpoint
    init_logging()
    train_script.train()


if __name__ == "__main__":
    validate_il_dataset_before_training()
    run_lerobot_train()
