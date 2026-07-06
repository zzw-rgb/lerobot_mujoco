"""
统一的模仿学习训练入口。

策略由 YAML 中的 ``policy.type`` 选择，目前支持：

- ``act``：Action Chunking Transformer
- ``diffusion``：Diffusion Policy

示例：

    python il/train_il.py --config_path config/il/act_franka.yaml
    python il/train_il.py --config_path config/il/diffusion_franka.yaml

本文件直接复用当前 LeRobot 版本的通用训练器，因此优化器、
学习率调度、断点续训和检查点格式与 VLA 训练流程保持一致。
"""

import os
import sys


# Windows 中文终端默认 GBK，LeRobot 帮助/日志中包含数学符号时会编码失败。
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from lerobot.common.utils.utils import init_logging
from lerobot.scripts import train as train_script


# LeRobot 用符号链接维护 checkpoints/last。Windows 未开启开发者模式时可能无权创建，
# 但数字编号的检查点已经完整保存；这时只警告而不让训练在收尾阶段失败。
_update_last_checkpoint = train_script.update_last_checkpoint


def _safe_update_last_checkpoint(checkpoint_dir):
    try:
        return _update_last_checkpoint(checkpoint_dir)
    except OSError as error:
        print(f"Warning: could not create checkpoints/last symlink: {error}")
        print(f"The numbered checkpoint is still valid: {checkpoint_dir}")
        return checkpoint_dir


train_script.update_last_checkpoint = _safe_update_last_checkpoint
train = train_script.train


if __name__ == "__main__":
    init_logging()
    train()
