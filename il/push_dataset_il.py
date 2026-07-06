"""
把本地采集的 ACT 数据集（./demo_data）上传到 Hugging Face Hub。

运行前需要先登录 Hugging Face（拥有写权限的 access token）：
    huggingface-cli login
    # 或较新版本的 huggingface_hub：
    hf auth login

运行方式：
    conda activate lerobot
    python il/push_dataset_il.py
"""

import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 目标仓库 id，格式为 "<Hugging Face 用户名或组织名>/<数据集名>"。
# 不写用户名（比如只写 "franka_pnp"）也能传，Hub 会自动挂到当前登录账号名下，
# 但写全 "用户名/仓库名" 更清楚，建议改成你自己的用户名。
REPO_ID = 'a3124371940/franka_pnp'
ROOT = './demo_data'
PRIVATE = False  # True：仓库设为私有，只有自己/被邀请的人能看；False：公开数据集

dataset = LeRobotDataset(REPO_ID, root=ROOT)
dataset.push_to_hub(private=PRIVATE)
print(f"上传完成：https://huggingface.co/datasets/{dataset.repo_id}")
