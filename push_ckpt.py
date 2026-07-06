"""把整个 ./ckpt 目录增量上传到 Hugging Face 模型仓库。

首次使用前运行：
    hf auth login

默认用法：
    python push_ckpt.py

重复运行时，Hugging Face 会比较文件内容，只上传新增或发生变化的文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parent
IGNORE_PATTERNS = [
    ".cache/**",
    "**/.cache/**",
    "**/__pycache__/**",
    "**/*.pyc",
    # last 通常是指向数字检查点的符号链接；数字目录已经会上传，无需再传一份。
    "**/checkpoints/last",
    "**/checkpoints/last/**",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally upload the complete ckpt directory to Hugging Face.")
    parser.add_argument("--repo_id", default="franka_ckpt", help="Repository name or owner/name.")
    parser.add_argument("--root", default="./ckpt", help="Local checkpoint directory.")
    parser.add_argument("--public", action="store_true", help="Create a public repository; default is private.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"找不到检查点目录：{root}")
    if not any(path.is_file() for path in root.rglob("*")):
        raise FileNotFoundError(f"检查点目录为空：{root}")

    api = HfApi()
    try:
        username = api.whoami()["name"]
    except Exception as error:
        raise RuntimeError("尚未登录 Hugging Face，请先运行：hf auth login") from error

    repo_id = args.repo_id if "/" in args.repo_id else f"{username}/{args.repo_id}"
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )

    print(f"本地目录：{root}")
    print(f"远端仓库：https://huggingface.co/{repo_id}")
    print("开始增量同步；已上传且内容未变化的文件会自动跳过。")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=root,
        path_in_repo="ckpt",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Sync checkpoints",
    )
    print(f"上传完成：https://huggingface.co/{repo_id}/tree/main/ckpt")


if __name__ == "__main__":
    main()
