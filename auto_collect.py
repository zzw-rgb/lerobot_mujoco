"""
自动采集 LeRobot 示教数据。

这个脚本用一个简单、可调的“专家轨迹”自动完成桌面抓杯子放盘子的任务：

    杯子上方 -> 下探 -> 闭合夹爪 -> 抬起 -> 盘子上方 -> 放下 -> 松爪 -> 抬起

注意：
    1. 专家内部用 eef_pose + IK 控制末端移动；
    2. 真正写入数据集的 action 仍然是 7 个关节角 + 1 个夹爪命令；
    3. 因此生成的数据可以直接给现有的 ACT / Diffusion Policy / VLA 训练脚本使用。

常用命令：

    # 平时直接运行即可；启动后选择是否无头运行以及 il / vla
    python auto_collect.py

    # 命令行参数仍然保留，临时覆盖配置时可以用
    python auto_collect.py --mode=il --num_demos=20 --root=./demo_data --force --headless
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from headless_gl import configure_headless_gl, option_from_argv, render_diagnostics


# Windows 中文终端可能仍使用 GBK；采集进度包含 ✓/✗，统一为 UTF-8 避免保存后打印崩溃。
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


# =============================================================================
# 用户配置区：日常只改这里，然后直接运行 python auto_collect.py
# =============================================================================

# 采集哪种数据：
#   "ask" = 运行后让你选择 il / vla
#   "il"  = ACT / Diffusion Policy 用的普通模仿学习数据
#   "vla" = π0 / SmolVLA 用的语言条件数据
AUTO_MODE = "ask"

# 想成功保存多少条 episode。调试专家轨迹时建议 3~5，正式采集再改成 50/100。
AUTO_NUM_DEMOS = 200

# 数据保存目录。
#   None：根据模式自动使用 ./demo_data 或 ./demo_data_language，直接覆盖/生成训练用数据。
#   "./demo_data"：直接生成给 IL 训练脚本用的数据。
#   "./demo_data_language"：直接生成给 VLA 训练脚本用的数据。
AUTO_DATASET_ROOT = None

# 目标数据集已存在时怎么处理：
#   "ask"       = 运行时询问：继续追加 / 删除重采 / 退出
#   "append"    = 直接继续追加，不覆盖旧数据
#   "overwrite" = 直接删除旧数据后重采
#   "exit"      = 直接退出
AUTO_EXISTING_DATASET_ACTION = "ask"

# 下面两个一般不用改；命令行临时指定 --force / --append 时才会覆盖上面的 ask 逻辑。
AUTO_FORCE = False

# True 表示在已有数据集后面追加；不能和 AUTO_FORCE 同时为 True。
AUTO_APPEND = False

# 运行时是否询问无头模式：
#   "ask" = 每次直接运行脚本时询问（推荐）
#   True  = 始终无头运行，不显示 MuJoCo 窗口
#   False = 始终显示 MuJoCo 窗口
AUTO_HEADLESS = "ask"

# "auto" 会按系统选择：Windows=WGL、Linux=EGL、macOS=CGL。
# Linux 没有可用的 NVIDIA EGL 时可改为 "osmesa"（需要系统安装 libosmesa6）。
AUTO_GL_BACKEND = "auto"
# 多 GPU Linux 服务器只有 EGL 选错卡时才填写物理显卡编号；通常保持 None。
AUTO_EGL_DEVICE_ID = None

# 本地窗口显示；AUTO_HEADLESS=True 时会自动关闭。
AUTO_RENDER = True

# 保存自动采集过程的视频。设为 None 就不保存视频。
AUTO_VIDEO_DIR = "./outputs/auto_collect"

# 随机种子与尝试次数。AUTO_MAX_ATTEMPTS=0 表示自动设为 AUTO_NUM_DEMOS * 5。
AUTO_SEED = 0
AUTO_MAX_ATTEMPTS = 0

# 采集频率和画面尺寸。
AUTO_FPS = 20
AUTO_RENDER_WIDTH = 400
AUTO_RENDER_HEIGHT = 300
AUTO_IMAGE_WRITER_THREADS = 2      # 长时间采集更稳；想更快可改 4/8，但更吃内存

# 数据质量保护。
# 续采时如果发现尾部 episode 的 parquet 只有几十 KB，通常说明相机返回了黑图；
# 脚本会自动裁掉这段坏尾巴，再从最后一条正常 episode 继续采集。
AUTO_REPAIR_INVALID_TAIL = True
AUTO_MIN_EPISODE_BYTES = 1_000_000
AUTO_MIN_EPISODE_BYTES_PER_FRAME = 100_000
AUTO_MIN_IMAGE_BYTES = 1_000
AUTO_IMAGE_VALIDATION_SAMPLES = 16
AUTO_MIN_IMAGE_STD = 10.0  # 收紧黑图判定：原 2.0 太松，大面积黑图 std 略>2 会漏过
# 近黑像素占比阈值：一帧里像素 < AUTO_NEAR_BLACK_VALUE 的比例超过 AUTO_MAX_NEAR_BLACK_RATIO 即判坏图
AUTO_NEAR_BLACK_VALUE = 10
AUTO_MAX_NEAR_BLACK_RATIO = 0.5
# 熔断：连续这么多次因黑图失败就立即停止（窗口失活/渲染坏时避免白跑一长串坏 episode）
AUTO_MAX_CONSECUTIVE_BLACK = 3

# 专家轨迹参数：抓不住主要调这几个。
AUTO_GRASP_X_OFFSET = 0.0        # 当前场景保持 0；实测 ±0.05 都无法稳定完成 seed=0
AUTO_GRASP_Y_OFFSET = 0.050      # 世界坐标 y 正方向对应杯子无把手侧；seed=0 已实测成功
AUTO_GRASP_Z_OFFSET = 0.022      # 下探高度；还高就试 0.0，太低碰撞就试 0.02
AUTO_PLACE_Z_OFFSET = 0.065      # 放到盘子上方的高度
AUTO_HOVER_Z = 1.05              # 搬运时抬起高度

# VLA 可选固定指令。None 表示随机红杯/蓝杯。
# 可写："Place the red mug on the plate." 或 "Place the blue mug on the plate."
AUTO_INSTRUCTION = None


def choose_headless_before_imports() -> bool:
    """在导入 MuJoCo/LeRobot 前确定渲染模式，保证 GL 后端能够生效。"""
    if "--headless" in sys.argv:
        return True
    if "--no_headless" in sys.argv or "--no-headless" in sys.argv:
        return False

    configured = AUTO_HEADLESS
    if isinstance(configured, bool):
        return configured
    if str(configured).strip().lower() != "ask":
        raise ValueError('AUTO_HEADLESS must be "ask", True, or False.')

    # 被其他 Python 文件 import 或只查看 --help 时不能弹出交互问题。
    if __name__ != "__main__" or "--help" in sys.argv or "-h" in sys.argv:
        return False

    print("\n请选择采集显示模式：")
    print("  1) 无头模式：不显示窗口，适合服务器或长时间采集（推荐）")
    print("  2) 窗口模式：显示 MuJoCo 画面，适合观察专家轨迹")
    while True:
        try:
            choice = input("请输入 1/2（直接回车默认 1）：").strip().lower()
        except EOFError:
            print("未检测到交互终端，自动使用无头模式。")
            return True
        if choice in {"", "1", "y", "yes", "headless"}:
            return True
        if choice in {"2", "n", "no", "window", "gui"}:
            return False
        print("输入无效，请输入 1 或 2。")


_DEFAULT_HEADLESS = choose_headless_before_imports()

if _DEFAULT_HEADLESS:
    # 必须在导入 mujoco / OpenGL / 环境封装之前设置；这里不能用 setdefault，
    # 否则 shell 中遗留的 glfw/egl 会覆盖本次无头配置。
    _requested_gl_backend = option_from_argv(sys.argv, "--gl_backend", AUTO_GL_BACKEND)
    configure_headless_gl(_requested_gl_backend, AUTO_EGL_DEVICE_ID)


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


TASK_NAME_IL = "Put mug cup on the plate"
DEFAULTS = {
    "il": {
        "repo_id": "franka_pnp",
        "root": "./demo_data",
        "xml": "./asset/example_scene_y.xml",
    },
    "vla": {
        "repo_id": "franka_pnp_language",
        "root": "./demo_data_language",
        "xml": "./asset/example_scene_y2.xml",
    },
}


@dataclass
class AttemptResult:
    saved_index: int | None
    attempt: int
    seed: int
    success: bool
    frames: int
    task: str
    video: str | None = None
    reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatically collect IL/VLA demonstrations in MuJoCo.")
    parser.add_argument("--mode", choices=["ask", "il", "vla"], default=AUTO_MODE, help="Dataset type to collect.")
    parser.add_argument("--num_demos", type=int, default=AUTO_NUM_DEMOS, help="Number of successful episodes to save.")
    parser.add_argument("--root", default=AUTO_DATASET_ROOT, help="Dataset root. Defaults to demo_data/demo_data_language.")
    parser.add_argument("--repo_id", default=None, help="LeRobot dataset repo id.")
    parser.add_argument("--seed", type=int, default=AUTO_SEED, help="First random seed. Attempts use seed + attempt_id.")
    parser.add_argument("--max_attempts", type=int, default=AUTO_MAX_ATTEMPTS, help="0 means num_demos * 5.")
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=AUTO_APPEND, help="Append to an existing dataset.")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=AUTO_FORCE, help="Delete the existing root before collection.")
    parser.add_argument(
        "--existing_action",
        choices=["ask", "append", "overwrite", "exit"],
        default=AUTO_EXISTING_DATASET_ACTION,
        help="What to do when the dataset root already exists.",
    )

    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=_DEFAULT_HEADLESS, help="Use platform-appropriate offscreen rendering without a viewer window.")
    parser.add_argument("--no_headless", action="store_false", dest="headless", help=argparse.SUPPRESS)
    parser.add_argument(
        "--gl_backend",
        choices=sorted({"auto", "egl", "osmesa", "wgl", "cgl", "glfw"}),
        default=AUTO_GL_BACKEND,
        help="Offscreen OpenGL backend. auto selects WGL/EGL/CGL by platform.",
    )
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=AUTO_RENDER, help="Render the GUI window while collecting. Enabled by default unless --headless is used.")
    parser.add_argument("--no_render", action="store_true", help="Do not refresh the GUI window when running without --headless.")
    parser.add_argument("--render_width", type=int, default=AUTO_RENDER_WIDTH, help="Offscreen camera width.")
    parser.add_argument("--render_height", type=int, default=AUTO_RENDER_HEIGHT, help="Offscreen camera height.")
    parser.add_argument("--video_dir", default=AUTO_VIDEO_DIR, help="Optional directory for MP4 videos of saved episodes.")

    parser.add_argument("--fps", type=int, default=AUTO_FPS, help="Dataset FPS and video FPS.")
    parser.add_argument("--physics_steps", type=int, default=0, help="MuJoCo steps per recorded frame. 0 auto-computes from env HZ.")
    parser.add_argument("--max_delta", type=float, default=0.012, help="Max end-effector translation per recorded frame.")
    parser.add_argument("--pos_tol", type=float, default=0.012, help="Waypoint position tolerance.")
    parser.add_argument("--hover_z", type=float, default=AUTO_HOVER_Z, help="Safe z height for moving above objects.")
    parser.add_argument("--grasp_z_offset", type=float, default=AUTO_GRASP_Z_OFFSET, help="TCP z offset relative to mug body center while grasping.")
    parser.add_argument("--grasp_x_offset", type=float, default=AUTO_GRASP_X_OFFSET, help="World x offset from mug center while grasping.")
    parser.add_argument("--grasp_y_offset", type=float, default=AUTO_GRASP_Y_OFFSET, help="World y offset from mug center while grasping.")
    parser.add_argument("--grasp_xy_offset", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grasp_direction", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--place_z_offset", type=float, default=AUTO_PLACE_Z_OFFSET, help="TCP z offset relative to plate body center while placing.")
    parser.add_argument("--close_frames", type=int, default=25, help="Frames to keep gripper closed before lifting.")
    parser.add_argument("--open_frames", type=int, default=22, help="Frames to keep gripper open after placing.")
    parser.add_argument("--settle_frames", type=int, default=60, help="Frames to wait after final lift for success check.")
    parser.add_argument("--max_waypoint_frames", type=int, default=180, help="Safety cap per waypoint.")

    parser.add_argument("--instruction", default=AUTO_INSTRUCTION, help="VLA only. Fixed instruction, e.g. 'Place the red mug on the plate.'")
    return parser.parse_args()


def choose_mode_if_needed(args: argparse.Namespace) -> None:
    if args.mode != "ask":
        return
    print("\n请选择要自动采集的数据类型：")
    print("  1) il  - ACT / Diffusion Policy 用，保存到 ./demo_data")
    print("  2) vla - π0 / SmolVLA 用，保存到 ./demo_data_language")
    choice = input("输入 1 或 2（默认 1）：").strip()
    if choice in {"", "1", "il", "IL"}:
        args.mode = "il"
    elif choice in {"2", "vla", "VLA"}:
        args.mode = "vla"
    else:
        raise ValueError(f"无效选择：{choice!r}，请输入 1 或 2。")

    if args.root is None:
        args.root = DEFAULTS[args.mode]["root"]
    print(f"已选择：{args.mode}")
    print(f"数据集目录：{args.root}")


def choose_existing_dataset_action(root: Path) -> str:
    print(f"\n数据集目录已存在：{root}")
    print("请选择怎么处理：")
    print("  1) 继续采集：不覆盖，追加到已有数据集后面")
    print("  2) 删除重采：覆盖旧数据集")
    print("  3) 退出：什么都不改")
    choice = input("输入 1 / 2 / 3（默认 1）：").strip()
    if choice in {"", "1", "append", "继续"}:
        return "append"
    if choice in {"2", "overwrite", "覆盖", "delete"}:
        return "overwrite"
    if choice in {"3", "exit", "退出", "q", "Q"}:
        return "exit"
    raise ValueError(f"无效选择：{choice!r}，请输入 1 / 2 / 3。")


def count_meta_episodes(root: Path) -> int:
    episodes_file = root / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        return 0
    with episodes_file.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_dataset_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def expected_state_shape(mode: str) -> tuple[int]:
    """IL/VLA 统一保存 7 个实际关节角和夹爪命令。"""
    if mode not in DEFAULTS:
        raise ValueError(f"Unsupported dataset mode: {mode!r}")
    return (8,)


def validate_existing_dataset_schema(root: Path, mode: str) -> None:
    """阻止把关节状态帧追加到旧版末端位姿数据集中。"""
    info = load_dataset_info(root)
    state_feature = info.get("features", {}).get("observation.state", {})
    actual_shape = tuple(state_feature.get("shape", ()))
    expected_shape = expected_state_shape(mode)
    if actual_shape and actual_shape != expected_shape:
        raise ValueError(
            "现有数据集使用旧版 observation.state 结构："
            f"{actual_shape}，新版 {mode.upper()} 数据需要 {expected_shape}"
            "（7 个实际关节角 + 夹爪命令）。\n"
            "旧数据只有末端位姿，无法可靠恢复每帧实际关节状态，不能继续追加。"
            "请重新运行并选择“删除重采”，或使用 --force 覆盖旧数据集。"
        )


def episode_parquet_path(root: Path, episode_index: int, info: dict | None = None) -> Path:
    info = info or load_dataset_info(root)
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return root / data_path.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def sampled_indices(length: int, samples: int) -> list[int]:
    if length <= 0:
        return []
    if length <= samples:
        return list(range(length))
    if samples <= 1:
        return [0]
    return sorted({round(i * (length - 1) / (samples - 1)) for i in range(samples)})


def is_episode_file_complete(root: Path, episode_index: int, min_bytes: int = AUTO_MIN_EPISODE_BYTES) -> bool:
    path = episode_parquet_path(root, episode_index)
    if not path.exists():
        return False

    try:
        import pyarrow.parquet as pq

        rows = pq.read_metadata(path).num_rows
        if rows <= 0:
            return False

        # Normal embedded-image episodes in this scene are tens of MB. Black
        # frames compress to ~270 bytes each, so even a half-black episode has a
        # much lower bytes/frame ratio. This catches bad episodes quickly
        # without reading every image blob.
        bytes_per_frame = path.stat().st_size / rows
        if path.stat().st_size < min_bytes or bytes_per_frame < AUTO_MIN_EPISODE_BYTES_PER_FRAME:
            return False
        return True
    except Exception:
        return path.stat().st_size >= min_bytes


def truncate_dataset_tail(root: Path, keep_episodes: int) -> None:
    """Drop episode files and metadata from keep_episodes onward.

    This is used only for repairing a bad tail produced by a failed/black-camera
    append run. Good earlier episodes are left untouched.
    """
    meta_dir = root / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"
    info_path = meta_dir / "info.json"

    episodes = read_jsonl(episodes_path)
    kept_episodes = [ep for ep in episodes if int(ep.get("episode_index", -1)) < keep_episodes]
    write_jsonl(episodes_path, kept_episodes)

    stats = read_jsonl(stats_path)
    if stats:
        kept_stats = [item for item in stats if int(item.get("episode_index", -1)) < keep_episodes]
        write_jsonl(stats_path, kept_stats)

    info = load_dataset_info(root)
    if info:
        total_episodes = len(kept_episodes)
        total_frames = sum(int(ep.get("length", 0)) for ep in kept_episodes)
        info["total_episodes"] = total_episodes
        info["total_frames"] = total_frames
        if "splits" in info:
            info["splits"]["train"] = f"0:{total_episodes}"
        with info_path.open("w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=4)

    cleanup_orphan_episode_files(root, keep_episodes)


def repair_invalid_tail(root: Path) -> int:
    """Repair invalid episodes before appending.

    If all invalid episodes are at the tail, trimming is enough. If invalid
    episodes are mixed into the middle, build a compact copy that keeps all good
    episodes and reindexes them contiguously.
    """
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    if not episodes:
        return 0

    bad_indices: list[int] = []
    valid_records: list[dict] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if is_episode_file_complete(root, episode_index):
            valid_records.append(episode)
        else:
            bad_indices.append(episode_index)

    if not bad_indices:
        return len(episodes)

    first_bad = bad_indices[0]
    has_good_after_first_bad = any(int(ep["episode_index"]) > first_bad for ep in valid_records)

    if not has_good_after_first_bad:
        print(
            f"检测到 episode_{first_bad:06d} 之后存在异常/黑图 parquet，"
            f"将裁掉坏尾巴并从 {first_bad} 继续采集。"
        )
        truncate_dataset_tail(root, first_bad)
        return count_meta_episodes(root)

    print(
        f"检测到 {len(bad_indices)} 个黑图/异常 episode，且坏数据混在中间；"
        "将保留所有好 episode 并重新连续编号。"
    )
    compact_valid_dataset(root, valid_records, bad_indices)
    return count_meta_episodes(root)


def replace_arrow_column(table, name: str, values):
    import pyarrow as pa

    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        return table
    field = table.schema.field(column_index)
    return table.set_column(column_index, field, pa.array(values, type=field.type))


def compact_valid_dataset(root: Path, valid_records: list[dict], bad_indices: list[int]) -> None:
    """Create a clean dataset containing only valid episodes, then swap it in."""
    import pyarrow.parquet as pq

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tmp_root = root.with_name(f"{root.name}_repair_tmp_{timestamp}")
    backup_root = root.with_name(f"{root.name}_bad_backup_{timestamp}")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    if backup_root.exists():
        shutil.rmtree(backup_root)

    info = load_dataset_info(root)
    stats_by_index = {
        int(item["episode_index"]): item
        for item in read_jsonl(root / "meta" / "episodes_stats.jsonl")
        if "episode_index" in item
    }

    (tmp_root / "data").mkdir(parents=True, exist_ok=True)
    (tmp_root / "meta").mkdir(parents=True, exist_ok=True)
    tasks_path = root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        shutil.copy2(tasks_path, tmp_root / "meta" / "tasks.jsonl")

    new_episodes: list[dict] = []
    new_stats: list[dict] = []
    global_frame_index = 0

    for new_episode_index, old_episode in enumerate(valid_records):
        old_episode_index = int(old_episode["episode_index"])
        old_path = episode_parquet_path(root, old_episode_index, info)
        table = pq.ParquetFile(old_path).read()
        frame_count = table.num_rows

        table = replace_arrow_column(table, "frame_index", list(range(frame_count)))
        table = replace_arrow_column(table, "episode_index", [new_episode_index] * frame_count)
        table = replace_arrow_column(table, "index", list(range(global_frame_index, global_frame_index + frame_count)))

        new_path = episode_parquet_path(tmp_root, new_episode_index, info)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, new_path)

        new_episodes.append(
            {
                "episode_index": new_episode_index,
                "tasks": old_episode.get("tasks", []),
                "length": frame_count,
            }
        )
        old_stats = stats_by_index.get(old_episode_index)
        if old_stats is not None:
            copied_stats = json.loads(json.dumps(old_stats))
            copied_stats["episode_index"] = new_episode_index
            new_stats.append(copied_stats)
        global_frame_index += frame_count

    write_jsonl(tmp_root / "meta" / "episodes.jsonl", new_episodes)
    if new_stats:
        write_jsonl(tmp_root / "meta" / "episodes_stats.jsonl", new_stats)

    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = global_frame_index
    if "splits" in info:
        info["splits"]["train"] = f"0:{len(new_episodes)}"
    with (tmp_root / "meta" / "info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=4)

    shutil.move(str(root), str(backup_root))
    shutil.move(str(tmp_root), str(root))
    print(f"已备份原数据集到：{backup_root}")
    print(f"已剔除坏 episode：{bad_indices}")
    print(f"保留好 episode：{len(new_episodes)} 条")


def cleanup_orphan_episode_files(root: Path, valid_episodes: int) -> None:
    """Remove episode files written after metadata failed to update.

    If save_episode crashes after writing parquet but before meta.save_episode,
    the dataset can contain episode_000XYZ.parquet files whose index is not in
    meta/episodes.jsonl. Those files should be deleted before appending.
    """
    pattern = re.compile(r"episode_(\d+)")
    removed = 0
    for path in list(root.rglob("episode_*")):
        match = pattern.search(path.stem)
        if not match:
            continue
        episode_index = int(match.group(1))
        if episode_index >= valid_episodes:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
    images_dir = root / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
        removed += 1
    if removed:
        print(f"已清理 {removed} 个未完整写入的临时/孤儿 episode 文件。")


def resize_image(image: np.ndarray, size: int = 256) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((size, size)))


def validate_camera_image(image: np.ndarray, camera_name: str) -> None:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise RuntimeError(f"{camera_name} image has invalid shape: {array.shape}")
    image_std = float(array.std())
    near_black_ratio = float((array < AUTO_NEAR_BLACK_VALUE).mean())
    if image_std < AUTO_MIN_IMAGE_STD or near_black_ratio > AUTO_MAX_NEAR_BLACK_RATIO:
        raise RuntimeError(
            f"{camera_name} image looks invalid/black: std={image_std:.3f}, "
            f"near_black_ratio={near_black_ratio:.2f}. "
            "Please restart the MuJoCo viewer or switch GL backend/headless mode."
        )


def create_or_load_dataset(args: argparse.Namespace) -> LeRobotDataset:
    root = Path(args.root or DEFAULTS[args.mode]["root"])
    repo_id = args.repo_id or DEFAULTS[args.mode]["repo_id"]
    args.dataset_action = "create"

    if root.exists():
        if args.force:
            action = "overwrite"
        elif args.append:
            action = "append"
        elif args.existing_action == "ask":
            action = choose_existing_dataset_action(root)
        else:
            action = args.existing_action

        if action == "exit":
            raise SystemExit("已退出，未修改数据集。")
        if action == "overwrite":
            print(f"删除旧数据集：{root}")
            shutil.rmtree(root)
            args.dataset_action = "overwrite"
        elif action == "append":
            print(f"继续采集，不覆盖旧数据：{root}")
            args.dataset_action = "append"
            validate_existing_dataset_schema(root, args.mode)
            valid_episodes = count_meta_episodes(root)
            if AUTO_REPAIR_INVALID_TAIL:
                valid_episodes = repair_invalid_tail(root)
            cleanup_orphan_episode_files(root, valid_episodes)
            return LeRobotDataset(repo_id, root=root)

    state_shape = expected_state_shape(args.mode)
    if args.mode == "il":
        obj_init_shape = (6,)
    else:
        obj_init_shape = (9,)

    print(f"Creating dataset: repo_id={repo_id!r}, root={root}")
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="franka",
        fps=args.fps,
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["action"],
            },
            "obj_init": {
                "dtype": "float32",
                "shape": obj_init_shape,
                "names": ["obj_init"],
            },
        },
        image_writer_threads=AUTO_IMAGE_WRITER_THREADS,
        image_writer_processes=0,
    )


def safe_clear_episode_buffer(dataset: LeRobotDataset) -> None:
    """Clear LeRobot's temporary episode buffer even after a failed save_episode.

    save_episode mutates the buffer before computing stats. If a MemoryError
    happens during stats, the official clear_episode_buffer can fail because the
    buffer is half-converted. Resetting through create_episode_buffer is safe for
    the next attempt.
    """
    try:
        dataset.clear_episode_buffer()
    except Exception as exc:
        print(f"清理临时 episode buffer 时遇到小问题，已强制重置：{exc}")
        dataset.episode_buffer = dataset.create_episode_buffer()


def make_env(args: argparse.Namespace):
    xml_path = DEFAULTS[args.mode]["xml"]
    if args.mode == "il":
        from mujoco_env.SimpleEnv1 import SimpleEnv

        env = SimpleEnv(
            xml_path,
            seed=args.seed,
            action_type="eef_pose",
            state_type="joint_angle",
            headless=args.headless,
            render_width=args.render_width,
            render_height=args.render_height,
        )
    else:
        from mujoco_env.SimpleEnv2 import SimpleEnv2

        env = SimpleEnv2(
            xml_path,
            seed=args.seed,
            action_type="eef_pose",
            state_type="joint_angle",
            headless=args.headless,
            render_width=args.render_width,
            render_height=args.render_height,
        )
    return env


def preflight_headless_renderer(args: argparse.Namespace) -> None:
    """在改动数据集目录前验证离屏上下文和两路相机，失败时给出可操作提示。"""
    print(f"Headless preflight: {render_diagnostics()}")
    env = None
    try:
        env = make_env(args)
        agent_image, wrist_image = env.grab_image()
        validate_camera_image(agent_image, "agent camera preflight")
        validate_camera_image(wrist_image, "wrist camera preflight")
        print(
            "Headless renderer OK: "
            f"agent={agent_image.shape}/std={float(np.std(agent_image)):.2f}, "
            f"wrist={wrist_image.shape}/std={float(np.std(wrist_image)):.2f}"
        )
    except Exception as exc:
        backend = os.environ.get("MUJOCO_GL", "unknown")
        fallback = (
            "Linux EGL 失败时请安装/检查 NVIDIA EGL；CPU 服务器可改用 "
            "--gl_backend=osmesa，并安装 libosmesa6。"
            if sys.platform.startswith("linux")
            else "请保持 --gl_backend=auto；Windows 会自动使用 WGL。"
        )
        raise RuntimeError(
            f"Headless renderer preflight failed with MUJOCO_GL={backend!r}. "
            f"{fallback}\nDiagnostics: {render_diagnostics()}"
        ) from exc
    finally:
        if env is not None:
            if hasattr(env, "close"):
                env.close()
            else:
                env.env.close_viewer()


def get_task(env, mode: str) -> str:
    if mode == "il":
        return TASK_NAME_IL
    return env.instruction


def get_target_body(env, mode: str) -> str:
    if mode == "il":
        return "body_obj_mug_5"
    return env.obj_target


def infer_grasp_offset(env, target_body: str, p_mug: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Return the TCP xy offset used to grasp the mug side.

    In this scene the handle-free side is reached with a positive world-frame
    y offset. Keeping x/y independent makes the expert easier to tune from the
    config block at the top of this file.
    """
    _ = (env, target_body, p_mug)
    return np.array([args.grasp_x_offset, args.grasp_y_offset], dtype=np.float64)


def clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm < 1e-9:
        return vec
    return vec / norm * max_norm


def open_video_writer(path: Path, width: int, height: int, fps: int):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def write_video_frame(
    writer,
    agent_image: np.ndarray,
    wrist_image: np.ndarray,
    label: str,
    width: int,
    height: int,
) -> None:
    import cv2

    left = cv2.resize(agent_image, (width, height))
    right = cv2.resize(wrist_image, (width, height))
    frame = np.concatenate([left, right], axis=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, "agent view", (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(
        frame,
        "wrist view",
        (agent_image.shape[1] + 10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    writer.write(frame)


def record_frame(
    *,
    env,
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    action: np.ndarray,
    task: str,
    physics_steps: int,
    writer=None,
    saved_count: int = 0,
) -> int:
    agent_image, wrist_image = env.grab_image()
    validate_camera_image(agent_image, "agent camera")
    validate_camera_image(wrist_image, "wrist camera")
    agent_256 = resize_image(agent_image)
    wrist_256 = resize_image(wrist_image)

    # observation 必须来自下发动作之前的时刻。IL/VLA 统一使用 7 个实际关节角
    # + 夹爪命令：它与绝对关节 action 处于同一空间，也不会出现欧拉角 ±pi 跳变。
    observation_state = env.get_joint_state().astype(np.float32)

    # step() 返回的是当前实际状态，不是动作标签。真正的监督目标是 IK/控制器
    # 本帧下发的绝对关节目标 + 归一化夹爪命令。
    env.step(action)
    command_action = env.get_command_action().astype(np.float32)

    dataset.add_frame(
        {
            "observation.image": agent_256,
            "observation.wrist_image": wrist_256,
            "observation.state": observation_state,
            "action": command_action,
            "obj_init": env.obj_init_pose.astype(np.float32),
        },
        task=task,
    )

    for _ in range(physics_steps):
        env.step_env()

    if writer is not None:
        agent_video, wrist_video = env.grab_image()
        write_video_frame(writer, agent_video, wrist_video, task, args.render_width, args.render_height)

    if args.render and not args.headless:
        try:
            env.render(idx=saved_count, total=args.num_demos)
        except TypeError:
            env.render(idx=saved_count)
    return 1


def hold(
    *,
    env,
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    gripper: float,
    frames: int,
    task: str,
    physics_steps: int,
    writer=None,
    saved_count: int = 0,
) -> int:
    total = 0
    for _ in range(frames):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper], dtype=np.float32)
        total += record_frame(
            env=env,
            dataset=dataset,
            args=args,
            action=action,
            task=task,
            physics_steps=physics_steps,
            writer=writer,
            saved_count=saved_count,
        )
    return total


def move_to(
    *,
    env,
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    target: np.ndarray,
    gripper: float,
    task: str,
    physics_steps: int,
    writer=None,
    saved_count: int = 0,
) -> int:
    total = 0
    target = target.astype(np.float64)
    stable = 0
    for _ in range(args.max_waypoint_frames):
        actual_p, _ = env.env.get_pR_body(body_name="tcp_link")
        target_delta = target - env.p0
        actual_delta = target - actual_p
        if np.linalg.norm(target_delta) < args.pos_tol and np.linalg.norm(actual_delta) < args.pos_tol:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
        dpos = clip_norm(target_delta, args.max_delta)
        action = np.concatenate([dpos, np.zeros(3), np.array([gripper])]).astype(np.float32)
        total += record_frame(
            env=env,
            dataset=dataset,
            args=args,
            action=action,
            task=task,
            physics_steps=physics_steps,
            writer=writer,
            saved_count=saved_count,
        )
    return total


def run_expert_episode(
    *,
    env,
    dataset: LeRobotDataset,
    args: argparse.Namespace,
    seed: int,
    saved_count: int,
    physics_steps: int,
    video_path: Path | None,
) -> AttemptResult:
    env.reset(seed=seed)
    if args.mode == "vla" and args.instruction:
        env.set_instruction(args.instruction)

    task = get_task(env, args.mode)
    target_body = get_target_body(env, args.mode)
    p_mug = env.env.get_p_body(target_body).copy()
    p_plate = env.env.get_p_body("body_obj_plate_11").copy()
    grasp_offset_xy = infer_grasp_offset(env, target_body, p_mug, args)
    grasp_xy = p_mug[:2] + grasp_offset_xy
    place_xy = p_plate[:2] + grasp_offset_xy

    tmp_video = None
    writer = None
    if video_path is not None:
        tmp_video = video_path.with_name(video_path.stem + ".tmp" + video_path.suffix)
        writer = open_video_writer(tmp_video, args.render_width, args.render_height, args.fps)

    frames = 0
    try:
        hover_mug = np.array([grasp_xy[0], grasp_xy[1], args.hover_z], dtype=np.float64)
        grasp = np.array([grasp_xy[0], grasp_xy[1], p_mug[2] + args.grasp_z_offset], dtype=np.float64)
        lift_mug = np.array([grasp_xy[0], grasp_xy[1], args.hover_z], dtype=np.float64)
        hover_plate = np.array([place_xy[0], place_xy[1], args.hover_z], dtype=np.float64)
        place = np.array([place_xy[0], place_xy[1], p_plate[2] + args.place_z_offset], dtype=np.float64)
        retreat = np.array([place_xy[0], place_xy[1], args.hover_z], dtype=np.float64)

        for target, gripper in [
            (hover_mug, 0.0),
            (grasp, 0.0),
        ]:
            frames += move_to(
                env=env,
                dataset=dataset,
                args=args,
                target=target,
                gripper=gripper,
                task=task,
                physics_steps=physics_steps,
                writer=writer,
                saved_count=saved_count,
            )

        frames += hold(
            env=env,
            dataset=dataset,
            args=args,
            gripper=1.0,
            frames=args.close_frames,
            task=task,
            physics_steps=physics_steps,
            writer=writer,
            saved_count=saved_count,
        )

        for target, gripper in [
            (lift_mug, 1.0),
            (hover_plate, 1.0),
            (place, 1.0),
        ]:
            frames += move_to(
                env=env,
                dataset=dataset,
                args=args,
                target=target,
                gripper=gripper,
                task=task,
                physics_steps=physics_steps,
                writer=writer,
                saved_count=saved_count,
            )

        frames += hold(
            env=env,
            dataset=dataset,
            args=args,
            gripper=0.0,
            frames=args.open_frames,
            task=task,
            physics_steps=physics_steps,
            writer=writer,
            saved_count=saved_count,
        )

        frames += move_to(
            env=env,
            dataset=dataset,
            args=args,
            target=retreat,
            gripper=0.0,
            task=task,
            physics_steps=physics_steps,
            writer=writer,
            saved_count=saved_count,
        )

        success = False
        for _ in range(args.settle_frames):
            frames += hold(
                env=env,
                dataset=dataset,
                args=args,
                gripper=0.0,
                frames=1,
                task=task,
                physics_steps=physics_steps,
                writer=writer,
                saved_count=saved_count,
            )
            if env.check_success():
                success = True
                break

        if writer is not None:
            writer.release()
            writer = None

        if success:
            dataset.save_episode()
            final_video = None
            if tmp_video is not None:
                video_path.parent.mkdir(parents=True, exist_ok=True)
                if video_path.exists():
                    video_path.unlink()
                tmp_video.rename(video_path)
                final_video = str(video_path)
            return AttemptResult(
                saved_index=saved_count,
                attempt=-1,
                seed=seed,
                success=True,
                frames=frames,
                task=task,
                video=final_video,
            )

        safe_clear_episode_buffer(dataset)
        if tmp_video is not None and tmp_video.exists():
            tmp_video.unlink()
        return AttemptResult(
            saved_index=None,
            attempt=-1,
            seed=seed,
            success=False,
            frames=frames,
            task=task,
            reason="success_check_failed",
        )
    except Exception as exc:
        if writer is not None:
            writer.release()
        safe_clear_episode_buffer(dataset)
        if tmp_video is not None and tmp_video.exists():
            tmp_video.unlink()
        return AttemptResult(
            saved_index=None,
            attempt=-1,
            seed=seed,
            success=False,
            frames=frames,
            task=task,
            reason=repr(exc),
        )


def load_report_results(dataset_root: Path) -> list[dict]:
    report_path = dataset_root / "auto_collect_report.json"
    if not report_path.exists():
        return []
    try:
        with report_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return list(payload.get("results", []))
    except Exception:
        return []


def next_attempt_start(dataset_root: Path, fallback: int) -> int:
    """续采时从历史报告延续 attempt 计数。

    每次尝试的随机种子是 ``seed + attempt``。若 attempt 只从已保存条数重新数起，
    上一轮里“尝试过但失败”的 attempt 编号会被重复使用，对应 seed 的物体布局
    与已保存 episode 完全相同，数据集里会出现重复演示。
    """
    attempts = [int(item.get("attempt", -1)) for item in load_report_results(dataset_root)]
    if not attempts:
        return fallback
    return max(max(attempts) + 1, fallback)


def save_report(dataset_root: Path, results: list[AttemptResult]) -> Path:
    """把本轮结果与历史报告合并写盘，保留完整的 attempt/seed 使用记录。"""
    report_path = dataset_root / "auto_collect_report.json"
    combined = load_report_results(dataset_root) + [asdict(r) for r in results]
    payload = {
        "attempts": len(combined),
        "saved": sum(1 for r in combined if r.get("success")),
        "results": combined,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return report_path


def cleanup_images_dir(dataset: LeRobotDataset) -> None:
    images_dir = dataset.root / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)


def main() -> None:
    args = parse_args()
    if args.grasp_xy_offset is not None:
        # Backward compatibility for old one-line commands. The new config uses
        # independent x/y offsets, and this scene only needs x.
        args.grasp_x_offset = args.grasp_xy_offset
    if args.force and args.append:
        raise ValueError("--force and --append cannot be used together.")
    if args.headless:
        args.render = False
    elif not args.no_render:
        args.render = True
    choose_mode_if_needed(args)
    if args.mode == "il" and args.instruction:
        raise ValueError("--instruction is only valid for --mode=vla.")

    # 先验证无头渲染，再执行 --force 删除/创建数据集；即使 EGL/WGL 配置失败，
    # 也不会误删用户已有数据。
    if args.headless:
        preflight_headless_renderer(args)

    dataset = create_or_load_dataset(args)
    env = make_env(args)
    physics_steps = args.physics_steps or max(1, int(round(env.env.HZ / args.fps)))
    existing_episodes = int(getattr(dataset, "num_episodes", 0)) if getattr(args, "dataset_action", "") == "append" else 0
    remaining_episodes = max(args.num_demos - existing_episodes, 0)
    max_attempts = args.max_attempts or max(remaining_episodes, 1) * 5
    video_dir = Path(args.video_dir) if args.video_dir else None

    print(f"Mode: {args.mode}")
    print(f"Dataset root: {dataset.root}")
    print(f"Target successful episodes: {args.num_demos}")
    if existing_episodes:
        print(f"Existing episodes: {existing_episodes}，将继续补到 {args.num_demos}")
    print(f"Max attempts: {max_attempts}")
    print(f"Physics steps per frame: {physics_steps}")
    print(f"Headless: {args.headless}")
    if args.headless:
        print(f"Headless GL: {render_diagnostics()}")

    if not args.headless:
        print(
            "注意：非 headless 采集时相机图从窗口帧缓冲读回，采集期间若窗口被遮挡/"
            "最小化/锁屏/息屏会导致后续全部黑图；长时间无人值守采集强烈建议加 --headless。"
        )

    results: list[AttemptResult] = []
    saved = existing_episodes
    # seed = args.seed + attempt；attempt 计数从历史报告延续，
    # 避免续采时重跑上一轮已用过的 seed、采出布局完全重复的 episode。
    attempt_start = next_attempt_start(Path(dataset.root), existing_episodes if existing_episodes else 0)
    consecutive_black = 0  # 连续黑图失败计数，用于熔断
    try:
        for attempt in range(attempt_start, attempt_start + max_attempts):
            if saved >= args.num_demos:
                break
            seed = args.seed + attempt
            video_path = None
            if video_dir is not None:
                video_path = video_dir / f"{args.mode}_episode_{saved:06d}_seed{seed}.mp4"

            result = run_expert_episode(
                env=env,
                dataset=dataset,
                args=args,
                seed=seed,
                saved_count=saved,
                physics_steps=physics_steps,
                video_path=video_path,
            )
            result.attempt = attempt
            results.append(result)

            if result.success:
                saved += 1
                consecutive_black = 0
                print(f"✓ saved {saved}/{args.num_demos} | seed={seed} | frames={result.frames} | task={result.task}")
            else:
                print(f"× failed attempt={attempt} | seed={seed} | frames={result.frames} | reason={result.reason}")
                # 熔断：连续多次因黑图/无效相机图失败，通常是窗口失活或渲染后端坏了，
                # 再跑下去只会攒一长串坏数据，直接停下报警。
                if result.reason and "invalid/black" in result.reason:
                    consecutive_black += 1
                    if consecutive_black >= AUTO_MAX_CONSECUTIVE_BLACK:
                        print(
                            f"熔断：已连续 {consecutive_black} 次相机黑图，停止采集。"
                            "请检查 MuJoCo 窗口是否被遮挡/最小化/锁屏，或改用 --headless 重新采集。"
                        )
                        break
                else:
                    consecutive_black = 0
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux 上把 free 的堆还给 OS，压制 RSS 增长
            except Exception:
                pass  # 非 Linux/无 libc 时忽略
    finally:
        if hasattr(env, "close"):
            env.close()
        else:
            env.env.close_viewer()
        cleanup_images_dir(dataset)

    report_path = save_report(Path(dataset.root), results)
    print(f"Done. Saved {saved}/{args.num_demos} successful episodes.")
    print(f"Report: {report_path}")
    if saved < args.num_demos:
        print("提示：如果成功率低，可以先调 --grasp_z_offset 或 --place_z_offset，例如 0.02/0.04/0.06 试几组。")


if __name__ == "__main__":
    main()
