"""
自动采集 LeRobot 示教数据。

这个脚本用一个简单、可调的“专家轨迹”自动完成桌面抓杯子放盘子的任务：

    杯子上方 -> 下探 -> 闭合夹爪 -> 抬起 -> 盘子上方 -> 放下 -> 松爪 -> 抬起

注意：
    1. 专家内部用 eef_pose + IK 控制末端移动；
    2. 真正写入数据集的 action 仍然是 7 个关节角 + 1 个夹爪命令；
    3. 因此生成的数据可以直接给现有的 ACT / Diffusion Policy / VLA 训练脚本使用。

常用命令：

    # 平时直接运行即可；启动后选择 il / vla
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
from dataclasses import asdict, dataclass
from pathlib import Path


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

# 服务器无桌面改成 True；本地想看 MuJoCo 窗口就保持 False。
AUTO_HEADLESS = False

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

# 专家轨迹参数：抓不住主要调这几个。
AUTO_GRASP_X_OFFSET = 0.0        # 抓取点相对杯子中心的 x 偏移；方向反了就改成 -0.060
AUTO_GRASP_Y_OFFSET = 0.060          # 抓取点相对杯子中心的 y 偏移；这个场景通常保持 0
AUTO_GRASP_Z_OFFSET = 0.012        # 下探高度；还高就试 0.0，太低碰撞就试 0.02
AUTO_PLACE_Z_OFFSET = 0.08         # 放到盘子上方的高度
AUTO_HOVER_Z = 1.05                # 搬运时抬起高度

# VLA 可选固定指令。None 表示随机红杯/蓝杯。
# 可写："Place the red mug on the plate." 或 "Place the blue mug on the plate."
AUTO_INSTRUCTION = None


if "--headless" in sys.argv or (AUTO_HEADLESS and "--no_headless" not in sys.argv and "--no-headless" not in sys.argv):
    # 必须在导入 mujoco / 环境封装之前设置。
    os.environ.setdefault("MUJOCO_GL", "egl")


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
from PIL import Image


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

    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=AUTO_HEADLESS, help="Use EGL offscreen rendering instead of a GLFW window.")
    parser.add_argument("--no_headless", action="store_false", dest="headless", help=argparse.SUPPRESS)
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
    parser.add_argument("--close_frames", type=int, default=28, help="Frames to keep gripper closed before lifting.")
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


def create_or_load_dataset(args: argparse.Namespace) -> LeRobotDataset:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

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
            cleanup_orphan_episode_files(root, count_meta_episodes(root))
            return LeRobotDataset(repo_id, root=root)

    if args.mode == "il":
        state_shape = (6,)
        obj_init_shape = (6,)
    else:
        state_shape = (7,)
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

    The current mug only needs a world-frame x offset. Keeping x/y independent
    makes the expert easier to tune from the config block at the top of this
    file.
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
    agent_256 = resize_image(agent_image)
    wrist_256 = resize_image(wrist_image)

    if args.mode == "il":
        observation_state = env.get_ee_pose().astype(np.float32)
    else:
        observation_state = None

    joint_q = env.step(action).astype(np.float32)
    if args.mode == "vla":
        observation_state = joint_q[:7].astype(np.float32)

    dataset.add_frame(
        {
            "observation.image": agent_256,
            "observation.wrist_image": wrist_256,
            "observation.state": observation_state,
            "action": joint_q,
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


def save_report(dataset_root: Path, results: list[AttemptResult]) -> Path:
    report_path = dataset_root / "auto_collect_report.json"
    payload = {
        "attempts": len(results),
        "saved": sum(1 for r in results if r.success),
        "results": [asdict(r) for r in results],
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

    results: list[AttemptResult] = []
    saved = existing_episodes
    attempt_start = existing_episodes if existing_episodes else 0
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
                print(f"✓ saved {saved}/{args.num_demos} | seed={seed} | frames={result.frames} | task={result.task}")
            else:
                print(f"× failed attempt={attempt} | seed={seed} | frames={result.frames} | reason={result.reason}")
            gc.collect()
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
