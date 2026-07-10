"""
================================================================================
本脚本作用：在「语言条件环境」里采集「带自然语言指令」的机器人演示数据集。
================================================================================

【一句话说明】
用键盘遥控（teleop）机械臂，把指定颜色的杯子放到盘子上，并把整个操作过程
（图像 + 机器人状态 + 动作 + 那句自然语言指令）录下来，存成数据集，
之后拿去训练「能听懂人话的机器人模型」（VLA 模型，如 π0 / SmolVLA）。

--------------------------------------------------------------------------------
【几个零基础概念，先讲清楚】

1) 什么是「遥操作 / teleop」？
   就是人用键盘（WASD 等）手动操控机器人，机器人不会自己动。
   我们一边操控、一边把每一帧的画面和动作记录下来，攒成「示范数据」，
   将来让模型「照着学」。这套流程和采集 ACT 数据时是一样的。

2) 什么是「语言条件（language-conditioned）」环境？
   普通采集：任务是固定的，比如永远「把杯子放到盘子上」，不需要告诉机器人做什么。
   语言条件：场景里有「多个不同颜色的杯子」+ 盘子，每一回合的任务都附带一句
   自然语言指令，例如 "Place the red mug on the plate."（把红色杯子放到盘子上）。
   也就是说，到底操作哪个杯子，是由这句「话」决定的。

3) 和普通采集到底差在哪？（核心区别，就一点）
   录数据时，每一帧除了图像/状态/动作，还要多存一个 task 字段 —— 也就是那句指令。
   这样模型学到的不是「闭眼把杯子放盘子」，而是「听懂这句话 → 做出对应动作」。

4) 为什么需要这种数据？
   因为我们想要的机器人是「能听懂指令」的：你说一句话，它照着做。
   只有训练数据里成对地包含了「指令 + 对应操作」，模型才学得会这种对应关系。

5) 什么是 VLA 模型？
   VLA = Vision-Language-Action（视觉-语言-动作）模型。
   它的输入是「看到的画面（Vision）」+「听到的指令（Language）」，
   输出是「机器人该怎么动（Action）」。π0、SmolVLA 都属于这一类。
   本脚本采集的数据，正是用来喂给这类模型训练的。

--------------------------------------------------------------------------------
【运行方式】（需要图形界面，因为要弹出仿真窗口用键盘操控）
    conda activate lerobot
    python vla/collect_data_language.py

采集完数据后，用 train_vla.py 这个脚本来训练 VLA 模型。

【按键说明】与采集 ACT 数据的脚本完全相同（详见下方键盘控制部分）。
提示：本脚本原为教程笔记本，后转写为可直接运行的 .py 脚本。
================================================================================
"""

# === 运行环境自举（本脚本位于子文件夹，确保从任何目录都能正常运行）===
# 它需要导入上一级目录的 mujoco_env，并访问项目根目录下的 ./asset、./demo_data、./ckpt 等资源。
# 下面把“项目根目录”（本文件所在目录的上一级）加入模块搜索路径，并把工作目录切换到根目录，
# 这样无论你在哪个目录下执行（比如 cd 进子文件夹，或从项目根目录运行）都不会找不到文件。
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)


# ======================================================================
# # 通过键盘采集演示数据
#
# 为给定的环境采集演示数据。
# 这里的「演示数据」就是人手动操控机器人完成任务的全过程录像（含动作）。
# 任务是抓取一个杯子并把它放到盘子上。当杯子位于盘子上、夹爪张开且末端执行器位于杯子上方时，环境会判定为成功。
# 注意：本环境是「语言条件」环境，场景里有多个不同颜色的杯子，
# 具体要操作哪个杯子由当前回合的自然语言指令（instruction）决定。
#
# 使用 WASD 控制 xy 平面，RF 控制 z 轴，QE 控制倾斜，方向键控制其余的旋转。
#
# SPACEBAR（空格键）会切换夹爪的状态，Z 键会重置环境并丢弃当前回合的数据。
#
# 对于叠加显示的图像，
# - 右上：智能体视角（Agent View）
# - 右下：第一人称视角（Egocentric View）
# - 左上：左侧视角（Left Side View）
# - 左下：俯视视角（Top View）
# ======================================================================

# --- 导入需要用到的工具库 ---
import sys
import random
import numpy as np                                  # 数值计算库（处理数组、向量）
import os                                           # 操作系统接口（这里用来判断文件夹是否存在）
from PIL import Image                               # 图像处理库（这里用来把画面缩放到 256x256）
from mujoco_env.SimpleEnv2 import SimpleEnv2            # 本项目的仿真环境类（语言条件版的「多杯子+盘子」场景）
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # LeRobot 的数据集类，负责按标准格式存数据
import glfw                                             # 读取键盘按键（Backspace 删除键用）

# --- 一些可调节的参数（采集前先设好）---

# 随机种子：控制每次运行时物体的摆放位置。
# 固定成一个数（如 0），每次运行杯子/盘子的位置都一样，便于复现、调试；
# 设成 None 则每次运行物体位置都随机化，数据更丰富多样。
SEED = 0
# SEED = None <- 取消注释这一行可随机化物体位置

REPO_NAME = 'franka_pnp_language'      # 数据集名字（repo id），后续训练时按这个名字找数据
NUM_DEMO = 20 # 要采集的演示数量    # 打算采集多少个回合（episode），采够这么多就自动停止
ROOT = "./demo_data_language" # 保存演示数据的根目录   # 数据存到这个文件夹里

xml_path = './asset/example_scene_y2.xml'   # 仿真场景描述文件（MuJoCo 的 XML，定义了机器人、杯子、盘子等）
# 定义环境：根据上面的场景文件创建仿真环境实例。
# state_type='joint_angle' 表示机器人状态用「关节角度」来表示。
PnPEnv = SimpleEnv2(xml_path, seed = SEED, state_type = 'joint_angle')

# ======================================================================
# ## 定义数据集特征并创建你的数据集！
#
# 这里要先「声明」每一帧数据里都要存哪些字段、每个字段是什么形状/类型。
# LeRobotDataset 会按这个声明，把数据存成训练时能直接读取的标准格式。
# 下面 features 里的每一项就是一个字段：
#   - observation.image       ：主相机看到的画面（机器人视角）
#   - observation.wrist_image ：手腕相机看到的画面（夹爪上的相机）
#   - observation.state       ：机器人当前状态
#   - action                  ：这一帧执行的动作
#   - obj_init                ：物体初始位置（只是记录用，训练时不用）
# 注意：那句「自然语言指令」不在这里声明，而是在每帧调用 add_frame 时
#      通过 task=... 单独传入（这正是「语言条件」采集比普通采集多出来的关键一步）。
# 数据集包含的内容如下：
# ```
# fps = 20,
# features={
#     "observation.image": {
#         "dtype": "image",
#         "shape": (256, 256, 3),
#         "names": ["height", "width", "channels"],
#     },
#     "observation.wrist_image": {
#         "dtype": "image",
#         "shape": (256, 256, 3),
#         "names": ["height", "width", "channel"],
#     },
#     "observation.state": {
#         "dtype": "float32",
#         "shape": (8,),
#         "names": ["state"], # 7 个关节角度 + 当前夹爪状态
#     },
#     "action": {
#         "dtype": "float32",
#         "shape": (8,),
#         "names": ["action"], # 7 个关节角度和 1 个夹爪
#     },
#     "obj_init": {
#         "dtype": "float32",
#         "shape": (9,),
#         "names": ["obj_init"], # 红杯+蓝杯+盘子各 3 个数的初始位置，训练中不使用。
#     },
# },
# ```
#
#
# 这会在 './demo_data_language' 文件夹中创建数据集，其结构如下：
#
# ```
# .
# ├── data
# │   ├── chunk-000
# │   │   ├── episode_000000.parquet
# │   │   └── ...
# ├── meta
# │   ├── episodes.jsonl
# │   ├── info.json
# │   ├── stats.json
# │   └── tasks.jsonl
# └──
# ```
# ======================================================================

# 先处理「数据集文件夹是否已存在」的情况：
# 如果之前已经采集过、文件夹还在，就问用户要不要删掉重来。
#   输入 y  -> 删除旧数据，重新开始采集（create_new 保持 True）
#   输入其他 -> 不删，改为「在已有数据集上继续」（create_new 设为 False）
create_new = True
if os.path.exists(ROOT):
    print(f"目录 {ROOT} 已存在。")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False


# 根据上面的判断，要么「新建一个空数据集」，要么「加载已有数据集继续采集」。
if create_new:
    # LeRobotDataset.create(...)：按照前面声明的 features 新建一个空数据集。
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root = ROOT,
                robot_type="franka",                   # 机器人型号标识
                fps=20, # 每秒 20 帧                 # 录制帧率：每秒存 20 帧
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
                        "shape": (8,),
                        "names": ["state"], # 7 个关节角度 + 当前夹爪状态
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (8,),
                        "names": ["action"], # 7 个关节角度和 1 个夹爪
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (9,),
                        "names": ["obj_init"], # 仅为物体的初始位置，训练中不使用。
                    },
                },
                # Windows 子进程会重新执行本脚本，递归创建 MuJoCo 场景并耗尽内存。
                image_writer_threads=10,
                image_writer_processes=0,
        )
else:
    # 不新建，直接打开磁盘上已有的数据集，后续采集的新回合会追加进去。
    print("从已有数据集加载")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

# ======================================================================
# ## 键盘控制
# 这一段只是说明书：列出每个按键控制机器人的哪个动作。
# 这套遥操作（teleop）按键和采集 ACT 数据时完全一样，会用就行。
# 你可以用键盘遥操作机器人并采集数据集
# ```
# ---------     -----------------------
#    w       ->        backward
# s  a  d        left   forward   right
# ---------      -----------------------
# 在 x, y 平面内
#
# ---------
# R: 向上移动
# F: 向下移动
# ---------
# 在 z 轴方向
#
# ---------
# Q: 向左倾斜
# E: 向右倾斜
# UP: 向上看
# Down: 向下看
# Right: 向右转
# Left: 向左转
# ---------
# 用于旋转
#
# ---------
# SPACEBAR: 切换夹爪状态
# --------
#
# ---------
# z: 重置
# --------
# ```
# 重置环境会移除当前演示的缓存数据并重新开始采集。
# ======================================================================

# ======================================================================
# ### 现在让我们遥操作机器人并采集数据吧！
#
# **要收到成功信号，你必须松开夹爪并向上移动到杯子上方！**
# ======================================================================

# --- 准备主循环用到的几个变量 ---
action = np.zeros(7)        # 当前动作，先初始化为「全 0」（即不动）。7 维 = 末端位姿增量(dx,dy,dz,droll,dpitch,dyaw) + 1 个夹爪状态
episode_id = 0              # 已经采集完成的回合计数，从 0 开始
record_flag = False # 当机器人开始移动时才开始记录
                    # 这个开关的作用：避免把「人还没开始操作」时的静止画面也录进去。
                    # 一旦检测到机器人动了，就翻成 True，从那一刻起才正式记录数据。

# 主循环：只要仿真窗口还开着，且还没采够 NUM_DEMO 个回合，就一直跑。
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()       # 推进底层物理仿真（让画面持续刷新、保持流畅）
    if PnPEnv.env.loop_every(HZ=20):    # 控制采集节奏：每秒只执行 20 次下面的逻辑（对应 fps=20）
        # 删除键（Backspace）：立即从数据集删掉最近保存的一条回合，用于误按回车存错后当场补救。
        # 做法：裁掉磁盘上最后一条 episode，再重载数据集使内存与磁盘一致，然后重置环境可继续采集。
        if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_BACKSPACE):
            if episode_id <= 0:
                print("[删除] 还没有已保存的回合可删。")
            else:
                from auto_collect import truncate_dataset_tail
                root_path = dataset.root
                try:
                    if getattr(dataset, "image_writer", None) is not None:
                        dataset.stop_image_writer()   # 先停异步图像写，避免文件占用/双写
                except Exception:
                    pass
                truncate_dataset_tail(root_path, episode_id - 1)   # 删掉最后一条（保留 episode_id-1 条）
                dataset = LeRobotDataset(REPO_NAME, root=root_path)  # 重载，使内存对象与磁盘一致
                try:
                    dataset.start_image_writer(num_processes=0, num_threads=4)  # 重新开启图像写，便于继续采集
                except Exception:
                    pass
                episode_id -= 1
                record_flag = False
                PnPEnv.reset()   # 顺便重置环境，准备重新采这一条
                print(f"[已删除] 最近一条回合已删除，现有 {episode_id} 条；环境已重置，可重新采集。")
        # 检查该回合是否完成：由人按回车手动确认，不做自动成功判定，采集质量完全由人把关。
        if PnPEnv.is_finish_pressed():   # 按回车=手动确认本轮完成、存盘、进入下一轮
            # 这一回合成功了 -> 把整段录像存盘，然后重置环境，准备下一回合。
            dataset.save_episode()      # 把缓存的这一整回合写入数据集
            PnPEnv.reset()              # 重置环境（重新随机摆放物体、机器人归位、换一句新指令）
            episode_id += 1             # 完成回合数 +1
        # 遥操作机器人：读取键盘输入，得到这一帧的动作 action 和「是否要求重置」的标志 reset。
        action, reset  = PnPEnv.teleop_robot()
        # 如果机器人此前一直没动（record_flag 还是 False），而现在动作不全为 0，
        # 说明人开始操作了 -> 打开记录开关，从这一帧起才存数据。
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("开始记录")
        if reset:
            # 用户按了 'z' 键要求重置：放弃当前这回合（不保存），重新来过。
            # 重置环境并清空回合缓冲区
            # 可以通过按下 'z' 键来完成
            # PnPEnv.reset(seed=SEED)
            PnPEnv.reset()                  # 重置环境
            dataset.clear_episode_buffer()  # 丢掉这回合已经攒在缓存里的帧（不写入磁盘）
            record_flag = False             # 记录开关复位，等下次机器人再动时重新打开
        # 推进环境一步
        # 获取末端执行器位姿和图像
        agent_image,wrist_image = PnPEnv.grab_image()   # 抓取这一帧的两路相机画面：主相机 + 手腕相机
        # # 缩放到 256x256
        # 把两张图都统一缩放到 256x256，保证存进数据集的图像尺寸一致（模型要求固定输入尺寸）。
        agent_image = Image.fromarray(agent_image)      # numpy 数组 -> PIL 图像对象
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))    # 缩放主相机画面
        wrist_image = wrist_image.resize((256, 256))    # 缩放手腕相机画面
        agent_image = np.array(agent_image)             # 再转回 numpy 数组
        wrist_image = np.array(wrist_image)
        # 观测来自动作下发前；状态包含 7 个实际关节角和当前夹爪命令。
        robot_state = PnPEnv.get_joint_state().astype(np.float32)
        # step() 返回当前状态，动作监督必须改用真正下发的目标命令。
        PnPEnv.step(action)
        command_action = PnPEnv.get_command_action().astype(np.float32)
        if record_flag:
            # 只有在记录开关打开后，才把这一帧写入数据集。
            # 将该帧添加到数据集
            # ★关键：注意最后的 task=PnPEnv.instruction —— 这就是「语言条件」采集的精髓！
            #   它把当前回合那句自然语言指令（如 "Place the red mug on the plate."）
            #   和这一帧的图像/状态/动作绑定在一起存下来。
            #   普通（非语言条件）采集不会存这个字段；正是因为存了它，
            #   训练出来的 VLA 模型才能学会「听懂指令并做出对应动作」。
            #   （PnPEnv.instruction 是英文字符串，因为它是要喂给模型的输入，保持英文原样不翻译。）
            dataset.add_frame( {
                    "observation.image": agent_image,           # 主相机画面
                    "observation.wrist_image": wrist_image,     # 手腕相机画面
                    "observation.state": robot_state,           # 7 个实际关节角 + 当前夹爪状态
                    "action": command_action,                   # 7 个目标关节角 + 1 个夹爪命令
                    "obj_init": PnPEnv.obj_init_pose,           # 物体初始位置（仅记录，训练不用）
                    # "task": PnPEnv.instruction,               # 指令不在这里传，而是用下面的 task= 参数传
                }, task = PnPEnv.instruction                    # ← 把自然语言指令作为本帧的任务标签存入
            )
        PnPEnv.render(teleop=True, idx=episode_id)      # 刷新画面显示（含遥操作叠加层、当前回合编号）

PnPEnv.env.close_viewer()       # 循环结束（采够了或关了窗口），关闭仿真窗口

# 清理 images 文件夹
# 数据集写入过程中会产生一个临时的 images 文件夹，采集完成后把它删掉即可。
import shutil
images_dir = dataset.root / 'images'
if images_dir.exists():
    shutil.rmtree(images_dir)
