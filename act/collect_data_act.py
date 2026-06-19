"""
========================================================================
步骤 1：用键盘遥操作采集示教数据（单物体抓放任务）。
========================================================================

【这个脚本到底在干嘛？（一句话版）】
    你坐在电脑前，用键盘像玩游戏一样操控一只虚拟（仿真里的）机械臂，
    把一个杯子抓起来、放到盘子上。整个过程中，程序会一帧一帧地把
    “机器人看到的画面 + 机器人当时的姿态 + 你按键产生的动作”记录下来，
    存成一份“示教数据”。以后我们就拿这份数据去训练 AI 模型（ACT 模型），
    让机器人学会自己完成这个抓放动作，不再需要人来遥控。

【几个零基础名词，先用大白话扫盲】
    - 仿真环境(simulation/MuJoCo)：一个“虚拟世界”，里面有机械臂、杯子、盘子，
      物理规律（重力、碰撞）都模拟得跟真实差不多。好处是不怕摔坏、可以无限次重来。
    - 遥操作(teleoperation)：人用键盘/手柄“远程”操控机器人，这里就是你按键控制机械臂。
    - 示教数据(demonstration)：人亲手做一遍给机器看的“标准答案”，AI 照着学。
    - 末端执行器(end-effector)：机械臂最前端那只“手”（夹爪），末端指的就是手腕末端那个点。
    - 夹爪(gripper)：机械臂的“手指”，能张开/合拢，用来夹住或松开物体。
    - 回合(episode)：完整做一次任务（从开始到把杯子放上盘子）算一个回合。
    - LeRobot 数据集：一种业界通用的机器人数据保存格式（由 HuggingFace 的 LeRobot 项目定义），
      存好后可以直接喂给训练脚本使用。

【运行方式（需要图形界面，会弹出 MuJoCo 窗口并读取键盘）】
    conda activate lerobot
    python act/collect_data_act.py

【遥操作按键速查】
    W/S 前后  A/D 左右  R/F 上下  Q/E 倾斜  方向键 俯仰/偏转
    空格 切换夹爪开合   Z 重置并丢弃当前回合

提示：本脚本原为教程笔记本（一个个单元格的 Jupyter notebook），现已转成普通 .py 脚本，
自上而下顺序执行即可。采集好的数据之后用训练脚本 train_vla.py 来训练 ACT 模型。
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
# # 通过键盘采集示教数据
#
# 为给定环境采集示教数据。
# 本任务是抓取一个杯子并把它放到盘子上。当杯子位于盘子上、夹爪张开、且末端执行器位于杯子上方时，环境判定任务成功。
#
# <img src="./media/teleop.gif" width="480" height="360">
#
# 使用 WASD 控制 xy 平面，RF 控制 z 轴，QE 控制倾斜，方向键（ARROW）控制其余的旋转。
#
# 空格键（SPACEBAR）切换夹爪状态，Z 键重置环境并丢弃当前回合的数据。
#
# 对于叠加显示的图像：
# - 右上：智能体视角（Agent View）
# - 右下：第一人称视角（Egocentric View）
# - 左上：左侧视角（Left Side View）
# - 左下：俯视视角（Top View）
# ======================================================================

# ---- 导入用到的库 ----
import sys
import random
import numpy as np                  # 数值计算库，这里主要用来处理数组（图像、动作向量等）
import os                           # 操作系统相关，用来判断文件夹是否存在等
from PIL import Image               # 图像处理库，用来把图片缩放到统一尺寸
# SimpleEnv 是本教程自带的“仿真环境”封装类，负责管理 MuJoCo 里的机械臂、杯子、盘子、相机等。
from mujoco_env.SimpleEnv1 import SimpleEnv
# LeRobotDataset 是 LeRobot 提供的数据集类，负责把采集到的每一帧按规定格式存盘。
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ---- 随机种子设置 ----
# “随机种子(seed)”决定了“随机”出来的结果。固定同一个种子，每次运行物体（杯子/盘子）
# 摆放的位置都完全一样，方便调试和复现；设成 None 则每次位置随机。
# 如果想随机化物体位置，把它设为 None
# 如果固定随机种子，每次的物体位置都会相同
SEED = 0
# SEED = None <- 取消注释这一行即可随机化物体位置

# ---- 数据集相关的基本配置 ----
REPO_NAME = 'omy_pnp'               # 数据集名字（repo_id），起个标识用，omy 是机械臂型号、pnp=pick and place 抓放
NUM_DEMO = 1 # 要采集的示教数据数量（这里只采集 1 个回合；想多采就调大）
ROOT = "./demo_data" # 保存示教数据的根目录（数据最终存到当前目录下的 demo_data 文件夹）

# 任务的文字描述。注意：这个字符串会被一起存进数据集，训练时模型也会读到它，
# 所以不要随意改动（改了就和已有数据对不上了）。
TASK_NAME = 'Put mug cup on the plate'
xml_path = './asset/example_scene_y.xml'   # 场景定义文件：描述了仿真世界里有哪些物体、相机、机械臂
# 定义环境
# 创建仿真环境实例：传入场景文件、随机种子，state_type='joint_angle' 表示
# 用“关节角度”来描述机器人状态（机械臂有好几个关节，每个关节转了多少度）。
PnPEnv = SimpleEnv(xml_path, seed = SEED, state_type = 'joint_angle')

# ======================================================================
# ## 定义数据集特征并创建你的数据集！
#
# 【什么是“特征(features)”？】
#   就是告诉 LeRobot：我这份数据集里，每一帧都要存哪几样东西、每样东西
#   长什么样（是图片还是数字？尺寸多大？数据类型是什么？）。相当于先把
#   “表格的列”和“每列的格式”定义清楚，后面才能往里填数据。
#
# 【下面这几样东西分别是什么（重点扫盲）】
#   - observation.image：相机拍到的画面（智能体视角），256x256 的彩色图（3 通道=RGB）。
#       “observation(观测)”=机器人在某一刻“看到/感知到”的信息。
#   - observation.wrist_image：装在机械臂手腕上的相机拍到的画面（第一人称/近距离视角）。
#   - observation.state：机器人当前的“状态”，这里是末端执行器的位姿，6 个数字：
#       x,y,z（在空间中的位置）+ roll,pitch,yaw（绕三个轴的旋转角度，即朝向）。
#   - action：“动作”，即这一帧机器人实际执行的指令，7 个数字 = 6 个关节角 + 1 个夹爪开合。
#       简单理解：observation 是“看到啥”，action 是“于是做了啥”，AI 就是学这个对应关系。
#   - obj_init：物体（杯子）的初始位置，仅作记录，训练时不用。
#
# 数据集的构成如下：
# ```
# fps = 20,                # fps=每秒帧数，20 表示每秒记录 20 帧（见下方 HZ=20 解释）
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
#         "shape": (6,),
#         "names": ["state"], # x, y, z, roll, pitch, yaw
#     },
#     "action": {
#         "dtype": "float32",
#         "shape": (7,),
#         "names": ["action"], # 6 joint angles and 1 gripper
#     },
#     "obj_init": {
#         "dtype": "float32",
#         "shape": (6,),
#         "names": ["obj_init"], # just the initial position of the object. Not used in training.
#     },
# },
# ```
#
#
# 这会在 './demo_data' 文件夹下生成数据集，其结构如下所示：
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

# ---- 先处理“数据集文件夹已经存在”的情况 ----
# create_new 是一个开关：True 表示“要新建一份数据集”，False 表示“沿用旧的”。
create_new = True
if os.path.exists(ROOT):
    # 如果保存目录已经存在（说明之前采集过），问问用户要不要删掉重来，
    # 避免新旧数据混在一起。
    print(f"目录 {ROOT} 已存在。")
    ans = input("Do you want to delete it? (y/n) ")   # 等待用户输入 y 或 n
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)        # 删除整个旧目录及里面所有文件
    else:
        create_new = False         # 用户选择保留，则不新建，改为加载旧数据集


if create_new:
    # 真正创建一份全新的、空的 LeRobot 数据集。
    # 这里把上面说明过的“特征(features)”一项项填进去，相当于把表格的列定义好。
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,        # 数据集标识名
                root = ROOT,              # 存到哪个文件夹
                robot_type="omy",         # 机器人型号标记
                fps=20, # 每秒 20 帧（每秒记录 20 张“快照”）
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
                        "shape": (6,),
                        "names": ["state"], # x, y, z, roll, pitch, yaw
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"], # 6 个关节角和 1 个夹爪
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["obj_init"], # 仅为物体的初始位置，不用于训练。
                    },
                },
                image_writer_threads=10,    # 用 10 个线程异步写图片（图片多，多开几个线程存得快）
                image_writer_processes=5,   # 配合上面，用 5 个进程并行写图片
        )
else:
    # 用户选择保留旧数据，则直接把已有数据集加载进来（在它后面继续追加新回合）。
    print("从已有数据集加载")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

# ======================================================================
# ## 键盘控制
# 你可以用键盘遥操作机器人并采集数据集
#
# 【怎么理解下面这些按键？】
#   想象你在操控机械臂前端那只“手”（末端执行器）在三维空间里移动+转向：
#   - WASD 控制它在水平桌面（x-y 平面）上前后左右滑动；
#   - R/F 控制它上下升降（z 轴，即离桌面高低）；
#   - Q/E 和方向键控制它“转头/倾斜”（改变朝向，即 roll/pitch/yaw 三种旋转）；
#   - 空格键让夹爪“张开<->合拢”来回切换（夹住杯子或松手放下）；
#   - z 键放弃本回合、把环境重置回初始状态重新开始。
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
# SPACEBAR: 切换夹爪
# --------
#
# ---------
# z: 重置
# --------
# ```
# 重置环境会清除当前示教数据的缓存并重新开始采集。
# ======================================================================

# ======================================================================
# ### 现在让我们遥操作机器人并采集数据吧！
#
# 【成功判定逻辑（很重要，否则采不到“成功”的数据）】
#   程序怎么知道你完成任务了？它会同时检查三件事：
#     (1) 杯子已经稳稳落在盘子上；
#     (2) 夹爪是张开的（你已经松手放下杯子，而不是还夹着）；
#     (3) 末端执行器（那只手）位于杯子上方（已经抬手离开）。
#   三个条件都满足，才算这一回合“成功”，数据才会被保存。
#
# **所以：放下杯子后，记得松开夹爪并向上移动到杯子上方，才能收到成功信号！**
# ======================================================================

# ======================================================================
# ====== 主循环：一边遥操作机器人，一边采集数据 ======
# ======================================================================

# action：本帧要执行的动作，先初始化成 7 个 0（什么都不做）。np.zeros(7) 生成 [0,0,0,0,0,0,0]。
action = np.zeros(7)
# episode_id：已经采集完成的回合数，从 0 开始数，达到 NUM_DEMO 就停。
episode_id = 0
# record_flag：是否“正在记录”的开关。一开始为 False，等你真正动了机械臂才开始录，
# 这样可以避免把开头“发呆不动”的无效帧也存进去。
record_flag = False # 当机器人开始移动时再开始记录

# 循环条件：仿真窗口还开着（人没关掉） 且 还没采够 NUM_DEMO 个回合。
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    # 让仿真世界往前推进一小步（物理引擎更新一次，比如重力让杯子下落一点点）。
    PnPEnv.step_env()
    # loop_every(HZ=20)：控制“每秒只执行 20 次”下面的采集逻辑。
    # 仿真内部跑得很快（一秒可能上千步），但我们只想每秒记录 20 帧（即每隔 1/20=0.05 秒记一次），
    # 所以用它来“限速”。HZ=20 就是 20 赫兹=每秒 20 次的意思。
    if PnPEnv.env.loop_every(HZ=20):
        # 检查当前回合是否结束（任务是否成功完成）
        # check_success 返回 True 表示：杯子已在盘子上、夹爪张开、且末端在杯子上方 —— 判定成功。
        done = PnPEnv.check_success()
        if done:
            # 成功了！把这一整个回合的数据正式存盘，然后重置环境，准备采下一回合。
            # save_episode：把刚才一帧帧攒下来的数据打包写成一个 episode 文件。
            dataset.save_episode()
            # reset：把仿真“重置”——机械臂回到初始姿势，杯子盘子重新摆放，相当于“重开一局”。
            PnPEnv.reset(seed = SEED)
            episode_id += 1        # 已完成回合数 +1
        # 遥操作机器人：读取你此刻按下的键，换算成动作。
        # action 是“末端执行器位姿的增量(移动/旋转多少) + 夹爪状态”，reset 是“你是否按了 z 键想重置”。
        action, reset  = PnPEnv.teleop_robot()
        # 一旦检测到你真的动了（动作之和不为 0），就打开记录开关，开始正式录制。
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("开始记录")
        if reset:
            # 你按了 'z' 键：放弃当前这一回合，重置环境并清空已经攒着的（还没存盘的）数据。
            # 重置环境并清空回合缓冲区
            # 可以通过按 'z' 键来触发
            PnPEnv.reset(seed=SEED)
            # PnPEnv.reset()
            dataset.clear_episode_buffer()   # 清掉缓冲区里这回合的临时数据（相当于“这局不要了”）
            record_flag = False              # 关掉记录开关，等下次重新动起来再录
        # 采集这一帧的“观测”信息：
        # 获取末端执行器位姿（6 个数：x,y,z + roll,pitch,yaw），作为机器人的 state。
        ee_pose = PnPEnv.get_ee_pose()
        # 抓取两路相机画面：agent_image=智能体视角全景，wrist_image=手腕相机近景。
        agent_image,wrist_image = PnPEnv.grab_image()
        # ---- 把图像统一缩放到 256x256 ----
        # 为什么要缩放？因为数据集特征里写死了图像是 256x256；相机原始尺寸可能更大，
        # 统一成同一尺寸，训练时模型才好处理（输入大小要一致）。
        # Image.fromarray：把 numpy 数组形式的图转成 PIL 图像对象，才能调用 resize。
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))   # 缩放到 256x256
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)            # 再转回 numpy 数组，方便后续存储
        wrist_image = np.array(wrist_image)
        # 把动作 action 真正下发给机器人执行，并让仿真前进一步；
        # 返回 joint_q = 执行后机器人的 7 个关节量（6 关节角 + 1 夹爪），作为本帧记录的 action。
        joint_q = PnPEnv.step(action)
        if record_flag:
            # 只有在“记录中”才把这一帧加入数据集。
            # add_frame：往当前回合的缓冲区里追加“一帧”数据（一帧=某一时刻的观测+动作）。
            # 等整回合采完、判定成功后，前面的 save_episode 会把这些帧一起存盘。
            dataset.add_frame( {
                    "observation.image": agent_image,       # 全景相机画面
                    "observation.wrist_image": wrist_image, # 手腕相机画面
                    "observation.state": ee_pose,           # 末端位姿（机器人当前状态）
                    "action": joint_q,                      # 本帧执行的动作（关节角+夹爪）
                    "obj_init": PnPEnv.obj_init_pose,        # 物体初始位置（仅记录，训练不用）
                    # "task": TASK_NAME,
                }, task = TASK_NAME                          # 任务文字描述
            )
        # 把当前画面渲染显示到 MuJoCo 窗口里（teleop=True 表示叠加显示遥操作相关的提示画面）。
        PnPEnv.render(teleop=True)

# 循环结束（采够了或人关掉了窗口），关闭仿真窗口。
PnPEnv.env.close_viewer()

# ---- 收尾清理 ----
# 采集过程中临时图片会落在 images 文件夹里，存盘后这个中间文件夹就没用了，删掉它节省空间。
# 清理 images 文件夹
import shutil
shutil.rmtree(dataset.root / 'images')
