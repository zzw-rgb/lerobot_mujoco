"""
步骤 4：在仿真中"部署"训练好的 ACT 策略（rollout）。

【这个脚本到底在干嘛？——用大白话讲】
机器学习分两个阶段：
  1) 训练（training）：拿大量人类演示数据去"教"模型怎么做某件事，得到一个"学会了的模型"，
     这个学好的模型会被存成文件，叫做"检查点（checkpoint）"。本项目里训练 ACT 用的是 train_act.py
     （另外 train_vla.py 用于训练别的策略）。
  2) 部署/推理（deployment / inference）：把训练好的检查点加载回来，让它"真正干活"。
本脚本做的就是第 2 件事：把训练好的 ACT 策略加载进来，放到 MuJoCo 这个物理仿真器里，
让它一步步地自己控制机械臂，把杯子抓起来放到盘子上。

【什么是 rollout（推演 / 跑一遍）？】
rollout 就是"让策略从头到尾控制机器人把任务做一遍"的过程：
每一小步，策略先"看一眼"当前画面和机器人状态，然后"决定"下一个动作，机器人执行，
画面随之改变，再看一眼、再动一下……如此循环，直到任务完成。这种"看一眼-动一下"
不断反馈的方式叫"闭环控制（closed-loop control）"，就像人闭着眼睛抓不准东西、
必须睁着眼边看边调整一样。

从 ./ckpt/act_franka 加载检查点，在 MuJoCo 抓放环境中自动执行并判断成功。

运行方式（需要 GPU，且 ./ckpt/act_franka 下要有 train_act.py 训练好的检查点）：
    conda activate lerobot
    python act/deploy_act.py

提示：本脚本原为教程笔记本（notebook），现已整理成普通 Python 脚本，自上而下顺序执行即可。
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
# # 部署训练好的策略
#
# 在仿真中部署训练好的策略。
# ======================================================================

# ----------------------------------------------------------------------
# 导入需要用到的工具库（import 就是"先把要用的工具箱搬进来"）。
# 这里既有 lerobot（一个开源机器人学习框架）里的各种组件，也有处理张量/图像的通用库。
# ----------------------------------------------------------------------
# LeRobotDataset / LeRobotDatasetMetadata：读取数据集本身、以及数据集的"元信息"
# （元信息 = 关于数据的数据，比如有哪些特征字段、各特征的均值方差等统计量）。
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np  # 数值计算库，本脚本主要用它来处理动作数组
from lerobot.common.datasets.utils import write_json, serialize_dict
# ACTConfig：ACT 策略的"配置类"（决定模型怎么搭、怎么推理）；ACTPolicy：ACT 策略本体。
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType  # 用于区分哪些是"动作"特征、哪些是"观测"特征
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch  # PyTorch：深度学习框架，模型的输入输出都用它的"张量(tensor)"表示
from PIL import Image  # 图像处理库，用来把数组转成图片对象、做缩放等
import torchvision  # 提供图像的常用变换（如把图片转成张量）

# ======================================================================
# ## 加载策略
# ======================================================================

# device 指定模型在哪块硬件上计算。'cuda' 表示用 NVIDIA 显卡（GPU），速度快；
# 若没有 GPU 可改成 'cpu'（会慢很多）。模型和它的输入数据必须放在同一个 device 上。
device = 'cuda'

# 读取数据集的"元信息"。注意：部署阶段我们并不需要训练数据本身，
# 但需要它的元信息——尤其是各特征的统计量（均值/方差），因为模型在训练时
# 对输入做过"归一化（标准化）"，部署时必须用同样的统计量做同样的预处理，否则模型会"看不懂"。
dataset_metadata = LeRobotDatasetMetadata("franka_pnp", root='./demo_data')
# 把数据集里的字段整理成"策略能识别的特征定义"，每个特征带有类型/形状等信息。
features = dataset_to_policy_features(dataset_metadata.features)
# 从所有特征里挑出"动作（ACTION）"类的，作为模型的【输出】特征（即模型要预测什么）。
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
# 剩下的（非动作）特征就是模型的【输入】特征（即模型靠什么来做判断：机器人状态、摄像头画面等）。
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# 这里特意把"腕部相机图像"从输入特征里去掉：本配置只用主视角相机 + 机器人状态来决策。
input_features.pop("observation.wrist_image")
# 策略通过一个配置类来初始化，本例中为 `ACTConfig`。在这个示例里，
# 我们直接使用默认值，因此除了输入/输出特征之外不需要传入其他参数。
# 使用时序集成（temporal ensemble）来让轨迹预测更平滑
# 几个关键参数通俗解释：
#   chunk_size=10           ：ACT 一次会"预测未来 10 步"的动作（像下棋时一次想好接下来几步）。
#   n_action_steps=1        ：但每次实际只执行其中 1 步，然后重新看画面再预测——这就是闭环控制。
#   temporal_ensemble_coeff ：把"对同一时刻在不同时间点做的多次预测"加权融合，让动作更平滑、不抖。
cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size= 10, n_action_steps=1, temporal_ensemble_coeff = 0.9)
# 计算各特征在时间轴上的相对时间戳偏移（ACT/扩散类策略需要知道"取哪些时刻的数据"）。
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
# 现在可以用这个配置和数据集统计量来实例化我们的策略。
# from_pretrained('./ckpt/act_franka', ...)：这一步就是"加载训练好的检查点"——
# 把之前 train_act.py 训练出来、存在 ./ckpt/act_franka 文件夹里的模型权重读回来，得到一个能用的策略。
# dataset_stats 把上面说的归一化统计量交给策略，确保推理时的预处理和训练时一致。
policy = ACTPolicy.from_pretrained('./ckpt/act_franka', config = cfg, dataset_stats=dataset_metadata.stats)
# 把整个模型搬到指定硬件（GPU/CPU）上。
policy.to(device)

# ======================================================================
# ## 加载环境
# ======================================================================

# 加载仿真"环境"。MuJoCo 是一个物理仿真器，能在电脑里模拟出机械臂、桌子、杯子、盘子，
# 以及它们之间的接触、重力等物理效果——相当于一个"虚拟实验场"，让策略在里面安全地练手/演示。
from mujoco_env.SimpleEnv1 import SimpleEnv
# 场景的 XML 文件：描述了这个虚拟世界里有哪些物体、它们的位置和外观（相当于"舞台布景图纸"）。
xml_path = './asset/example_scene_y.xml'
# 创建抓放（Pick-and-Place，简称 PnP）环境实例。
# action_type='joint_angle' 表示：策略输出的动作被理解为"机械臂各关节的角度"。
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

# ======================================================================
# ## 执行策略推演（Roll-Out）
# ======================================================================

# ---- 推演前的准备工作 ----
step = 0                 # 步数计数器，从 0 开始；后面用它来计算时间戳。
PnPEnv.reset(seed=0)     # 把环境恢复到初始状态（杯子/盘子摆回起点）。seed=0 固定随机种子，保证每次场景一致、结果可复现。
policy.reset()           # 重置策略的"内部记忆/动作队列"。ACT 会缓存预测好的动作序列，开始新一轮前必须清空，避免用到上一轮的残留。
policy.eval()            # 把模型切到"评估/推理模式"。训练时模型会开启 Dropout、BatchNorm 更新等随机行为；
                         # 推理时必须关掉这些，让输出稳定、确定——就像考试时不能再"边学边改"。
save_image = True        # 一个标志位（本脚本里仅作记录用途，不影响主流程）。
img_transform = torchvision.transforms.ToTensor()  # 定义图像变换：把 PIL 图片转成 PyTorch 张量，并自动把像素值从 0~255 归一化到 0~1。

# ---- 主循环：只要仿真窗口还开着，就持续运行 ----
# 这是一个典型的"闭环控制"循环：不停地"推进仿真 → 到点了就看一眼、决策、动一下"。
while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()  # 让物理仿真向前推进一个极小的时间片（物理引擎的内部步进，频率很高）。
    # loop_every(HZ=20)：每秒只触发 20 次（即每 1/20 秒做一次决策）。
    # 为什么不每个物理步都决策？因为物理步进非常密集，没必要也来不及每步都让神经网络推理一次；
    # 以固定频率（这里 20Hz）"看一眼再动一下"既够用又高效。
    if PnPEnv.env.loop_every(HZ=20):
        # 检查任务是否已完成
        success = PnPEnv.check_success()  # 由环境判断"杯子是否已经放到盘子上"。
        if success:
            print('成功')
            # 重置环境与动作队列
            policy.reset()        # 清空策略缓存的动作，准备从头再来。
            PnPEnv.reset(seed=0)  # 环境复位，开始新一轮演示。
            step = 0
            save_image = False
        # 获取环境当前状态
        # state 是机器人末端执行器（夹爪）的位姿（位置+姿态），属于策略的"本体感觉"输入之一。
        state = PnPEnv.get_ee_pose()
        # 从环境中获取当前图像
        # 这相当于机器人"睁眼看世界"：拿到主视角相机画面和腕部相机画面（都是 numpy 数组形式的图片）。
        image, wirst_image = PnPEnv.grab_image()
        # 下面对图像做预处理：转成 PIL 图片 → 缩放到 256x256 → 转成张量。
        # 关键：这套预处理必须和"训练时喂给模型的图像处理方式"完全一致，
        # 否则模型在训练时见到的是一种格式、部署时见到另一种，就会"水土不服"导致行为异常。
        image = Image.fromarray(image)        # numpy 数组 → PIL 图片对象
        image = image.resize((256, 256))      # 统一缩放到模型期望的输入尺寸
        image = img_transform(image)          # PIL 图片 → 张量（并把像素归一化到 0~1）
        wrist_image = Image.fromarray(wirst_image)
        wrist_image = wrist_image.resize((256, 256))
        wrist_image = img_transform(wrist_image)
        # 把这一时刻的所有观测打包成一个字典，按策略约定的"键名"组织好喂给模型。
        # 键名（如 'observation.state'、'observation.image'）必须与训练时使用的字段名一一对应，
        # 模型才能对号入座地取用每一项。unsqueeze(0) 是给张量加一个"批次维度"，
        # 因为模型习惯一次处理"一批"数据，这里这一批里只有 1 个样本。
        data = {
            'observation.state': torch.tensor([state]).to(device),       # 机器人状态（末端位姿）
            'observation.image': image.unsqueeze(0).to(device),          # 主视角相机画面
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),  # 腕部相机画面
            'task': ['Put mug cup on the plate'],                        # 任务的文字指令："把马克杯放到盘子上"
            'timestamp': torch.tensor([step/20]).to(device)             # 当前时间戳（秒）= 步数 / 频率(20Hz)
        }
        # 选取一个动作
        # select_action 是推理的核心：输入"当前画面 + 机器人状态 + 任务指令"，
        # 模型在内部思考后输出"下一步该怎么动"（一组关节角度目标）。
        # 配合前面 chunk_size=10/n_action_steps=1，它内部一次预测多步、但每次只交还 1 步给我们执行。
        action = policy.select_action(data)
        # 把动作从 GPU 张量转回普通的 numpy 数组（环境只认得普通数组）：
        # [0] 取出这一批里唯一的样本；.cpu() 搬回内存；.detach() 切断梯度（推理不需要求导）；.numpy() 转格式。
        action = action[0].cpu().detach().numpy()
        # 在环境中执行一步
        _ = PnPEnv.step(action)  # 把动作下发给仿真机器人，让它真正动起来（执行这一步）。
        PnPEnv.render()          # 把最新画面绘制到窗口上，方便我们肉眼观看 rollout 过程。
        step += 1                # 步数 +1。
        success = PnPEnv.check_success()  # 动作执行后再检查一次是否成功。
        if success:
            print('成功')
            break  # 任务完成，跳出主循环，结束本次演示。
