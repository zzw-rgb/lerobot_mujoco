"""
============================================================================
可视化“语言条件（language-conditioned）”数据集
============================================================================

【这个脚本是干嘛的？一句话版本】
    把我们之前录制好的机器人操作数据，在仿真里“重新播放一遍”给你看，
    并且在画面上把当时下达的“自然语言指令”（比如 “把红色方块放进盒子里”）
    一起显示出来。

【先搞懂几个名词（说人话）】
  - 数据集(dataset)：就是一段段录好的机器人操作录像 + 各种传感器数据
    （机械臂每个关节的角度、相机拍到的画面、物体初始位置……）。
  - 回放(replay)：好比看回放录像。我们不是让机器人真的去思考、去决策，
    而是把当初录下来的“动作序列”一帧一帧喂给仿真环境，让仿真里的机械臂
    照着做一遍。这样就能直观检查：我录的这段数据到底对不对、好不好看。
  - 语言条件(language-conditioned)：和“普通数据可视化”相比，这份数据
    每个回合还额外存了一句“任务指令”（即语言）。本脚本的关键区别就在于：
    会把这句指令通过 set_instruction() 显示到仿真画面上。
    可以理解为：普通可视化只放“动作录像”，这里还在录像角落打上一行字幕，
    告诉你“这一条录像当时是要它干什么”。

【这份数据将来用来做什么？】
    用来训练 VLA（Vision-Language-Action，视觉-语言-动作）模型——也就是
    让模型“看着画面 + 读懂一句话指令 -> 输出机械臂动作”。训练脚本是
    train_vla.py。本脚本只负责“看数据”，不负责训练。

运行方式（需要图形界面 / 能弹出仿真窗口的桌面环境）：
    conda activate lerobot
    python vla/visualize_data_language.py

提示：本脚本原为教程笔记本，已转换成可直接从上往下顺序执行的普通 .py 脚本。
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
# ### [可选] 下载已采集的数据集
# ======================================================================

# 【Hugging Face 是什么？】可以把它想成“AI 界的 GitHub / 网盘”：很多人会
# 把训练好的模型、采集好的数据集公开放在上面，供别人免费下载使用。
# 下面这条命令做的事，就是把别人(用户名 Jeongeun)分享的一份语言条件数据集
# 整个克隆(下载)到本地，免去你自己重新采集数据的麻烦。
#
# 【这一段只是说明，不会自动执行】下面用 ''' ''' 包起来的是一段普通文字
# （文档字符串），Python 运行时不会把它当命令执行；真正要下载时，请你手动
# 复制最后那行 git clone 命令，到“终端/命令行”里去敲。
'''
如果你想使用已采集好的数据集，请从 Hugging Face 下载。
'''
# [终端命令] git clone https://huggingface.co/datasets/Jeongeun/omy_pnp_language

# ======================================================================
# # 可视化你的数据
#
# <img src="./media/data_v2.gif" width="480" height="360">
#
# 基于重建的仿真场景可视化你的动作。
#
# 主仿真画面是在回放（replay）动作。
#
# 右上角和右下角叠加的图像来自数据集。
# ======================================================================
# 【上面这段是原笔记本里的图文说明，转成 .py 后保留为注释】用大白话翻译一下：
#   - 屏幕中间那个大窗口：仿真里的机械臂，正在“照着录像”一帧帧重演当初的动作；
#   - 屏幕角落叠加的小图：是数据集里真实录下来的相机画面(机位视角/手腕视角)；
#   - 把两者放一起看，就能对照检查“录的数据”和“仿真重演”是否一致、合理。

# 【导入要用到的工具】
#   - LeRobotDataset：LeRobot 框架定义的“数据集”类。负责把硬盘上那一堆
#     录制文件，读成程序里方便取用的对象（之后用下标就能取出每一帧的数据）。
#   - numpy：科学计算库，这里主要用来处理图像的数值数组（缩写惯例为 np）。
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict

# 【告诉程序去哪里读数据】ROOT 就是“数据集存放的文件夹路径”。
# 默认指向自己采集的演示数据 ./demo_data_language。
ROOT = "./demo_data_language" # 保存演示数据的根目录
# 如果你已从 Hugging Face 下载了数据集，可以把 root 设置为该数据集所在的目录
# 想改用前面下载的官方示例数据，就把上面这行注释掉，改成 ROOT = './omy_pnp_language'
# ROOT = './omy_pnp_language' # 如果想使用提供的示例数据，请改用 root = './omy_pnp_language'！

# 【真正把数据集“装载”进来】
# 第一个参数 'omy_pnp_language' 是数据集名字（omy=机器人型号，pnp=pick and place
# 抓取-放置任务，language=带语言指令）；root 告诉它去本地哪个文件夹找。
# 装载之后，dataset 就像一本翻开的相册，可以按帧取出动作、图像、指令等内容。
dataset = LeRobotDataset('omy_pnp_language', root=ROOT) # 如果想使用提供的示例数据，请改用 root = './omy_pnp_language'！

# 如果你想使用已采集好的数据集，请从 Hugging Face 下载。
# dataset = LeRobotDataset('omy_pnp_language', root='omy_pnp_language')

# ======================================================================
# ## 加载数据集
# ======================================================================

# torch 即 PyTorch，深度学习框架。这里借用它的“数据加载”工具来按帧取数据。
import torch

# 【先理解“回合(episode)”和“帧(frame)”】
#   - 一个 episode = 一段完整的操作录像（例如“抓一次方块放进盒子”的全过程）。
#   - 一个 frame = 这段录像里的一帧（某个瞬间的动作和画面）。
#   - 一个数据集里通常存了很多段 episode，每段又由很多帧连续组成。
#
# 【这个 Sampler(采样器)是干嘛的？】
# 整个数据集是“所有 episode 的所有帧”混在一起按编号排列的。如果我们只想
# 回放“第几段”这一段录像，就需要一个“挑帧器”：只把属于这一段的帧编号挑出来，
# 而且要按顺序(从头到尾)给出去——这样回放才不会乱序。EpisodeSampler 做的就是这件事。
class EpisodeSampler(torch.utils.data.Sampler):
    """
    针对单个回合（episode）的采样器：只产出指定 episode 范围内的帧编号，按顺序输出。
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        # 数据集记录了每段 episode 在“全体帧”里的起止编号。
        # from_idx = 这段录像的第一帧编号；to_idx = 最后一帧的下一个编号。
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        # frame_ids 就是这段录像所有帧编号的连续区间，例如 range(120, 245)。
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        # 被遍历时，逐个吐出这段录像的帧编号（保证顺序）。
        return iter(self.frame_ids)

    def __len__(self) -> int:
        # 这段录像一共有多少帧（后面用来判断“是否已经放完一轮”）。
        return len(self.frame_ids)

# 选择你想要可视化的回合（episode）索引。0 表示第 1 段录像；想看别的就改这个数字。
episode_index = 0

# 用上面的采样器，锁定“只取第 episode_index 段录像”的那些帧。
episode_sampler = EpisodeSampler(dataset, episode_index)

# 【DataLoader 是什么？】把它想成一台“自动上菜的传送带”：你告诉它从哪个
# 数据集取、按什么顺序取(sampler)、一次端几份(batch_size)，它就一帧一帧
# 帮你把数据准备好递过来，省得你自己手动算编号、读文件。
#   - num_workers=1：用 1 个后台帮工去读数据（够用即可）。
#   - batch_size=1：一次只取 1 帧（因为回放就是一帧一帧播）。
#   - sampler=episode_sampler：取数顺序由我们上面的采样器决定（只取这一段、按顺序）。
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=1,
    sampler=episode_sampler,
)

# ======================================================================
# ## 在仿真中可视化你的数据集
# ======================================================================

# 【搭建仿真舞台】SimpleEnv2 是一个 MuJoCo 仿真环境（带物理引擎的虚拟世界），
# 我们要在这个虚拟世界里重演机械臂动作。
#   - xml_path：场景描述文件，定义了桌子、机械臂、待抓取物体等长什么样、放哪儿。
#   - action_type='joint_angle'：动作的形式是“关节角度”——即每一帧告诉机械臂
#     它的每个关节应该转到多少度（而不是末端坐标之类的其它表示方式）。
from mujoco_env.y_env2 import SimpleEnv2
xml_path = './asset/example_scene_y2.xml'
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')

# step：计数器，记录“这一段录像已经播到第几帧了”。
step = 0
# iter_dataloader：把传送带(dataloader)变成一个“逐帧取”的迭代器，
# 之后每调用一次 next() 就拿到下一帧。
iter_dataloader = iter(dataloader)
# reset()：把仿真世界恢复到初始状态（机械臂归位、物体复位），准备开播。
PnPEnv.reset()

# 【主循环：只要仿真窗口还开着，就一直循环】
# 这是整个脚本的“放映机”。它不停地推进仿真、按节奏取出下一帧动作并执行、
# 再把画面渲染出来，从而让你看到一段流畅的回放。
while PnPEnv.env.is_viewer_alive():
    # step_env()：让物理引擎往前走一个极小的时间步（保持仿真画面平滑、不卡顿）。
    PnPEnv.step_env()
    # loop_every(HZ=20)：节流阀。物理引擎跑得很快，但我们的数据是按 20Hz
    # （每秒 20 帧）录的，所以这里控制成“每秒只取 20 次数据”，让回放速度对得上。
    if PnPEnv.env.loop_every(HZ=20):
        # 从传送带取出“下一帧”的全部数据（动作、图像、指令、物体初始位姿等都在 data 里）。
        data = next(iter_dataloader)
        if step == 0:
            # 【这一段是“语言条件可视化”的关键，普通可视化没有这一步】
            # data['task'] 里存的就是这段录像对应的自然语言指令字符串，
            # 例如 "pick up the red cube and place it in the box"。
            instruction = data['task'][0]
            # set_instruction()：把这句指令“贴”到仿真画面上当字幕显示出来，
            # 让你一眼看到“这段录像当初要机器人做什么”。
            PnPEnv.set_instruction(instruction)
            # 根据数据集重置物体的位姿
            # set_obj_pose()：按录制时的真实初始摆放，把仿真里的物体摆到对应位置，
            # 这样重演才和当初的场景一致。这里把 9 个数拆成 3 组(每组 3 个 xyz)分别传入。
            PnPEnv.set_obj_pose(data['obj_init'][0,:3], data['obj_init'][0,3:6], data['obj_init'][0,6:9])
        # 从数据集中取出动作
        # .numpy()：把动作从 PyTorch 张量转成 numpy 数组，方便交给仿真执行。
        action = data['action'].numpy()
        # step(action[0])：让仿真里的机械臂执行这一帧的动作（核心“回放”动作）。
        obs = PnPEnv.step(action[0])

        # 将数据集中的图像可视化叠加到 rgb_overlay 上
        # 【下面这段：把数据集里录的真实相机画面，叠加到仿真窗口角落显示】
        # observation.image     = 第三人称机位相机看到的画面（rgb_agent）
        # observation.wrist_image = 装在机械手腕上的相机看到的画面（rgb_ego）
        # 数据里图像数值是 0~1 的小数，这里 *255 还原成常见的 0~255 像素亮度范围。
        PnPEnv.rgb_agent = data['observation.image'][0].numpy()*255
        PnPEnv.rgb_ego = data['observation.wrist_image'][0].numpy()*255
        # astype(np.uint8)：把数值转成 8 位整数（0~255），这是图像的标准像素格式。
        PnPEnv.rgb_agent = PnPEnv.rgb_agent.astype(np.uint8)
        PnPEnv.rgb_ego = PnPEnv.rgb_ego.astype(np.uint8)
        # 3 256 256 -> 256 256 3
        # 【调整图像的“维度顺序”】数据里图像是 (通道3, 高256, 宽256) 的排法，
        # 而显示/绘图通常要求 (高, 宽, 通道) 的排法，transpose 就是把维度重新排序。
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1,2,0))
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1,2,0))
        # rgb_side：侧面视角这里没有真实数据，就用一张全黑图(全 0)占位。
        PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
        # render()：把这一帧（机械臂姿态 + 角落叠加图 + 指令字幕）真正画到窗口上。
        PnPEnv.render()
        # 帧计数 +1，表示又播了一帧。
        step += 1

        if step == len(episode_sampler):
            # 从头开始
            # 这段录像已经播完最后一帧了 -> 重新拿一条新的传送带、复位仿真、计数清零，
            # 于是这段录像会循环播放（loop），方便你反复观察。
            iter_dataloader = iter(dataloader)
            PnPEnv.reset()
            step = 0
    # PnPEnv

# 窗口被关掉后跳出循环：关闭仿真查看器，释放资源。
PnPEnv.env.close_viewer()

# 【可选/收尾步骤】把本地数据集上传回 Hugging Face（分享给别人或备份）。
# upload_large_folder=True 表示用“适合大文件夹”的方式上传。
# 注意：需要你已登录 Hugging Face 且有该仓库的写权限，否则这步会报错——
# 若只是看数据、不打算上传，可以忽略此步（或在自己版本里不执行它）。
dataset.push_to_hub(upload_large_folder=True)
