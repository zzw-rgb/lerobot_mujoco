"""
步骤 2：回放并可视化已采集的数据集（教学注释版，零基础也能看懂）。

================================ 这个脚本到底在做什么？ ================================
我们之前用遥操作（或脚本）采集了一批机器人“抓取-放置”的演示数据，
这些数据存成了 LeRobot 数据集（里面记录了：每一帧机械臂的动作、相机拍到的图像、
物体初始摆放位置等）。

但是“存下来的数据到底对不对、采得好不好”光看一堆数字是看不出来的。
于是这个脚本做一件事：把数据“重新播放一遍”给你看。

具体来说：
  1. 在电脑里重新搭一个和采集时一模一样的 MuJoCo 仿真场景（虚拟的机械臂+桌面+物体）；
  2. 把数据集里记录的动作，一帧一帧地“喂”给仿真里的机械臂，让它照着重做一遍；
  3. 同时把数据集里当时相机拍到的真实图像，叠加显示在仿真画面的角落，方便你左右对照。
这就叫“回放（replay）”——好比把行车记录仪的录像倒回去重新放一遍，检查有没有问题。

================================ 为什么要在“仿真里”重放？ ================================
直接看图片只能看到“相机当时拍到什么”，看不到机械臂的动作轨迹是否合理。
放进仿真里重放，就能直观看到机械臂是怎么动的、抓没抓到、动作有没有抖动或跳变，
从而判断这条数据值不值得拿去训练模型。如果回放时机械臂行为很怪，
说明这条采集数据质量差，应该剔除。

================================ 运行方式（需要图形界面） ================================
    conda activate lerobot
    python act/visualize_data_act.py

  - 运行后会弹出一个 MuJoCo 窗口，自动循环回放选中的那一回合（episode）数据，
    关掉窗口即结束。
  - 想改看第几回合，修改下面的 episode_index 变量即可。
  - 确认数据没问题后，就可以用训练脚本 train_vla.py 拿这批数据去训练策略模型了。

提示：本脚本原为教程笔记本（notebook），现整理成普通 Python 脚本，自上而下顺序执行即可。
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
# # 可视化你的数据
#
# <img src="./media/data.gif" width="480" height="360">
#
# 在重建的仿真场景中可视化你的动作。
#
# 主仿真画面正在回放动作。
#
# 右上角和右下角叠加的图像来自数据集。
# ======================================================================

# --- 导入需要用到的工具 ---
# LeRobotDataset：LeRobot 框架提供的“数据集对象”，负责把硬盘上采集的数据读进来，
#                 并能按帧（frame）逐条取出动作、图像等内容。
# numpy（简写 np）：科学计算库，这里主要用来处理图像数组（改数据类型、调整维度顺序等）。
# write_json / serialize_dict：把统计信息（均值、方差等）整理成可写入磁盘的格式并存成 json。
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict

# 打开我们之前采集好的数据集。
#   - 第一个参数 'omy_pnp' 是数据集名字（omy 机械臂的 pick-and-place 抓放任务）。
#   - root 指定数据存放的文件夹。
dataset = LeRobotDataset('omy_pnp', root='./demo_data') # 如果你想使用提供的示例数据，请改为 root = './demo_data_example'！

# ======================================================================
# ## 加载数据集
# ======================================================================

# torch（PyTorch）：深度学习框架。这里只借用它的两个“取数据小工具”：
#   - Sampler（采样器）：决定“按什么顺序、取哪些帧”。
#   - DataLoader（数据加载器）：真正把数据一条条/一批批从数据集里搬出来的传送带。
import torch

# 先解释两个关键词：
#   episode（回合）：一段完整的演示。比如“机械臂从抓起方块到放进盒子”这一整套过程，
#                    就是一个 episode。数据集里通常存了很多个 episode。
#                    每个 episode 又由很多“帧（frame）”组成，一帧就是某个时刻的快照
#                    （那一刻的动作 + 相机图像）。
#   采样器（Sampler）：一个完整数据集里所有 episode 的帧是混在一起、连续编号的。
#                      我们一次只想看某一个 episode，所以需要一个采样器，专门告诉
#                      DataLoader：“只取属于第 N 个 episode 的那一段帧编号就行”。
class EpisodeSampler(torch.utils.data.Sampler):
    """
    针对单个 episode 的采样器：只挑出指定 episode 对应的帧编号范围。
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        # dataset.episode_data_index 里记着每个 episode 在整个数据集中
        # 的起始帧编号（from）和结束帧编号（to）。
        # 比如第 0 个 episode 占用帧 0~149，第 1 个占用 150~320，以此类推。
        from_idx = dataset.episode_data_index["from"][episode_index].item()  # 该回合起始帧编号
        to_idx = dataset.episode_data_index["to"][episode_index].item()      # 该回合结束帧编号（不含）
        # range(from, to) 就是这个回合所有帧的编号序列，回放时按顺序逐帧取出。
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        # DataLoader 会调用它，逐个吐出帧编号。
        return iter(self.frame_ids)

    def __len__(self) -> int:
        # 这个回合一共有多少帧（后面用来判断“是否已经放到最后一帧”）。
        return len(self.frame_ids)

# 选择你想要可视化的 episode 索引（0 表示第一个回合，想看别的就改这个数字）
episode_index = 0

# 用上面的采样器创建一个只针对该回合的采样器实例。
episode_sampler = EpisodeSampler(dataset, episode_index)
# 创建数据加载器（传送带）：
#   - dataset：数据来源。
#   - num_workers=1：用 1 个后台线程搬数据。
#   - batch_size=1：每次只取 1 帧（回放是一帧一帧放的，所以一次取一帧最自然）。
#   - sampler=episode_sampler：用我们的采样器，保证只取选中回合的那些帧。
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=1,
    sampler=episode_sampler,
)

# ======================================================================
# ## 在仿真中可视化你的数据集
# ======================================================================

# SimpleEnv：项目自带的“仿真环境”类，负责加载 MuJoCo 场景、推进物理、渲染画面、
#            执行动作等。可以把它理解成一个“虚拟实验室”。
from mujoco_env.y_env import SimpleEnv
# xml_path：场景描述文件。MuJoCo 用 XML 文件来定义机械臂、桌子、物体长什么样、摆在哪。
# 这个文件描述的场景应当与当初采集数据时一致，回放才有意义。
xml_path = './asset/example_scene_y.xml'
# 创建仿真环境：
#   - action_type='joint_angle' 表示动作的含义是“各个关节的目标角度”
#     （也就是直接告诉机械臂每个关节应转到多少度），这要和数据集里记录的动作格式一致。
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

# step：计数器，记录当前已经回放到这个回合的第几帧。
step = 0
# iter_dataloader：把“传送带”变成一个可以反复调用 next() 逐帧取数据的迭代器。
#                  每调用一次 next(iter_dataloader) 就拿到下一帧的数据。
iter_dataloader = iter(dataloader)
# 把仿真复位到初始状态（机械臂回到起始姿态等）。
PnPEnv.reset()

# 主循环：只要 MuJoCo 的查看窗口还开着，就一直运行。
# 关掉那个弹出的窗口，整个循环（以及程序）就结束。
while PnPEnv.env.is_viewer_alive():
    # step_env()：推进一小步物理仿真。物理引擎跑得很快（每秒很多步），
    #             这样画面才平滑、物体下落/碰撞才真实。
    PnPEnv.step_env()

    # loop_every(HZ=20)：节流器，控制“多久才喂一帧新动作”。
    #   物理仿真步频很高，但数据集里的动作是按 20Hz（每秒 20 帧）采集的，
    #   所以这里限制为每秒只取并执行 20 个新动作，保持回放速度和采集时一致。
    #   下面这段 if 里的代码，每秒大约只进入 20 次。
    if PnPEnv.env.loop_every(HZ=20):
        # 从“传送带”取出下一帧数据（包含这一帧的动作、当时的相机图像、物体初始位姿等）。
        data = next(iter_dataloader)
        if step == 0:
            # 只在回合的第一帧做一次：把仿真里的物体摆到“当初采集时的初始位置和朝向”。
            #   为什么必须这么做？因为同一套动作只有在物体起始摆放相同的前提下，
            #   重放出来才会和采集时一致——否则机械臂照搬动作却抓了个空。
            #   data['obj_init'] 前 3 个数是位置(x,y,z)，后面的数是朝向(四元数)。
            PnPEnv.set_obj_pose(data['obj_init'][0,:3], data['obj_init'][0,3:])
        # 取出这一帧记录的动作。.numpy() 把 PyTorch 张量转成 numpy 数组，便于后续使用。
        action = data['action'].numpy()
        # 把动作交给仿真执行。action[0] 是去掉 batch 维后这一帧真正的动作向量。
        # step() 返回 obs（执行后的观测），这里仅回放、不需要用它。
        obs = PnPEnv.step(action[0])

        # ---------- 把数据集里“当时拍到的真实图像”叠加显示到仿真画面上 ----------
        # 叠加图（overlay）：就是把数据集中的图像贴在仿真窗口的角落里，
        #   让你一边看仿真里机械臂怎么动、一边对照当时相机拍到了什么，方便核对采集质量。
        # observation.image：固定视角（agent/外部）相机图像；observation.wrist_image：腕部相机图像。
        # 数据集里图像像素值是 0~1 的小数，这里乘 255 还原成常规的 0~255 范围。
        PnPEnv.rgb_agent = data['observation.image'][0].numpy()*255
        PnPEnv.rgb_ego = data['observation.wrist_image'][0].numpy()*255
        # 转成 uint8（0~255 的整数），这是图像显示通常要求的数据类型。
        PnPEnv.rgb_agent = PnPEnv.rgb_agent.astype(np.uint8)
        PnPEnv.rgb_ego = PnPEnv.rgb_ego.astype(np.uint8)
        # 3 256 256 -> 256 256 3
        # 数据集里图像是“通道在前”(C,H,W)的排列，而显示需要“通道在后”(H,W,C)，
        # 所以用 transpose 调换维度顺序：把 (3,256,256) 变成 (256,256,3)。
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1,2,0))
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1,2,0))
        # side（侧视）相机这里没有对应数据，用一张全黑图占位即可。
        PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
        # 真正把这一帧画出来（含机械臂状态 + 角落里叠加的相机图像）。
        PnPEnv.render()
        step += 1  # 帧计数 +1

        if step == len(episode_sampler):
            # 已经放到这个回合的最后一帧 —— 循环播放：重新拿一条新传送带、复位仿真、
            # 计数归零，从头再放一遍（如此往复，直到你关掉窗口）。
            iter_dataloader = iter(dataloader)
            PnPEnv.reset()
            step = 0

# 关闭 MuJoCo 查看窗口，释放资源。
PnPEnv.env.close_viewer()

# ======================================================================
# ### [可选] 为其他版本保存 stats.json
# ======================================================================

# 这一小段是可选的，和上面的可视化没有直接关系。
# stats（统计信息）：数据集里各项数据的均值、标准差、最大/最小值等。
#   训练模型时常用它来对数据做归一化（把数值缩放到统一范围，便于训练）。
# 这里把统计信息重新导出成 stats.json 存到数据集的 meta 文件夹，
# 方便不同版本/工具读取（有些版本期望磁盘上存在这个文件）。
stats = dataset.meta.stats                     # 取出数据集自带的统计信息
PATH = dataset.root / 'meta' / 'stats.json'    # 目标保存路径
stats = serialize_dict(stats)                  # 转成可写入 json 的纯 Python 结构

write_json(stats, PATH)                         # 写入磁盘
