"""
================================================================================
本脚本作用（一句话）：教机器人“看到画面就知道该怎么动手臂”。
================================================================================

它在我们自己录制的示教数据上，训练一个叫 ACT（Action Chunking Transformer，
动作分块 Transformer）的模型，然后把训练好的模型（检查点）存下来，并评估它
“预测的动作”和“专家真实动作”差多少。

【几个零基础名词，先用大白话讲清楚】
- 策略(policy)：就是一个神经网络。给它当前看到的画面+机器人状态，它就吐出
  “接下来该怎么动”。可以理解成机器人的“大脑/驾驶员”。
- 示教数据(demonstration)：人类（专家）事先操作机器人完成任务录下来的录像，
  里面同时记录了“每一刻看到的画面”和“专家当时做的动作”。这就是教材。
- 训练(training)：让神经网络一遍遍地看这些示教数据，不断微调自己的内部参数，
  目标是让它“预测的动作”越来越接近“专家当时的动作”。像学生反复对答案订正。
- ACT 的核心思想——动作分块(action chunking)：普通做法是“看一帧、预测下一步”，
  一步一步走，容易抖、容易累积误差。ACT 改成“看一眼，一次性预测未来连续一小段
  动作”（比如未来 10 步），更平滑、更稳。这“一小段”的长度就是 chunk_size。

训练完成后，检查点（即存到硬盘上的模型）保存到 ./ckpt/act_y，
并在数据集上评估“预测动作”与“真值(ground-truth，即专家真实动作)”的平均误差。

运行方式（需要 NVIDIA GPU，约 30~60 分钟）：
    conda activate lerobot
    python act/train_act.py
（如果想训练 π0 / SmolVLA 这类视觉-语言-动作大模型，请改用 train_vla.py。）

提示：本脚本原为教程笔记本，已整理成普通 Python 脚本，自上而下顺序执行即可。
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
# # 在你的数据集上训练动作分块Transformer (ACT)
# 在你的自定义数据集上训练 ACT 模型。在本示例中,我们将 chunk_size(动作块大小)设为 10。
# ======================================================================

# ----------------------------------------------------------------------
# 导入需要用到的工具库。
# torch 是深度学习框架 PyTorch，负责张量计算、自动求导、神经网络这些底层工作。
# 下面那几行 lerobot.* 是这个机器人学习框架提供的现成模块：数据集、ACT 模型、配置等。
# ----------------------------------------------------------------------
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
import torchvision

# 指定用 GPU（"cuda" 是 NVIDIA 显卡的计算接口）来训练。
# 神经网络的训练涉及海量矩阵运算，GPU 比 CPU 快几十上百倍，所以这里把计算放到显卡上。
device = torch.device("cuda")

# 离线训练的步数(本示例只进行离线训练)。
# “一步(step)”= 取出一批数据、算一次预测误差、调整一次网络参数。
# 步数越多，模型学得通常越充分；这里设 3000 步。
# 可按需调整。需要约 5000 步才能得到值得评估的结果。
training_steps = 3000
# 每隔多少步在屏幕上打印一次当前损失(loss)，方便我们观察训练是否在变好。
log_freq = 100

# ======================================================================
# ## 策略配置与初始化
#
# chunk_size(动作块大小) = 10
# ======================================================================

# 从零开始训练时(即不从预训练策略加载),在创建策略之前需要指定两样东西:
#   - 输入/输出形状:用于正确设置策略的尺寸
#     （网络要先知道“喂进来的画面/状态多大、要吐出来的动作有几维”，才能搭好结构）
#   - 数据集统计量:用于输入/输出的归一化与反归一化
#     （统计量=数据集里各项数值的均值、标准差等；下面会讲归一化为什么需要）

# 先读取数据集的“元信息”（metadata，即关于数据的说明：有哪些字段、各自的形状、统计量等），
# 但还没真正把图像数据全部载入内存。"omy_pnp" 是数据集名字，root 是它在硬盘上的位置。
dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
# 把数据集里的字段翻译成“策略能理解的特征(feature)描述”，每个特征带有类型和形状信息。
features = dataset_to_policy_features(dataset_metadata.features)
# 输出特征 = 类型为 ACTION（动作）的那些字段，也就是网络要预测/输出的东西。
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
# 输入特征 = 除动作之外的其余字段（比如相机画面、机器人关节状态），也就是网络要“看”的东西。
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# 剔除“腕部相机”这一路画面。腕部相机装在机械臂手腕上，视角随手臂晃动、看到的范围很窄，
# 这里为了简化、让模型主要依赖固定视角的主相机，所以把它从输入里去掉。
input_features.pop("observation.wrist_image")
# 策略通过一个配置类来初始化,这里是 `ACTConfig`。在本示例中,
# 我们直接使用默认值,因此除了输入/输出特征外无需传入其他参数。
# 关键超参数说明（超参数 = 训练前由人设定、训练中不改变的设置）：
#   chunk_size=10     ：一次预测未来 10 步连续动作（前面讲的“动作分块”长度）。
#   n_action_steps=10 ：预测出的这 10 步动作，实际拿去执行多少步（这里也是 10）。
cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size= 10, n_action_steps=10)
# 根据 chunk_size 推算出“每条样本要取连续哪几帧的动作”，这样数据集才能按动作分块的方式取数据。
# delta_timestamps 可理解为“相对当前时刻的一串时间偏移”，告诉数据集要打包未来这一小段。
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
# 现在可以用该配置和数据集统计量来实例化我们的策略。
# dataset_stats（数据集统计量）会被存进策略里，专门用于“归一化(normalize)”：
#   归一化 = 把各项数值缩放到大小相近的范围（比如都变成均值0、标准差1附近）。
#   为什么需要？不同物理量量级差很多（角度可能上百，速度可能零点几），
#   直接喂给网络会让训练不稳定、难收敛。归一化后大家“尺度一致”，网络更好学。
policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
# 把策略切换到“训练模式”（会启用 dropout 等只在训练时用的机制）。
policy.train()
# 把策略（网络参数）搬到 GPU 上，和数据放在同一个设备，才能一起做运算。
policy.to(device)

# ======================================================================
# ## 加载数据集
# ======================================================================

from torchvision import transforms

# 下面定义一个“图像数据增强”小工具：给图片加一点点随机噪声。
# 数据增强(data augmentation) = 在训练时对输入做些微小随机扰动，
# 让模型见到的画面更多样，从而更“皮实”、不死记硬背，泛化能力更强。
class AddGaussianNoise(object):
    """
    向张量(tensor，可理解为多维数组，这里装的是图像像素)添加高斯噪声。
    高斯噪声 = 服从正态分布的随机小扰动，mean 是均值、std 是噪声强度(标准差)。
    """
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 添加噪声:张量仍然保持为张量。
        # 生成一份和图像同样大小的随机噪声，再叠加到原图上。
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# 创建一个变换流水线:先将 PIL 图像转换为张量,然后添加噪声。
# Compose 就是把多个变换“串成一条流水线”，数据依次经过每一步。
# 这里：先加噪声(std=0.02 噪声很小)，再用 clamp(0,1) 把像素值裁回 [0,1] 合法范围
# （加噪声后个别值可能跑出范围，裁一下避免出问题）。
transform = transforms.Compose([
    AddGaussianNoise(mean=0., std=0.02),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])

# 接着我们用这些 delta_timestamps 配置来实例化数据集。
# 这次是真正把数据集准备好用于训练：每取一条样本，会按 delta_timestamps 打包好一小段动作，
# 并对图像套用上面定义的 transform 做增强。
dataset = LeRobotDataset("omy_pnp", delta_timestamps=delta_timestamps, root='./demo_data', image_transforms=transform)

# 然后为离线训练创建优化器和数据加载器。
# 优化器(optimizer) = 负责“怎么调整网络参数”的算法。Adam 是最常用的一种，
# 它会根据每个参数过去的梯度自动调整步子，通常又快又稳。
# lr=1e-4（即 0.0001）是学习率(learning rate)：每次调整参数迈多大的步子。
#   太大→容易跑过头、训练发散；太小→学得太慢。1e-4 是个常见的稳妥取值。
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
# 数据加载器(DataLoader) = 自动从数据集里成批取数据、打乱顺序、并行预读取的工具。
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,          # 用 4 个子进程在后台并行读数据，避免 GPU 等着挨饿
    batch_size=64,          # 批量大小：每次同时喂 64 条样本一起训练（批量越大越稳，但更吃显存）
    shuffle=True,           # 每轮把数据顺序打乱，防止模型记住“出场顺序”而学偏
    pin_memory=device.type != "cpu",  # 用 GPU 时锁页内存，能让数据搬到显卡更快
    drop_last=True,         # 最后一批若凑不满 64 条就丢弃，保证每批大小一致
)

# ======================================================================
# ## 训练
#
# 训练得到的检查点将保存在 './ckpt/act_y' 文件夹中。
# ======================================================================

# 运行训练循环。
# 训练就是不断重复下面这 5 个动作，直到走满 training_steps 步：
#   ①取一批数据 → ②让网络预测并算出损失 → ③反向传播算梯度
#   → ④优化器据梯度更新参数 → ⑤清空梯度，准备下一批。
# 这样循环成千上万次，网络的预测就会越来越贴近专家动作。
step = 0          # 记录当前已经训练了多少步
done = False      # 是否训练结束的标志
while not done:
    for batch in dataloader:   # 每次从数据加载器拿出一批(64条)样本
        # 把这一批数据里的张量都搬到 GPU 上（非张量的字段原样保留），和网络在同一设备才能算。
        inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        # 前向传播(forward)：网络看着这批数据做预测，并直接算出损失 loss。
        # 损失(loss) = 衡量“预测动作”和“专家真实动作”差多少的一个数字，越小代表预测越准。
        loss, _ = policy.forward(inp_batch)
        # 反向传播(backward)：从损失出发，自动算出“每个参数该往哪个方向、改多少”才能让损失变小（即梯度）。
        loss.backward()
        # 让优化器按刚算出的梯度，真正去更新一次网络参数（朝着损失更小的方向迈一小步）。
        optimizer.step()
        # 清空本次累计的梯度，否则它会和下一批的梯度叠加，导致更新出错。
        optimizer.zero_grad()

        # 每隔 log_freq 步打印一次损失，正常情况下这个数字应整体呈下降趋势。
        if step % log_freq == 0:
            print(f"步数: {step} 损失: {loss.item():.3f}")
        step += 1
        if step >= training_steps:   # 走满预定步数就收工
            done = True
            break

# 将策略保存到磁盘。
# 保存下来的模型就叫“检查点(checkpoint)”，存到 ./ckpt/act_y 文件夹。
# 以后做推理/部署时，直接从这里把训练好的模型加载回来即可，不用重新训练。
policy.save_pretrained('./ckpt/act_y')

# ======================================================================
# ## 测试推理
#
# 要在数据集上评估策略,你可以计算预测动作与数据集中真值(ground-truth)动作之间的误差。
# ======================================================================

import torch

# 一段示教叫一个“回合(episode)”，比如“完整抓取并放置一次”就是一个 episode。
# 数据集把很多回合的帧首尾相连存在一起，下面这个采样器(Sampler)的作用是：
# 只挑出“某一个指定回合”的那些帧，让我们能按时间顺序、单独评估这一整段。
class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        # 查出第 episode_index 个回合在整个数据集里“从第几帧到第几帧”。
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

# 把策略切到“评估模式”：关闭 dropout 等训练专用机制，让输出稳定、可复现。
policy.eval()
actions = []      # 收集模型预测出的动作
gt_actions = []   # 收集对应的专家真值动作(ground-truth)，用来对比
images = []       # 收集对应画面（这里仅留存，未必用到）
episode_index = 0 # 这里挑第 0 个回合来评估
episode_sampler = EpisodeSampler(dataset, episode_index)
# 评估用的数据加载器：batch_size=1 且不打乱，逐帧按时间顺序喂入，模拟真实“一帧接一帧”地决策。
test_dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=1,
    shuffle=False,
    pin_memory=device.type != "cpu",
    sampler=episode_sampler,   # 用上面的采样器，限定只跑这一个回合
)
# reset() 清空策略内部的“动作分块缓存”等状态，从这一回合的开头干净地开始。
policy.reset()
for batch in test_dataloader:
    # 同样把数据搬到 GPU。
    inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    # select_action：根据当前这一帧画面，让策略给出“此刻该执行的那一步动作”。
    # （ACT 内部其实预测了未来一整块动作，这里每次取出该执行的那一步返回。）
    action = policy.select_action(inp_batch)
    actions.append(action)
    # 取这一帧对应的专家真值动作（[:,0,:] 表示取这一小段里的第 0 步，即当下这一步）。
    gt_actions.append(inp_batch["action"][:,0,:])
    images.append(inp_batch["observation.image"])
# 把整段回合每一帧的结果拼成一个大张量，方便整体比较。
actions = torch.cat(actions, dim=0)
gt_actions = torch.cat(gt_actions, dim=0)
# 计算“平均绝对误差(MAE)”：先对每一维取 |预测 - 真值| 的绝对值，再求全部的平均。
# 通俗讲就是“预测动作平均偏离专家动作多少”，这个数字越小说明模型学得越好。
print(f"动作平均误差: {torch.mean(torch.abs(actions - gt_actions)).item():.3f}")

'''
绘制预测动作(actions)与真值动作(gt_actions)
把上面的数字误差画成曲线图，更直观：每一维动作都画一条“预测”和一条“真值”，
两条线贴得越近，说明模型学得越好。
'''
import matplotlib.pyplot as plt
# action_dim=7：动作有 7 个维度（这里通常是机械臂 6 个关节 + 1 个夹爪开合）。
action_dim = 7

# 建 7 个上下排列的子图，每个子图画一维动作。
fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10))

for i in range(action_dim):
    # 注意：张量在 GPU 上，要先 .cpu() 搬回内存、.detach() 断开梯度、再转成 numpy 才能交给 matplotlib 画图。
    axs[i].plot(actions[:, i].cpu().detach().numpy(), label="pred")   # pred=模型预测
    axs[i].plot(gt_actions[:, i].cpu().detach().numpy(), label="gt")  # gt=专家真值
    axs[i].legend()   # 显示图例，区分两条线
plt.show()
