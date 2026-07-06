"""
步骤 8：部署微调好的 SmolVLA（轻量级 VLA）策略。

================== 写给零基础读者的总览 ==================
【什么是 VLA 模型？】
VLA = Vision-Language-Action（视觉-语言-动作）。
你可以把它想象成机器人的“大脑”：它一只眼睛看画面（Vision，摄像头图像），
一只耳朵听人话（Language，比如“把红色方块放进盘子里”），
然后嘴里不说话、直接动手（Action，输出机械臂该怎么转动的关节角度）。
传统机器人需要工程师把每个动作写死成程序；而 VLA 模型是“学”出来的——
喂给它大量“看到什么画面 + 听到什么指令 -> 应该做什么动作”的示范数据，
它就能举一反三，听懂新指令、应对新画面。

【什么是 SmolVLA？它有什么特点？】
SmolVLA 里的 “Smol” 是 “small（小）” 的俏皮写法，意思就是“小号 VLA”。
- 轻量级：参数量少、算力需求低，普通/显存有限的机器（甚至消费级显卡）也跑得动。
- 能力不打折太多：照样能看图、听语言指令、输出动作，完成抓取-放置这类任务。
打个比方：如果说大模型像一台需要专业机房的超级计算机，
SmolVLA 就像一台能塞进书包的笔记本电脑——便携、省电，日常任务够用。

【SmolVLA 与 π0（pi0）的区别？】
π0 是另一款更大、更强的 VLA 模型。两者“接口”几乎一样
（都吃 图像 + 机器人状态 + 语言指令，都吐 动作），区别主要在“块头”：
SmolVLA 更小、更省显存、推理更快，但模型容量小一些；
π0 更大、更强，但更吃硬件。本系列教程里两者可以无缝替换，
只是把模型类从 Pi0Policy 换成了 SmolVLAPolicy。

【本脚本做的事，一句话概括】
加载我们事先“微调”好的 SmolVLA 权重（微调 = 在通用模型基础上，
用自己采集的抓放示范数据再训练一遍，让它学会这个具体任务），
然后把它放进一个仿真环境里，让它边看画面边听指令，一步步把物体抓起来放好。

本脚本在语言条件环境中加载微调权重并执行。
（“语言条件”= 机器人的行为由一句语言指令来决定/引导。）

训练 SmolVLA 请在终端运行：
    python vla/train_vla.py --config_path=config/vla/smolvla_franka.yaml

部署运行方式（需要 GPU + 图形界面）：
    conda activate lerobot
    python vla/deploy_smolvla.py

注意：脚本中以 [终端命令] 标注的行（pip/python vla/train_vla.py）原是
教程笔记本里的 shell 命令，已注释掉，请按需在终端单独执行。
提示：本脚本原为教程笔记本，这里整理成了带详细中文注释的 Python 脚本。
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
# # 部署训练好的 SmolVLA 策略
#
# 在仿真环境中部署训练好的策略。
# ======================================================================

# 下面这几行是“安装依赖库”的命令，原本写在笔记本里用一个感叹号直接运行。
# 在普通 Python 脚本里它们不会自动执行，所以加了 [终端命令] 前缀提醒你：
# 请把它们复制到终端里手动跑一次（只需第一次配置环境时执行）。
# - transformers：HuggingFace 的模型库，SmolVLA 内部用到它的语言/视觉组件。
# - num2words：把数字转成英文单词（某些文本处理会用到）。
# - accelerate：HuggingFace 的训练/推理加速库，帮忙调度 GPU。
# - safetensors：一种安全又快速的模型权重文件格式。
# [终端命令] pip install transformers==4.50.3
# [终端命令] pip install num2words
# [终端命令] pip install accelerate
# [终端命令] pip install safetensors>=0.4.3

# ======================================================================
# ### [可选] 下载数据集
# ======================================================================

'''
数据集（一堆“看到的画面 + 当时的指令 + 机器人当时做的动作”的录像，模型靠模仿它学会任务）
由 vla/collect_data_language.py 采集得到。已开源一份现成数据集，不想自己采集可以直接下载：
  git clone https://huggingface.co/datasets/a3124371940/franka_pnp_language
若已用 vla/push_dataset_language.py 把自己采集的数据上传到了 Hugging Face，
同样可以用 git clone 把它下载到别的机器上。
'''

# ======================================================================
# ## 步骤 2. 训练模型
# ======================================================================

# 这一步是“训练/微调”模型：用上面的数据集把 SmolVLA 教会本任务。
# 训练很耗时也耗显卡，通常在另一台带 GPU 的机器上单独跑一次即可，
# 跑完会得到权重文件（见后面 from_pretrained 加载的那个路径）。
# 部署阶段（本脚本）不需要重新训练，直接加载现成权重就行。
# [终端命令] python vla/train_vla.py --config_path=config/vla/smolvla_franka.yaml

# ======================================================================
# ## 步骤 3. 部署
# ======================================================================

# 下面是“导入工具箱”：把要用到的现成库和功能搬进来，方便后面调用。
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # 读取数据集及其元信息（统计量、特征定义等）
import numpy as np  # 数值计算库，处理数组/矩阵
from lerobot.common.datasets.utils import write_json, serialize_dict  # 数据序列化工具（本脚本未直接用到，随教程保留）
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig  # SmolVLA 的“配置”类，描述模型怎么搭、吃什么吐什么
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # SmolVLA 的“策略”类，也就是模型本体（能 select_action）
from lerobot.configs.types import FeatureType  # 特征类型枚举（区分哪些是“观测/输入”，哪些是“动作/输出”）
from lerobot.common.datasets.factory import resolve_delta_timestamps  # 根据配置算出需要哪些时间点的数据（时序相关）
from lerobot.common.datasets.utils import dataset_to_policy_features  # 把数据集里的字段，翻译成模型能理解的“特征”定义
import torch  # PyTorch 深度学习框架，模型在它上面跑
from PIL import Image  # 图像处理库，把数组变成图片对象、缩放等
import torchvision  # PyTorch 的视觉工具库，下面用它做图像预处理

# 指定模型在哪块硬件上运行：'cuda' 表示用 NVIDIA GPU（显卡），跑得快。
# 如果机器没有 GPU，这里通常要改成 'cpu'（但 VLA 模型用 CPU 会非常慢）。
device = 'cuda'

# 读取数据集的“元信息”（metadata）：不读全部录像，只读它的说明书——
# 比如每个字段叫什么、是图像还是关节角、各字段的均值/方差等统计量。
# 这些统计量后面用来做“归一化”（把数据缩放到模型习惯的范围）。
# 这里用 try/except 兜底：先试一个目录名，找不到就退而求其次试另一个目录名，
# 这样无论数据集放在 './demo_data_language' 还是 './franka_pnp_language' 都能加载成功。
try:
    dataset_metadata = LeRobotDatasetMetadata("franka_pnp_language", root='./demo_data_language')
except:
    dataset_metadata = LeRobotDatasetMetadata("franka_pnp_language", root='./franka_pnp_language')
# 把数据集里的字段翻译成模型认识的“特征”定义。
features = dataset_to_policy_features(dataset_metadata.features)
# 从所有特征里挑出“动作（ACTION）”类型的，作为模型的【输出】——也就是模型要预测什么。
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
# 剩下的（不是动作的）都当作模型的【输入】——也就是图像、关节状态这些“观测”。
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# 策略通过一个配置类来初始化，本例中即 `SmolVLAConfig`。在这个示例里，
# 我们直接使用默认值，因此除了输入/输出特征外无需传入其他参数。
# 使用时序集成（temporal ensemble）来得到更平滑的轨迹预测
#
# 通俗讲：cfg 就像给模型填的一张“规格表”，告诉它输入有哪些、输出有哪些。
# - chunk_size=5：模型一次性预测“未来 5 步”的动作（成块预测，而不是一步一问，更稳更连贯）。
# - n_action_steps=5：这预测出的 5 步动作里，实际拿去执行多少步。
cfg = SmolVLAConfig(input_features=input_features, output_features=output_features, chunk_size= 5, n_action_steps=5)
# 根据配置推算出：模型每次需要历史/未来哪几个时间点的数据（时序对齐用）。
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

# 现在我们可以用这个配置和数据集统计信息来实例化策略。
#
# 【加载微调权重】from_pretrained 的意思是“把训练好的模型从硬盘里读出来”。
# - 第一个参数是权重文件夹路径：'./ckpt/smolvla_franka/checkpoints/last/pretrained_model'
#   就是步骤 2 训练完保存下来的“成果”，里面是学好的网络参数。
# - dataset_stats 把数据集统计量交给模型，让它知道怎么归一化输入、还原输出。
# 这一步之后，policy 就是一个“学成归来、可以直接干活”的模型对象了。
policy = SmolVLAPolicy.from_pretrained('./ckpt/smolvla_franka/checkpoints/last/pretrained_model', dataset_stats=dataset_metadata.stats)
# 若已用 policy.push_to_hub 把自己训练好的策略上传到了 Hugging Face，
# 可以像下面这样直接从远端仓库加载，换成自己的 repo_id：
# policy = SmolVLAPolicy.from_pretrained("<你的用户名>/franka_pnp_smolvla", config=cfg, dataset_stats=dataset_metadata.stats)
# 把模型搬到指定硬件（GPU/CPU）上，之后所有计算都在那块硬件上进行。
policy.to(device)

# 创建仿真环境：相当于在电脑里搭一个“虚拟实验室”，里面有机械臂和待抓取的物体。
# 我们先在仿真里验证模型，安全又省钱，不用担心真机摔坏。
from mujoco_env.SimpleEnv2 import SimpleEnv2  # MuJoCo 是一款物理仿真引擎；SimpleEnv2 是基于它封装的抓放环境
xml_path = './asset/example_scene_y2.xml'  # 场景描述文件：定义了机械臂、桌子、物体的位置和外观
# action_type='joint_angle' 表示我们用“关节角度”来控制机械臂
# （即直接告诉每个关节转到多少度，而不是告诉末端去哪个坐标）。
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')  # PnP = Pick and Place（抓取并放置）

# 【图像预处理】模型不能直接“看”原始照片，需要先把图片整理成它习惯的格式。
# 这里定义一个预处理流水线，把摄像头拍到的图片转换成模型能吃的张量(tensor)。
from torchvision import transforms
# 方法 1：使用 torchvision.transforms
def get_default_transform(image_size: int = 224):
    """
    返回一个 torchvision 变换，它会：
     转换为 FloatTensor 并将像素值从 [0,255] 缩放到 [0.0,1.0]

    通俗解释：
    - 普通图片每个像素的颜色值是 0~255 的整数（比如纯红 = 255）。
    - 神经网络更喜欢 0.0~1.0 的小数，数值小、计算稳定。
    - ToTensor() 一步到位：既把图片变成 PyTorch 张量，又顺手把 0~255 缩放到 0.0~1.0。
    - 同时它会把图片维度从“高×宽×通道(HWC)”调整成模型要的“通道×高×宽(CHW)”。
    （参数 image_size 这里实际没被用到，只是保留接口；真正的缩放在下面调用处用 resize 完成。）
    """
    return transforms.Compose([  # Compose = 把多个步骤串成一条流水线，依次执行
        transforms.ToTensor(),  # PIL [0–255] -> FloatTensor [0.0–1.0]，形状为 C×H×W
    ])

# ===================== 准备开跑前的初始化 =====================
step = 0                       # 步数计数器，记录这一回合已经执行了多少步
PnPEnv.reset(seed=0)           # 把环境恢复到初始状态；seed=0 固定随机种子，保证每次开局物体摆放一致，便于复现
policy.reset()                 # 清空模型内部的“记忆”（比如它缓存的待执行动作队列），开始干净的一回合
policy.eval()                  # 把模型切到“评估/推理模式”：关闭训练专用的随机行为(如 dropout)，输出更稳定
save_image = True              # 一个开关标志（本脚本里仅作状态标记，控制是否处于首回合录像阶段）
IMG_TRANSFORM = get_default_transform()  # 实例化上面定义的图像预处理流水线，循环里反复用它

# ===================== 闭环执行（边看边想边动） =====================
# 这是整个部署的核心——一个不断循环的“感知-决策-执行”回路，俗称“闭环控制”：
#   看一眼画面 -> 模型想一下该怎么动 -> 真的去动 -> 再看一眼新画面 ……如此往复。
# 之所以叫“闭环”，是因为执行后的新画面又反馈回来影响下一步决策，形成闭合的环，
# 这样即使中途有偏差，下一轮也能根据最新画面纠正过来。
while PnPEnv.env.is_viewer_alive():  # 只要仿真窗口还开着（没被关掉），就一直循环
    PnPEnv.step_env()                # 推进一小步物理仿真（让时间往前走、物体按物理规律运动）
    # 物理仿真步进得很快，但我们不必每一步都让模型做决策。
    # loop_every(HZ=20) 表示“每秒只触发 20 次”进入下面的决策逻辑，既够用又省算力。
    if PnPEnv.env.loop_every(HZ=20):
        # 检查任务是否完成（比如物体是否已经被放到了目标位置）
        success = PnPEnv.check_success()
        if success:
            print('成功')
            # 任务完成了，就把一切归零，开始新的一回合：
            # 重置环境与动作队列
            policy.reset()   # 清空模型缓存的动作，避免上一回合的残留影响新回合
            PnPEnv.reset()   # 重新摆放物体、复位机械臂
            step = 0         # 步数清零
            save_image = False
        # ---------- 第 1 步：感知（收集模型需要的输入）----------
        # 获取环境的当前状态：机械臂各关节当前的角度。取前 7 个 [:7]，对应 7 个关节。
        state = PnPEnv.get_joint_state()[:7]
        # 从环境获取当前图像：返回两路画面——
        #   image 是“第三人称/全局相机”看到的场景，wirst_image 是装在机械臂手腕上的相机看到的近景。
        #   （手腕相机能看清要抓的物体细节，全局相机能看清整体布局，两者互补。）
        #   注：变量名 wirst_image 是原代码里的拼写，保持不动。
        image, wirst_image = PnPEnv.grab_image()
        # 把全局相机图像做预处理：数组 -> PIL 图片 -> 缩放到 256×256 -> 转成 [0,1] 张量
        image = Image.fromarray(image)        # numpy 数组转成 PIL 图片对象
        image = image.resize((256, 256))      # 统一缩放到 256×256，保证尺寸固定、和训练时一致
        image = IMG_TRANSFORM(image)          # 走预处理流水线，变成模型能吃的张量
        # 手腕相机图像做同样的预处理
        wrist_image = Image.fromarray(wirst_image)
        wrist_image = wrist_image.resize((256, 256))
        wrist_image = IMG_TRANSFORM(wrist_image)
        # ---------- 第 2 步：把所有输入打包成一个字典喂给模型 ----------
        # unsqueeze(0) 是在最前面加一个“批次维度”：模型习惯一次处理一批数据，
        # 即使我们只有 1 张图，也要包装成“1 张图的批”（形状从 C×H×W 变成 1×C×H×W）。
        # .to(device) 把数据也搬到 GPU 上，和模型待在同一块硬件才能一起算。
        data = {
            'observation.state': torch.tensor([state]).to(device),       # 关节状态
            'observation.image': image.unsqueeze(0).to(device),          # 全局相机图像
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),  # 手腕相机图像
            # 【必须提供语言指令】这就是 VLA 里的 “L(Language)”：
            # 告诉机器人“这一回合要干什么”，比如“把红色方块放进盘子”。
            # 没有这句指令，模型不知道目标是什么，就无法正确行动。
            # PnPEnv.instruction 是环境根据当前任务自动给出的指令文本（保持原样，不要改）。
            'task': [PnPEnv.instruction],
        }
        # ---------- 第 3 步：决策（模型推理出该做什么动作）----------
        # select_action 是 VLA 里的 “A(Action)”：模型看完图、读完指令后，吐出该执行的动作。
        # 选择一个动作
        action = policy.select_action(data)
        # 模型输出的是一批/多维结果，这里取出第 0 条样本的前 8 个数值 [0,:8]
        # （对应 7 个关节角 + 1 个夹爪开合），并搬回 CPU、脱离计算图、转成普通 numpy 数组，
        # 这样才能交给仿真环境去执行。
        action = action[0,:8].cpu().detach().numpy()
        # ---------- 第 4 步：执行（让机械臂真的动起来）----------
        # 在环境中执行一步
        _ = PnPEnv.step(action)   # 把动作下发给仿真，机械臂按这个动作移动
        PnPEnv.render()           # 把最新画面画到屏幕上，方便我们肉眼观察
        step += 1                 # 步数 +1
        # 执行完再检查一次是否成功；成功就跳出整个循环，结束部署。
        success = PnPEnv.check_success()
        if success:
            print('成功')
            break

# 下面这段是“把训练好的模型上传到 Hugging Face 云端分享给别人”的代码，
# 默认被注释掉了（不会执行）。如果你想分享自己的模型，去掉注释并填上你自己的仓库名即可。
# - repo_id：云端仓库的名字（用户名/仓库名）。
# - commit_message：本次上传的说明文字（类似 git 提交说明）。
# policy.push_to_hub(
#     repo_id='<你的用户名>/franka_pnp_smolvla',
#     commit_message='Add trained policy for PnP task',
# )
