"""
步骤 7：部署微调好的 π0（pi0）视觉-语言-动作（VLA）策略。

================================ 先科普几个名词（零基础友好）================================
什么是 VLA（Vision-Language-Action，视觉-语言-动作）模型？
  打个比方：传统机器人控制就像“按死的菜谱做菜”，只会做被写死的那一道菜。
  而 VLA 模型更像一个“能听懂话的厨师助手”：
    - Vision（视觉）：用摄像头“看”当前桌面/物体长什么样（输入图像）；
    - Language（语言）：能“听懂”一句自然语言指令，比如“把红色方块放进盒子里”；
    - Action（动作）：综合“看到的”和“听到的”，直接输出机器人该怎么动（关节角度等）。
  也就是说，同一个模型，换一句指令就能做不同的事，不需要为每个任务重写程序。

π0（pi0）这个 VLA 模型有什么特点？
  1) 它“三种信息一起看”：图像 + 机器人当前状态（关节角度等）+ 自然语言指令；
  2) 它输出的不是“下一帧单个动作”，而是“一小段连续的动作序列”（动作块 chunk）；
  3) 它生成动作用的是“流匹配（flow matching）”技术——可以粗略理解为：
     模型不是一次拍脑袋猜出动作，而是从“随机噪声”出发，一步步把噪声“流动/雕刻”
     成一条平滑、合理的动作轨迹（类似扩散模型“去噪”的思路）。这样动作更顺、更稳。

本脚本干什么？
  加载我们已经“微调（fine-tune）”好的 π0 权重，放进一个“语言条件”的仿真环境里，
  让机器人按自然语言指令真正去执行“抓取-放置（Pick and Place, PnP）”任务。
  “语言条件”的意思是：环境会给出一句任务指令（instruction），策略必须读这句话才知道要干嘛。
==============================================================================================

训练 π0 请在终端运行（train_vla.py 即原 train_model.py 改名而来）：
    python vla/train_vla.py --config_path pi0_omy.yaml

部署运行方式（需要 GPU + 图形界面）：
    conda activate lerobot
    python vla/deploy_pi0.py

注意：脚本中以 [终端命令] 标注的行（pip/git/python vla/train_vla.py）原是
notebook 的 shell 命令，已注释掉，请按需在终端单独执行。
提示：本脚本原为教程笔记本，这里转成了带详细注释的 .py 脚本方便逐行学习。
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
# # 部署训练好的 pi0 策略
#
# <img src="./media/rollout2.gif" width="480" height="360">
#
# 在仿真环境中部署训练好的策略。
# ======================================================================

# 下面两行是“环境准备”用的安装命令（原本是 notebook 里的 shell 命令）。
# 它们不是 Python 代码，请在终端单独执行；这里用 # 注释掉避免被当成代码运行。
#   - pytest：测试工具（部分依赖会用到）；
#   - transformers==4.50.3：HuggingFace 的 Transformers 库，π0 内部用到它，
#     这里固定到 4.50.3 这个版本是为了避免版本不兼容导致的报错。
# [终端命令] pip install pytest
# [终端命令] pip install transformers==4.50.3

# ======================================================================
# # 训练 pi0 并部署
# ======================================================================

# ======================================================================
# ### [可选] 下载数据集
# ======================================================================

'''
如果你想使用已采集好的数据集,请从 Hugging Face 下载。
（数据集 = 一堆“演示录像”：人/专家先示范怎么做任务，模型从中学会模仿。
  这个 omy_pnp_language 数据集带语言指令，正好用来训练“听话”的 VLA 策略。）
'''
# 下面这行用 git 把数据集仓库整个克隆到本地（同样是终端命令，不是 Python）。
# [终端命令] git clone https://huggingface.co/datasets/Jeongeun/omy_pnp_language

# ======================================================================
# ## 步骤 1. 修改配置文件 pi0_omy.yaml
# ======================================================================

# ======================================================================
# pi0_omy.yaml 文件
# ```
# dataset:
#   repo_id: omy_pnp
#   root: ./omy_pnp
# policy:
#   type : pi0
#   chunk_size: 5
#   n_action_steps: 5
# save_checkpoint: true
# output_dir: ./ckpt/pi0_omy
# batch_size: 16
# job_name : pi0_omy
# resume: false
# seed : 42
# num_workers: 8
# steps: 20_000
# eval_freq: -1 # No evaluation
# log_freq: 50
# save_checkpoint: true
# save_freq: 5_000
# use_policy_training_preset: true
#
# wandb:
#   enable: true
#   project: pi0_omy
#   entity: <YOUR ENTITY for wandb>
#   disable_artifact: true
# ```
# ======================================================================

# ======================================================================
# ## 步骤 2. 训练模型。
# 该代码在 A100 上测试通过。
# ======================================================================

# 步骤 2 的训练命令：在终端里运行下面这行，开始用配置文件训练 π0。
# （train_vla.py 就是原来的 train_model.py，只是改了个更贴切的名字：train VLA = 训练 VLA 模型）
# [终端命令] python vla/train_vla.py --config_path pi0_omy.yaml

# ======================================================================
# ## 步骤 3. 部署
# ======================================================================

# ---------------------------- 导入所需的库 ----------------------------
# 下面这些 import 把要用到的“工具箱”引进来。逐个说明：
#   - LeRobotDataset / LeRobotDatasetMetadata：LeRobot 框架里读“数据集”和“数据集元信息”的类。
#     元信息（metadata）= 数据集的“说明书”，比如有哪些特征、每个特征的统计量（均值/方差）等。
#   - numpy：科学计算库，处理数组/数值（这里主要把张量转成 numpy 数组）。
#   - PI0Config / PI0Policy：π0 策略的“配置类”和“模型类”。Config 负责描述模型该怎么搭，
#     Policy 就是真正能“看图听话出动作”的那个 π0 模型本体。
#   - FeatureType：枚举类型，用来区分一个特征到底是“输入观测”还是“输出动作（ACTION）”。
#   - resolve_delta_timestamps：根据配置算出“需要取哪些时间点的数据”（π0 输出动作序列，会涉及时间）。
#   - dataset_to_policy_features：把“数据集里的特征”翻译成“策略能理解的特征描述”。
#   - torch：PyTorch 深度学习框架，模型和张量都跑在它上面。
#   - PIL.Image：图像处理库，用来把摄像头画面包装成图片对象、做缩放等。
#   - torchvision：PyTorch 的视觉工具库，下面用它做图像预处理（转成张量）。
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision

# ======================================================================
# ### 加载策略
# ======================================================================

# 指定模型运行在 GPU 上（'cuda' 表示用 NVIDIA 显卡）。π0 模型较大，几乎必须有 GPU。
device = 'cuda'

# 读取数据集的“元信息（说明书）”。这里其实不是要拿训练数据，而是要拿两样关键东西：
#   1) 特征定义（有哪些观测、哪些动作、各自形状）；
#   2) 统计量 stats（每个特征的均值/方差等），后面给模型做“归一化”要用到。
# try/except 是在做“路径兜底”：先试 ./demo_data_language 这个目录，
# 找不到就退而求其次用 ./omy_pnp_language，哪个存在就用哪个，避免因路径不同而报错。
try:
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./demo_data_language')
except:
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./omy_pnp_language')
# 把数据集特征翻译成“策略能懂的特征描述”。
features = dataset_to_policy_features(dataset_metadata.features)
# 从所有特征里挑出“动作（ACTION）”作为输出特征：这是模型要预测/产出的东西。
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
# 剩下的（不是动作的）就都当作“输入特征”：也就是各种观测，比如图像、机器人状态等。
input_features = {key: ft for key, ft in features.items() if key not in output_features}
# 策略通过一个配置类来初始化,这里使用 PI0Config。本示例中
# 我们只使用默认值,因此除了输入/输出特征外无需传入其他参数。
# 通过时间集成(temporal ensemble)使轨迹预测更加平滑
# 解释两个关键参数（VLA 输出的是“一小段动作”而非单步，所以才需要它们）：
#   - chunk_size=5     ：模型一次预测“5 步”的动作块（chunk）。π0 是“成段”地规划动作的。
#   - n_action_steps=5 ：这 5 步里，实际拿去执行的步数（这里也是 5，即预测多少就执行多少）。
cfg = PI0Config(input_features=input_features, output_features=output_features, chunk_size= 5, n_action_steps=5)
# 根据配置推算“需要哪些时间点的数据偏移”。因为涉及动作序列，模型对“时间”是敏感的。
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

# 现在加载我们“微调好”的 π0 权重，实例化出真正可用的策略对象。
# from_pretrained = “加载预训练/已训练好的权重”：相当于把训练阶段学到的全部本领（参数）
# 一次性装进模型里，省去从零训练。这里从本地 checkpoints/last（最近一次保存的检查点）读取。
# dataset_stats 把数据集统计量交给策略，模型内部会用它对输入做归一化、对输出做反归一化，
# 保证“部署时的数据处理方式”和“训练时”一致——否则模型会因为“看到的数据尺度不对”而失灵。
policy = PI0Policy.from_pretrained('./ckpt/pi0_omy/checkpoints/last/pretrained_model', dataset_stats=dataset_metadata.stats)
# 如果你没有资源进行训练,可以从 hub 加载已训练好的策略。
# （下面这行被注释掉了：它演示如何直接从 Hugging Face 在线仓库下载别人训练好的同款策略。）
# policy = PI0Policy.from_pretrained("Jeongeun/omy_pnp_pi0", config=cfg, dataset_stats=dataset_metadata.stats)
# 把模型搬到 GPU 上（前面 device='cuda'），这样推理才会用显卡加速。
policy.to(device)

# ======================================================================
# ### 部署策略
# ======================================================================

# ---------------------------- 创建仿真环境 ----------------------------
# 用 MuJoCo（一个物理仿真器）搭一个虚拟的桌面抓放场景。
# xml_path 指向场景描述文件（里面写了机器人、桌子、物块等长什么样、放哪里）。
# action_type='joint_angle' 表示：我们给环境的动作是“关节角度”，即直接告诉每个关节该转到多少度。
from mujoco_env.y_env2 import SimpleEnv2
xml_path = './asset/example_scene_y2.xml'
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')

# ---------------------------- 图像预处理 ----------------------------
# 为什么要做图像预处理？因为模型“吃”的不是普通图片，而是规定格式的张量（Tensor）。
# 摄像头给出的是 0~255 的整数像素、且是 PIL 图片对象；而神经网络通常需要：
#   (1) 浮点数；(2) 数值缩放到 0.0~1.0；(3) 维度顺序是 C×H×W（通道在前）。
# 下面这个变换就负责把“原始图片”转成“模型能直接用的张量”，让训练和部署的输入口径一致。
from torchvision import transforms
# 方法 1:使用 torchvision.transforms
def get_default_transform(image_size: int = 224):
    """
    返回一个 torchvision 变换,功能为:
     将图像转换为 FloatTensor,并把像素值从 [0,255] 缩放到 [0.0,1.0]
    """
    # transforms.Compose([...]) 把多个变换“串成流水线”，图片依次经过每一步处理。
    # 这里只放了一步 ToTensor：它一口气完成“类型转浮点 + 缩放到 0~1 + 调整成 C×H×W”。
    return transforms.Compose([
        transforms.ToTensor(),  # PIL [0–255] -> FloatTensor [0.0–1.0],形状为 C×H×W
    ])

# ---------------------------- 部署前的初始化 ----------------------------
step = 0                       # 步数计数器，从 0 开始数本回合执行了多少步。
PnPEnv.reset(seed=0)           # 重置环境到初始状态；seed=0 固定随机种子，保证每次开局物体摆放一致、可复现。
policy.reset()                 # 重置策略的“内部记忆/动作队列”。π0 成段输出动作并排队执行，开局要清空这个队列。
policy.eval()                  # 切到“推理模式（评估模式）”：关闭 dropout 等只在训练时用的机制，让输出稳定。
save_image = True              # 一个标志位（脚本里保留），用于控制是否保存图像。
IMG_TRANSFORM = get_default_transform()  # 提前准备好上面定义的图像预处理流水线，循环里反复用。

# ---------------------------- 闭环执行（核心循环）----------------------------
# “闭环（closed-loop）”是什么意思？打个比方：边走边看路，而不是闭着眼睛走完全程。
# 每一拍：先“看图 + 读状态 + 听指令”得到最新观测 → 让 π0 算出该怎么动 → 真的去动一步
#         → 环境因此变化 → 再回到开头重新观测……如此循环。
# 因为每一步都根据“最新现实”来决策，所以即使中途有偏差也能被不断纠正，这就是闭环的好处。
#
# 关键提醒（和上一节 ACT 部署的最大区别！）：
#   ACT 只看图像+状态就能动；而 π0 是 VLA 模型，必须额外喂一句“语言指令 task”，
#   它要先“听懂要做什么任务”才能输出对的动作。所以下面 data 字典里多了 'task' 字段——
#   这正是“语言条件（language-conditioned）”策略的标志。少了它，π0 就不知道该干嘛。
while PnPEnv.env.is_viewer_alive():   # 只要可视化窗口还开着（没被关掉），就一直循环。
    PnPEnv.step_env()                 # 推进物理仿真一个底层小步（让世界按物理规律往前走一点）。
    if PnPEnv.env.loop_every(HZ=20):  # 控制“决策频率”：约每秒 20 次（20Hz）才执行一次下面的决策逻辑。
        # 检查任务是否完成
        success = PnPEnv.check_success()
        if success:
            print('成功')
            # 任务成功了：把策略和环境都重置，开始新一回合（相当于“再来一局”）。
            # 重置环境和动作队列
            policy.reset()
            PnPEnv.reset()
            step = 0
            save_image = False
        # 获取环境的当前状态：取机器人前 6 个关节的角度，作为“本体感知”观测交给模型。
        state = PnPEnv.get_joint_state()[:6]
        # 从环境中获取当前图像：返回两路画面——一路是固定的全局相机，一路是装在手腕上的相机。
        # （手腕相机 wrist camera：跟着机械臂末端走，能看清要抓的物体细节。注意原变量名拼写为 wirst_image，保持不改。）
        image, wirst_image = PnPEnv.grab_image()
        # 下面对两路图像做同样的预处理：包成 PIL 图片 -> 缩放到 256x256 -> 转成模型要的张量。
        image = Image.fromarray(image)
        image = image.resize((256, 256))
        image = IMG_TRANSFORM(image)
        wrist_image = Image.fromarray(wirst_image)
        wrist_image = wrist_image.resize((256, 256))
        wrist_image = IMG_TRANSFORM(wrist_image)
        # 把这一拍的所有观测打包成一个字典，喂给策略。字典的“键名”必须和训练时一致，模型才认得：
        #   'observation.state'        ：机器人关节状态（注意外面套了一层 [ ] 变成批量维度，再转 GPU 张量）。
        #   'observation.image'        ：全局相机图像；unsqueeze(0) 在最前面补一个“批大小=1”的维度。
        #   'observation.wrist_image'  ：手腕相机图像，同样补批维度。
        #   'task'                     ：★最关键★ 自然语言指令本身（来自环境的 PnPEnv.instruction），
        #                                π0 靠它“听懂要做什么”。这就是 VLA / 语言条件策略相比 ACT 多出来的输入。
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': [PnPEnv.instruction],
        }
        # 选择一个动作：把观测交给 π0，它内部用“流匹配”从噪声雕出一段平滑动作，
        # 并按队列每次吐出“当前这一步”该执行的动作。select_action 屏蔽了这些细节，直接给你一步动作。
        action = policy.select_action(data)
        # 模型输出是 GPU 上的张量；这里取第 0 个样本的前 7 维（机器人需要的动作维度），
        # 搬回 CPU、断开梯度追踪、转成普通的 numpy 数组，方便交给仿真环境执行。
        action = action[0,:7].cpu().detach().numpy()
        # 在环境中执行一步：机器人真的按这个动作动一下，世界随之改变（下一拍会观测到新画面）。
        _ = PnPEnv.step(action)
        PnPEnv.render()   # 刷新可视化画面，让我们肉眼能看到机器人在动。
        step += 1         # 步数 +1。
        # 动完再查一次是否成功；成功就打印并跳出整个循环，结束部署。
        success = PnPEnv.check_success()
        if success:
            print('成功')
            break

# 下面这段被注释掉了（默认不执行）：它演示如何把你训练好的策略“上传”到 Hugging Face Hub，
# 方便分享或日后再下载。repo_id 是远程仓库名，commit_message 是这次上传的说明。
# 只有当你真的想公开/备份自己的模型时，才取消注释来用。
# policy.push_to_hub(
#     repo_id='Jeongeun/omy_pnp_pi0',
#     commit_message='Add trained policy for PnP task',
# )
