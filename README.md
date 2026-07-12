# LeRobot + MuJoCo 机械臂操作：运行指南与代码说明

本文档面向想在自定义仿真环境中**采集示教数据、训练并部署视觉-语言-动作（VLA）/模仿学习策略**的读者。第一次使用请先按下面的“快速运行”操作；后续章节再解释环境、数据和模型原理。

整个项目围绕一条主线展开：

> 在 MuJoCo 里搭一个桌面抓放（pick-and-place）场景 → 用键盘遥操作机械臂采集示教数据 → 训练动作策略（ACT / Diffusion Policy / π0 / SmolVLA）→ 放回仿真自动执行并评估。

机器人为 **Franka Emika Panda**，7 自由度机械臂加一个平行夹爪，任务是"把杯子放到盘子上"。

---

## 快速运行（第一次使用先看这里）

每次打开新终端，先激活环境并进入项目根目录：

```bash
conda activate lerobot
cd /path/to/lerobot_mujoco
```

然后从下面两条路线中选择一条。**同一次实验只需选择一种模型配置，不需要把四种模型全部训练一遍。**

### 四个模型常用命令速查

ACT：

```bash
python il/train_il.py --config_path=config/il/act_franka.yaml
python il/deploy_il.py --config_path=config/il/act_franka.yaml
CUDA_VISIBLE_DEVICES=7 python il/deploy_il.py --config_path=config/il/act_franka.yaml --checkpoint=./ckpt/act_franka/checkpoints/100000/pretrained_model --device=cuda --seed=0 --max_steps=2000 --headless
```

Diffusion Policy：

```bash
python il/train_il.py --config_path=config/il/diffusion_franka.yaml
python il/deploy_il.py --config_path=config/il/diffusion_franka.yaml
CUDA_VISIBLE_DEVICES=7 python il/deploy_il.py --config_path=config/il/diffusion_franka.yaml --checkpoint=./ckpt/diffusion_franka/checkpoints/100000/pretrained_model --device=cuda --seed=0 --max_steps=2000 --headless
```

π0：

```bash
python vla/train_vla.py --config_path=config/vla/pi0_franka.yaml
CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --num_processes=3 --main_process_port=29501 vla/train_vla.py --config_path=config/vla/pi0_franka.yaml
python vla/deploy_vla.py --config_path=config/vla/pi0_franka.yaml
CUDA_VISIBLE_DEVICES=7 python vla/deploy_vla.py --config_path=config/vla/pi0_franka.yaml --checkpoint=./ckpt/pi0_franka_v2/checkpoints/040000/pretrained_model --device=cuda --seed=0 --max_steps=2000 --headless
```

SmolVLA：

```bash
python vla/train_vla.py --config_path=config/vla/smolvla_franka.yaml
CUDA_VISIBLE_DEVICES=5,6,7 accelerate launch --num_processes=3 --main_process_port=29502 vla/train_vla.py --config_path=config/vla/smolvla_franka.yaml
python vla/deploy_vla.py --config_path=config/vla/smolvla_franka.yaml
CUDA_VISIBLE_DEVICES=7 python vla/deploy_vla.py --config_path=config/vla/smolvla_franka.yaml --checkpoint=./ckpt/smolvla_franka_v2/checkpoints/030000/pretrained_model --device=cuda --seed=0 --max_steps=2000 --headless
```

无头部署不写 `--video` 时会自动按模型保存到 `output/act/`、`output/diffusion/`、`output/pi0/` 或 `output/smolvla/`。

### 路线 A：IL 模仿学习（推荐先跑通）

```bash
# 0. 一致性自检：验证场景 XML、策略配置、已有数据集是否满足代码假设。
#    改过机械臂/场景 XML 或 YAML 配置后，先跑这一步再采集/训练。
python verify_setup.py

# 1. 采集并检查无语言指令的示教数据
python il/collect_data_il.py
# 如果不想手动遥操作，也可以用自动专家采集；参数在 auto_collect.py 顶部改：
# python auto_collect.py
python il/visualize_data_il.py

# 2. 二选一训练：ACT 或 Diffusion Policy
python il/train_il.py --config_path=config/il/act_franka.yaml
# python il/train_il.py --config_path=config/il/diffusion_franka.yaml

# 3. 使用与训练相同的配置部署
python il/deploy_il.py --config_path=config/il/act_franka.yaml
# python il/deploy_il.py --config_path=config/il/diffusion_franka.yaml
```

### 路线 B：VLA 语言条件策略

```bash
# 1. 采集并检查带语言指令的示教数据
python vla/collect_data_language.py
# 自动专家采集语言数据：运行后输入 2 选择 vla
# python auto_collect.py
python vla/visualize_data_language.py

# 2. 二选一训练，再用统一 VLA 入口部署
python vla/train_vla.py --config_path=config/vla/pi0_franka.yaml
python vla/deploy_vla.py --config_path=config/vla/pi0_franka.yaml

# SmolVLA 方案：
# python vla/train_vla.py --config_path=config/vla/smolvla_franka.yaml
# python vla/deploy_vla.py --config_path=config/vla/smolvla_franka.yaml
```

默认输出位置：IL 数据在 `./demo_data`，VLA 数据在 `./demo_data_language`，训练检查点在 `./ckpt`。完整安装、按键说明、配置字段和故障排查见后续章节。

---

## 目录

1. [整体架构与目录结构](#一整体架构与目录结构)
2. [环境安装](#二环境安装)
3. [核心概念与数据流](#三核心概念与数据流)
4. [底层代码模块详解（mujoco_env）](#四底层代码模块详解mujoco_env)
5. [完整工作流程与运行步骤](#五完整工作流程与运行步骤)
6. [模型原理详解](#六模型原理详解)
7. [配置文件说明](#七配置文件说明)
8. [常见问题（FAQ）](#八常见问题faq)
9. [致谢与参考](#九致谢与参考)

---

## 一、整体架构与目录结构

项目分为三层：**仿真底层（mujoco_env）** → **任务环境封装（SimpleEnv / SimpleEnv2）** → **教程脚本（按用途命名的 .py 脚本 + 训练脚本）**。

```
lerobot_mujoco/
├── mujoco_env/                 # 仿真底层与环境封装（核心 Python 代码）
│   ├── mujoco_parser.py        # MuJoCo 解析器/仿真封装（本项目地基）
│   ├── utils.py                # 通用工具：随机采样、图像处理、轨迹插值、XML 等
│   ├── ik.py                   # 逆运动学（IK）求解：增广雅可比 + 阻尼最小二乘
│   ├── transforms.py           # 坐标/旋转变换：欧拉角、四元数、齐次矩阵、点云等
│   ├── SimpleEnv1.py           # 单物体抓放环境 SimpleEnv（1 个杯子 + 1 个盘子）
│   └── SimpleEnv2.py           # 语言条件多物体环境 SimpleEnv2（红/蓝 2 个杯子 + 1 个盘子）
│
├── il/                         # 【流水线一】模仿学习（ACT / Diffusion Policy）
│   ├── collect_data_il.py      #   步骤1：键盘遥操作采集示教数据
│   ├── visualize_data_il.py    #   步骤2：回放/可视化已采集数据
│   ├── train_il.py             #   步骤3：根据 YAML 训练 ACT / Diffusion
│   ├── deploy_il.py            #   步骤4：根据同一 YAML 自动部署
│   ├── eval_offline_il.py      #   [排查] 离线开环评估：预测动作 vs 示教动作
│   └── push_dataset_il.py      #   [可选] 上传示教数据集
│
├── vla/                         # 【流水线二】语言条件 VLA（π0 / SmolVLA）全流程
│   ├── collect_data_language.py    #   步骤5：采集带语言指令的数据
│   ├── visualize_data_language.py  #   步骤6：可视化语言条件数据
│   ├── train_vla.py            #   训练 π0 / SmolVLA 的通用入口
│   ├── deploy_vla.py           #   统一部署 π0 / SmolVLA（窗口/无头/多种子评估）
│   ├── deploy_pi0.py           #   旧版 π0 教程部署脚本
│   ├── deploy_smolvla.py       #   旧版 SmolVLA 教程部署脚本
│   └── push_dataset_language.py    #   [可选] 把采集好的数据集上传到 Hugging Face
│
├── config/                     # 按训练路线分类的配置文件
│   ├── il/
│   │   ├── act_franka.yaml         # ACT 训练/部署配置
│   │   └── diffusion_franka.yaml   # Diffusion Policy 训练/部署配置
│   └── vla/
│       ├── pi0_franka.yaml         # π0 训练配置
│       └── smolvla_franka.yaml     # SmolVLA 训练配置
├── requirements.txt            # 依赖清单
├── push_ckpt.py                # 增量上传整个 ckpt 到 Hugging Face
├── auto_collect.py             # 自动专家采集 IL/VLA 数据（可无头运行）
├── verify_setup.py             # 场景/配置/数据集一致性自检（改过 XML 或配置后先跑）
│
└── asset/                      # 仿真资源
    ├── franka_panda/           # Franka Panda 机械臂的 MJCF 模型与网格
    ├── tabletop/                # 桌面、相机、各类可抓取物体
    ├── objaverse/                # 杯子/盘子网格（plate_11 需解压）
    ├── example_scene_y.xml     # 单物体场景（SimpleEnv 使用）
    └── example_scene_y2.xml    # 语言条件多物体场景（SimpleEnv2 使用）

# 运行后会自动生成（已在 .gitignore 中忽略，不在仓库里）：
#   demo_data/            IL 流水线采集的数据集
#   demo_data_language/   VLA 流水线采集的语言条件数据集
#   ckpt/                 训练得到的模型检查点
```

**两条流水线（两个文件夹）：**

- **`il/`**：非语言模仿学习路线，共用一份示教数据，可用 YAML 选择 **ACT** 或 **Diffusion Policy**。顺序为 `collect_data_il → visualize_data_il → train_il → deploy_il`。
- **`vla/`**：进阶路线，采集带自然语言指令的数据，训练并部署 **π0 / SmolVLA** 这类视觉-语言-动作（VLA）模型（能听懂"把红色杯子放到盘子上"）。顺序为 `collect_data_language → visualize_data_language → train_vla → deploy_pi0 / deploy_smolvla`。

> **运行约定**：所有脚本统一**在项目根目录**下用 `python 文件夹/脚本名.py` 的方式运行（例如 `python il/train_il.py --config_path=config/il/act_franka.yaml`）。LeRobot 的配置参数要写成 `--config_path=文件路径`（等号不能省略）。每个脚本顶部都加了一小段"环境自举"代码：它会自动把项目根目录加入模块搜索路径、并把工作目录切到根目录。

**依赖关系一览：**

```
教程脚本 (各步骤 .py 脚本 / train_il.py / train_vla.py)
        │  调用
        ▼
SimpleEnv1 / SimpleEnv2  (SimpleEnv1.py / SimpleEnv2.py)   ← 任务级封装：reset / step / teleop / 成功判定
        │  组合
        ▼
MuJoCoParserClass (mujoco_parser.py)             ← 仿真级封装：加载模型 / 步进 / 渲染 / 相机
        │  依赖
        ▼
ik.py（逆运动学） + transforms.py（坐标变换） + utils.py（工具）
        │  底层
        ▼
MuJoCo 3.1.6 物理引擎
```

---

## 二、环境安装

本章手把手带你从零搭好运行环境。**第一次跑这类项目的同学，照着从头做到尾即可。**

> **环境前提**
> - 操作系统：Linux（推荐 Ubuntu 20.04/22.04）或 Windows 均可；**遥操作采集**需要带显示器的机器。
> - Python：**3.12**（本项目当前实际测试版本）。
> - MuJoCo：**3.1.6**（由 `requirements.txt` 锁定，无需单独装）。
> - GPU：**训练 ACT / Diffusion / π0 / SmolVLA 需要 NVIDIA 显卡**（依赖按 CUDA 12.4 的 PyTorch 2.6 配置）；只想"回放数据/看仿真"则用 CPU 也能跑。

### 0. 准备工作：装 Conda 与 Git

如果你电脑里还没有 Conda 和 Git，先装好它们（已装过可跳过）：

- **Conda**：去 [Miniconda 官网](https://docs.conda.io/en/latest/miniconda.html) 下载对应系统的安装包装上。它用来创建互不干扰的"虚拟环境"。
- **Git**：去 [git-scm.com](https://git-scm.com/downloads) 下载安装。它用来下载本项目和 LeRobot 代码。

装完后，打开终端（Windows 用"Anaconda Prompt"，Linux 用普通终端），输入下面两行能看到版本号就说明装好了：

```bash
conda --version
git --version
```

### 1. 下载本项目代码

```bash
git clone <本仓库地址> lerobot_mujoco
cd lerobot_mujoco
```

> 之后所有命令，默认你**都在这个项目根目录 `lerobot_mujoco/` 下执行**。

### 2. 创建并激活虚拟环境

```bash
conda create -n lerobot python=3.12 -y   # 新建一个名为 lerobot 的环境
conda activate lerobot                   # 激活它（之后每次开新终端都要先执行这句）
```

> 激活成功后，命令行最前面会出现 `(lerobot)` 字样。**之后跑任何脚本前，都要先确认 `(lerobot)` 已激活。**

### 3. 安装所有依赖（含 LeRobot）

本项目的核心库 **LeRobot 不要用 `pip install lerobot` 直接装**（官方源版本变动快、容易和本项目其他依赖冲突）。我们已经在 `requirements.txt` 里**锁定了一个经过验证、可用的 LeRobot 指定提交（commit）**，连同 PyTorch、MuJoCo 等一并安装，一条命令搞定：

```bash
pip install -r requirements.txt
```

> 首次安装会下载 PyTorch/CUDA、PyArrow 和 LeRobot 等较大依赖，根据网络速度通常需要 **20～90 分钟**。只要终端仍在持续显示 `Downloading`、`Building` 或安装进度就不是卡死；完成时会出现 `Successfully installed ...`。

这条命令会做这些事（无需手动操作，了解即可）：

| 依赖 | 版本 | 用途 |
|------|------|------|
| `mujoco` | 3.1.6 | 物理仿真引擎 |
| `torch / torchvision / torchaudio` | 2.6.0（cu124） | 深度学习框架（CUDA 12.4） |
| `transformers` | 4.50.3 | π0/SmolVLA 的视觉语言骨干 |
| `diffusers` | 由 LeRobot 安装 | Diffusion Policy 的 DDPM/DDIM 噪声调度器 |
| `lerobot` | 锁定的 git commit | 数据集格式、策略实现、训练工具 |
| `datasets` | 3.4.1 | 数据集底层 |
| `pyautogui / matplotlib / scipy` | — | 遥操作、绘图、数值工具 |

> 训练前请运行 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`。
> 若显示 `+cpu False`，说明当前是 CPU 版 PyTorch；请重装 CUDA 12.4 版：
> `pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124`。

> **关于 LeRobot 是怎么装进来的**：`requirements.txt` 最后一行
> `git+https://github.com/huggingface/lerobot.git@10b7b35...#egg=lerobot`
> 表示 pip 会自动从 GitHub 克隆 LeRobot 仓库、切到那个指定 commit 再安装。**因此安装时需要联网，并且能正常访问 GitHub。** 如果这一步卡住或失败，多半是网络问题，见 [FAQ Q7](#八常见问题faq)。

**验证安装是否成功**（无报错即 OK）：

```bash
python -c "import mujoco, torch, lerobot; print('mujoco', mujoco.__version__); print('cuda ok:', torch.cuda.is_available())"
```

- 能打印出 `mujoco 3.1.6` 说明仿真库就绪。
- `cuda ok: True` 说明显卡可用（可训练）；若是 `False`，仿真/回放仍能跑，但训练会很慢或失败，请检查显卡驱动与 CUDA。

### 4. 解压物体资源（必做）

杯子/盘子的网格模型以压缩包形式提供，**必须先解压**，否则加载场景时会报找不到 `plate_11` 的错误：

```bash
cd asset/objaverse
unzip plate_11.zip      # Windows 无 unzip 命令时，直接用资源管理器右键“解压到当前文件夹”
cd ../..                # 回到项目根目录
```

解压后 `asset/objaverse/` 下应能看到 `plate_11/` 文件夹。

### 5. 关于"无显示器"的服务器

- **需要图形界面（有显示器/桌面）的脚本**：所有遥操作采集（`collect_data_il.py`、`collect_data_language.py`）和带窗口回放脚本。
- **纯命令行服务器即可**：训练脚本、上传脚本，以及带 `--headless` 的自动采集/部署脚本。无头模式使用离屏渲染，并把两路相机结果保存为 MP4。

> 如果只有一台无显示器的 GPU 服务器，可以在本地采集数据，或用 `push_dataset_il.py` / `push_dataset_language.py` 上传后在服务器训练。
> 脚本会在导入 MuJoCo 前自动选择后端：Windows 使用 WGL、Linux 使用 EGL、macOS 使用 CGL。Linux CPU 服务器没有 EGL 时，可安装 `libosmesa6` 并临时增加 `--gl_backend=osmesa`。

### 6. 用 Git 把本机修改同步到 Linux 服务器

先在修改代码的电脑上提交并推送。本次无头部署涉及以下文件：

```bash
git add .gitignore README.md requirements.txt auto_collect.py il/deploy_il.py vla/deploy_vla.py mujoco_env/SimpleEnv1.py mujoco_env/SimpleEnv2.py mujoco_env/eval_utils.py
git commit -m "feat: add automatic data collection"
git push origin main
```

然后在 Linux 服务器的项目目录拉取更新：

```bash
cd ~/all_users/z_work/lerobot_mujoco && git pull --ff-only origin main
```

如果 Linux 端提示本地修改会被覆盖，先把服务器上的修改安全暂存，再拉取：

```bash
cd ~/all_users/z_work/lerobot_mujoco && git stash push -u -m "linux-backup-before-update" && git pull --ff-only origin main
```

可以用 `git stash list` 查看备份；只有确实需要恢复服务器旧修改时才运行 `git stash pop`。`demo_data`、`ckpt`、`output` 和 `outputs` 已被 `.gitignore` 忽略，拉取代码不会覆盖数据集、检查点或视频。

本次新增了 OpenCV 视频写入依赖，Linux 更新代码后安装一次：

```bash
pip install opencv-python
```

---

## 三、核心概念与数据流

### 1. 任务定义

- **场景**：桌面上有一个/多个杯子和一个盘子，Franka Panda 机械臂固定在桌边（基座抬高到桌面高度 z=0.8）。
- **目标**：把杯子抓起来放到盘子上。
- **成功判定**（`SimpleEnv1.py` / `SimpleEnv2.py` 的 `check_success`，**仅在部署/评估脚本里使用，采集阶段不会自动判定**）：需要同时满足并**连续保持约 1.25 秒（20Hz 下 25 帧）**——杯子与盘子水平距离 < 0.09 m、竖直高度差 < 0.09 m、夹爪已张开（`finger_joint1 > 0.03`）、末端执行器抬升到 0.9 m 以上、且杯子基本静止（速度 < 0.01）。用连续帧计数是为了避免"杯子刚一碰到盘子就草草判成功"。
- **采集阶段的"完成"判定**：不依赖上面的自动检测，而是**由人按回车（Enter）手动确认**这一回合做得不错、可以保存——质量把关完全交给操作者。

### 2. 数据集格式（LeRobot Dataset）

采集到的数据以 **LeRobotDataset v2.1** 格式保存，`parquet` 存逐帧数据，`meta/` 存元信息，`fps=20`，`robot_type="franka"`。ACT（单物体）和语言条件（双杯子）两条流水线的字段形状不完全一样：

**ACT 数据集**（`./demo_data`，`repo_id="franka_pnp"`）：

| 字段 | 类型 | 形状 | 含义 |
|------|------|------|------|
| `observation.image` | image | (256, 256, 3) | 智能体视角（agentview）相机图 |
| `observation.wrist_image` | image | (256, 256, 3) | 腕部（第一人称 egocentric）相机图 |
| `observation.state` | float32 | (8,) | 7 个当前实际关节角 + 当前夹爪状态 |
| `action` | float32 | (8,) | 本帧下发的 7 个目标关节角 + 夹爪命令（0=张开，1=闭合） |
| `obj_init` | float32 | (6,) | 杯子初始位置(3) + 盘子初始位置(3)，仅记录，不参与训练 |

**语言条件数据集**（`./demo_data_language`，`repo_id="franka_pnp_language"`）：

| 字段 | 类型 | 形状 | 含义 |
|------|------|------|------|
| `observation.image` | image | (256, 256, 3) | 智能体视角相机图 |
| `observation.wrist_image` | image | (256, 256, 3) | 腕部相机图 |
| `observation.state` | float32 | (8,) | 7 个当前实际关节角 + 当前夹爪状态 |
| `action` | float32 | (8,) | 本帧下发的 7 个目标关节角 + 夹爪命令 |
| `obj_init` | float32 | (9,) | 红杯(3) + 蓝杯(3) + 盘子(3) 的初始位置，仅记录 |
| `task` | str | — | 自然语言指令，如 `"Place the red mug on the plate."`，每回合随机在红/蓝之间挑一个 |

数据集目录结构：

```
demo_data/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
└── meta/
    ├── episodes.jsonl     # 每条回合的索引/长度/任务
    ├── episodes_stats.jsonl
    ├── info.json          # 全局信息：fps、特征定义、总帧数等
    ├── stats.json         # 各特征的均值/方差（用于归一化）
    └── tasks.jsonl        # 任务（语言指令）列表
```

> **现成的 Franka 数据集**：本项目采集的数据已开源在 Hugging Face，可直接下载使用，不用自己从头采集：
> - ACT 数据集：[a3124371940/franka_pnp](https://huggingface.co/datasets/a3124371940/franka_pnp)
> - 语言条件数据集：[a3124371940/franka_pnp_language](https://huggingface.co/datasets/a3124371940/franka_pnp_language)
>
> 下载方式：`git clone https://huggingface.co/datasets/a3124371940/franka_pnp`（语言版把仓库名换成 `franka_pnp_language`），再把对应脚本里的 `root` 指向下载下来的目录即可。
> 注意：Hugging Face 上流传的 `Jeongeun/omy_pnp*` 系列是给旧版 6 自由度 OMY 机械臂采的，动作维度是 7（Franka 是 8），**不能直接使用**。采集新数据后可用 `push_dataset_il.py` / `push_dataset_language.py` 上传。
>
> **数据格式修复提示**：旧版采集器曾把“动作下发前的当前关节位置”误写成 `action`，并且状态里没有夹爪命令。修复后的 IL/VLA 状态分别为 7/8 维，旧数据和旧检查点仅供回放验证，不能和新数据混合训练。请覆盖重采并重新训练；`auto_collect.py` 会拒绝向旧结构数据集继续追加。

### 3. 动作空间与状态空间

环境封装支持多种动作/状态表示，由构造参数 `action_type` / `state_type` 控制：

- **`action_type`**：
  - `eef_pose`：动作是末端位姿的增量 `[dx,dy,dz,droll,dpitch,dyaw, gripper]`，内部用 IK 解算出关节角（**遥操作采集时使用**）。
  - `joint_angle`：动作直接是 7 个目标关节角 + 夹爪。
  - `delta_joint_angle`：动作是关节角增量 + 夹爪。
- **`state_type`**：`joint_angle`（关节角）、`ee_pose`（末端位姿）或 `delta_q`（关节角增量）。

> 采集时人通过键盘控制"末端往哪个方向移动一点"（`eef_pose` 增量），但**存进数据集 / 喂给策略学习的 `action` 是 8 维的“7 个目标关节角 + 夹爪命令”**——也就是当前观测之后由 IK 真正下发的关节空间指令，而不是动作执行前的当前关节状态。

> **夹爪语义**：对外（数据集、策略）统一用 `0=张开，1=闭合`；仿真内部 MuJoCo 的执行器控制量是 `255=张开，0=闭合`（tendon 驱动两指联动），由 `SimpleEnv.step()` 内部做换算，写代码时只需要关心对外的 0/1 语义。

---

## 四、底层代码模块详解（mujoco_env）

### 4.1 `mujoco_parser.py` —— 仿真封装地基

整个项目的"地基"，核心是 **`MuJoCoParserClass`** 类，另含两个辅助类 `MinimalCallbacks` 与 `MuJoCoMinimalViewer`（负责底层 GLFW 窗口、鼠标/键盘回调与渲染）。它把"用 MuJoCo 写仿真"所需的零碎操作封装成一套统一接口：

- **模型加载与信息**：`_parse_xml`（解析刚体/关节/几何体/执行器/相机/传感器并建立名称→索引映射）、`print_info`、`print_body_joint_info`。
- **前向仿真与状态**：`reset`、`step`（施加控制并调用 `mj_step` 推进物理）、`forward`（正运动学，只更新姿态不推进时间）、`get_state`/`set_state`/`store_state`/`restore_state`、`solve_inverse_dynamics`（逆动力学）。
- **位姿读写**：`get_/set_p/R/T_body`（刚体位置/旋转/齐次变换）、`base_body`、`mocap`、`joint`、`geom`、`site`、`sensor`、`cam` 等系列。
- **相机与渲染**：`init_viewer`/`set_viewer`、`grab_rgbd_img`、`get_egocentric_rgb(d_pcd)`（腕部相机）、`get_fixed_cam_rgb(d_pcd)`（固定相机，如 agentview/sideview/topview）、深度图转点云。
- **可视化绘制**：`plot_T`/`plot_sphere`/`plot_box`/`plot_cylinder`/`plot_arrow`/`plot_traj`/`plot_contact_info`，以及文本与 RGB 图叠加层（`viewer_text_overlay`/`viewer_rgb_overlay`，用于在窗口角落显示多视角画面和文字提示；同一 `loc` 多次调用会按行累加，不同 `loc` 才互不干扰）。
- **IK 基础量**：`get_J_body`/`get_J_geom`（雅可比矩阵）、`get_ik_ingredients`（位姿误差向量）、`damped_ls`（阻尼最小二乘求关节增量 `dq`）。
- **键盘/鼠标交互**：`check_key_pressed`、`is_key_pressed_once`（按一次触发一次）/`is_key_pressed_repeat`（按住持续触发）、左右键双击拾取 3D 坐标、`compensate_gravity`（重力补偿）等。

### 4.2 `ik.py` —— 逆运动学求解

实现"给定末端目标位姿，反解机械臂关节角"的功能，方法是**增广雅可比 + 阻尼最小二乘**迭代：

- `init_ik_info` / `add_ik_info`：构建并登记 IK 目标（目标刚体/几何体的目标位置 `p_trgt` 与目标旋转 `R_trgt`）。
- `get_dq_from_ik_info`：把所有目标的雅可比与位姿误差堆叠成增广方程，按指定关节列筛选后调用阻尼最小二乘求解关节增量 `dq`。
- `plot_ik_info`：在窗口中可视化当前位姿与目标位姿。
- `solve_ik`：**顶层入口**。迭代调用上述函数，每步对关节角做范围裁剪、正运动学更新，直到位姿误差收敛或达到最大迭代步数，返回求得的关节角。`SimpleEnv1.py` / `SimpleEnv2.py` 的 `reset` 和 `eef_pose` 模式下的 `step` 都依赖它。

### 4.3 `transforms.py` —— 坐标与旋转变换工具

纯 NumPy 实现的一组刚体变换函数：

- 齐次变换分解：`t2pr`/`t2p`/`t2r`（4×4 变换矩阵 → 位置 p / 旋转 R）。
- 欧拉角 ↔ 旋转矩阵：`rpy2r`/`rpy2r_order`/`r2rpy`（滚转/俯仰/偏航）。
- 旋转矩阵 ↔ 四元数：`r2quat`/`quat2r`。
- 位姿 → 齐次矩阵：`pr2t`；旋转矩阵 → 角速度向量：`r2w`。
- 几何工具：`skew`（反对称矩阵）、`rodrigues`（罗德里格斯公式，轴角 → 旋转矩阵）、两点构造旋转、`align_z_axis`（z 轴对齐）。
- 视觉相关：`meters2xyz`（深度图 → 点云）、`R_yuzf2zuxf`/`T_yuzf2zuxf`（坐标系约定转换）。

### 4.4 `utils.py` —— 通用工具集

- **随机采样与场景布置**：`sample_xyzs`/`sample_xys`（采样若干互不重叠、满足最小间距的位置）、`ObjectSpawner`（在仿真中随机生成托盘与物体并赋予不碰撞位置）。
- **索引检索**：`get_idxs`/`get_idxs_contain`/`get_idxs_closest_ndarray`/`get_consecutive_subarrays`（按相等、子串、最近邻、连续段查索引）。
- **轨迹与运动学**：`finite_difference_matrix`、`get_A_vel_acc_jerk`、`get_interp_const_vel_traj_nd`（匀速插值轨迹）、`check_vel_acc_jerk_nd`（检查速度/加速度/加加速度）。
- **几何变换**：`compute_view_params`（由相机位姿算观察方位）、`unit_vector`、`rotation_matrix`（绕任意轴的 4×4 旋转矩阵）。
- **图像与可视化**：`get_colors`、`load_image`/`save_png`、`imshows`（多图并排）、`depth_to_gray_img`、`add_title_to_img`（给图加标题，遥操作叠加层用到）。
- **XML 与其他**：`get_xml_string_from_path`、`prettify`（美化 MuJoCo XML）、`TicTocClass`（计时器）、`get_monitor_size`、`sleep`。

### 4.5 `SimpleEnv1.py` —— 单物体抓放环境 `SimpleEnv`

把底层 `MuJoCoParserClass` 进一步封装成"任务级"环境，接口风格接近 Gym：

| 方法 | 职责 |
|------|------|
| `__init__(xml_path, action_type, state_type, seed)` | 加载场景 XML、建查看器、reset |
| `init_viewer` | 初始化可视化窗口（距离、仰角、叠加层等） |
| `reset(seed)` | 用 IK 把机械臂移到初始位姿（home 关节角 `[0, -0.785, 0, -2.356, 0, 1.571, 0.785]`），随机布置物体，预步进 100 次让物体稳定落下 |
| `step(action)` | 按 `action_type` 解算关节指令（`eef_pose` 走 IK）、组合夹爪指令，返回新状态 |
| `step_env()` | 真正调用物理引擎推进一步 |
| `grab_image()` | 抓取 agentview / egocentric / sideview 三路相机图 |
| `render(teleop, idx, total)` | 绘制末端标记、把多视角图叠加到窗口；遥操作模式额外显示侧视图和按键状态；`idx`/`total` 用于显示采集进度 |
| `teleop_robot()` | 读取键盘，生成末端增量动作（见下方按键表） |
| `get_joint_state` / `get_ee_pose` / `get_delta_q` | 三种状态表示 |
| `check_success()` | 判定任务是否稳定完成（仅用于部署/评估，采集阶段不调用） |
| `get_obj_pose` / `set_obj_pose` | 读写杯子/盘子位姿 |
| `is_finish_pressed()` | 检测回车键——采集时用来手动确认"这一回合可以保存了" |

**遥操作按键表**（`teleop_robot` / 采集脚本主循环）：

| 按键 | 作用 |
|------|------|
| `W / S` | 沿 x 轴前 / 后移 |
| `A / D` | 沿 y 轴左 / 右移 |
| `R / F` | 沿 z 轴上 / 下移 |
| `Q / E` | 绕 z 轴倾斜（左/右） |
| `↑ / ↓` | 绕 x 轴俯仰 |
| `← / →` | 绕 y 轴偏转 |
| `空格` | 切换夹爪开/合 |
| `回车 Enter` | 确认本回合完成、保存并进入下一回合 |
| `Z` | 重置环境并丢弃当前回合 |

### 4.6 `SimpleEnv2.py` —— 语言条件多物体环境 `SimpleEnv2`

在 `SimpleEnv` 基础上扩展为**语言条件**任务，主要差异：

- 场景固定一个盘子，放置**红、蓝两个杯子**（`body_obj_mug_5` / `body_obj_mug_6`）。
- 通过 `set_instruction(given=None)` 每次 `reset` 时**随机**在红/蓝之间挑一个，拼出指令 `"Place the {color} mug on the plate."`，并据此设定目标杯子（`given` 传入具体字符串则可手动指定，供调试/补采某个颜色用）。
- `render` 时额外在画面**顶部**叠加当前语言指令文字（避免和左下角的 tick/按键调试信息挤在一起看不清），并显示回合（Episode）序号。

其余方法（`reset`/`step`/`teleop_robot`/`grab_image`/`check_success` 等）职责与 `SimpleEnv` 一致。它服务于 π0、SmolVLA 这类需要"听懂指令"的策略。

---

## 五、完整工作流程与运行步骤

### 运行前必读

1. **每次开新终端，先激活环境、再进项目根目录**：
   ```bash
   conda activate lerobot          # 命令行前应出现 (lerobot)
   cd /path/to/lerobot_mujoco      # 换成你的项目实际路径
   ```
2. **所有脚本都在项目根目录下用 `python 子文件夹/脚本名.py` 运行**，例如 `python il/train_il.py --config_path=config/il/act_franka.yaml`。
   每个脚本顶部都有"环境自举"代码，会自动把根目录加入搜索路径、并切到根目录，所以 `import mujoco_env`、`./asset`、`./demo_data`、`./ckpt` 等路径都能正确找到——你不用关心当前在哪个目录，只要保证从根目录调用即可。
3. **遥操作脚本（步骤 1、5）会弹出 MuJoCo 窗口进入键盘控制循环**，按回车保存当前回合，关闭窗口或采够数量即结束。
4. **回放脚本（步骤 2、6）支持 M/N/Q 键**：M 切到下一个回合、N 切到上一个回合（两端循环衔接）、Q 结束程序。

### 我该按什么顺序跑？（两条路线，任选其一先跑通）

> **路线 A · 模仿学习——ACT / Diffusion Policy，无需语言：**
> 步骤 1 采集 → 步骤 2 回放检查 → 步骤 3 训练 → 步骤 4 部署。
>
> ```bash
> python il/collect_data_il.py                                      # 步骤1：采集
> python auto_collect.py                                               # 可选：自动采集，参数在文件顶部改
> python il/visualize_data_il.py                                    # 步骤2：回放
> python il/train_il.py --config_path=config/il/act_franka.yaml        # 步骤3A：训练 ACT
> python il/train_il.py --config_path=config/il/diffusion_franka.yaml  # 步骤3B：训练 Diffusion
> python il/deploy_il.py --config_path=config/il/act_franka.yaml       # 步骤4A：部署 ACT
> python il/deploy_il.py --config_path=config/il/diffusion_franka.yaml # 步骤4B：部署 Diffusion
> ```
>
> **路线 B · 进阶——π0 / SmolVLA，语言条件 VLA：**
> 步骤 5 采集 → 步骤 6 回放 → `train_vla.py` 训练 → `deploy_pi0/​smolvla` 部署。
>
> ```bash
> python vla/collect_data_language.py                          # 步骤5：采集带语言指令的数据
> python auto_collect.py                                          # 可选：自动采集 VLA，运行后输入 2
> python vla/visualize_data_language.py                        # 步骤6：回放核对
> python vla/train_vla.py --config_path=config/vla/pi0_franka.yaml # 训练 π0（需 GPU）
> python vla/deploy_pi0.py                                     # 部署 π0
> ```

> **不想自己采集数据？** 可以直接下载现成的 Franka 数据集（[a3124371940/franka_pnp](https://huggingface.co/datasets/a3124371940/franka_pnp) / [franka_pnp_language](https://huggingface.co/datasets/a3124371940/franka_pnp_language)），跳过步骤 1/5，把 `root` 指向下载目录直接进行步骤 2 之后的流程，详见[第三章](#三核心概念与数据流)。
>
> **没有 GPU 怎么办？** 目前没有现成的 Franka 版预训练检查点可以下载（旧的 OMY 检查点动作维度不兼容，直接加载会出错），训练这一步必须有 GPU。仿真、遥操作、回放这几步用 CPU 也能跑，可以先在无 GPU 机器上把数据采好、用 `push_dataset_*.py` 传到 Hugging Face，再拿到有 GPU 的机器上训练。

下面按教程顺序逐个讲解每个步骤的命令与原理。

### 步骤 1：采集示教数据（`collect_data_il.py`）

```bash
python il/collect_data_il.py      # 需要显示器；弹出窗口后用键盘遥操作
```

用键盘遥操作机械臂完成"把杯子放到盘子上"，并存成 LeRobot 数据集。窗口四角叠加了多路相机画面：右上为智能体视角（Agent View）、右下为腕部第一人称视角（Egocentric View）、左上为左侧视角（Side View），中间还会显示当前按键和已保存回合数（`Saved N/M`）。

流程：

1. `SimpleEnv(xml_path, state_type='joint_angle')` 创建单物体环境。
2. `LeRobotDataset.create(...)` 定义数据集特征（`fps=20`，`robot_type="franka"`，特征字段见[第三章](#2-数据集格式lerobot-dataset)）。
3. 主循环以 20 Hz 运行：`teleop_robot()` 读键盘 → 每帧 `get_joint_state()` + `grab_image()` 取状态和图像（缩放到 256×256）→ `step(action)` 解算关节角 → `dataset.add_frame(...)` 记录这一帧。
4. **按回车手动确认**这一回合做得不错后 `save_episode()` 保存该回合；按 `Z` 则 `clear_episode_buffer()` 丢弃重来。质量把关全靠操作者自己判断，环境不会自动帮你判定成功与否。

数据默认存到 `./demo_data`。采集数量建议参考[第八章 FAQ](#八常见问题faq) 里给的数量建议。

**可选：自动采集 IL 数据（不用键盘遥操作）**

如果你只是想快速生成一批可训练数据，可以直接让脚本专家自动抓放：

```bash
python auto_collect.py
```

脚本会先让你选择无头模式或窗口模式，然后选择采集 IL / VLA 数据。日常不用在命令行输入一长串参数，直接改 [auto_collect.py](auto_collect.py) 顶部的“用户配置区”即可。最常改这些：

- `AUTO_MODE`：默认 `"ask"`，运行后输入 `1` 采 IL、输入 `2` 采 VLA；也可以固定成 `"il"` 或 `"vla"`。
- `AUTO_NUM_DEMOS`：要保存多少条成功 episode。
- `AUTO_DATASET_ROOT`：默认 `None`，IL 会保存到 `./demo_data`，VLA 会保存到 `./demo_data_language`。
- `AUTO_EXISTING_DATASET_ACTION`：目标数据集已存在时怎么处理。默认 `"ask"`，运行时会让你选：继续追加 / 删除重采 / 退出。
- `AUTO_FORCE`：命令行强制覆盖开关，默认 `False`，一般不用改。
- `AUTO_HEADLESS`：默认 `"ask"`，每次运行时选择无头/窗口模式；也可以固定成 `True`（无头）或 `False`（窗口）。命令行的 `--headless` / `--no-headless` 会跳过询问并临时覆盖它。
- `AUTO_GL_BACKEND`：默认 `"auto"`，自动选择 Windows/WGL、Linux/EGL、macOS/CGL；通常不用改。
- `AUTO_EGL_DEVICE_ID`：Linux 多 GPU 服务器只有 EGL 选错显卡时才填写，通常保持 `None`。
- `AUTO_VIDEO_DIR`：保存检查视频；设为 `None` 就不保存。
- `AUTO_IMAGE_WRITER_THREADS`：图片写入线程数。默认 `2` 更省内存；想更快可改 `4/8`，但长时间采集更容易吃内存。
- `AUTO_GRASP_X_OFFSET`：抓取点的世界坐标 x 偏移；当前场景保持 `0.0`。
- `AUTO_GRASP_Y_OFFSET`：抓取点的世界坐标 y 偏移；当前场景的无把手侧使用 `0.05`，已用 seed 0 实测成功。
- `AUTO_GRASP_Z_OFFSET`：下探高度；还高就试 `0.0`，太低碰撞就试 `0.02`。
- `AUTO_PLACE_Z_OFFSET`：放杯时的高度。

如果采到一半因为内存不足中断，例如已经显示 `✓ saved 116/200`，重新运行：

```bash
python auto_collect.py
```

选择同一个模式后，当它提示数据集目录已存在，输入 `1` 选择“继续采集”，脚本会读取已有 episode 数，不覆盖旧数据，并继续补到 `AUTO_NUM_DEMOS`。

> 如果目录是旧版 6/7 维末端位姿状态数据，脚本会明确报错并要求选择 `2` 删除重采。新版 IL/VLA 都使用 8 维“7 个实际关节角 + 夹爪”；旧数据无法可靠恢复逐帧实际关节角，不能简单追加或转换。

### 步骤 2：回放与可视化数据（`visualize_data_il.py`）

```bash
python il/visualize_data_il.py    # 需要显示器；默认读取步骤1采集的 ./demo_data
```

读取已采集的数据集，在重建的仿真场景里**回放动作**并核对：

1. `LeRobotDataset` 加载数据集；自定义 `EpisodeSampler` 选取单条回合构建 DataLoader（`num_workers=0`，避免 Windows 下多进程 DataLoader 重新执行整个脚本报错）。
2. `SimpleEnv` 按数据集里的物体初始位姿重置场景。
3. 逐帧取出 `action` 驱动机器人回放，同时把数据集中记录的相机图像作为叠加图显示在窗口角落，直观比对"记录的画面"与"重放的仿真"。
4. 窗口内按 **M** 切下一个回合、**N** 切上一个回合（两端循环衔接）、**Q** 结束程序。

### 步骤 3：选择并训练 IL 策略（`train_il.py`）

```bash
python il/train_il.py --config_path=config/il/act_franka.yaml
python il/train_il.py --config_path=config/il/diffusion_franka.yaml
```

两种策略共用 LeRobot 通用训练器：YAML 解析 → 数据集时序采样 → 创建策略/优化器 → 训练 → 定期保存检查点。

训练入口会先检查 `demo_data`：`observation.state/action` 必须都是 8 维，Parquet 数量必须与 `meta/info.json` 一致，并逐个核对真实 Arrow 维度。这样可以在训练前直接拦截旧 6/7 维数据或 Hugging Face 重复上传留下的额外 episode。

- ACT：`chunk_size=50`、`n_action_steps=1`、`temporal_ensemble_coeff=0.01`，每帧闭环并对重叠动作块做时间集成。
- Diffusion：`n_obs_steps=2`、`horizon=16`、`n_action_steps=8`，用条件 1D U-Net 逐步去噪生成动作序列。
- 两者都训练 100,000 步（每 10,000 步保存一个检查点），目录名：`./ckpt/act_franka` 和 `./ckpt/diffusion_franka`。不要从旧 6/7 维检查点续训。训练量说明：约 5 万帧数据下，20k 步 × batch 16 只相当于约 5~6 个 epoch，策略往往还远未拟合；loss 曲线走平不代表闭环成功率走平，建议用多个中间检查点分别跑 `deploy_il.py` 比较成功率。

### 步骤 4：部署 IL 策略（`deploy_il.py`）

```bash
python il/deploy_il.py --config_path=config/il/act_franka.yaml
python il/deploy_il.py --config_path=config/il/diffusion_franka.yaml
```

`deploy_il.py` 从同一份 YAML 获取策略类型和 `output_dir`，自动定位最新检查点，再从检查点的 `input_features` 生成它需要的相机/状态输入。20 Hz 闭环中调用 `policy.select_action()`，并以 `action_type="joint_angle"` 将模型输出解释为“7 个绝对关节角 + 夹爪”；同一脚本即可部署 ACT 或 Diffusion。

无桌面 Linux 服务器可直接离屏运行并保存 MP4（MP3 只能保存音频，不能保存画面）：

```bash
CUDA_VISIBLE_DEVICES=7 python il/deploy_il.py --config_path=config/il/act_franka.yaml --checkpoint=./ckpt/act_franka/checkpoints/100000/pretrained_model --device=cuda --seed=0 --max_steps=2000 --headless
```

无头模式不会创建 GLFW 窗口，默认以 `400×300` 分辨率分别渲染主相机和腕部相机，再横向合成为视频。`--max_steps` 在无头模式下必须大于 0；视频按 `--control_hz`（默认 20 FPS）写入。默认输出会按模型分文件夹保存，例如 ACT 是 `output/act/act_seed0.mp4`，Diffusion 是 `output/diffusion/diffusion_seed0.mp4`，汇总文件也在对应文件夹中，例如 `output/act/act_summary.json`。无头模式下首轮结束后（无论成败）会继续评估 `--random_seeds`（默认 10）个随机种子并统计成功率；添加 `--random_seeds=0` 可只跑首轮。需要降低渲染开销时可添加 `--render_width=320 --render_height=240`。如果想指定其它路径，仍可添加 `--video=自定义路径.mp4`。

> **训练效果排查（`eval_offline_il.py`）**：闭环成功率低时，先做离线开环评估——把数据集里已录制的观测逐帧喂给检查点，对比预测动作与示教动作，不依赖 MuJoCo 渲染：
>
> ```bash
> python il/eval_offline_il.py --config_path=config/il/act_franka.yaml
> python il/eval_offline_il.py --config_path=config/il/diffusion_franka.yaml --num_episodes=5
> ```
>
> 输出各回合的关节 MAE（弧度）、夹爪命令准确率、首次闭合/张开时机相对示教的偏移帧数。判读：示教数据相邻帧关节增量约 0.03 rad——离线 MAE 明显大于该量级 → 拟合不足，优先加训练步数/检查数据；离线误差小但闭环失败 → 多为闭环漂移或夹爪时机问题，优先加数据覆盖，或按 `act_franka.yaml` 注释切换为分块开环执行对比。

### 步骤 5：采集语言条件数据（`collect_data_language.py`）

```bash
python vla/collect_data_language.py     # 需要显示器；数据默认存到 ./demo_data_language
```

与步骤 1 类似，但环境换成 `SimpleEnv2`（`example_scene_y2.xml`，红/蓝两个杯子 + 盘子）。**关键区别**：

- 每回合 `reset` 时环境会**随机**挑一个颜色，拼出指令（如 `"Place the red mug on the plate."`），显示在窗口**顶部**。
- 每帧通过 `dataset.add_frame(..., task=PnPEnv.instruction)` 写入这句自然语言指令，使数据集支持训练能听懂指令的语言条件策略。

按键操作、按回车手动确认保存的流程与步骤 1 完全相同。

**可选：自动采集 VLA 语言数据**

直接运行脚本，启动后输入 `2` 选择 VLA：

```bash
python auto_collect.py
```

默认每个 episode 会随机选择红杯或蓝杯，并把对应英文指令写进 `task` 字段。想固定只采红杯或蓝杯，就把 `AUTO_INSTRUCTION` 改成 `"Place the red mug on the plate."` 或 `"Place the blue mug on the plate."`。

### 步骤 6：可视化语言条件数据（`visualize_data_language.py`）

```bash
python vla/visualize_data_language.py   # 需要显示器；默认读取 ./demo_data_language
```

与步骤 2 类似，但针对语言条件数据集，用 `SimpleEnv2` 逐帧回放，叠加显示智能体视角与腕部相机图，并在顶部还原当时的语言指令，核对采集质量。同样支持 **M/N/Q** 键切换回合、结束程序。

### 步骤 7：训练并部署 π0（`train_vla.py` + `deploy_vla.py`）

**训练（GPU 机）：**

```bash
python vla/train_vla.py --config_path=config/vla/pi0_franka.yaml   # 训练 π0，检查点存到 ./ckpt/pi0_franka_v2
```

> 首次运行会自动从 Hugging Face 下载 π0 预训练权重 `lerobot/pi0`（需联网，体积较大）。训练前请确认配置文件里 `dataset.root` 指向的数据集已存在（自己采集的 `./demo_data_language`）。

**有窗口部署（统一入口）：**

```bash
python vla/deploy_vla.py --config_path=config/vla/pi0_franka.yaml
```

加载微调好的 π0 策略，在 `SimpleEnv2` 语言条件环境中执行。与 ACT 部署的**关键区别**：观测字典里必须额外提供语言指令（`task` 字段，取自 `PnPEnv.instruction`），π0 据此决定要操作哪个杯子。

### 步骤 8：训练并部署 SmolVLA（`train_vla.py` + `deploy_vla.py`）

**训练（GPU 机）：**

```bash
python vla/train_vla.py --config_path=config/vla/smolvla_franka.yaml   # 训练 SmolVLA，检查点存到 ./ckpt/smolvla_franka_v2
```

> 首次运行会自动下载 SmolVLA 基座权重 `lerobot/smolvla_base`（需联网）。

**有窗口部署（统一入口）：**

```bash
python vla/deploy_vla.py --config_path=config/vla/smolvla_franka.yaml
```

与步骤 7 流程一致，换成更轻量的 SmolVLA 策略。

**VLA 无头部署与自动多种子评估：**

```bash
# π0
CUDA_VISIBLE_DEVICES=7 python vla/deploy_vla.py --config_path=config/vla/pi0_franka.yaml --checkpoint=./ckpt/pi0_franka_v2/checkpoints/040000/pretrained_model --device=cuda --seed=0 --max_steps=2000 --headless

# SmolVLA
CUDA_VISIBLE_DEVICES=7 python vla/deploy_vla.py --config_path=config/vla/smolvla_franka.yaml --checkpoint=./ckpt/smolvla_franka_v2/checkpoints/030000/pretrained_model --device=cuda --seed=0 --max_steps=2000 --headless
```

首轮结束后（无论成败）会自动再跑 10 个随机种子统计成功率。每段视频顶部包含当前语言指令，视频默认保存到 `output/pi0/` 或 `output/smolvla/`，最终生成同目录下的 `pi0_summary.json` / `smolvla_summary.json`，记录每个 seed、指令、步数、视频路径和总成功率。若只想固定测试红杯，可添加 `--instruction="Place the red mug on the plate."`；蓝杯同理。

### `train_vla.py` —— 通用训练入口

π0 和 SmolVLA 共用这一脚本：

- 通过 `@parser.wrap()` 装饰器，把命令行 `--config_path=xxx.yaml` 的内容解析进 `TrainPipelineConfig` 类型的 `cfg`。
- `train(cfg)`：校验配置 → 初始化 wandb / 随机种子 / 设备 → 创建数据集、（可选）评估环境、策略、优化器与调度器 → 构建 DataLoader → 进入**离线训练主循环**：取 batch → 搬到 GPU → `update_policy` 前向+反向+优化 → 按频率记录日志、保存检查点、（可选）评估。`cfg.policy.type` 为 `"pi0"` / `"smolvla"` 时会分别自动把 `pretrained_path` 设成 `lerobot/pi0` / `lerobot/smolvla_base` 这两个官方基座权重。
- `update_policy(...)`：单步更新。可选混合精度（AMP）下前向算 loss → 反向传播 → **先反缩放梯度再做梯度裁剪** → `grad_scaler` 执行优化器步进并更新缩放因子 → 清零梯度 → 推进学习率调度器 → 对支持的策略更新内部缓冲（如 EMA）→ 返回训练指标。

检查点默认存到 `output_dir`（如 `./ckpt/pi0_franka_v2`）。

### 数据集/策略上传到 Hugging Face（`push_dataset_il.py` / `push_dataset_language.py`）

采集好数据、或训练出满意的策略后，可以上传到自己的 Hugging Face 账号，方便备份或在别的机器上复用：

```bash
# 先登录一次（需要有写权限的 access token）
hf auth login          # 老版本 huggingface_hub 用 huggingface-cli login

python il/push_dataset_il.py             # 上传 ./demo_data
python vla/push_dataset_language.py      # 上传 ./demo_data_language
```

两个脚本都在文件顶部留了 `REPO_ID`（目标仓库名，格式 `用户名/数据集名`）和 `PRIVATE`（是否设为私有仓库）两个变量，改成自己的即可。策略（训练好的模型）上传则见 `deploy_pi0.py` / `deploy_smolvla.py` 结尾处被注释掉的 `policy.push_to_hub(...)` 示例。

### 上传全部训练检查点（`push_ckpt.py`）

```bash
hf auth login
python push_ckpt.py
```

脚本默认把整个 `./ckpt` 增量上传到当前 Hugging Face 账号下的私有模型仓库 `franka_ckpt`，远端保留 `ckpt/act_franka/...`、`ckpt/diffusion_franka/...` 等完整目录结构。再次运行时只同步新增或内容发生变化的文件；缓存目录以及指向数字检查点的 `checkpoints/last` 重复链接不会上传。

自定义仓库名或改为公开仓库时使用：

```bash
python push_ckpt.py --repo_id=你的用户名/仓库名 --public
```

### 从 Hugging Face 下载全部检查点

仓库为私有时，先在目标电脑登录同一个 Hugging Face 账号，然后下载：

```bash
hf auth login
hf download a3124371940/franka_ckpt --local-dir ./hf_ckpt
```

下载后会保留上传时的完整目录结构，例如 ACT 模型位于 `./hf_ckpt/ckpt/act_franka/checkpoints/100000/pretrained_model`。如果检查点编号不同，可以查找实际路径：

```bash
# Linux
find ./hf_ckpt -type d -name pretrained_model

# Windows PowerShell
Get-ChildItem -Path ./hf_ckpt -Recurse -Directory -Filter pretrained_model
```

使用 CPU 部署下载的 ACT 模型：

```bash
python il/deploy_il.py --config_path=config/il/act_franka.yaml --checkpoint=./hf_ckpt/ckpt/act_franka/checkpoints/100000/pretrained_model --device=cpu --seed=0 --max_steps=2000
```

---

## 六、模型原理详解

### 6.1 逆运动学：增广雅可比 + 阻尼最小二乘

机械臂控制的基本问题：已知想让末端到达的位姿，求关节角。本项目用**迭代式数值 IK**：

1. **正运动学** $x = f(q)$ 给出当前末端位姿；**雅可比** $J = \partial f / \partial q$ 描述"关节微动 → 末端微动"的线性关系：$\dot{x} = J\dot{q}$。
2. 设当前位姿与目标的误差为 $e$（位置误差 + 由 `r2w` 得到的姿态误差），希望 $J\,\Delta q = e$。
3. 直接求逆在奇异位形附近会数值爆炸，故用**阻尼最小二乘（Damped Least Squares，又称 Levenberg-Marquardt）**：
   $$\Delta q = J^\top (JJ^\top + \lambda^2 I)^{-1} e$$
   阻尼项 $\lambda^2 I$ 牺牲一点精度换取在奇异点附近的数值稳定。代码见 `mujoco_parser.py::damped_ls`。
4. **增广**：当有多个目标（多个刚体/几何体），把各自的 $J$ 和 $e$ 纵向堆叠成一个大方程一起解，即 `ik.py::get_dq_from_ik_info`。
5. `solve_ik` 反复迭代步骤 2-4，每步裁剪到关节限位并更新正运动学，直到误差收敛。

> 本项目里 IK 有两处用途：`reset` 时把机械臂摆到固定初始位姿（末端竖直朝下）；遥操作 `eef_pose` 模式下把"末端位姿增量"实时转成关节角。

### 6.2 ACT：动作分块 Transformer

**ACT（Action Chunking Transformer）** 是模仿学习中应对"误差累积"和"演示数据多模态/非马尔可夫"的经典方法（出自 ALOHA 论文）。核心思想：

- **动作分块（Action Chunking）**：策略不逐帧预测单个动作，而是**一次预测未来一段动作序列**（长度 `chunk_size`，本项目为 10）。执行时一次推理可走多步，减少高频闭环带来的复合误差，也更好地建模"成套动作"。
- **CVAE 结构**：训练时用一个 VAE 编码器把"专家动作序列"压成隐变量 $z$，缓解人类演示中的随机性/多模态；推理时 $z$ 取先验均值（0），由 Transformer 编码器-解码器结合 ResNet 视觉特征、关节状态生成动作块。
- **输入**：相机图像（经 ResNet 主干）+ 机器人状态；**输出**：未来 `chunk_size` 步动作。

本项目把 ACT 当作最直接的入门策略：不需要语言输入，根据数据集自动使用相机与末端状态。

### 6.3 Diffusion Policy：逐步去噪生成动作

**Diffusion Policy** 把未来动作序列看成一个从噪声中恢复的信号。训练时对专家动作加噪并学习去噪；推理时用相机图像和机器人状态作为条件，通过多次反向去噪得到平滑动作块。它对多模态示教很有优势，但推理通常比 ACT 慢。

### 6.4 π0（pi-zero）：流匹配 VLA

**π0** 是一个**视觉-语言-动作（Vision-Language-Action, VLA）**大模型：

- 以视觉语言模型 **PaliGemma** 为骨干理解图像与文字，外接一个 **Gemma 动作专家** 头输出动作。
- 用 **流匹配（Flow Matching）** 这种连续生成方法产生平滑的高频动作序列（相比离散自回归更适合连续控制）。
- 同时吃三类输入：**图像 + 机器人状态 + 自然语言指令**，因此能完成"把*红色*杯子放到盘子上"这类需要语言区分目标的任务。

本项目通过 `lerobot/pi0` 预训练权重做微调（`train_vla.py` 在 `cfg.policy.type=="pi0"` 时自动设定 `pretrained_path='lerobot/pi0'`），再在 Franka 语言条件数据集上学习具体技能。

### 6.5 SmolVLA：轻量 VLA

**SmolVLA** 是 Hugging Face 推出的**轻量级 VLA**，设计目标是用较小的参数量和算力即可训练/部署，同样接收图像 + 状态 + 语言指令并输出动作。基座为 `lerobot/smolvla_base`。它与 π0 的部署流程几乎一致，区别主要在模型规模与资源占用——适合显存/算力有限的场景。

---

## 七、配置文件说明

`config/il/act_franka.yaml` / `config/il/diffusion_franka.yaml` 供 `train_il.py` 和 `deploy_il.py` 共用；`config/vla/pi0_franka.yaml` / `config/vla/smolvla_franka.yaml` 供 `train_vla.py` 读取。训练时都通过 `--config_path` 指定，公共字段含义：

统一使用 8 维关节状态后的推荐配置如下：

| 模型 | 预测长度 | 每次执行 | 每卡 batch | 训练步数 | 检查点目录 |
|---|---:|---:|---:|---:|---|
| ACT | 50 | 1 | 16 | 100,000 | `ckpt/act_franka` |
| Diffusion | 16 | 8 | 64 | 100,000 | `ckpt/diffusion_franka` |
| π0 | 50 | 50 | 8 | 40,000 | `ckpt/pi0_franka_v2` |
| SmolVLA | 50 | 10 | 16 | 30,000 | `ckpt/smolvla_franka_v2` |

> `batch_size` 是每张 GPU 的 batch；例如 π0 使用 3 张卡时全局 batch 为 24。当前锁定的 LeRobot π0 实现要求 `n_action_steps == chunk_size`。ACT 每帧重规划并做时间集成，Diffusion 每 8 帧重规划；IL 每 10,000 步、VLA 每 5,000 步保存一次，建议用同一组随机种子比较中间检查点的成功率。Diffusion 的视觉编码器按官方结构从零训练（GroupNorm），当前锁定的 LeRobot 版本不允许它与 ImageNet 预训练权重同时开启。

```yaml
dataset:
  repo_id: franka_pnp               # 数据集仓库 ID
  root: ./demo_data                 # 本地数据集根目录
policy:
  type: act                         # act / diffusion / pi0 / smolvla
  device: cuda
  chunk_size: 50                    # 预测 50 步动作；Diffusion 改用 horizon / n_obs_steps
  n_action_steps: 1
  temporal_ensemble_coeff: 0.01
output_dir: ./ckpt/act_franka       # 保持原始检查点目录名
batch_size: 16
job_name: act_franka
resume: false                       # 是否从已有检查点续训
seed: 42
num_workers: 4                      # Linux 训练用 4；Windows 下需改回 0
steps: 100_000                      # ACT/DP 均按 100k 训练
eval_freq: -1                       # -1 表示训练中不评估
log_freq: 50                        # 每 50 步记录一次日志
save_freq: 10_000                   # 每 10000 步存一次，方便按成功率选模型
use_policy_training_preset: true    # 使用策略自带的训练预设（优化器/调度器等）
wandb:
  enable: true
  project: act_franka
  entity: null                      # 使用当前临时登录的 W&B 账号
  disable_artifact: true
```

> 切换 IL 策略时直接换 YAML。Diffusion 的 `horizon` 必须能被 U-Net 下采样倍数整除；默认配置为 16。部署脚本会自动从 YAML 对应的 `output_dir` 找最新检查点。

---

## 八、常见问题（FAQ）

**Q1：训练时报 `PicklingError: Can't pickle <function <lambda> ...>`？**
A：DataLoader 多进程无法序列化 lambda。把 `num_workers` 设为 `0`（YAML 里改 `num_workers: 0`，或脚本里 DataLoader 的 `num_workers=0`）。

**Q2：MuJoCo 窗口打不开 / 远程服务器报 GLFW 错误？**
A：遥操作与带窗口脚本需要图形界面；自动采集和部署请加 `--headless`。脚本会先执行相机预检并打印实际 GL 后端。Linux/NVIDIA 默认 EGL；若报 `EGL_NOT_INITIALIZED`，先检查 NVIDIA 驱动/EGL，CPU 服务器可执行 `sudo apt install libosmesa6` 后增加 `--gl_backend=osmesa`。Windows 不应强制 EGL，保持默认 `--gl_backend=auto`，脚本会使用 WGL。

**Q3：Windows 上运行 `visualize_*.py` 报 `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`？**
A：这是 DataLoader `num_workers>0` 在 Windows 下用 `spawn` 方式起子进程、子进程重新执行整个脚本导致的。本仓库的可视化脚本已经把 `num_workers` 设为 `0`，如果你自己改动过这个参数改回 `0` 即可。

**Q4：必须自己手动采集数据吗？还能下载现成数据集/预训练权重吗？**
A：数据集可从 [a3124371940/franka_pnp](https://huggingface.co/datasets/a3124371940/franka_pnp)（IL）和 [a3124371940/franka_pnp_language](https://huggingface.co/datasets/a3124371940/franka_pnp_language)（语言条件）下载。预训练权重目前没有现成的，仍需 GPU 训练。自己采集的数据可用 `push_dataset_il.py` / `push_dataset_language.py` 上传。

**Q5：采集多少条演示数据比较合适？**
A：ACT / Diffusion（单一固定任务）建议至少 50~100 条；Diffusion 对数据覆盖度更敏感，条件允许可采 100~200 条。语言条件数据集建议红/蓝两种指令各 30~50 条。先用 10~20 条跑通全流程，再扩大数据量。

**Q6：MuJoCo 版本有要求吗？**
A：必须 **3.1.6**，其他版本可能与本项目的解析器/资源不兼容。

**Q7：ACT、Diffusion、π0、SmolVLA 该选哪个入门？**
A：先用 **ACT** 跑通全流程；然后在同一 `demo_data` 上换 `diffusion_franka.yaml` 对比 Diffusion。需要语言指令再上 π0 / SmolVLA。

**Q8：能换成自己的机器人/物体吗？**
A：可以。准备好对应的 MJCF 模型放入 `asset/`，仿照 `example_scene_y*.xml` 拼场景，并相应调整 `SimpleEnv1.py` / `SimpleEnv2.py` 里的刚体名（如 `tcp_link`、`body_obj_*`、`finger_joint1`）、关节名（`joint1`~`joint7`）和成功判定逻辑里的数值阈值。

**Q9：`pip install -r requirements.txt` 卡在安装 LeRobot 或 GitHub 连不上？**
A：最后一行依赖需要从 GitHub 克隆 LeRobot。请确认网络能访问 GitHub；国内网络可考虑配置代理，或先手动 `git clone` LeRobot 仓库、`git checkout` 到 `requirements.txt` 里指定的 commit，再用 `pip install -e .` 本地安装，其余依赖照常 `pip install -r requirements.txt`。PyTorch 那几行从 `download.pytorch.org` 下载，慢的话可换用 CUDA 12.4 对应的国内镜像。

**Q10：从 Hugging Face 上传/下载数据集很慢或失败？**
A：`push_dataset_*.py` 内部用的是 `huggingface_hub`，运行前先 `hf auth login`（或旧版 `huggingface-cli login`）。下载慢可设置镜像端点，例如运行脚本前执行 `export HF_ENDPOINT=https://hf-mirror.com`（Windows PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`）。

**Q11：提示找不到 `plate_11` / 加载场景报缺文件？**
A：忘记解压物体资源了。回到[环境安装步骤 4](#4-解压物体资源必做)解压 `asset/objaverse/plate_11.zip`。

**Q12：夹爪把杯子挤飞了 / 抓不稳总是滑脱，怎么调？**
A：这是物理参数问题，不是代码 bug。可以在 `asset/franka_panda/panda.xml` 里调整夹爪执行器（`actuator8`）的刚度 `kp`（越大夹得越死、也越容易把圆柱形物体挤飞，越小越"软"）和 `forcerange`（最大夹持力），或者调整杯子与手指之间的摩擦系数（`friction`）——摩擦太小会打滑脱手。没有固定的"标准答案"，需要跑几次采集/部署观察实际手感再微调。

---

## 九、致谢与参考

- 本项目的采集/训练/部署流程基于 [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)，在此基础上把机械臂由 OMY 迁移到 Franka Panda。
- Franka Panda 机械臂模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 的 `franka_emika_panda`。
- 训练/数据流程参考 [LeRobot 官方示例](https://github.com/huggingface/lerobot/tree/main/examples)。
- 杯子/盘子网格来自 [Objaverse](https://objaverse.allenai.org/)。

---

> 本文档随代码注释一同维护：`mujoco_env/`、`il/`、`vla/` 下的教程脚本可结合源码对照阅读。
