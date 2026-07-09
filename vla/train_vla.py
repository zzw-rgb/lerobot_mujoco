#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
本脚本（train_vla.py，原名 train_model.py）是 LeRobot 的通用训练入口，
用于训练两种 VLA（Vision-Language-Action，视觉-语言-动作）策略：

    - pi0     （π0）
    - smolvla （SmolVLA）

所谓 VLA 策略，简单说就是一个神经网络：它同时"看"摄像头图像、"读"任务的
语言指令、"感知"机器人当前状态，然后输出机器人接下来要执行的动作。本脚本做的
事情就是：用一批已经录好的示教数据（dataset），通过反复"看数据-预测-纠错"的
方式，把这个网络训练好。

运行命令（在配置文件里写好所有超参数，然后用 --config_path 指定它）：

    # π0 单卡
    python vla/train_vla.py --config_path=config/vla/pi0_franka.yaml

    # π0 多卡（例：3 张卡）
    CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --num_processes=3 --main_process_port=29501 vla/train_vla.py --config_path=config/vla/pi0_franka.yaml

    # SmolVLA 单卡
    python vla/train_vla.py --config_path=config/vla/smolvla_franka.yaml

    # SmolVLA 多卡（例：3 张卡）
    CUDA_VISIBLE_DEVICES=5,6,7 accelerate launch --num_processes=3 --main_process_port=29502 vla/train_vla.py --config_path=config/vla/smolvla_franka.yaml

整体流程概览：
    1. 读取 yaml 配置 → 构造 cfg（由 @parser.wrap() 自动完成，见 train 函数说明）
    2. 创建数据集 dataset 与数据加载器 dataloader
    3. 创建策略 policy（pi0 / smolvla 会自动加载各自的官方预训练权重）
    4. 创建优化器 optimizer 与学习率调度器 lr_scheduler
    5. 进入离线训练主循环：取一批数据 → 前向算损失 → 反向求梯度 →
       更新参数 → 记录日志 → 定期存检查点 / 评估
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

import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


# =============================================================================
# update_policy：对策略执行"一次"参数更新（即训练主循环里的一个 step）
# -----------------------------------------------------------------------------
# 职责：拿到一批已经搬到 GPU 上的数据 batch，完成"前向 → 反向 → 优化"这一整套
#       标准训练步骤，并把本次更新产生的指标（loss、梯度范数、学习率、耗时）记录
#       到 train_metrics 里返回。
#
# 完整流程：
#   1. policy.train()                 切到训练模式（启用 dropout、BN 统计更新等）
#   2. accelerator.autocast()+forward 前向传播，得到本批数据的损失 loss
#   3. accelerator.backward(loss)     反向传播求梯度；多卡下在此处自动对各卡梯度做同步
#   4. accelerator.clip_grad_norm_    梯度裁剪，防止梯度爆炸导致训练发散
#   5. optimizer.step()               按梯度更新网络参数（真正"学习"的一步）
#   6. lr_scheduler.step()            推进学习率调度（让学习率随训练逐步变化）
#
# 混合精度（AMP）与梯度缩放由 accelerator 统一接管：以 mixed_precision="bf16" 创建
# Accelerator 时 autocast 与梯度反缩放/裁剪会自动生效，为 "no" 时等价于全精度训练，
# 因此这里不再手动维护 GradScaler。
# =============================================================================
def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()  # 切换到训练模式
    # accelerator.autocast()：按创建 Accelerator 时设定的 mixed_precision 决定是否
    # 启用半精度（bfloat16）前向；设为 "no" 时它是“什么都不做”的上下文，等价于全精度。
    with accelerator.autocast():
        # 前向传播：策略内部会读取 batch 里的图像/状态/语言/目标动作，算出损失 loss。
        # loss 越小，代表策略预测的动作越接近示教数据里的真实动作。
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # 反向传播。accelerator.backward 会在混合精度下自动完成梯度缩放；在多卡（DDP）下，
    # 它还负责把各进程的梯度做 all-reduce 同步，使每张卡上的参数保持一致。
    accelerator.backward(loss)

    # 梯度裁剪：把所有参数梯度拼成一个整体向量，若其范数超过 grad_clip_norm，就按比例
    # 缩小到该阈值，防止偶发的超大梯度把参数“一步带飞”（梯度爆炸）。accelerator 会在
    # 裁剪前自动反缩放梯度（若启用了混合精度），返回值是裁剪前的梯度范数，可作监控。
    grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

    optimizer.step()       # 真正按梯度更新一次网络参数
    optimizer.zero_grad()  # 清空梯度，否则梯度会在下一次 backward 时累加

    # 在每个 batch（而非每个 epoch）上推进 PyTorch 的学习率调度器，
    # 使学习率按预设策略（如 warmup + 衰减）随训练步数平滑变化。
    if lr_scheduler is not None:
        lr_scheduler.step()

    # 用于可能需要更新内部缓冲区的情况（例如 TDMPC 中的指数滑动平均 EMA）。多卡下
    # policy 被 DDP 包装，需先 unwrap 取回原始策略再调用其自定义方法。
    unwrapped_policy = accelerator.unwrap_model(policy)
    if has_method(unwrapped_policy, "update"):
        unwrapped_policy.update()

    # 记录本次更新的训练指标，供主循环打印日志、上传 wandb 使用。
    # .item() 把 GPU 上的标量张量转成普通 Python 数值。
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item() if grad_norm is not None else 0.0
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


# =============================================================================
# train：整个训练流程的总指挥（程序入口最终调用的就是它）
# -----------------------------------------------------------------------------
# 职责：把"配置 → 数据 → 模型 → 优化器 → 训练循环 → 评估/保存"这一整条流水线
#       串起来，是本脚本真正干活的主函数。
#
# 关于 @parser.wrap()：
#   这是 LeRobot 提供的装饰器。它会拦截命令行参数（比如我们传的
#   --config_path=config/vla/pi0_franka.yaml），把 yaml 配置文件解析、校验后，自动构造成一个
#   TrainPipelineConfig 对象，并以 cfg 这个参数注入进来。也就是说，我们在命令行
#   写的 --config_path，最终就变成了函数里能直接用的 cfg.xxx（如 cfg.steps、
#   cfg.batch_size、cfg.policy 等）。所以调用时只写 train()，cfg 由装饰器填好。
#
# 大致步骤：
#   配置校验与日志 → 设随机种子 → 选设备 → 建数据集/(可选)评估环境 →
#   建策略 → 建优化器与调度器 → (可选)从检查点恢复 → 建 dataloader →
#   进入离线训练主循环。
# =============================================================================
@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()  # 校验配置合法性（参数齐全、取值合理等）

    # === 多卡/分布式训练初始化（Accelerate）===
    # Accelerator 统一管理进程组、设备分配、混合精度与梯度同步。用普通
    # `python vla/train_vla.py ...` 启动时它自动退化为单卡；用
    # `accelerate launch --num_processes=N vla/train_vla.py ...` 启动时，会拉起 N 个
    # 进程做分布式数据并行（DDP）：每个进程各绑定一张 GPU、各持一份完整模型副本，
    # 反向传播时自动对各卡梯度做 all-reduce 同步，使参数始终保持一致。
    #   - mixed_precision：由 cfg.policy.use_amp 决定，开启即用 bfloat16 混合精度。
    #   - find_unused_parameters=True：允许前向中存在未参与计算的参数（pi0 的部分
    #     子模块在某些配置下不参与反向），避免 DDP 因“有参数没收到梯度”而报错；
    #     若确认没有未用参数，可改为 False 略微提速。
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.policy.use_amp else "no",
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    # DDP：把本进程分到的设备写回配置，让 make_policy 按各自的 local_rank 把模型
    # 加载到对应 GPU（rank0→cuda:0、rank1→cuda:1…）；否则所有进程都会按
    # cfg.policy.device（默认 cuda:0）把模型挤到 0 号卡，导致 0 号卡 OOM。
    cfg.policy.device = str(accelerator.device)

    # 仅主进程打印完整配置，避免多卡下每个进程各刷一遍日志。
    if accelerator.is_main_process:
        logging.info(pformat(cfg.to_dict()))  # 打印最终生效的完整配置，便于复现

    # wandb（Weights & Biases）是一个在线实验跟踪平台，可把训练过程中的 loss、
    # 学习率、评估成功率、视频等实时上传到网页看板，方便可视化和对比实验。
    # 只在主进程创建与上传，否则多个进程会向同一次实验记录重复写入。
    if accelerator.is_main_process and cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if accelerator.is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # 设置随机种子，让数据打乱、参数初始化等随机过程可复现（同样的种子 → 同样的结果）。
    if cfg.seed is not None:
        set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 创建数据集 dataset：封装了所有录制好的示教轨迹（每一帧含图像、机器人状态、
    # 语言指令、对应的真实动作等），是训练的"教材"。
    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # 创建用于在仿真数据训练过程中评估检查点的环境。
    # 对于真实世界数据，则无需创建环境，因为评估是在 train.py 之外完成的，
    # 改用 eval.py，并配合 gym_dora 环境和 dora-rs。
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # 创建策略 policy（也就是要训练的那个 VLA 神经网络）。
    # 这里根据策略类型自动设定 pretrained_path（预训练权重路径）：pi0 和 smolvla
    # 都不是从零随机初始化训练，而是先从 HuggingFace Hub 下载官方发布的预训练基座
    # （lerobot/pi0、lerobot/smolvla_base），再在我们自己的数据上微调（fine-tune）。
    # 这样能借助大规模预训练得到的通用能力，用很少的数据就训出可用的策略。
    logging.info("Creating policy")
    if cfg.policy.type == "pi0":
        cfg.policy.pretrained_path = 'lerobot/pi0'
    elif cfg.policy.type == 'smolvla':
        cfg.policy.pretrained_path = 'lerobot/smolvla_base'
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,  # 传入数据集元信息，让策略知道输入输出的维度/统计量等
    )

    # 创建优化器 optimizer 与学习率调度器 lr_scheduler：
    #   - optimizer（如 AdamW）：负责根据梯度更新网络参数，决定"怎么改、改多少"。
    #   - lr_scheduler：在训练过程中动态调整学习率（如先 warmup 升温再逐步衰减），
    #     学习率即每一步参数更新的"步幅大小"。
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # 混合精度训练的梯度缩放已交由 accelerator 统一接管，这里不再单独创建 GradScaler。

    step = 0  # 已经完成的策略更新次数（每次 = 一轮前向 + 反向 + 优化）

    # 断点续训：从已有检查点恢复训练步数、优化器与调度器状态，接着上次继续训。
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # 为离线训练创建数据加载器 dataloader。
    # dataloader 的作用是：把 dataset 自动切成一小批一小批（batch），并打乱顺序、
    # 多进程并行预读，源源不断地"喂"给训练循环。"离线训练"是指只用这份固定录好的
    # 数据集反复学习，训练过程中机器人不与环境实时交互（与"在线强化学习"相对）。
    if hasattr(cfg.policy, "drop_n_last_frames"):
        # 某些策略需要预测未来连续若干帧动作，每条轨迹末尾几帧凑不齐完整目标序列，
        # 用 EpisodeAwareSampler 把这些尾帧丢掉（drop_n_last_frames），并按轨迹采样。
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True  # 否则就普通随机打乱
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,  # 后台预读数据的进程数，越多越能避免 GPU 等数据
        batch_size=cfg.batch_size,    # 每批样本数
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",  # 锁页内存，加快数据从 CPU 拷到 GPU 的速度
        drop_last=False,
    )

    # 用 Accelerate 包装策略、优化器与数据加载器：多卡（DDP）下各卡各持一份完整模型、
    # 反向时自动做梯度全归约，并把数据在各进程间分片；取出的 batch 会自动搬到本进程 GPU。
    # lr_scheduler 不交给 accelerator，保持按训练步手动 step，使学习率曲线与单卡口径一致。
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    # cycle(...) 把 dataloader 包成"无限循环"的迭代器：遍历完整个数据集后会自动
    # 从头再来，这样主循环就能一直按 step 取数据，而不必关心 epoch 边界。
    dl_iter = cycle(dataloader)

    policy.train()  # 进入训练模式

    # 定义要跟踪的训练指标（AverageMeter 会自动计算这些值的滑动平均，用于日志展示）：
    #   loss 损失、grad_norm 梯度范数、lr 学习率、update_s 单步更新耗时、
    #   dataloading_s 取数据耗时。
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    # ---------------- 离线训练主循环 ----------------
    # 共训练 cfg.steps 步（若从检查点续训则从 step 开始）。每一步处理一个 batch，
    # 完成一次参数更新；并按设定的频率打印日志、保存检查点、执行评估。
    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)  # 取一个 batch 的数据
        train_tracker.dataloading_s = time.perf_counter() - start_time  # 记录取数据耗时

        # 把 batch 里的张量从 CPU 搬到 GPU（device）。non_blocking 配合上面的
        # pin_memory 可实现异步拷贝，让数据传输和计算尽量重叠以提速。
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # 核心：对这一批数据做一次完整的"前向 → 反向 → 优化"更新（见上方函数说明）。
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator,
            lr_scheduler=lr_scheduler,
        )

        # 注意：评估和保存检查点发生在第 `step` 次训练更新完成*之后*，因此在这里
        # 递增 `step`。
        step += 1
        train_tracker.step()
        # 根据频率判断本步是否需要：打印日志 / 保存检查点 / 执行评估。
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps  # 最后一步也强制保存
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # —— 记录日志：打印到控制台，并（若启用）上传到 wandb 看板 ——
        # 多卡下只由主进程打印与上传，避免每个进程各记一遍。
        if is_log_step and accelerator.is_main_process:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        # —— 保存检查点（checkpoint）——
        # 检查点就是把"当前训练进度"完整存盘：模型权重 + 优化器/调度器状态 + 步数 + 配置。
        # 有了它，既能随时拿出来部署/推理，也能在中断后从这里断点续训（配合 cfg.resume）。
        if cfg.save_checkpoint and is_saving_step:
            # 先让所有进程在此对齐（等各卡都跑完这一步），再由主进程统一存盘。
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                # unwrap_model 从 DDP 包装里取回原始 pi0 策略再保存，使检查点与单卡
                # 训练产出的格式完全一致（可直接用于部署或断点续训）。
                unwrapped_policy = accelerator.unwrap_model(policy)
                # 存盘前把 device 从带索引的 "cuda:N"（DDP 下各进程各不相同）规整为通用的
                # "cuda"：否则 checkpoint 的 config.json 会写入 "cuda:0" 这类带索引的值，部署时
                # lerobot 用 draccus 重新解析会因不认带索引的 device 而实例化失败。存完立即恢复，
                # 训练过程不受影响。
                _saved_pol_dev = unwrapped_policy.config.device
                _saved_cfg_dev = cfg.policy.device
                if str(_saved_pol_dev).startswith("cuda"):
                    unwrapped_policy.config.device = "cuda"
                if str(_saved_cfg_dev).startswith("cuda"):
                    cfg.policy.device = "cuda"
                try:
                    save_checkpoint(
                        checkpoint_dir, step, cfg,
                        unwrapped_policy, optimizer, lr_scheduler,
                    )
                finally:
                    unwrapped_policy.config.device = _saved_pol_dev
                    cfg.policy.device = _saved_cfg_dev
                update_last_checkpoint(checkpoint_dir)  # 更新指向"最新检查点"的快捷链接
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

        # —— 可选评估：仅当配置了仿真环境 cfg.env 时才做 ——
        # 在仿真里让当前策略实际跑若干个 episode，统计平均回报、成功率，并录制视频，
        # 从而直观衡量"训练到现在策略到底好不好用"（而不仅看 loss 数值）。
        # 多卡下评估仅由主进程执行；如需严格的多卡评估请改用独立的 eval 脚本。
        if cfg.env and is_eval_step and accelerator.is_main_process:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            # 评估时用 torch.no_grad() 关闭梯度计算（只前向、不训练，省显存更快）；
            # 同时复用与训练一致的精度设置。
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    accelerator.unwrap_model(policy),
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()  # 关闭仿真环境，释放资源
    if accelerator.is_main_process:
        logging.info("End of training")


if __name__ == "__main__":
    init_logging()  # 初始化日志格式
    # 调用 train()，但实参 cfg 由 @parser.wrap() 从命令行 --config_path 自动解析注入，
    # 因此这里不需要、也不能手动传参。
    # 单卡运行：
    #   python vla/train_vla.py --config_path=config/vla/pi0_franka.yaml
    # 多卡运行（N=参与训练的 GPU 数量，分布式数据并行 DDP）：
    #   accelerate launch --num_processes=N vla/train_vla.py --config_path=config/vla/pi0_franka.yaml
    train()
