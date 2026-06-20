"""
================================================================================
逆运动学（IK, Inverse Kinematics）求解模块
================================================================================

【先搞懂两个基本概念】
- 正运动学（Forward Kinematics）：已知每个关节转了多少度（关节角），算出机械臂
  末端（手爪）此刻在空间里的位置和朝向。这是“正着算”，结果唯一、好算。
- 逆运动学（Inverse Kinematics）：反过来——已经知道想让手爪去哪个位置、摆成什么
  姿态（目标位姿），反推每个关节应该各转到多少度。这是“倒着推”，通常没有简单公式，
  需要用迭代法一点点逼近。本文件干的就是这件事。

【本文件用的求解思路：增广雅可比 + 阻尼最小二乘 的迭代法】
- 雅可比矩阵 J：描述“关节角微小变化”与“末端位姿微小变化”之间的线性关系，即
  “末端的微动 ≈ J × 关节的微动”。它就像一个“换算表”，告诉你每个关节稍微动一点，
  手爪会朝哪个方向、动多少。
- 增广（augmented）：当我们同时有多个目标（比如要管好几个 body 的位置/姿态），
  就把它们各自的雅可比和误差“竖着堆叠”成一个大矩阵、一个大向量，一起求解。
- 我们想解的是：J × dq = err（err 是“当前位姿”到“目标位姿”的差距）。
  也就是求一个关节增量 dq，让末端朝着缩小误差的方向走一步。
- 直接求 J 的逆会有问题：当机械臂处于“奇异位形”（某些方向几乎动不了）时，
  J 接近不可逆，解会数值爆炸（dq 变得超大、乱跳）。
- 阻尼最小二乘（Damped Least Squares）：在求解时加一个“阻尼项”，相当于给解
  套了个“安全带”，宁可走得保守一点，也不让 dq 在奇异点附近爆炸。代价是收敛稍慢，
  但数值稳定得多。

【整体流程】
  init_ik_info()        -> 建一个空的 IK 目标登记表（字典）
  add_ik_info(...)      -> 往登记表里加一个目标（某个 body/geom 要到达的目标位置/姿态）
  get_dq_from_ik_info() -> 根据登记表，堆叠雅可比与误差，算出这一步的关节增量 dq
  plot_ik_info(...)     -> 把当前位姿与目标位姿画出来，便于可视化调试
  solve_ik(...)         -> 顶层入口：反复迭代上面的步骤，直到误差足够小（收敛）
================================================================================
"""
import sys
import numpy as np
from .utils import (
    get_colors,
    get_idxs,
)

# 逆运动学辅助工具
def init_ik_info():
    """
        初始化 IK（逆运动学）信息
        用法:
        ik_info = init_ik_info()
        ...
        add_ik_info(ik_info,body_name='BODY_NAME',p_trgt=P_TRGT,R_trgt=R_TRGT)
        ...
        for ik_tick in range(max_ik_tick):
            dq,ik_err_stack = get_dq_from_ik_info(
                env = env,
                ik_info = ik_info,
                stepsize = 1,
                eps = 1e-2,
                th = np.radians(10.0),
                joint_idxs_jac = joint_idxs_jac,
            )
            qpos = env.get_qpos()
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            env.forward(q=qpos)
            if np.linalg.norm(ik_err_stack) < 0.05: break
    """
    # ik_info 是一个“目标登记表”字典，几个列表一一对应（第 i 个目标的信息都在各列表的第 i 项）：
    #   'body_names': 每个目标关联的 body（刚体）名字；不针对 body 时为 None
    #   'geom_names': 每个目标关联的 geom（几何体）名字；不针对 geom 时为 None
    #   'p_trgts'   : 每个目标的目标位置（3 维坐标 [x,y,z]）；只控姿态不控位置时为 None
    #   'R_trgts'   : 每个目标的目标旋转（3x3 旋转矩阵，表示朝向）；只控位置时为 None
    #   'n_trgt'    : 当前已登记的目标个数
    ik_info = {
        'body_names':[],
        'geom_names':[],
        'p_trgts':[],
        'R_trgts':[],
        'n_trgt':0,
    }
    return ik_info

def add_ik_info(
        ik_info,
        body_name = None,
        geom_name = None,
        p_trgt    = None,
        R_trgt    = None,
    ):
    """
        添加 IK 信息
        往 ik_info 登记表里追加一个 IK 目标。
        参数：
            ik_info  : init_ik_info() 返回的登记表字典（会被就地修改）
            body_name: 要控制的 body 名字（控制某个刚体到达目标时填它）
            geom_name: 要控制的 geom 名字（控制某个几何体时填它）
            p_trgt   : 目标位置 [x,y,z]；若只想约束朝向、不约束位置则留 None
            R_trgt   : 目标朝向（3x3 旋转矩阵）；若只想约束位置、不约束朝向则留 None
        注：body_name 与 geom_name 通常二选一；p_trgt 与 R_trgt 可单独给、也可都给。
    """
    # 把这个目标的各项信息分别追加到对应列表的末尾，保持各列表下标一一对应
    ik_info['body_names'].append(body_name)
    ik_info['geom_names'].append(geom_name)
    ik_info['p_trgts'].append(p_trgt)
    ik_info['R_trgts'].append(R_trgt)
    ik_info['n_trgt'] = ik_info['n_trgt'] + 1 # 目标计数 +1

def get_dq_from_ik_info(
        env,
        ik_info,
        stepsize       = 1,
        eps            = 1e-2,
        th             = np.radians(1.0),
        joint_idxs_jac = None,
    ):
    """
        基于增广雅可比方法计算关节增量 delta q
        做一步迭代：根据登记表里的所有目标，算出关节该走的一小步 dq。
        参数：
            env           : 仿真环境对象，提供雅可比、阻尼最小二乘等计算
            ik_info        : 目标登记表（见 init_ik_info）
            stepsize       : 步长，缩放 dq 的大小（越大走得越快但可能越过头）
            eps            : 阻尼系数，越大越稳但收敛越慢（奇异点附近防爆炸的关键）
            th             : 误差阈值，控制阻尼如何随误差大小调整
            joint_idxs_jac : 只允许哪些关节参与求解（其余关节列被置零、保持不动）
        返回：
            dq           : 本步算出的关节增量（让末端朝目标靠近一小步）
            ik_err_stack : 当前所有目标的误差堆叠成的向量（用来判断是否收敛）
    """
    # 逐个目标计算它的雅可比 J 和位姿误差 ik_err，分别收集到列表里
    J_list,ik_err_list = [],[]
    for ik_idx,(ik_body_name,ik_geom_name) in enumerate(zip(ik_info['body_names'],ik_info['geom_names'])):
        ik_p_trgt = ik_info['p_trgts'][ik_idx]
        ik_R_trgt = ik_info['R_trgts'][ik_idx]
        IK_P = ik_p_trgt is not None # 该目标是否约束位置
        IK_R = ik_R_trgt is not None # 该目标是否约束姿态（朝向）
        # 向环境索要“IK 原料”：当前位姿对应的雅可比 J，以及当前位姿到目标的误差 ik_err
        J,ik_err = env.get_ik_ingredients(
            body_name = ik_body_name,
            geom_name = ik_geom_name,
            p_trgt    = ik_p_trgt,
            R_trgt    = ik_R_trgt,
            IK_P      = IK_P,
            IK_R      = IK_R,
        )
        J_list.append(J)
        ik_err_list.append(ik_err)

    # 把所有目标的雅可比竖着堆叠成一个大矩阵 J_stack（增广），误差也拼成一个长向量
    # 这样就能用“一次求解”同时照顾到全部目标
    J_stack      = np.vstack(J_list)   # 形状约为 (所有目标误差维度之和, 关节总数)
    ik_err_stack = np.hstack(ik_err_list) # 形状约为 (所有目标误差维度之和,)

    # 仅选取属于待使用关节的雅可比矩阵列
    # 做法：把 J_stack 整体清零，只把“允许活动的关节”那几列填回去。
    # 效果是其余关节对应的列全为 0，求解时它们的 dq 自然为 0，即保持不动。
    if joint_idxs_jac is not None:
        J_stack_backup = J_stack.copy()
        J_stack = np.zeros_like(J_stack)
        J_stack[:,joint_idxs_jac] = J_stack_backup[:,joint_idxs_jac]

    # 通过阻尼最小二乘计算 dq
    # 想解的方程是 J_stack · dq = ik_err（让末端朝缩小误差的方向走一步）。
    # 直接求逆在奇异位形附近会数值爆炸，故用带阻尼(eps)的最小二乘求一个稳健的近似解。
    dq = env.damped_ls(J_stack,ik_err_stack,stepsize=stepsize,eps=eps,th=th)
    return dq,ik_err_stack

def plot_ik_info(
        env,
        ik_info,
        axis_len   = 0.05,
        axis_width = 0.005,
        sphere_r   = 0.01,
        ):
    """
        绘制 IK 信息
        把每个 IK 目标的“当前位姿”和“目标位姿”都画到仿真画面里，方便直观调试：
        看手爪现在在哪、目标在哪、两者差多远。纯可视化，不影响求解结果。
        参数 axis_len/axis_width 控制坐标轴箭头长短粗细，sphere_r 控制标记小球半径。
    """
    # 给每个目标分配一种彩虹色，便于在画面里区分不同目标
    colors = get_colors(cmap_name='gist_rainbow',n_color=ik_info['n_trgt'])
    for ik_idx,(ik_body_name,ik_geom_name) in enumerate(zip(ik_info['body_names'],ik_info['geom_names'])):
        color = colors[ik_idx]
        ik_p_trgt = ik_info['p_trgts'][ik_idx]
        ik_R_trgt = ik_info['R_trgts'][ik_idx]
        IK_P = ik_p_trgt is not None
        IK_R = ik_R_trgt is not None

        if ik_body_name is not None:
            # 绘制当前位姿
            env.plot_body_T(
                body_name   = ik_body_name,
                plot_axis   = IK_R,
                axis_len    = axis_len,
                axis_width  = axis_width,
                plot_sphere = IK_P,
                sphere_r    = sphere_r,
                sphere_rgba = color,
                label       = '' # ''/ik_body_name
            )
            # 绘制目标位姿
            if IK_P:
                env.plot_sphere(p=ik_p_trgt,r=sphere_r,rgba=color,label='')
                env.plot_line_fr2to(p_fr=env.get_p_body(body_name=ik_body_name),p_to=ik_p_trgt,rgba=color)
            if IK_P and IK_R:
                env.plot_T(p=ik_p_trgt,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)
            if not IK_P and IK_R: # 仅旋转
                p_curr = env.get_p_body(body_name=ik_body_name)
                env.plot_T(p=p_curr,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)
            
        if ik_geom_name is not None:
            # 绘制当前位姿
            env.plot_geom_T(
                geom_name   = ik_geom_name,
                plot_axis   = IK_R,
                axis_len    = axis_len,
                axis_width  = axis_width,
                plot_sphere = IK_P,
                sphere_r    = sphere_r,
                sphere_rgba = color,
                label       = '' # ''/ik_geom_name
            )
            # 绘制目标位姿
            if IK_P:
                env.plot_sphere(p=ik_p_trgt,r=sphere_r,rgba=color,label='')
                env.plot_line_fr2to(p_fr=env.get_p_geom(geom_name=ik_geom_name),p_to=ik_p_trgt,rgba=color)
            if IK_P and IK_R:
                env.plot_T(p=ik_p_trgt,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)
            if not IK_P and IK_R: # 仅旋转
                p_curr = env.get_p_geom(geom_name=ik_geom_name)
                env.plot_T(p=p_curr,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)

def solve_ik(
        env,
        joint_names_for_ik,
        body_name_trgt,
        q_init          = None, # IK 从该初始位姿开始求解
        p_trgt          = None,
        R_trgt          = None,
        max_ik_tick     = 1000,
        ik_err_th       = 1e-2,
        restore_state   = True,
        ik_stepsize     = 1.0,
        ik_eps          = 1e-2,
        ik_th           = np.radians(1.0),
        verbose         = False,
        verbose_warning = True,
        reset_env       = False,
        render          = False,
        render_every    = 1,
    ):
    """
        求解逆运动学（IK）——顶层入口
        给定“想让某个 body 到达的目标位置 p_trgt / 目标朝向 R_trgt”，反复迭代，
        直到末端与目标的误差小于阈值（收敛），最终返回求得的关节角。

        关键参数：
            env                : 仿真环境对象
            joint_names_for_ik : 允许用来求解的关节名列表（只动这些关节）
            body_name_trgt     : 要控制去够目标的那个 body 名字
            q_init             : 迭代的初始关节角（给个好的起点能更快收敛）
            p_trgt / R_trgt    : 目标位置 / 目标朝向（旋转矩阵）
            max_ik_tick        : 最多迭代多少步（防止死循环）
            ik_err_th          : 误差阈值，误差小于它就认为收敛、提前结束
            restore_state      : 求解前备份仿真状态、求解后恢复（IK 只为算角度，
                                 不希望它真的改变环境当前状态）
            ik_stepsize/ik_eps/ik_th : 透传给单步求解的步长/阻尼/阈值
            render / render_every    : 是否实时渲染、每隔几步渲染一次
        返回：
            q_curr       : 求得的关节角（最终解）
            ik_err_stack : 最后一次的误差向量
            ik_info      : 本次使用的目标登记表
    """
    # 重置
    if reset_env:
        env.reset()
    if render:
        env.init_viewer()
    # 关节索引
    joint_idxs_jac = env.get_idxs_jac(joint_names=joint_names_for_ik)
    joint_idxs_fwd = env.get_idxs_fwd(joint_names=joint_names_for_ik)
    # 关节范围
    q_mins = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),0]
    q_maxs = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),1]
    # 保存 MuJoCo 状态
    if restore_state:
        env.store_state()
    # 初始 IK 位姿
    if q_init is not None:
        env.forward(q=q_init,joint_idxs=joint_idxs_fwd,increase_tick=False)
    # 初始化 IK 信息
    ik_info = init_ik_info()
    add_ik_info(
        ik_info  = ik_info,
        body_name= body_name_trgt,
        p_trgt   = p_trgt,
        R_trgt   = R_trgt, 
    )
    # 迭代循环
    # 每一轮做四件事：1) 算这一步该走的关节增量 dq；2) 更新关节角并裁剪到合法范围；
    # 3) 用正运动学把机械臂摆到新角度、看末端到哪了；4) 算误差，够小就停。
    q_curr = env.get_qpos_joints(joint_names=joint_names_for_ik) # 取当前关节角作为起点
    for ik_tick in range(max_ik_tick):
        dq,ik_err_stack = get_dq_from_ik_info(
            env            = env,
            ik_info        = ik_info,
            stepsize       = ik_stepsize,
            eps            = ik_eps,
            th             = ik_th,
            joint_idxs_jac = joint_idxs_jac,
        )
        q_curr = q_curr + dq[joint_idxs_jac] # 更新
        q_curr = np.clip(q_curr,q_mins,q_maxs) # 裁剪到关节范围
        env.forward(q=q_curr,joint_idxs=joint_idxs_fwd,increase_tick=False) # 正运动学
        ik_err = np.linalg.norm(ik_err_stack) # IK 误差
        if ik_err < ik_err_th: break # 终止条件
        if verbose:
            print ("[%d/%d] IK 误差 ik_err:[%.3f]"%(ik_tick,max_ik_tick,ik_err))
        if render:
            if ik_tick%render_every==0:
                plot_ik_info(env,ik_info)
                env.render()
    # 若 IK 误差过大则打印提示
    if verbose_warning and ik_err > ik_err_th:
        print ("ik_err:[%.4f] 高于阈值 ik_err_th:[%.4f]。"%
               (ik_err,ik_err_th))
        print ("你可能需要增大 max_ik_tick:[%d]"%
               (max_ik_tick))
    # 恢复此前备份的状态
    if restore_state:
        env.restore_state()
    # 关闭查看器
    if render:
        env.close_viewer()
    # 返回结果
    return q_curr,ik_err_stack,ik_info
