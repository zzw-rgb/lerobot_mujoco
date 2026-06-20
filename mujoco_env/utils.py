"""
================================================================================
通用工具集模块 (utils.py)
================================================================================
本文件是 MuJoCo 机械臂仿真教程项目里的"百宝箱",收集了写仿真/采数据时经常用到
的零碎小工具函数。零基础读者可以把它当成一个"工具抽屉",按下面几大类去找:

1. 随机采样与场景布置
   - sample_xyzs / sample_xys:随机撒一堆点,且保证点之间不会挨得太近(互不重叠)。
   - ObjectSpawner:在仿真场景里把托盘和物体随机摆到桌上(互不碰撞、随机朝向)。

2. 索引检索(在列表/数组里找东西的位置)
   - get_idxs / get_idxs_contain / get_idxs_closest_ndarray / get_consecutive_subarrays

3. 轨迹与运动学(让机械臂动得平滑)
   - finite_difference_matrix / get_A_vel_acc_jerk:用矩阵算速度、加速度、加加速度。
   - get_interp_const_vel_traj_nd:把几个路点插值成一条"匀速"轨迹。
   - check_vel_acc_jerk_nd:检查一条轨迹动得快不快、猛不猛。

4. 几何变换(三维空间里的方向、旋转)
   - compute_view_params:由相机/目标位置算出看相机参数(方位角、仰角等)。
   - np_uv / unit_vector:把向量变成长度为 1 的单位向量。
   - rotation_matrix:绕某根轴旋转的旋转矩阵。

5. 图像与可视化
   - get_colors / load_image / save_png / imshows / depth_to_gray_img / add_title_to_img

6. XML 与其他杂项
   - get_xml_string_from_path / prettify:读取/美化 MuJoCo 的 XML 模型文件。
   - TicTocClass:秒表计时器。get_monitor_size:获取屏幕尺寸。sleep:暂停一会儿。

注意:很多函数返回的数组都标注了"形状",例如 [L x D] 表示一个二维数组,有 L 行、
D 列;常见含义是 L 个时间步、每步 D 个维度(比如机械臂的 D 个关节角度)。
================================================================================
"""
import os
import pyautogui
import sys
import time
import numpy as np
# import cvxpy as cp
# import shapely as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import tkinter as tk
import xml.etree.ElementTree as ET
from scipy.spatial.distance import cdist
from PIL import Image
from xml.dom import minidom
from functools import partial
from io import BytesIO
import math
from .transforms import t2p, rpy2r
import cv2
from PIL import ImageDraw, ImageFont
def trim_scale(x,th):
    """
        裁剪缩放(等比例限幅)

        大白话:如果数组 x 里绝对值最大的那个数超过了阈值 th,就把整个数组按同一比例
        缩小,使得最大绝对值刚好等于 th;否则原样返回。这样既"压住了"过大的值,
        又保持了各元素之间的相对比例(方向不变),常用于限制速度/力等不要超标。

        参数:
            x  : 任意 numpy 数组。
            th : 阈值(threshold),允许的最大绝对值。
        返回:
            缩放后的数组(不会修改原数组)。
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def compute_view_params(
        camera_pos,
        target_pos,
        up_vector = np.array([0,0,1]),
    ):
    """根据三维空间中的相机位姿,计算观察器的方位角(azimuth)、距离(distance)、仰角(elevation)和注视点(lookat)。

    大白话:你想把虚拟相机放在某个位置去拍某个目标。MuJoCo 的查看器不是直接吃"相机
    坐标",而是用"方位角 + 仰角 + 距离 + 注视点"这套球面参数来描述视角。本函数就是把
    "相机在哪、看哪"换算成这套查看器能接受的参数。
      - 方位角 azimuth:在水平面内绕一圈的角度(左右转头),单位度。
      - 仰角 elevation:抬头/低头的角度,单位度。
      - 距离 distance:相机到目标的直线距离。
      - 注视点 lookat:相机盯着看的那个点(就是 target_pos)。

    Args:
        camera_pos (np.ndarray): 相机位置的三维数组。
        target_pos (np.ndarray): 目标位置的三维数组。
        up_vector (np.ndarray): 上方向向量的三维数组。

    Returns:
        tuple: 包含方位角、距离、仰角和注视点的元组。
    """
    # 计算相机到目标的向量及距离
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    # 计算方位角和仰角
    azimuth = np.arctan2(cam_to_target[1], cam_to_target[0])
    azimuth = np.rad2deg(azimuth) # [度]
    elevation = np.arcsin(cam_to_target[2] / distance)
    elevation = np.rad2deg(elevation) # [度]

    # 计算注视点
    lookat = target_pos

    # 计算相机朝向矩阵:用"从相机指向目标的方向"作为 z 轴,再借助上方向向量叉乘
    # 出互相垂直的 x、y 轴,凑成一个三维坐标系(此处算出但未被返回,仅作完整性保留)。
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # 返回计算结果
    return azimuth, distance, elevation, lookat

def get_idxs(list_query,list_domain):
    """
        获取两个列表或 ndarray 之间相互对应的索引

        大白话:对于 list_domain 里的每个元素,如果它也出现在 list_query 里,就找出它
        在 list_query 中的位置(下标)。常用于"我有一个总的名字清单 list_query,现在
        想知道其中一部分名字 list_domain 各自排在第几号"。

        参数:
            list_query  : 被查找的总清单(在它里面找位置)。
            list_domain : 想要定位的若干元素。
        返回:
            idxs : 一个整数列表,是这些元素在 list_query 中的下标。
    """
    if isinstance(list_query,list) and isinstance(list_domain,list):
        idxs = [list_query.index(item) for item in list_domain if item in list_query]
    else:
        print("[get_idxs] 输入应为 'List' 类型。")
    return idxs

def get_idxs_contain(list_query,list_substring):
    """
        按"子串包含"关系查找索引

        大白话:在字符串列表 list_query 里,找出那些"包含 list_substring 中任意一个
        片段"的元素的下标。和 get_idxs 不同,这里不要求完全相等,只要"含有"即可。
        例如清单里有 'body_obj_can_1',而 list_substring=['can'],它就会被选中。

        参数:
            list_query     : 字符串列表(在里面逐个检查)。
            list_substring : 若干关键词片段,只要命中其一即算匹配。
        返回:
            idxs : 命中元素在 list_query 中的下标列表。
    """
    idxs = [i for i, s in enumerate(list_query) if any(sub in s for sub in list_substring)]
    return idxs

def get_colors(n_color=10,cmap_name='gist_rainbow',alpha=1.0):
    """
        获取多种互相区分明显的颜色

        大白话:一次性生成 n_color 种颜色,常用于画图时给多条曲线/多个物体上不同的色。
        做法是从 matplotlib 的"色谱(colormap)"上等间隔取色——色谱就像一条彩虹条,
        从一端均匀地采若干个点,得到一组渐变但彼此可分的颜色。

        参数:
            n_color   : 想要几种颜色。
            cmap_name : 色谱名字(默认 'gist_rainbow' 即彩虹色)。
            alpha     : 透明度(此参数当前未实际使用)。
        返回:
            colors : 长度为 n_color 的列表,每个元素是 (R,G,B,A) 四元组。
    """
    colors = [plt.get_cmap(cmap_name)(idx) for idx in np.linspace(0,1,n_color)]
    for idx in range(n_color):
        color = colors[idx]
        colors[idx] = color
    return colors

def sample_xyzs(n_sample=1,x_range=[0,1],y_range=[0,1],z_range=[0,1],min_dist=0.1,xy_margin=0.0):
    """
        在三维空间中随机采样若干个互不重叠的点

        大白话:在指定的长方体范围(x、y、z 各自的区间)里随机撒 n_sample 个点,并保证
        任意两点之间的距离都不小于 min_dist。典型用途:采集数据时,把杯子、盘子等物体
        随机摆到桌面上,且彼此不会叠在一起。

        做法:逐个采点,每采一个就反复随机直到它离已采的所有点都足够远(拒绝采样)。

        参数:
            n_sample  : 要采几个点。
            x_range / y_range / z_range : 各坐标轴的取值区间 [最小值, 最大值]。
            min_dist  : 任意两点之间允许的最小距离(米)。
            xy_margin : 在 x、y 方向上向内收缩的边距,避免点贴着边界(z 不受此影响)。
        返回:
            xyzs : 形状 [n_sample x 3] 的数组,每行是一个点的 (x, y, z) 坐标。
    """
    xyzs = np.zeros((n_sample,3))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0]+xy_margin,high=x_range[1]-xy_margin)
            y_rand = np.random.uniform(low=y_range[0]+xy_margin,high=y_range[1]-xy_margin)
            z_rand = np.random.uniform(low=z_range[0],high=z_range[1])
            xyz = np.array([x_rand,y_rand,z_rand])
            if p_idx == 0: break
            devc = cdist(xyz.reshape((-1,3)),xyzs[:p_idx,:].reshape((-1,3)),'euclidean')
            if devc.min() > min_dist: break # 物体之间的最小距离
        xyzs[p_idx,:] = xyz
    return xyzs

class ObjectSpawner:
    """
        物体生成器:在 MuJoCo 仿真场景里随机布置托盘和若干物体

        大白话:每次开始采集数据前,我们希望桌面上的托盘和物体(杯子、罐子、瓶子等)
        都摆在随机但合理的位置,而且彼此不碰撞、朝向也随机。这个类就负责干这件事:
        先随机放一个托盘,再把其余物体一个个放到不重叠的位置,并给每个物体一个随机的
        水平旋转角。这样每条数据的初始场景都不一样,训练出的模型泛化性更好。

        用法:
            spawner = ObjectSpawner(env)   # env 是仿真环境
            spawner.spawn_objects()        # 随机布置一次场景
    """
    def __init__(self, env):
        """
        env: 一个环境实例,需提供以下方法:
            - get_body_names(prefix)
            - set_p_base_body(body_name, p)
            - set_R_base_body(body_name, R)
        """
        self.env = env

    def spawn_objects(self):
        # 随机布置整个场景:先放托盘,再放其余物体(每个物体不与他人/托盘重叠,且随机朝向)。
        # --- 生成托盘 ---
        # 使用提供的采样函数采样托盘位置。
        tray_xyz = sample_xyzs(
            n_sample=1,
            x_range=[0.3, 0.7],
            y_range=[-0.35, 0.35],
            z_range=[0.82, 0.82],
            min_dist=0.1,
            xy_margin=0.00
        )[0]
        self.env.set_p_base_body(body_name='body_obj_tray_5', p=tray_xyz)
        
        # 随机选择托盘朝向(若发生旋转则交换尺寸维度)
        if np.random.rand() > 0.5:
            # 将托盘绕 z 轴旋转 90°
            self.env.set_R_base_body(body_name='body_obj_tray_5',
                                     R=rpy2r(np.deg2rad([0, 0, 90])))

        # --- 获取需要生成的物体名称(排除托盘)---
        obj_names = self.env.get_body_names(prefix='body_obj_')
        if 'body_obj_tray_5' in obj_names:
            obj_names.remove('body_obj_tray_5')

        # 用于记录已放置物体的列表,以避免碰撞。
        placed_positions = []

        # 为每个物体生成一个不发生碰撞的位置和随机旋转。
        for name in obj_names:
            # 根据启发式规则设置 x 范围:名称中含 "can" 的物体使用受限范围。
            if 'can' in name or 'bottle' in name:
                x_range = [0.5, 0.7]
                z = 0.9
            else:
                x_range = [0.3, 0.6]
                z = 0.82
            y_range = [-0.35, 0.35]


            # 寻找一个不与已放置物体重叠的位置。
            pos = self._get_non_colliding_position(
                placed_positions=placed_positions,
                x_range=x_range,
                y_range=y_range,
                min_dist=0.1,
                tray_xyz=tray_xyz  # 如有需要,可选择性地避开托盘区域。
            )
            placed_positions.append(pos)
            # 设置物体位置(为简化处理,使用与托盘相同的 z 值)。
            self.env.set_p_base_body(body_name=name, p=[pos[0], pos[1], z])

            # 可选地赋予一个随机旋转。
            angle = np.random.uniform(0, 360)
            self.env.set_R_base_body(body_name=name, R=rpy2r(np.deg2rad([0, 0, angle])))

    def _get_non_colliding_position(self, placed_positions, x_range, y_range, min_dist, tray_xyz):
        """尝试采样一个不与已放置物体(或托盘)发生碰撞的位置。
           若在固定次数的尝试后仍未找到有效位置,则抛出 ValueError。"""
        max_attempts = 100
        tray_margin = 0.3  # 定义一个边距,必要时用于避开托盘中心区域。
        for attempt in range(max_attempts):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            candidate = np.array([x, y])

            collision = False
            # 检查与已放置物体之间的距离。
            for pos in placed_positions:
                if np.linalg.norm(candidate - np.array(pos)) < min_dist:
                    collision = True
                    break
            # 可选:检查候选位置是否过于靠近托盘中心。
            if np.linalg.norm(candidate - np.array(tray_xyz[:2])) < tray_margin:
                collision = True
            if not collision:
                return candidate
        raise ValueError("Could not find a non-colliding position after {} attempts".format(max_attempts))


def sample_xys(n_sample=1,x_range=[0,1],y_range=[0,1],min_dist=0.1,xy_margin=0.0):
    """
        在二维平面上随机采样若干个互不重叠的点

        和 sample_xyzs 几乎一样,只是少了 z 维度——只在水平面 (x, y) 上撒点,保证两两
        之间距离不小于 min_dist。适合"高度固定、只关心平面位置"的摆放场景。

        参数:含义同 sample_xyzs(去掉 z_range)。
        返回:
            xys : 形状 [n_sample x 2] 的数组,每行是一个点的 (x, y) 坐标。
    """
    xys = np.zeros((n_sample,2))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0]+xy_margin,high=x_range[1]-xy_margin)
            y_rand = np.random.uniform(low=y_range[0]+xy_margin,high=y_range[1]-xy_margin)
            xy = np.array([x_rand,y_rand])
            if p_idx == 0: break
            devc = cdist(xy.reshape((-1,3)),xys[:p_idx,:].reshape((-1,3)),'euclidean')
            if devc.min() > min_dist: break # 物体之间的最小距离
        xys[p_idx,:] = xy
    return xys

def save_png(img,png_path,verbose=False):
    """
        把图像保存为 PNG 文件

        大白话:给一张图(numpy 数组)和一个保存路径,就帮你存成 png。如果路径所在的
        文件夹还不存在,会自动先把文件夹建好。verbose=True 时会打印提示信息。

        参数:
            img      : 图像数组(如 [H x W x 3] 的 RGB 图)。
            png_path : 目标文件路径,例如 'out/img_0.png'。
            verbose  : 是否打印"已生成/已保存"提示。
    """
    directory = os.path.dirname(png_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        if verbose:
            print ("[%s] 已生成。"%(directory))
    # 保存为 png
    plt.imsave(png_path,img)
    if verbose:
        print ("[%s] 已保存。"%(png_path))

def finite_difference_matrix(n, dt, order):
    """
    构造"有限差分"矩阵,用来对一串等间隔采样的数值求导(求变化率)。

    大白话:假设我们记录了某个量(比如某个关节角度)在等时间间隔下的 n 个数值,排成
    一列。我们想知道它变化得有多快——
      - 一阶导数 = 速度(位置每秒变多少);
      - 二阶导数 = 加速度(速度每秒变多少);
      - 三阶导数 = 加加速度 jerk(加速度变化得猛不猛,越大越"颠")。
    连续函数靠求导,离散数据则靠"相邻点相减再除以时间间隔"来近似,这就是"有限差分"。
    本函数把这种"相邻相减"的规则打包成一个 n×n 矩阵 A,之后只要算 A @ 数据列,
    就能一次性得到每个时刻的导数,非常方便。

    n: 点的数量(数据有多少个时间步)
    dt: 时间间隔(相邻两个点相差多少秒)
    order: 阶数 (1=速度, 2=加速度, 3=加加速度/jerk)
    返回: 形状 [n x n] 的差分矩阵(末尾几行用"后向差分"补齐,避免越界)。
    """
    # 阶数
    if order == 1:  # 速度
        coeffs = np.array([-1, 1])
    elif order == 2:  # 加速度
        coeffs = np.array([1, -2, 1])
    elif order == 3:  # 加加速度(jerk)
        coeffs = np.array([-1, 3, -3, 1])
    else:
        raise ValueError("Order must be 1, 2, or 3.")

    # 填充矩阵
    mat = np.zeros((n, n))
    for i in range(n - order):
        for j, c in enumerate(coeffs):
            mat[i, i + j] = c

    # (可选)使用后向差分处理边界条件
    if order == 1:  # 速度
        mat[-1, -2:] = np.array([-1, 1])  # 后向差分
    elif order == 2:  # 加速度
        mat[-1, -3:] = np.array([1, -2, 1])  # 后向差分
        mat[-2, -3:] = np.array([1, -2, 1])  # 后向差分
    elif order == 3:  # 加加速度(jerk)
        mat[-1, -4:] = np.array([-1, 3, -3, 1])  # 后向差分
        mat[-2, -4:] = np.array([-1, 3, -3, 1])  # 后向差分
        mat[-3, -4:] = np.array([-1, 3, -3, 1])  # 后向差分

    # 返回
    return mat / (dt ** order)

def get_A_vel_acc_jerk(n=100,dt=1e-2):
    """
        一次性拿到速度、加速度、加加速度(jerk)三个差分矩阵

        大白话:这是 finite_difference_matrix 的小封装,帮你把 1/2/3 阶差分矩阵都建好。
        之后对一条轨迹数据列 q,分别算 A_vel @ q、A_acc @ q、A_jerk @ q 就能得到
        速度、加速度、加加速度。

        参数:
            n  : 轨迹点数。
            dt : 相邻两点的时间间隔(秒)。
        返回:
            A_vel, A_acc, A_jerk:三个 [n x n] 矩阵。
    """
    A_vel  = finite_difference_matrix(n,dt,order=1)
    A_acc  = finite_difference_matrix(n,dt,order=2)
    A_jerk = finite_difference_matrix(n,dt,order=3)
    return A_vel,A_acc,A_jerk

def get_idxs_closest_ndarray(ndarray_query,ndarray_domain):
    """
        在一个数组里,为另一个数组的每个值找"最接近的那个"的下标

        大白话:对于 ndarray_domain 里的每个数 x,在 ndarray_query 中找到与 x 数值最接近
        的元素,返回它的下标。典型用途:已知插值后的密集时间轴 ndarray_query,想知道
        原始几个路点的时间 ndarray_domain 各自落在密集时间轴的第几个位置。

        参数:
            ndarray_query  : 被搜索的数组(在它里面找最近的)。
            ndarray_domain : 若干目标值。
        返回:
            一个下标列表,长度与 ndarray_domain 相同。
    """
    return [np.argmin(np.abs(ndarray_query-x)) for x in ndarray_domain]

def get_interp_const_vel_traj_nd(
        anchors, # [L x D]
        vel = 1.0,
        HZ  = 100,
        ord = np.inf,
    ):
    """
        把几个路点连成一条"匀速"的密集轨迹(线性插值)

        大白话:你给出几个关键路点(anchors),比如机械臂依次要经过的几个姿态。本函数把
        相邻路点之间用直线连起来,并按固定的速度 vel 重新"铺"出一串细密的中间点,使得
        机械臂沿途以大致恒定的速度移动。距离远的两个路点之间会被铺上更多中间点(因为
        匀速走完更长的路要花更多时间),这样动作就不会忽快忽慢。

        关键思路:
          1. 算出相邻路点之间的距离;
          2. 距离 ÷ 速度 = 走这段要花的时间,累加得到每个路点的"到达时刻" times_anchor;
          3. 在 0 到总时长之间按采样频率 HZ 均匀铺出密集时间轴 times_interp;
          4. 对每个维度分别做一维线性插值,得到每个密集时刻的取值。

        参数:
            anchors : 形状 [L x D] 的路点数组,L 个路点、每个 D 维(如 D 个关节角)。
            vel     : 期望的移动速度(决定铺多密、总共多长时间)。
            HZ      : 采样频率(每秒铺多少个点)。
            ord     : 计算路点间距离时用的范数(默认 np.inf 即各维差值的最大值)。
        返回:
            times_interp   : [L_interp] 密集时间轴。
            anchors_interp : [L_interp x D] 插值后的密集轨迹。
            times_anchor   : [L] 每个原始路点对应的时刻。
            idxs_anchor    : 每个原始路点在密集轨迹中最接近的下标。
    """
    L = anchors.shape[0]
    D = anchors.shape[1]
    # 第 1 步:逐段计算相邻路点之间的距离(第 0 个点没有"上一个",距离记为 0)。
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev,p_curr = anchors[tick-1,:],anchors[tick,:]
            dists[tick] = np.linalg.norm(p_prev-p_curr,ord=ord)
    # 第 2 步:距离 ÷ 速度 = 每段耗时,累加(cumsum)得到每个路点的到达时刻。
    times_anchor = np.cumsum(dists/vel) # [L]
    # 第 3 步:按频率 HZ 在 0~总时长之间均匀铺出密集时间轴。
    L_interp     = int(times_anchor[-1]*HZ)
    times_interp = np.linspace(0,times_anchor[-1],L_interp) # [L_interp]
    # 第 4 步:对每一维单独做一维线性插值,填出每个密集时刻的取值。
    anchors_interp  = np.zeros((L_interp,D)) # [L_interp x D]
    for d_idx in range(D): # 对每个维度
        anchors_interp[:,d_idx] = np.interp(times_interp,times_anchor,anchors[:,d_idx])
    # 顺带记录:每个原始路点落在密集轨迹的哪个位置(便于事后对应)。
    idxs_anchor = get_idxs_closest_ndarray(times_interp,times_anchor)
    return times_interp,anchors_interp,times_anchor,idxs_anchor


def check_vel_acc_jerk_nd(
        times, # [L]
        traj, # [L x D]
        verbose = True,
        factor  = 1.0,
    ):
    """
        检查一条 n 维轨迹动得快不快、猛不猛(速度/加速度/加加速度)

        大白话:给定时间轴 times 和轨迹 traj,本函数对每一维分别算出整条轨迹的最大速度、
        最大加速度、最大加加速度,以及起点/终点速度,用来判断这条轨迹是否平滑、会不会
        让机械臂动作太剧烈。verbose=True 时把这些数值打印出来方便人工检查。

        参数:
            times   : [L] 各时间步对应的时刻(要求等间隔)。
            traj    : [L x D] 轨迹,L 个时间步、每步 D 维。
            verbose : 是否打印检查结果。
            factor  : 打印时给数值乘的缩放系数(仅影响显示,不改变返回值)。
        返回:
            vel_inits, vel_finals, max_vels, max_accs, max_jerks:
            五个列表,每个长度为 D,分别是各维的起始速度、终止速度、最大速度、
            最大加速度、最大加加速度。
    """
    L,D = traj.shape[0],traj.shape[1]
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=len(times),dt=times[1]-times[0])
    vel_inits,vel_finals,max_vels,max_accs,max_jerks = [],[],[],[],[]
    for d_idx in range(D):
        traj_d = traj[:,d_idx]
        vel = A_vel @ traj_d
        acc = A_acc @ traj_d
        jerk = A_jerk @ traj_d
        vel_inits.append(vel[0])
        vel_finals.append(vel[-1])
        max_vels.append(np.abs(vel).max())
        max_accs.append(np.abs(acc).max())
        max_jerks.append(np.abs(jerk).max())

    # 打印
    if verbose:
        print ("正在检查 L:[%d]xD:[%d] 轨迹的速度、加速度和加加速度 (factor:[%.2f])。"%
               (L,D,factor))
        for d_idx in range(D):
            print (" 维度 dim:[%d/%d]: v_init:[%.2e] v_final:[%.2e] v_max:[%.2f] a_max:[%.2f] j_max:[%.2f]"%
                   (d_idx,D,
                    factor*vel_inits[d_idx],factor*vel_finals[d_idx],
                    factor*max_vels[d_idx],factor*max_accs[d_idx],factor*max_jerks[d_idx])
                )

    # 返回
    return vel_inits,vel_finals,max_vels,max_accs,max_jerks

        
def np_uv(vec):
    """
        把向量变成单位向量(长度归一化为 1,只保留方向)

        大白话:向量既有方向又有长度。很多时候我们只关心"指向哪",不关心"多长",
        就把它除以自己的长度,得到长度为 1 的"单位向量"。本函数还做了保护:如果向量
        几乎是零向量(长度过小),直接返回默认方向 [0,0,1],避免除以 0 出错。

        参数:vec —— 任意向量。
        返回:同方向、长度为 1 的向量(零向量时返回 [0,0,1])。
    """
    x = np.array(vec)
    len = np.linalg.norm(x)
    if len <= 1e-6:
        return np.array([0,0,1])
    else:
        return x/len    
    
def uv_T_joi(T_joi,joi_fr,joi_to):
    """
        求"从一个关节指向另一个关节"的方向(单位向量)

        说明:T_joi 是一个字典,键是关节/部位名,值是该部位的 4x4 位姿矩阵(包含位置和
        姿态)。JOI 是 "Joints/Joint Of Interest"(关注的关节/部位)的简称。t2p() 从位姿
        矩阵里取出位置(平移)。本函数算"从 joi_fr 指向 joi_to 的单位方向向量"。

        参数:
            T_joi  : {部位名: 4x4 位姿矩阵} 的字典。
            joi_fr : 起点部位名。
            joi_to : 终点部位名。
        返回:长度为 1 的方向向量。
    """
    return np_uv(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))

def len_T_joi(T_joi,joi_fr,joi_to):
    """
        求两个关节/部位之间的直线距离

        含义同 uv_T_joi,但这里返回的是两点间的距离(长度),而不是方向。
        常用于量某段连杆/肢段有多长。

        参数:同 uv_T_joi。
        返回:两部位位置之间的欧氏距离(标量)。
    """
    return np.linalg.norm(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))

def get_consecutive_subarrays(array,min_element=1):
    """
        把一串整数下标按"是否连续"切成若干段

        大白话:给一串(通常已排序的)整数,例如 [2,3,4, 7,8, 20],本函数会在"不连续"
        的地方断开,切成 [2,3,4]、[7,8]、[20] 这样的连续小段。常用于:某个条件在一段
        连续帧里成立(比如手一直握着杯子),想把这些连续区间一段段拎出来。
        只保留长度不小于 min_element 的段。

        参数:
            array       : 整数数组(一般是已排序的下标)。
            min_element : 子段至少要有几个元素才保留。
        返回:连续子数组组成的列表。
    """
    # np.diff 求相邻差,差不等于 1 的位置就是"断点",在那里切开。
    split_points = np.where(np.diff(array) != 1)[0] + 1
    subarrays = np.split(array,split_points)    
    return [subarray for subarray in subarrays if len(subarray) >= min_element]

def load_image(image_path):
    """
        将图像加载为 ndarray(uint8)
    """
    return np.array(Image.open(image_path))

def imshows(img_list,title_list,figsize=(8,2),fontsize=8):
    """
        把多张图像并排显示在一行里

        大白话:传入一组图和对应的一组标题,本函数用 matplotlib 把它们横向排成一行
        显示出来,方便一眼对比(比如同时看 RGB 图、深度图、分割图)。

        参数:
            img_list   : 图像列表。
            title_list : 标题列表(与 img_list 一一对应)。
            figsize    : 画布大小。
            fontsize   : 标题字号。
    """
    n_img = len(img_list)
    plt.figure(figsize=(8,2))
    for img_idx in range(n_img):
        img   = img_list[img_idx]
        title = title_list[img_idx]
        plt.subplot(1,n_img,img_idx+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title,fontsize=fontsize)
    plt.show()
    
def depth_to_gray_img(depth,max_val=10.0):
    """
        把深度图转换成可显示的灰度图

        大白话:深度图里每个像素存的是"该点离相机多远"(单位米,是浮点数),没法直接当
        普通图片看。本函数先把过远的距离截断到 max_val,再把距离线性映射到 0~255 的
        灰度值,最后复制成三通道(R=G=B),变成一张能直接显示/保存的灰度图。
        近处暗、远处亮(具体取决于数值分布)。

        参数:
            depth   : [H x W] 的浮点深度图。
            max_val : 超过这个距离的都按 max_val 处理(截断,防止极远点拉坏对比度)。
        返回:[H x W x 3] 的 uint8 灰度图。
    """
    depth_clip = np.clip(depth,a_min=0.0,a_max=max_val) # float 类型
    img = np.tile(255*depth_clip[:,:,np.newaxis]/depth_clip.max(),(1,1,3)).astype(np.uint8) # uint8 类型
    return img

def get_monitor_size():
    """
        获取显示器尺寸
    """
    w,h = pyautogui.size()
    return w,h
    
def get_xml_string_from_path(xml_path):
    """
        读取一个 XML 文件,返回它的字符串内容

        大白话:MuJoCo 用 XML 文件(.xml/.mjcf)描述机器人和场景。本函数把指定路径的
        XML 读进来、解析后再转回完整字符串,方便后续修改或拼接。

        参数:xml_path —— XML 文件路径。
        返回:该 XML 的字符串。
    """
    # 解析 XML 文件
    tree = ET.parse(xml_path)

    # 获取 XML 的根元素
    root = tree.getroot()

    # 将 ElementTree 对象转换为字符串
    xml_string = ET.tostring(root, encoding='unicode', method='xml')

    return xml_string

def prettify(elem):
    """
        返回该 Element 经美化排版后的 XML 字符串。

        大白话:程序生成的 XML 往往挤成一团、没有缩进,很难读。本函数给它加上整齐的
        缩进换行(并去掉多余空行),变成人类好读的格式。常用于打印/保存拼好的 MuJoCo
        模型时让文件更清晰。

        参数:elem —— 一个 XML Element 节点。
        返回:美化后的 XML 字符串。
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")

    # 去除多余的空白(空行)
    lines = [line for line in pretty_xml.splitlines() if line.strip()]
    return "\n".join(lines)

class TicTocClass(object):
    """
        计时器(Tic toc)—— 像秒表一样测量一段代码跑了多久

        大白话:tic() 表示"开始计时",toc() 表示"停止并读数"。名字来自钟表"嘀-嗒"声。
        典型用法:在耗时操作前调用 tic(),操作后调用 toc(verbose=True),它会自动选择
        合适的单位(毫秒/秒/分钟)把用时打印出来。print_every 可控制每隔几次才打印一次,
        避免循环里刷屏。

        tictoc = TicTocClass()
        tictoc.tic()
        ~~
        tictoc.toc()
    """
    def __init__(self,name='tictoc',print_every=1):
        """
            初始化
        """
        self.name         = name
        self.time_start   = time.time()
        self.time_end     = time.time()
        self.print_every  = print_every
        self.time_elapsed = 0.0
        self.cnt          = 0 

    def tic(self):
        """
            开始计时(Tic)
        """
        self.time_start = time.time()

    def toc(self,str=None,cnt=None,print_every=None,verbose=False):
        """
            结束计时(Toc)
        """
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start
        if print_every is not None: self.print_every = print_every
        if verbose:
            if self.time_elapsed <1.0:
                time_show = self.time_elapsed*1000.0
                time_unit = 'ms'
            elif self.time_elapsed <60.0:
                time_show = self.time_elapsed
                time_unit = 's'
            else:
                time_show = self.time_elapsed/60.0
                time_unit = 'min'
            if cnt is not None: self.cnt = cnt
            if (self.cnt % self.print_every) == 0:
                if str is None:
                    print ("%s 已用时间:[%.2f]%s"%
                        (self.name,time_show,time_unit))
                else:
                    print ("%s 已用时间:[%.2f]%s"%
                        (str,time_show,time_unit))
        self.cnt = self.cnt + 1
        # 返回
        return self.time_elapsed
    
def sleep(sec):
    """
        暂停(休眠)指定的秒数

        大白话:让程序停一会儿再继续,常用于控制仿真节奏或等画面刷新。
        参数:sec —— 要暂停的秒数。
    """
    time.sleep(sec)
    
    

def unit_vector(data, axis=None, out=None):
    """
    返回沿指定轴按长度(即欧几里得范数)归一化后的 ndarray。

    示例:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): 待归一化的数据
        axis (None or int): 若指定,则确定沿数据的哪个具体轴进行归一化
        out (None or np.array): 若指定,则将计算结果存储到该变量中

    Returns:
        None or np.array: 若未指定 @out,则返回归一化后的向量;否则将输出存储到 @out 中
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.asarray(data)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """
    返回绕由点(point)和方向(direction)定义的轴进行旋转的矩阵。

    示例:
        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True

        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    Args:
        angle (float): 旋转的大小(角度幅值)
        direction (np.array): 旋转所绕的轴 (ax,ay,az)
        point (None or np.array): 若指定,则为旋转所绕的点 (x,y,z)

    Returns:
        np.array: 包含所需旋转的 4x4 齐次矩阵
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # 绕单位向量的旋转矩阵
    R = np.array(((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # 旋转不绕原点进行
        point = np.asarray(point[:3], dtype=np.float32)
        M[:3, 3] = point - np.dot(R, point)
    return M


def add_title_to_img(img,text='Title',margin_top=30,color=(0,0,0),font_size=20,resize=True,shape=(300,300)):
    """
    为图像顶部添加一条文字标题

    大白话:在原图上方留出一条白边,把标题文字居中写上去,返回一张"带标题"的新图。
    常用于把多张相机图拼成图表时给每张图标注名字。可选先把图缩放到统一尺寸 shape。

    参数:
        img        : 原始图像(numpy 数组,RGB)。
        text       : 标题文字。
        margin_top : 顶部留白的高度(像素),标题就写在这条白边里。
        color      : 文字颜色 (R,G,B)。
        font_size  : 字号。
        resize     : 是否先把图缩放到 shape。
        shape      : 缩放后的目标尺寸 (宽, 高)。
    返回:带标题的新图(numpy 数组)。
    """
    # 调整尺寸
    img_copied = img.copy()
    if resize:
        img_copied = cv2.resize(img_copied,shape,interpolation=cv2.INTER_NEAREST)
    # 转换为 PIL 图像
    pil_img = Image.fromarray(img_copied)#
    width, height = pil_img.size
    new_height = margin_top + height
    # 创建带有顶部边距的新图像
    new_img = Image.new("RGB", (width, new_height),color=(255,255,255))
    # 粘贴原始图像
    new_img.paste(pil_img, (0, margin_top))
    # 绘制文本
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default(size=font_size)
    bbox = draw.textbbox((0,0),text,font=font)
    # 文本居中
    text_width  = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (margin_top - text_height) // 2
    # 绘制文本
    draw.text((x, y), text, font=font, fill=color)
    img_with_title = np.array(new_img)
    # 返回
    return img_with_title