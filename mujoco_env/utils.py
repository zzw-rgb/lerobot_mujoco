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
        裁剪缩放
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

    # 计算相机朝向矩阵
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # 返回计算结果
    return azimuth, distance, elevation, lookat

def get_idxs(list_query,list_domain):
    """
        获取两个列表或 ndarray 之间相互对应的索引
    """
    if isinstance(list_query,list) and isinstance(list_domain,list):
        idxs = [list_query.index(item) for item in list_domain if item in list_query]
    else:
        print("[get_idxs] 输入应为 'List' 类型。")
    return idxs

def get_idxs_contain(list_query,list_substring):
    """
        获取两个列表之间相互对应的索引
    """
    idxs = [i for i, s in enumerate(list_query) if any(sub in s for sub in list_substring)]
    return idxs

def get_colors(n_color=10,cmap_name='gist_rainbow',alpha=1.0):
    """
        获取多种不同的颜色
    """
    colors = [plt.get_cmap(cmap_name)(idx) for idx in np.linspace(0,1,n_color)]
    for idx in range(n_color):
        color = colors[idx]
        colors[idx] = color
    return colors

def sample_xyzs(n_sample=1,x_range=[0,1],y_range=[0,1],z_range=[0,1],min_dist=0.1,xy_margin=0.0):
    """
        在三维空间中采样若干点,并保证点之间满足最小距离要求
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
    def __init__(self, env):
        """
        env: 一个环境实例,需提供以下方法:
            - get_body_names(prefix)
            - set_p_base_body(body_name, p)
            - set_R_base_body(body_name, R)
        """
        self.env = env

    def spawn_objects(self):
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
        在三维空间中采样若干点,并保证点之间满足最小距离要求
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
        保存图像
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
    n: 点的数量
    dt: 时间间隔
    order: 阶数 (1=速度, 2=加速度, 3=加加速度/jerk)
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
        获取用于计算速度、加速度和加加速度(jerk)的矩阵
    """
    A_vel  = finite_difference_matrix(n,dt,order=1)
    A_acc  = finite_difference_matrix(n,dt,order=2)
    A_jerk = finite_difference_matrix(n,dt,order=3)
    return A_vel,A_acc,A_jerk

def get_idxs_closest_ndarray(ndarray_query,ndarray_domain):
    return [np.argmin(np.abs(ndarray_query-x)) for x in ndarray_domain]

def get_interp_const_vel_traj_nd(
        anchors, # [L x D]
        vel = 1.0,
        HZ  = 100,
        ord = np.inf,
    ):
    """
        获取经线性插值得到的匀速轨迹
        输出为 (times_interp, anchors_interp, times_anchor, idxs_anchor)
    """
    L = anchors.shape[0]
    D = anchors.shape[1]
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev,p_curr = anchors[tick-1,:],anchors[tick,:]
            dists[tick] = np.linalg.norm(p_prev-p_curr,ord=ord)
    times_anchor = np.cumsum(dists/vel) # [L]
    L_interp     = int(times_anchor[-1]*HZ)
    times_interp = np.linspace(0,times_anchor[-1],L_interp) # [L_interp]
    anchors_interp  = np.zeros((L_interp,D)) # [L_interp x D]
    for d_idx in range(D): # 对每个维度
        anchors_interp[:,d_idx] = np.interp(times_interp,times_anchor,anchors[:,d_idx])
    idxs_anchor = get_idxs_closest_ndarray(times_interp,times_anchor)
    return times_interp,anchors_interp,times_anchor,idxs_anchor


def check_vel_acc_jerk_nd(
        times, # [L]
        traj, # [L x D]
        verbose = True,
        factor  = 1.0,
    ):
    """
        检查 n 维轨迹的速度、加速度和加加速度(jerk)
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
        获取单位向量
    """
    x = np.array(vec)
    len = np.linalg.norm(x)
    if len <= 1e-6:
        return np.array([0,0,1])
    else:
        return x/len    
    
def uv_T_joi(T_joi,joi_fr,joi_to):
    """
        获取两个 JOI 位姿之间的单位向量
    """
    return np_uv(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))

def len_T_joi(T_joi,joi_fr,joi_to):
    """
        获取两个 JOI 位姿之间的长度(距离)
    """
    return np.linalg.norm(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))

def get_consecutive_subarrays(array,min_element=1):
    """
        从数组中获取连续的子数组
    """
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
        在一行中绘制多张图像
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
        将单通道 float 类型的深度图转换为三通道 uint8 类型的灰度图
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
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")

    # 去除多余的空白(空行)
    lines = [line for line in pretty_xml.splitlines() if line.strip()]
    return "\n".join(lines)

class TicTocClass(object):
    """
        计时器(Tic toc)
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
        休眠指定时间
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
    为图像添加标题
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