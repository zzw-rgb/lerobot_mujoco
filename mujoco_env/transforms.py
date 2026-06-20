"""
本文件：坐标与旋转变换工具箱（纯 NumPy 实现）
================================================

它是机械臂仿真里最常用的"数学小工具集合"，负责在不同的描述方式之间互相转换。
读这个文件前，先建立几个最基础的直观概念（完全不懂线性代数也能看懂）：

1) 位置 p：物体在空间里的坐标，就是 3 个数 (x, y, z)，形状记作 (3,)。

2) 旋转矩阵 R：描述"物体朝向哪个方向"的一张 3x3 表格（形状 (3,3)）。
   你可以把它理解为：物体自己的 x/y/z 三根轴，分别指向世界里的哪个方向。
   它只管"转向"，不管"在哪"。

3) 变换矩阵 T（也叫齐次变换矩阵）：把"朝向"和"位置"打包在一起的 4x4 表格（形状 (4,4)）。
   它的结构是：
       [ R(3x3)  p(3x1) ]
       [ 0  0  0    1    ]
   左上角 3x3 是旋转 R，右上角 3x1 是位置 p，最后一行固定是 [0,0,0,1]。
   一个 T 就能完整描述"一个物体在空间中的位置 + 朝向（即位姿 pose）"。

4) 欧拉角 rpy：用 3 个角度（roll 滚转 / pitch 俯仰 / yaw 偏航）来描述朝向，
   像飞机的"翻滚、抬头、转弯"。直观好懂，但有"万向锁"等坑。

5) 四元数 quaternion：用 4 个数 (w, x, y, z) 来表示旋转。
   它比欧拉角更稳定、能避免万向锁，是工程里表示旋转的常用方式。

下面每个函数就是在 p / R / T / rpy / 四元数 / 点云 等之间做转换。
"""
import numpy as np

def t2pr(T):
    """
        从变换矩阵 T 中分解出位置 p 和旋转矩阵 R
        （T 是 4x4 矩阵；本函数把它"拆开"成位置和朝向两部分）

        输入：
            T —— (4,4) 齐次变换矩阵
        返回：
            p —— (3,) 位置坐标 (x, y, z)
            R —— (3,3) 旋转矩阵（朝向）
    """
    p = T[:3,3]   # 取 T 的第 4 列的前 3 行 = 位置 p
    R = T[:3,:3]  # 取 T 的左上角 3x3 = 旋转矩阵 R
    return p,R

def t2p(T):
    """
        从变换矩阵 T 中提取位置 p
        （只要位置，不要朝向）

        输入：T —— (4,4) 变换矩阵
        返回：p —— (3,) 位置 (x, y, z)
    """
    p = T[:3,3]  # T 第 4 列前 3 行就是位置
    return p

def t2r(T):
    """
        从变换矩阵 T 中提取旋转矩阵 R
        （只要朝向，不要位置）

        输入：T —— (4,4) 变换矩阵
        返回：R —— (3,3) 旋转矩阵
    """
    R = T[:3,:3]  # T 左上角 3x3 就是旋转矩阵
    return R

def rpy2r(rpy_rad):
    """
        将弧度制的滚转、俯仰、偏航（roll, pitch, yaw）转换为旋转矩阵 R
        （把"3 个角度"这种好懂的朝向描述，变成机器人学里通用的 3x3 旋转矩阵）

        输入：
            rpy_rad —— (3,) 三个角度 [roll, pitch, yaw]，单位是弧度（不是度）
                       roll  绕 x 轴转（侧翻），pitch 绕 y 轴转（俯仰），yaw 绕 z 轴转（转头）
        返回：
            R —— (3,3) 旋转矩阵

        原理：分别绕 x、y、z 轴转三次，再把这三次旋转"乘"在一起。
              下面的大矩阵就是这三次旋转相乘后展开、化简得到的固定公式。
    """
    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
    # C 表示 cos（余弦），S 表示 sin（正弦）；phi/the/psi 分别对应 roll/pitch/yaw
    Cphi  = np.cos(roll)
    Sphi  = np.sin(roll)
    Cthe  = np.cos(pitch)
    Sthe  = np.sin(pitch)
    Cpsi  = np.cos(yaw)
    Spsi  = np.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def rpy2r_order(r0, order=[0,1,2]):
    """
        将弧度制的滚转、俯仰、偏航（roll, pitch, yaw）按指定顺序转换为旋转矩阵 R
        （和 rpy2r 类似，但允许你自己决定"先绕哪根轴、后绕哪根轴"的顺序）

        为什么顺序重要？因为旋转不像加法可以随便交换：先低头再左转，
        和先左转再低头，最终朝向是不一样的。所以转动顺序会影响结果。

        输入：
            r0    —— (3,) 三个角度 [a, b, c]，单位弧度
            order —— 列表，指定三次旋转分别用哪根轴。0=绕x轴, 1=绕y轴, 2=绕z轴
                     默认 [0,1,2] 表示先绕 x、再绕 y、最后绕 z
        返回：
            a —— (3,3) 旋转矩阵
    """
    c1 = np.cos(r0[0]); c2 = np.cos(r0[1]); c3 = np.cos(r0[2])  # 三个角的余弦
    s1 = np.sin(r0[0]); s2 = np.sin(r0[1]); s3 = np.sin(r0[2])  # 三个角的正弦
    a1 = np.array([[1,0,0],[0,c1,-s1],[0,s1,c1]])  # 绕 x 轴转 r0[0] 的基本旋转矩阵
    a2 = np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]])  # 绕 y 轴转 r0[1] 的基本旋转矩阵
    a3 = np.array([[c3,-s3,0],[s3,c3,0],[0,0,1]])  # 绕 z 轴转 r0[2] 的基本旋转矩阵
    a_list = [a1,a2,a3]
    # 按 order 指定的顺序，把三个基本旋转矩阵相乘（矩阵相乘 = 把多次旋转合成一次）
    a = np.matmul(np.matmul(a_list[order[0]],a_list[order[1]]),a_list[order[2]])
    assert a.shape == (3,3)  # 确认结果确实是 3x3
    return a

def r2rpy(R,unit='rad'):
    """
        将旋转矩阵转换为滚转、俯仰、偏航（roll, pitch, yaw）
        （rpy2r 的逆操作：从 3x3 旋转矩阵反推出 3 个角度，方便人类阅读）

        输入：
            R    —— (3,3) 旋转矩阵
            unit —— 返回角度的单位：'rad' 弧度（默认）或 'deg' 度数
        返回：
            out —— (3,) 三个角度 [roll, pitch, yaw]

        原理：从旋转矩阵特定位置的元素，用反正切函数 arctan2 把角度"解"出来。
              arctan2(y, x) 能根据正负号正确判断角度落在哪个象限。
    """
    roll  = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], (np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out

def r2quat(R):
    """
        将旋转矩阵转换为四元数。说明参见 rotation.py
        (https://gist.github.com/machinaut/dab261b78ac19641e91c6490fb9faa96)

        四元数是用 4 个数 (w, x, y, z) 来表示旋转的方式，比欧拉角更稳定、能避免万向锁。
        本函数把 (3,3) 旋转矩阵 R 转换成长度为 4 的四元数。

        实现细节（看不懂可跳过，不影响使用）：
            它先用 R 的各元素拼出一个 4x4 的对称矩阵 K，
            再求 K 的特征向量；其中最大特征值对应的特征向量，就是我们要的四元数。
            这是一种数值上很稳健的求解方法。
        输入：R —— (3,3) 旋转矩阵（也支持批量，形状 (...,3,3)）
        返回：q —— (4,) 四元数 [w, x, y, z]（批量时形状 (...,4)）
    """
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # 仅填充对称矩阵的下半部分
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: 对此进行向量化 —— 可能会更快
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # 使用厄米（Hermitian）特征向量与特征值以提升速度
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # 选取最大的特征向量，并重排为 w,x,y,z 顺序的四元数
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # 优先选取 w 为正的四元数
        # （q * -1 对应与 q 相同的旋转）
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q

def pr2t(p,R):
    """
        将位姿转换为变换矩阵
        （t2pr 的逆操作：把"位置 p + 朝向 R"重新拼装成一个 4x4 变换矩阵 T）

        输入：
            p —— (3,) 位置 (x, y, z)
            R —— (3,3) 旋转矩阵
        返回：
            T —— (4,4) 变换矩阵，结构为 [[R, p],[0,0,0,1]]
    """
    p0 = p.ravel() # 展平：确保 p 是一维的 (3,)，避免形状不对
    # np.block 像"拼积木"一样把几小块拼成大矩阵：
    #   上排 = [R(3x3)  p(3x1)]，下排 = [0 0 0   1]
    T = np.block([
        [R, p0[:, np.newaxis]],   # p0[:, np.newaxis] 把 (3,) 变成竖着的 (3,1) 列向量
        [np.zeros(3), 1]          # 最后一行固定 [0,0,0,1]
    ])
    return T

def r2w(R):
    """
        将旋转矩阵 R 转换为角速度向量 ω（也叫"轴角"表示）

        直观理解：任何旋转都可以看作"绕某一根轴、转某个角度"。
        本函数把这个旋转浓缩成一个 3 维向量 w：
            - w 的方向 = 那根旋转轴的方向
            - w 的长度 = 旋转的角度（弧度）
        这种表示在计算机器人运动、误差时很方便。

        输入：R —— (3,3) 旋转矩阵
        返回：w —— (3,) 轴角向量
    """
    # 取 R 的"反对称部分"，它编码了旋转轴的方向（这一步是固定数学套路）
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]],
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)  # el 的长度，反映旋转角度的大小
    if norm_el > 1e-10:
        # 一般情形：np.trace(R) 是 R 对角线之和，用它和 el 一起算出旋转角，再乘回轴方向
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        # 特例：几乎没有旋转（R 接近单位矩阵），角速度为 0
        w = np.array([[0, 0, 0]]).T
    else:
        # 特例：旋转角接近 180 度，上面的公式会失效，这里用专门公式处理
        w = np.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()  # 拉平成一维 (3,) 返回

def meters2xyz(depth_img,cam_matrix):
    """
        将深度图转换为点云

        深度图：相机拍的一张图，每个像素存的不是颜色，而是"这个点离相机多远"（米）。
        点云：把图上每个像素，根据它的深度和像素位置，还原成空间中真实的 3D 坐标点 (x,y,z)。
        本函数就是做这个"2D 深度图 -> 3D 点"的反投影。

        输入：
            depth_img  —— (H, W) 深度图，每个元素是该像素的深度（米）
            cam_matrix —— (3,3) 相机内参矩阵，描述相机的焦距和光心位置
        返回：
            xyz_img —— (H, W, 3) 点云，每个像素对应一个 3D 坐标
    """
    # 从相机内参矩阵里取出 4 个关键参数：
    fx = cam_matrix[0][0]  # x 方向焦距
    cx = cam_matrix[0][2]  # 光心（图像中心）横坐标
    fy = cam_matrix[1][1]  # y 方向焦距
    cy = cam_matrix[1][2]  # 光心纵坐标

    height = depth_img.shape[0]
    width = depth_img.shape[1]
    # 生成每个像素的行、列网格坐标，indices[...,0]=行号，indices[...,1]=列号
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)

    # 利用针孔相机模型反投影：已知像素位置和深度，求它在相机坐标系下的真实 3D 坐标
    z_e = depth_img                              # 深度直接作为 z（前方距离）
    x_e = (indices[..., 1] - cx) * z_e / fx      # 由列号 - 光心，乘深度除焦距，得到水平坐标
    y_e = (indices[..., 0] - cy) * z_e / fy      # 由行号 - 光心，得到竖直坐标

    # 注意 y_e 的顺序是反向的！（这里调整坐标轴朝向，得到 [前, 左, 上] 这样的约定）
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3]
    return xyz_img # [H x W x 3]

def get_rotation_matrix_from_two_points(p_fr,p_to):
    """
        由两个点计算旋转矩阵
        用途：想让某个物体（默认它原本"指向 z 轴方向"）转得正好从 p_fr 指向 p_to，
              本函数算出实现这个朝向所需的旋转矩阵。比如让箭头/夹爪对准目标点。

        输入：
            p_fr —— (3,) 起点坐标
            p_to —— (3,) 终点坐标
        返回：
            R —— (3,3) 旋转矩阵；它能把 z 轴方向转到 (p_to - p_fr) 的方向
    """
    p_a  = np.copy(np.array([1e-10,-1e-10,1.0]))   # 参考起始方向，约等于 z 轴 [0,0,1]（加微小扰动避免数值问题）
    if np.linalg.norm(p_to-p_fr) < 1e-8: # 若两点距离过近
        return np.eye(3)                 # 没有明确方向，直接返回单位矩阵（不旋转）
    p_b  = (p_to-p_fr)/np.linalg.norm(p_to-p_fr)   # 目标方向，归一化成长度为 1 的单位向量
    v    = np.cross(p_a,p_b)                        # 两个方向的叉乘 = 旋转轴
    S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])  # v 的反对称矩阵（罗德里格斯公式需要）
    if np.linalg.norm(v) == 0:
        R = np.eye(3,3)   # 两方向已平行，无需旋转
    else:
        # 罗德里格斯旋转公式：由旋转轴和夹角直接构造出旋转矩阵
        R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))
    return R

def skew(x):
    """
        构造反对称矩阵（也叫"斜对称矩阵"）

        作用：把一个 3 维向量 x 变成一个特殊的 3x3 矩阵 x_hat。
        它的妙处是：用矩阵乘法 x_hat @ y，效果等同于叉乘 x × y。
        这是旋转、角速度相关公式里反复出现的一个数学工具。

        输入：x —— (3,) 向量
        返回：x_hat —— (3,3) 反对称矩阵（满足 x_hat 的转置 = -x_hat）
    """
    x_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    return x_hat

def rodrigues(a=np.array([1,0,0]),q_rad=0.0):
    """
        基于罗德里格斯公式由"轴 + 角"计算旋转矩阵
        （和 r2w 相反：r2w 是矩阵->轴角，这里是轴角->矩阵）

        直观：告诉它"绕哪根轴 a，转多少角度 q_rad"，它就还原出对应的旋转矩阵。

        输入：
            a     —— (3,) 旋转轴方向，必须是单位向量（长度=1）
            q_rad —— 旋转角度，单位弧度
        返回：
            R —— (3,3) 旋转矩阵
    """
    a_norm = np.linalg.norm(a)  # 计算轴向量的长度，应当为 1
    if abs(a_norm-1) > 1e-6:
        # 若传入的轴不是单位向量，给出提示并返回"不旋转"
        print ("[rodrigues] a 的范数应为 1.0，而不是 [%.2e]。"%(a_norm))
        return np.eye(3)

    a = a / a_norm          # 再次归一化，确保是单位向量
    q_rad = q_rad * a_norm
    a_hat = skew(a)         # 把轴向量变成反对称矩阵，供公式使用

    # 罗德里格斯公式：R = I + sin(θ)·[a]× + (1-cos(θ))·[a]×²
    R = np.eye(3) + a_hat*np.sin(q_rad) + a_hat@a_hat*(1-np.cos(q_rad))
    return R

def R_yuzf2zuxf(R):
    """
        把"Y 轴朝上、Z 轴朝前"坐标系下的旋转矩阵 R，换算到"Z 轴朝上、X 轴朝前"坐标系

        背景：不同软件/数据集对"哪个轴朝上、哪个轴朝前"的约定不一样。
              例如 CMU-MoCap 动作捕捉数据用 Y 轴朝上，而机器人里常用 Z 轴朝上。
              直接混用会导致姿态错乱，所以需要这个转换。

        输入：R —— (3,3) 原坐标系下的旋转矩阵
        返回：转换到新坐标系约定后的 (3,3) 旋转矩阵
    """
    R_offset = rpy2r(np.radians([-90,0,-90]))  # 两套坐标系之间固定的"角度补偿"
    return R_offset@R                          # 左乘补偿矩阵完成坐标系切换

def T_yuzf2zuxf(T):
    """
        把"Y 轴朝上、Z 轴朝前"坐标系下的变换矩阵 T，换算到"Z 轴朝上、X 轴朝前"坐标系
        （和 R_yuzf2zuxf 同理，只是处理的是完整的 4x4 变换矩阵）

        输入：T —— (4,4) 原坐标系下的变换矩阵
        返回：转换后的 (4,4) 变换矩阵（位置 p 保持不变，只换算朝向 R）
    """
    p,R = t2pr(T)                            # 先拆出位置和朝向
    T = pr2t(p=p,R=R_yuzf2zuxf(R))           # 只对朝向 R 做坐标系转换，再拼回 T
    return T

def quat2r(q):
    """
        将四元数转换为旋转矩阵（r2quat 的逆操作）

        输入：q —— (4,) 四元数 [w, x, y, z]
        返回：(3,3) 旋转矩阵

        下面这个 3x3 矩阵就是由四元数四个分量代入固定公式展开得到的。
    """
    w, x, y, z = q  # 把四元数拆成 4 个分量
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
def align_z_axis(R):
    """
        对齐旋转矩阵 R 的 z 轴，使其指向世界的正上方 [0,0,1]

        作用：在保持物体大体朝向的同时，把它的 z 轴"扳正"到竖直朝上。
              常用于让某个坐标系的上方与世界上方对齐。

        输入：R —— (3,3) 旋转矩阵
        返回：调整后的 (3,3) 旋转矩阵，其 z 轴朝上
    """
    q = r2quat(R)        # 把当前旋转转成四元数，便于后面做四元数乘法
    z_axis = R[:, 2]     # R 的第 3 列就是物体当前的 z 轴方向

    # 计算旋转轴与旋转角：把当前 z 轴转到 [0,0,1] 需要绕哪根轴、转多少
    rotation_axis = np.cross(z_axis, [0, 0, 1])      # 叉乘得到旋转轴
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-15:  # z_axis 已经是 [0,0,1] 或 [0,0,-1]
        if z_axis[2] < 0:  # [0,0,-1] 的情形（正好朝下）
            return R @ quat2r([0, 1, 0, 0])  # 绕 x 轴旋转 180 度，把它翻上来
        else:
            return R                          # 已经朝上，无需调整

    rotation_axis /= rotation_axis_norm                # 旋转轴归一化为单位向量
    cos_theta = np.dot(z_axis, [0, 0, 1])              # 两向量点乘 = 夹角余弦
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))   # 反余弦求夹角（clip 防止浮点误差越界）

    # 计算"把 z 轴扳正"这一旋转对应的四元数 [w, x, y, z]
    q_rot = np.array([np.cos(theta/2)] + list(np.sin(theta/2) * rotation_axis))

    # 应用旋转：用四元数乘法 q_rot * q，把这次扳正叠加到原旋转上
    q_result = np.array([
        q_rot[0]*q[0] - q_rot[1]*q[1] - q_rot[2]*q[2] - q_rot[3]*q[3],
        q_rot[0]*q[1] + q_rot[1]*q[0] + q_rot[2]*q[3] - q_rot[3]*q[2],
        q_rot[0]*q[2] - q_rot[1]*q[3] + q_rot[2]*q[0] + q_rot[3]*q[1],
        q_rot[0]*q[3] + q_rot[1]*q[2] - q_rot[2]*q[1] + q_rot[3]*q[0]
    ])   # 以上四行是四元数相乘的标准展开公式，结果是合成后的新旋转四元数

    return quat2r(q_result)  # 把合成后的四元数再转回 (3,3) 旋转矩阵返回