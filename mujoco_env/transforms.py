import numpy as np

def t2pr(T):
    """
        从变换矩阵 T 中分解出位置 p 和旋转矩阵 R
    """
    p = T[:3,3]
    R = T[:3,:3]
    return p,R

def t2p(T):
    """
        从变换矩阵 T 中提取位置 p
    """
    p = T[:3,3]
    return p

def t2r(T):
    """
        从变换矩阵 T 中提取旋转矩阵 R
    """
    R = T[:3,:3]
    return R

def rpy2r(rpy_rad):
    """
        将弧度制的滚转、俯仰、偏航（roll, pitch, yaw）转换为旋转矩阵 R
    """
    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
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
    """
    c1 = np.cos(r0[0]); c2 = np.cos(r0[1]); c3 = np.cos(r0[2])
    s1 = np.sin(r0[0]); s2 = np.sin(r0[1]); s3 = np.sin(r0[2])
    a1 = np.array([[1,0,0],[0,c1,-s1],[0,s1,c1]])
    a2 = np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]])
    a3 = np.array([[c3,-s3,0],[s3,c3,0],[0,0,1]])
    a_list = [a1,a2,a3]
    a = np.matmul(np.matmul(a_list[order[0]],a_list[order[1]]),a_list[order[2]])
    assert a.shape == (3,3)
    return a

def r2rpy(R,unit='rad'):
    """
        将旋转矩阵转换为弧度制的滚转、俯仰、偏航（roll, pitch, yaw）
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
    """
    p0 = p.ravel() # 展平
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def r2w(R):
    """
        将旋转矩阵 R 转换为角速度向量 ω
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def meters2xyz(depth_img,cam_matrix):
    """
        将缩放后的深度图转换为点云
    """
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]
    
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)
    
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # 注意 y_e 的顺序是反向的！
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3]
    return xyz_img # [H x W x 3]

def get_rotation_matrix_from_two_points(p_fr,p_to):
    """
        由两个点计算旋转矩阵
    """
    p_a  = np.copy(np.array([1e-10,-1e-10,1.0]))
    if np.linalg.norm(p_to-p_fr) < 1e-8: # 若两点距离过近
        return np.eye(3)
    p_b  = (p_to-p_fr)/np.linalg.norm(p_to-p_fr)
    v    = np.cross(p_a,p_b)
    S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if np.linalg.norm(v) == 0:
        R = np.eye(3,3)
    else:
        R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))
    return R

def skew(x):
    """
        构造反对称矩阵
    """
    x_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    return x_hat

def rodrigues(a=np.array([1,0,0]),q_rad=0.0):
    """
        基于罗德里格斯公式由角速度向量计算旋转矩阵
    """
    a_norm = np.linalg.norm(a)
    if abs(a_norm-1) > 1e-6:
        print ("[rodrigues] a 的范数应为 1.0，而不是 [%.2e]。"%(a_norm))
        return np.eye(3)
    
    a = a / a_norm
    q_rad = q_rad * a_norm
    a_hat = skew(a)
    
    R = np.eye(3) + a_hat*np.sin(q_rad) + a_hat@a_hat*(1-np.cos(q_rad))
    return R

def R_yuzf2zuxf(R):
    """
        将（Y 轴朝上、Z 轴朝前，例如 CMU-MoCap）坐标系下的旋转矩阵 R 转换为（Z 轴朝上、X 轴朝前）坐标系
    """
    R_offset = rpy2r(np.radians([-90,0,-90]))
    return R_offset@R

def T_yuzf2zuxf(T):
    """
        将（Y 轴朝上、Z 轴朝前，例如 CMU-MoCap）坐标系下的变换矩阵 T 转换为（Z 轴朝上、X 轴朝前）坐标系
    """
    p,R = t2pr(T)
    T = pr2t(p=p,R=R_yuzf2zuxf(R))
    return T

def quat2r(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
def align_z_axis(R):
    """
        对齐旋转矩阵 R 的 z 轴
    """
    q = r2quat(R)
    z_axis = R[:, 2]
    
    # 计算旋转轴与旋转角
    rotation_axis = np.cross(z_axis, [0, 0, 1])
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-15:  # z_axis 已经是 [0,0,1] 或 [0,0,-1]
        if z_axis[2] < 0:  # [0,0,-1] 的情形
            return R @ quat2r([0, 1, 0, 0])  # 绕 x 轴旋转 180 度
        else:
            return R
    
    rotation_axis /= rotation_axis_norm
    cos_theta = np.dot(z_axis, [0, 0, 1])
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # 计算旋转四元数
    q_rot = np.array([np.cos(theta/2)] + list(np.sin(theta/2) * rotation_axis))

    # 应用旋转
    q_result = np.array([
        q_rot[0]*q[0] - q_rot[1]*q[1] - q_rot[2]*q[2] - q_rot[3]*q[3],
        q_rot[0]*q[1] + q_rot[1]*q[0] + q_rot[2]*q[3] - q_rot[3]*q[2],
        q_rot[0]*q[2] - q_rot[1]*q[3] + q_rot[2]*q[0] + q_rot[3]*q[1],
        q_rot[0]*q[3] + q_rot[1]*q[2] - q_rot[2]*q[1] + q_rot[3]*q[0]
    ])
    
    return quat2r(q_result)