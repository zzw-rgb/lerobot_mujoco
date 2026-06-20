# ============================================================================
# 这个文件是整个 MuJoCo 机械臂仿真教程的“地基”。
# 它把“用 MuJoCo 写仿真”时那些零碎、底层的操作，封装成一套统一、好用的接口。
#
# 几个最基础的概念（看完后面会反复用到）：
#   - MuJoCo：一个物理仿真引擎，可以模拟机器人/物体在虚拟世界里的运动、碰撞、重力等。
#   - model：场景/机器人的“静态定义”，从 XML 文件解析而来。它描述了有哪些刚体、关节、
#            几何体、执行器、相机、传感器，以及它们的尺寸、初始位置等“不随时间变化的”信息。
#   - data：当前“这一刻”的“动态状态”，比如每个关节现在转到了什么角度、速度多少、
#            受到多大的力等。仿真每推进一步，data 都会更新，而 model 一般保持不变。
#   - body（刚体）：构成机器人/场景的基本积木，比如机械臂的一节连杆。
#   - joint（关节）：连接两个刚体、允许它们相对运动的部件，比如可以转动的轴。
#   - geom（几何体）：刚体的“形状外壳”，用来做碰撞检测和显示（盒子、球、圆柱、网格等）。
#   - actuator（执行器）：给关节施加力/力矩的“马达”，控制信号 ctrl 就是发给它们的。
#   - site（标记点）：刚体上的一个“虚拟参考点”，不参与物理，常用来标记末端执行器位置等。
#   - 齐次变换矩阵 T：一个 4x4 矩阵，能同时表示“位置 + 朝向”。左上 3x3 是旋转矩阵 R，
#                     右上 3x1 是位置 p。后面经常用 T、R、p 这三个量描述物体的位姿。
# ============================================================================

import os
import time
import mujoco   # MuJoCo 物理仿真引擎的 Python 接口
import copy
import glfw     # 跨平台的窗口/输入库，用来开仿真窗口、接收鼠标键盘事件
import pathlib
import cv2      # OpenCV，做图像处理（缩放、显示等）
import numpy as np
from threading import Lock   # 线程锁，渲染时防止多线程同时改数据出错

# 记录当前 MuJoCo 版本号（拆成数字元组，便于后面按版本做兼容处理）
MUJOCO_VERSION = tuple(map(int,mujoco.__version__.split('.')))

# 从本项目的工具模块里导入一批数学/坐标变换相关的辅助函数：
#   t2p: 从齐次变换矩阵 T 取出位置 p；t2r: 取出旋转矩阵 R；pr2t: 由位置+旋转拼成 T；
#   r2quat/quat2r: 旋转矩阵与四元数互转；r2w: 旋转矩阵转角速度向量；rpy2r: 欧拉角转旋转矩阵；
#   meters2xyz: 深度图(米)转点云坐标；get_rotation_matrix_from_two_points: 由两点求朝向。
from .transforms import (
    t2p,
    t2r,
    pr2t,
    r2quat,
    quat2r,
    r2w,
    rpy2r,
    meters2xyz,
    get_rotation_matrix_from_two_points,
)
from .utils import (
    trim_scale,            # 限幅缩放：把向量按比例缩放到不超过某个最大长度
    compute_view_params,   # 由相机参数算出观察视角的辅助函数
    get_idxs,              # 在一个名字列表里找出目标名字对应的下标
    get_colors,            # 生成一组好看的颜色（画轨迹/多物体时用）
    get_monitor_size,      # 获取显示器分辨率
    TicTocClass,           # 简单的计时器（tic 开始、toc 结束）
)

# ============================================================================
# MinimalCallbacks：最小化的“输入回调”基类。
# 它本身不开窗口，只负责定义“当鼠标/键盘有动作时该怎么处理”的回调函数，
# 并保存一堆交互状态（哪个键被按下、是否双击、是否暂停等）。
# 下面的 MuJoCoMinimalViewer 会继承它，把这些回调挂到真正的窗口上。
# 这样做的好处：把“事件处理逻辑”和“窗口/渲染逻辑”分开，结构更清晰。
# ============================================================================
class MinimalCallbacks:
    def __init__(self, hide_menus):
        # __init__ 里把所有交互相关的状态变量初始化好（这些值会被回调函数不断更新）
        self._gui_lock                   = Lock()
        self._button_left_pressed        = False
        self._button_right_pressed       = False
        self._left_double_click_pressed  = False
        self._right_double_click_pressed = False
        self._last_left_click_time       = None
        self._last_right_click_time      = None
        self._last_mouse_x               = 0
        self._last_mouse_y               = 0
        self._paused                     = False
        self._render_every_frame         = True
        self._time_per_render            = 1/60.0
        self._run_speed                  = 1.0
        self._loop_count                 = 0
        self._advance_by_one_step        = False
        # 键盘
        self._key_pressed                = None
        self._is_key_pressed             = False
        # 键盘缓冲区
        self._key_pressed_set            = set()
        self._key_repeated_set           = set()

    def _key_callback(self, window, key, scancode, action, mods):
        """
            按键回调
        """

        # 按键状态标志
        is_key_pressed  = (action==glfw.PRESS)
        is_key_released = (action==glfw.RELEASE)
        is_key_repeated = (action==glfw.REPEAT)

        # 添加和移除按键
        if is_key_pressed:
            self._key_pressed_set.add(key)
        if is_key_repeated:
            self._key_repeated_set.add(key)
        if is_key_released:
            # 从按下集合和重复集合中移除（如果存在）
            self._key_pressed_set.discard(key)
            self._key_repeated_set.discard(key)

        # 暂停 / 恢复处理（空格键）
        # if is_key_pressed and (key==glfw.KEY_SPACE) and (self._paused is not None):
        #     self._paused = not self._paused

        # 退出（ESC 键）
        if (key==glfw.KEY_ESCAPE):
            glfw.set_window_should_close(self.window, True)

        # 保存按下的键（遗留代码）
        self._key_pressed    = key
        self._is_key_pressed = True

        # 返回
        return

    def _cursor_pos_callback(self, window, xpos, ypos):
        """
            鼠标移动回调：当鼠标在窗口里移动时被调用。
            作用：按住鼠标拖动来旋转/平移/缩放观察相机（或拖动物体施加扰动）。
            xpos, ypos 是鼠标当前像素坐标。
        """
        # 如果没有按住左键或右键，就什么都不做（只有按住拖动才需要响应）
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        # 是否同时按住 Shift 键（按住 Shift 会切换“水平/垂直”这两种操作模式）
        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        # 根据按下的是哪个键 + 是否按 Shift，决定这次拖动要做哪种操作：
        #   右键 -> 平移相机（水平/垂直）；左键 -> 旋转相机（水平/垂直）
        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        # 计算鼠标这一帧相对上一帧移动了多少（dx, dy），作为相机移动量
        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        # 加锁，避免渲染线程和这里同时改场景数据
        with self._gui_lock:
            # 如果当前正在“拖动物体施加扰动”，就移动扰动；否则移动观察相机
            if self.pert.active:
                mujoco.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert)
            else:
                mujoco.mjv_moveCamera(
                    self.model,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.cam)

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        """
            鼠标按键回调：鼠标按下/松开时被调用。
            主要做三件事：1) 记录左右键是否处于“按下”状态；
            2) 检测是否发生“双击”（用两次点击的时间差判断）；
            3) 处理“按住 Ctrl + 拖动”给物体施加力的扰动操作。
        """
        # 记录左/右键当前是否被按下（按下=True，松开=False）
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        # 记录按下时鼠标的位置，作为后续拖动的参考起点
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # 检测左键或右键双击
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        time_now = glfw.get_time()

        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = glfw.get_time()

            # 两次左键点击间隔在 0.01~0.3 秒之间，就认定为一次“双击”
            time_diff = (time_now - self._last_left_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = time_now

        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = glfw.get_time()

            time_diff = (time_now - self._last_right_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._right_double_click_pressed = True
            self._last_right_click_time = time_now

        # 设置扰动：按住 Ctrl 键时，可以用鼠标“抓住”物体并拖动施加外力
        key = mods == glfw.MOD_CONTROL
        newperturb = 0
        if key and self.pert.select > 0:
            # 右键：平移，左键：旋转
            if self._button_right_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE

            # 扰动开始时：重置参考量
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(
                    self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb
        # 3D 释放
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        """
            鼠标滚轮回调：滚动滚轮时拉近/拉远观察相机（缩放视图）。
            y_offset 是滚轮滚动量，乘以 -0.05 作为缩放步长。
        """
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)

# ============================================================================
# MuJoCoMinimalViewer：一个“最小化的可视化窗口”。
# 它负责真正开一个 GLFW 窗口，把 MuJoCo 的场景画出来，并接收鼠标键盘交互
# （旋转/平移/缩放视角、拖动物体、显示文字和小图等）。
# 它继承自上面的 MinimalCallbacks，复用那套输入回调逻辑。
# 主类 MuJoCoParserClass 内部会创建并使用这个 viewer 来显示画面。
# ============================================================================
class MuJoCoMinimalViewer(MinimalCallbacks):
    def __init__(
            self,
            model,            # MuJoCo 模型（场景静态定义）
            data,             # MuJoCo 数据（当前动态状态）
            mode              = 'window',   # 渲染模式，目前只支持开窗口 'window'
            title             = "MuJoCo Minimal Viewer",
            width             = None,
            height            = None,
            hide_menus        = True,
            maxgeom           = 10000,
            n_fig             = 1,
            perturbation      = True,
            use_rgb_overlay   = True,
            loc_rgb_overlay   = 'top right',
        ):
        # 先调用父类构造函数，初始化那一堆交互状态变量
        super().__init__(hide_menus)

        # 保存模型与数据的引用（注意是引用，外面更新 data，这里看到的也会同步更新）
        self.model = model
        self.data = data
        self.render_mode = mode
        if self.render_mode not in ['window']:
            raise NotImplementedError(
                "Invalid mode. Only 'window' is supported.")

        # 运行期间保持为 True
        self.is_alive = True

        self.CONFIG_PATH = pathlib.Path.joinpath(
            pathlib.Path.home(), ".config/mujoco_viewer/config.yaml")

        # glfw 初始化（启动窗口系统）
        glfw.init()

        # 如果没指定窗口宽/高，就默认用主显示器的分辨率（全屏大小）
        if not width:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if not height:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
            
        if self.render_mode == 'offscreen':
            glfw.window_hint(glfw.VISIBLE, 0)

        # 创建窗口
        self.maxgeom = maxgeom   # 场景里最多能画多少个几何体（包括我们额外画的箭头/球等）
        self.window = glfw.create_window(
            width, height, title, None, None)
        glfw.make_context_current(self.window)   # 把这个窗口设为当前 OpenGL 绘图目标
        glfw.swap_interval(1)                     # 开启垂直同步，避免画面撕裂

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
            self.window)

        # 仅在 'window' 模式下安装回调
        if self.render_mode == 'window':
            window_width, _ = glfw.get_window_size(self.window)
            self._scale = framebuffer_width * 1.0 / window_width

            # 设置回调
            glfw.set_cursor_pos_callback(
                self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(
                self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)

        # 创建渲染需要的几个核心对象：
        self.vopt = mujoco.MjvOption()    # 可视化选项（显示哪些元素、线框/接触点等开关）
        self.cam  = mujoco.MjvCamera()    # 观察相机（视角的位置、朝向、距离）
        self.scn  = mujoco.MjvScene(self.model, maxgeom=self.maxgeom)  # 场景缓冲区，存放要画的几何体
        self.pert = mujoco.MjvPerturb()   # 扰动对象（鼠标拖物体施加外力时用）

        # 渲染上下文：把场景真正画到 OpenGL 上所需的资源（字体大小用 150%）
        self.ctx = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        width, height = glfw.get_framebuffer_size(self.window)
        
        # 图表：可以在窗口上叠加显示实时曲线图（比如画关节角度随时间变化）
        self.n_fig = n_fig
        self.figs  = []
        for idx in range(self.n_fig):
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            fig.flg_extend = 1
            fig.figurergba = (1,1,1,0)
            fig.panergba   = (1,1,1,0.2)
            self.figs.append(fig)

        # 获取视口
        self.viewport = mujoco.MjrRect(
            0, 0, framebuffer_width, framebuffer_height)

        # 叠加层、标记
        self._overlay = {}
        self._markers = []

        # 用于叠加的 RGB 图像（遗留代码）
        self.use_rgb_overlay = use_rgb_overlay
        self.loc_rgb_overlay = loc_rgb_overlay

        # 用于叠加的 RGB 图像
        self.rgb_overlay_top_right    = None
        self.rgb_overlay_top_left     = None
        self.rgb_overlay_bottom_right = None
        self.rgb_overlay_bottom_left  = None

        # 扰动
        self.perturbation = perturbation

    def add_marker(self, **marker_params):
        """ 往待绘制列表里追加一个“标记”（比如一个球、箭头），下次渲染时画出来 """
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        """
            把一个标记真正写进场景缓冲区 scn。这是底层细节：从空闲槽位取一个 geom，
            填入它的类型、位置、朝向、尺寸、颜色等，让 MuJoCo 渲染时把它画出来。
        """
        # 场景里能画的几何体数量有上限，超了就报错
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(
                'Ran out of geoms. maxgeom: %d' %
                self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]
        # 默认值。
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        # g.matid = -1 # 新增（由 Jihwan，2025-02-27）
        """
            mujoco 3.2 版本不向后兼容
        """
        if MUJOCO_VERSION[1] == 1:
            """
                以下几行在 mujoco 3.2 版本下会报错
            """
            g.texid        = -1
            g.texuniform   = 0
            g.texrepeat[0] = 1
            g.texrepeat[1] = 1
        
        g.emission    = 0
        g.specular    = 0.5
        g.shininess   = 0.5
        g.reflectance = 0
        g.type        = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:]     = np.ones(3) * 0.1
        g.mat[:]      = np.eye(3)
        g.rgba[:]     = np.ones(4)

        for key, value in marker.items():
            # setattr(g, key, value)
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                # assert key == "label", "Only label is a string in mjtGeom."（mjtGeom 中只有 label 是字符串）
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)))
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)
            
        # 几何体数量加一
        self.scn.ngeom += 1
        return

    def apply_perturbations(self):
        """ 把用户用鼠标拖拽产生的扰动力，施加到物体上（先清零外力，再加扰动位姿和力） """
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def read_pixels(self, camid=None, depth=False):
        """
            从渲染缓冲读取像素，得到一张图片（离屏渲染时用；'window' 模式下不支持）。
            camid: 用哪个相机；depth=True 时同时返回深度图。
            返回的图像做了上下翻转（OpenGL 的像素是上下颠倒的）。
        """
        if self.render_mode == 'window':
            raise NotImplementedError(
                "Use 'render()' in 'window' mode.")

        if camid is not None:
            if camid == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camid

        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
            self.window)
        # 更新场景
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scn)
        # 渲染
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)
        shape = glfw.get_framebuffer_size(self.window)

        if depth:
            rgb_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            depth_img = np.zeros((shape[1], shape[0], 1), dtype=np.float32)
            mujoco.mjr_readPixels(rgb_img, depth_img, self.viewport, self.ctx)
            return (np.flipud(rgb_img), np.flipud(depth_img))
        else:
            img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
            return np.flipud(img)

    def add_overlay(
            self,
            loc     = 'bottom left',
            gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT,
            text1   = '',
            text2   = '',
        ):
        """
            添加叠加层：在窗口的某个角落叠加显示一行/两行文字（text1 主文字，text2 副文字）。
            loc 指定显示位置（上/上右/上左/下/下右/下左）。
            loc: ['top','top right','top left','bottom','bottom right','bottom left']
            用法:
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_TOPLEFT,text1='TopLeft')
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_TOP,text1='Top')
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,text1='TopRight')
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,text1='BottomLeft')
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_BOTTOM,text1='Bottom')
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,text1='BottomRight')
        """
        if loc is not None:
            if loc == 'top': gridpos = mujoco.mjtGridPos.mjGRID_TOP
            elif loc == 'top right': gridpos = mujoco.mjtGridPos.mjGRID_TOPRIGHT
            elif loc == 'top left': gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT
            elif loc == 'bottom': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOM
            elif loc == 'bottom right': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
            elif loc == 'bottom left': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
            
        if gridpos not in self._overlay:
            self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1
            self._overlay[gridpos][1] += text2    
        else:
            self._overlay[gridpos][0] += "\n" + text1
            self._overlay[gridpos][1] += "\n" + text2    
        # self._overlay[gridpos][0] += text1 + "\n"
        # self._overlay[gridpos][1] += text2 + "\n"
        
    def _create_overlay(self):
        """
            叠加层项目
        """
        topleft     = mujoco.mjtGridPos.mjGRID_TOPLEFT
        topright    = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft  = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
        
        # self.add_overlay(
        #     gridpos = topleft,
        #     text1   = "A",
        #     text2   = "B",
        # )
    
    def add_line(
            self,
            fig_idx    = 0,
            line_idx   = 0,
            xdata      = np.linspace(0,1,mujoco.mjMAXLINEPNT),
            ydata      = np.zeros(mujoco.mjMAXLINEPNT),
            linergb    = (0,0,1),
            linename   = 'Line Name',
            figurergba = (1,1,1,0),
            panergba   = (1,1,1,0.2),
        ):
        """
            向内部图表添加曲线
            用法:
                xdata = np.linspace(start=0.0,stop=10.0,num=100)
                ydata = np.sin(xdata)
                env.viewer.add_line(
                    fig_idx=0,line_idx=0,xdata=xdata,ydata=ydata,linergb=(1,0,0),linename='Line 1')
                xdata = np.linspace(start=0.0,stop=10.0,num=100)
                ydata = np.cos(xdata)
                env.viewer.add_line(
                    fig_idx=0,line_idx=1,xdata=xdata,ydata=ydata,linergb=(0,0,1),linename='Line 2')
        """
        fig = self.figs[fig_idx]
        fig.figurergba  = figurergba
        fig.panergba    = panergba
        L = len(xdata) # 此值不能超过 'mujoco.mjMAXLINEPNT'
        for i in range(L):
            fig.linedata[line_idx][2*i]   = xdata[i]
            fig.linedata[line_idx][2*i+1] = ydata[i]
        fig.linergb[line_idx]  = linergb
        fig.linename[line_idx] = linename
        fig.linepnt[line_idx]  = L
        
    def add_rgb_overlay(self,rgb_img_raw,fix_ratio=False):
        """
            设置要叠加显示的 RGB 小图（比如把腕部相机看到的画面贴到窗口角落）。
            会把图缩放到窗口的 1/4 大小；fix_ratio=True 时保持原图宽高比并居中、四周补黑边。
        """
        width,height = glfw.get_framebuffer_size(self.window)
        rgb_h,rgb_w = height//4,width//4
        self.rgb_overlay = np.zeros((rgb_h,rgb_w,3))
        (h,w) = self.rgb_overlay.shape[:2]
        if fix_ratio: # 固定宽高比
            h_raw, w_raw = rgb_img_raw.shape[:2]
            # 计算保持宽高比的缩放比例
            scale = min(w / w_raw, h / h_raw)
            new_w = int(w_raw * scale)
            new_h = int(h_raw * scale)
            # 在保持宽高比的同时缩放图像
            resized_img = cv2.resize(rgb_img_raw, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # 创建一个目标尺寸的黑色画布
            padded_img = np.zeros((h, w, 3), dtype=np.uint8)
            # 计算用于居中放置缩放后图像的左上角坐标
            x_offset = (w - new_w) // 2
            y_offset = (h - new_h) // 2
            # 将缩放后的图像放置到画布上
            padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
            rgb_img_rsz = padded_img  # 最终缩放并填充后的图像
        else:
            rgb_img_rsz = cv2.resize(rgb_img_raw,(w,h),interpolation=cv2.INTER_NEAREST)
        self.rgb_overlay = rgb_img_rsz

    def plot_rgb_overlay(self,rgb=None,loc='top right'):
        """
            把一张 RGB 图缩放后保存到指定角落（top/bottom + left/right），下次渲染时贴上去。
            和 add_rgb_overlay 类似，但支持四个角分别独立放图。
            loc:['top right','top left','bottom right','bottom left']
        """
        w_window,h_window = glfw.get_framebuffer_size(self.window)
        h_overlay,w_overlay = h_window//4,w_window//4
        rgb_overlay = np.zeros((h_overlay,w_overlay,3))
        # 固定宽高比
        h_raw,w_raw = rgb.shape[:2]
        # 计算保持宽高比的缩放比例
        scale = min(w_overlay/w_raw,h_overlay/h_raw)
        w_new = int(w_raw*scale)
        h_new = int(h_raw*scale)
        # 缩放
        rgb_resized = cv2.resize(rgb,(w_new,h_new),interpolation=cv2.INTER_NEAREST)
        # 创建一个目标尺寸的黑色画布
        rgb_padded = np.zeros((h_overlay,w_overlay,3),dtype=np.uint8)
        # 计算用于居中放置缩放后图像的左上角坐标
        x_offset = (w_overlay-w_new) // 2
        y_offset = (h_overlay-h_new) // 2
        # 将缩放后的图像放置到画布上
        rgb_padded[y_offset:y_offset+h_new,x_offset:x_offset+w_new] = rgb_resized
        # 保存 RGB 叠加图像
        if loc=='top right':
            self.rgb_overlay_top_right = rgb_padded
        elif loc=='top left':
            self.rgb_overlay_top_left = rgb_padded
        elif loc=='bottom right':
            self.rgb_overlay_bottom_right = rgb_padded
        elif loc=='bottom left':
            self.rgb_overlay_bottom_left = rgb_padded
        else:
            print ("RGB 叠加层位置无效。请使用 'top right'、'top left'、'bottom right' 或 'bottom left'。")

    def reset_rgb_overlay(self,loc=None):
        """
            清除叠加的 RGB 小图。loc=None 时清除四个角，否则只清除指定角。
            loc:['top right','top left','bottom right','bottom left']
        """
        if loc is None:
            self.rgb_overlay_top_right    = None
            self.rgb_overlay_top_left     = None
            self.rgb_overlay_bottom_right = None
            self.rgb_overlay_bottom_left  = None
        else:
            if loc=='top_right':
                self.rgb_overlay_top_right = None
            if loc=='top left':
                self.rgb_overlay_top_left = None
            if loc=='bottom right':
                self.rgb_overlay_bottom_right = None
            if loc=='bottom left':
                self.rgb_overlay_bottom_left = None
    
    def render(self):
        """
            渲染一帧：把当前 MuJoCo 场景画到窗口里。
            完整流程是“更新场景 -> 渲染几何体 -> 叠加文字/曲线/小图 -> 交换缓冲显示”。
            这是 viewer 最核心的方法，每个仿真步循环里都会被调用来刷新画面。
        """
        # 窗口必须还活着才能渲染
        if not self.is_alive:
            raise Exception(
                "GLFW window does not exist but you tried to render.")
        # 如果用户点了关闭按钮，就关闭窗口并返回
        if glfw.window_should_close(self.window):
            self.close()
            return

        # 内部函数 update：真正完成“更新场景 + 渲染”的所有步骤
        # mjv_updateScene, mjr_render, mjr_overlay
        def update():

            # 填充叠加层项目
            self._create_overlay()

            # 渲染开始
            render_start = time.time()
            width, height = glfw.get_framebuffer_size(self.window)
            self.viewport.width, self.viewport.height = width, height

            with self._gui_lock:
                # 更新场景：根据当前 data（物体位置等）把要画的几何体填进 scn 缓冲区
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)
                # 把我们额外添加的标记（球/箭头等）也加进场景
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # 渲染：把场景缓冲区里的几何体真正画到屏幕上
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)

                # 叠加层项目
                for gridpos, [t1, t2] in self._overlay.items():
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.ctx)
                    
                # 处理图表
                for idx,fig in enumerate(self.figs):
                    width_adjustment = width % 4
                    x = int(3 * width / 4) + width_adjustment
                    y = idx * int(height / 4)
                    viewport = mujoco.MjrRect(
                        x, y, int(width / 4), int(height / 4))
                    # 绘图
                    mujoco.mjr_figure(viewport, fig, self.ctx)

                # 叠加 RGB 图像（遗留代码）
                if self.use_rgb_overlay:
                    rgb_h,rgb_w = height//4,width//4
                    if self.loc_rgb_overlay == 'top right':
                        left   = 3*rgb_w
                        bottom = 3*rgb_h
                    elif self.loc_rgb_overlay == 'top left':
                        left   = 0*rgb_w
                        bottom = 3*rgb_h
                    elif self.loc_rgb_overlay == 'bottom right':
                        left   = 3*rgb_w
                        bottom = 0*rgb_h
                    elif self.loc_rgb_overlay == 'bottom left':
                        left   = 0*rgb_w
                        bottom = 0*rgb_h
                    else:
                        print ("RGB 叠加层位置无效。请使用 'top right'、'top left'、'bottom right' 或 'bottom left'。")
                    self.viewport_rgb_render = mujoco.MjrRect(
                        left   = left,
                        bottom = bottom,
                        width  = rgb_w,
                        height = rgb_h,
                    )
                    mujoco.mjr_drawPixels(
                        rgb      = np.flipud(self.rgb_overlay).flatten(),
                        depth    = None,
                        viewport = self.viewport_rgb_render,
                        con      = self.ctx,
                    )

                # 叠加 RGB 图像
                if self.rgb_overlay_top_right is not None:
                    h_overlay,w_overlay = self.rgb_overlay_top_right.shape[:2]
                    viewport_rgb_top_right = mujoco.MjrRect(
                        left   = 3*w_overlay,
                        bottom = 3*h_overlay,
                        width  = w_overlay,
                        height = h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb      = np.flipud(self.rgb_overlay_top_right).flatten(),
                        depth    = None,
                        viewport = viewport_rgb_top_right,
                        con      = self.ctx,
                    )
                if self.rgb_overlay_top_left is not None:
                    h_overlay,w_overlay = self.rgb_overlay_top_left.shape[:2]
                    viewport_rgb_top_left = mujoco.MjrRect(
                        left   = 0*w_overlay,
                        bottom = 3*h_overlay,
                        width  = w_overlay,
                        height = h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb      = np.flipud(self.rgb_overlay_top_left).flatten(),
                        depth    = None,
                        viewport = viewport_rgb_top_left,
                        con      = self.ctx,
                    )
                if self.rgb_overlay_bottom_right is not None:
                    h_overlay,w_overlay = self.rgb_overlay_bottom_right.shape[:2]
                    viewport_rgb_bottom_right = mujoco.MjrRect(
                        left   = 3*w_overlay,
                        bottom = 0*h_overlay,
                        width  = w_overlay,
                        height = h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb      = np.flipud(self.rgb_overlay_bottom_right).flatten(),
                        depth    = None,
                        viewport = viewport_rgb_bottom_right,
                        con      = self.ctx,
                    )
                if self.rgb_overlay_bottom_left is not None:
                    h_overlay,w_overlay = self.rgb_overlay_bottom_left.shape[:2]
                    viewport_rgb_bottom_left = mujoco.MjrRect(
                        left   = 0*w_overlay,
                        bottom = 0*h_overlay,
                        width  = w_overlay,
                        height = h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb      = np.flipud(self.rgb_overlay_bottom_left).flatten(),
                        depth    = None,
                        viewport = viewport_rgb_bottom_left,
                        con      = self.ctx,
                    )
                
                # 双缓冲：把刚画好的“后台缓冲”交换到屏幕上显示（避免画面闪烁）
                glfw.swap_buffers(self.window)
            glfw.poll_events()   # 处理累积的鼠标/键盘事件
            # 用指数平滑估计每帧渲染耗时（用于控制仿真与渲染速度的同步）
            self._time_per_render = 0.9 * self._time_per_render + \
                0.1 * (time.time() - render_start)

        if self._paused: # 如果已暂停：停在原地反复刷新画面，直到取消暂停或单步前进
            while self._paused:
                update()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            # 未暂停：根据仿真步长与渲染耗时的比例决定这次要刷新几帧
            self._loop_count += self.model.opt.timestep / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:  # 若设置成每帧都渲染，则固定刷新一次
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # 清除标记（标记是“一次性”的，画完就清空，下一帧重新添加）
        self._markers[:] = []

        # 清除叠加层文字
        self._overlay.clear()

        # 施加扰动（这一步是否应该放在 mj_step 之前？）
        if self.perturbation:
            self.apply_perturbations()

    def close(self):
        """ 关闭窗口：标记为已死，释放 GLFW 和渲染上下文资源 """
        self.is_alive = False
        glfw.terminate()
        self.ctx.free()


# ============================================================================
# MuJoCoParserClass：整个项目最核心的类，是“仿真封装地基”。
#
# 它把“用 MuJoCo 写仿真”时一堆零散、底层的操作打包成统一、好用的接口，
# 让上层教程代码可以用简单的方法名（如 env.step()、env.get_p_body('hand')）
# 来加载模型、推进物理、读写位姿、渲染画面、做逆运动学、处理键鼠交互等。
#
# 典型使用流程：
#   env = MuJoCoParserClass(rel_xml_path='...')  # 加载场景，解析模型
#   env.reset()                                  # 复位到初始状态
#   env.init_viewer()                            # 打开可视化窗口
#   while env.is_viewer_alive():                 # 主循环
#       env.step(ctrl=...)                       # 施加控制并推进一步物理
#       env.render()                             # 刷新画面
#
# 名字里的 "Parser（解析器）" 指它会解析 XML 模型文件，建立“名字 -> 索引”的映射表，
# 这样我们就能用人类可读的名字（如 'panda_hand'）去操作物体，而不必记一堆数字编号。
# ============================================================================
class MuJoCoParserClass(object):
    """
        MuJoCo 解析器类
    """
    def __init__(
            self,
            name          = None,         # 给这个仿真环境起的名字（不填则用模型名）
            rel_xml_path  = None,         # XML 模型文件的相对路径（二选一：路径 或 字符串）
            xml_string    = None,         # 直接以字符串形式给出的 XML 内容
            assets        = None,         # 配套资源（网格、贴图等），配合 xml_string 使用
            verbose       = True,         # 是否在初始化时打印模型信息
        ):
        """
            初始化 MuJoCo 解析器
        """
        self.name         = name
        self.rel_xml_path = rel_xml_path
        self.xml_string   = xml_string
        self.assets       = assets
        self.verbose      = verbose

        # 常量
        self.tick              = 0       # 仿真步计数器（每 step 一次加一）
        self.render_tick       = 0       # 渲染帧计数器
        self.use_mujoco_viewer = False   # 当前是否已打开可视化窗口

        # 解析 xml 文件：加载模型、建立各种名字到索引的映射（核心步骤，见 _parse_xml）
        if (self.rel_xml_path is not None) or (self.xml_string is not None):
            self._parse_xml()
        if self.name is None:
            self.name = self.model_name

        # 计时器（Tic-toc）
        self.tt = TicTocClass(name='env:[%s]'%(self.name))

        # 显示器尺寸
        self.monitor_width,self.monitor_height = get_monitor_size()

        # 打印信息
        if self.verbose:
            self.print_info()

        # 重置
        self.reset(step=True)

    def _parse_xml(self):
        """
            解析 XML 模型文件，这是整个类的“开机”步骤，非常关键。做的事情包括：
            1) 从 XML 路径或字符串加载出 model（静态定义）和 data（动态状态）；
            2) 统计场景里有多少刚体 body、关节 joint、几何体 geom、执行器 actuator、
               相机、传感器等，并把它们的“名字”和“整数索引”一一对应起来，
               存成一堆列表/字典（如 self.body_names、self.joint_names ...）。
            为什么要建这些映射？因为 MuJoCo 内部都用整数索引操作，而我们人更习惯用名字，
            有了映射表，后面就能写 env.get_p_body('hand') 这样直观的代码。
        """
        # 方式一：从 XML 文件路径加载
        if self.rel_xml_path is not None:
            self.full_xml_path = os.path.abspath(os.path.join(os.getcwd(),self.rel_xml_path))
            self.model         = mujoco.MjModel.from_xml_path(self.full_xml_path)

        # 方式二：直接从 XML 字符串加载（配合 assets 资源）
        if self.xml_string is not None:
            self.model = mujoco.MjModel.from_xml_string(xml=self.xml_string,assets=self.assets)

        # 解析 xml 模型名称
        parsed_strings = [s for s in self.model.names.split(b'\x00') if s]
        parsed_strings = [s.decode('utf-8') for s in parsed_strings]
        self.model_name = parsed_strings[0]

        self.data             = mujoco.MjData(self.model)   # 创建动态数据容器（当前状态）
        self.dt               = self.model.opt.timestep      # 每一仿真步代表多少“仿真时间”（秒）
        self.HZ               = int(1/self.dt)               # 仿真频率（每秒多少步）

        # 积分器：决定物理方程怎么往前“算”，常见有欧拉法(EULER)和四阶龙格库塔(RK4)
        # 积分器 (https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtintegrator)
        self.integrator       = self.model.opt.integrator
        if self.integrator == mujoco.mjtIntegrator.mjINT_EULER:
            self.integrator_name = 'EULER'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            self.integrator_name = 'RK4'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT:
            self.integrator_name = 'IMPLICIT'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            self.integrator_name = 'IMPLICITFAST'
        else:
            self.integrator_name = 'UNKNOWN'
        
        # 状态空间和动作空间
        # qpos：所有关节的“位置/角度”拼成的大向量；qvel：所有关节的“速度”向量。
        # 注意 nq 和 nv 不一定相等（自由关节用 4 元数表示朝向占 7 个 qpos，但只占 6 个 qvel）。
        self.n_qpos           = self.model.nq # 状态数量
        self.n_qvel           = self.model.nv # 速度数量（切空间维度）
        self.n_qacc           = self.model.nv # 加速度数量（切空间维度）

        # 几何体
        self.n_geom           = self.model.ngeom # 几何体数量
        self.geom_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,geom_idx)
                                 for geom_idx in range(self.model.ngeom)]

        # 网格
        self.n_mesh           = self.model.nmesh # 网格数量
        self.mesh_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_MESH,mesh_idx)
                                 for mesh_idx in range(self.model.nmesh)]

        # 刚体
        self.n_body           = self.model.nbody # 刚体数量
        self.body_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,body_idx)
                                 for body_idx in range(self.n_body)]
        self.body_masses      = self.model.body_mass # (kg)
        self.body_total_mass  = self.body_masses.sum()
        
        self.parent_body_names = []
        for b_idx in range(self.n_body):
            parent_id = self.model.body_parentid[b_idx]
            parent_body_name = self.body_names[parent_id]
            self.parent_body_names.append(parent_body_name)
            
        # 自由度
        self.n_dof            = self.model.nv # 自由度（= 雅可比矩阵的列数）
        self.dof_names        = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_DOF,dof_idx)
                                 for dof_idx in range(self.n_dof)]

        # 关节：下面把关节按类型分类。MuJoCo 关节主要有几种：
        #   FREE 自由关节（6 自由度，物体可自由漂浮，如一个被抓的方块）
        #   HINGE 旋转关节（绕轴转动，机械臂的关节多是这种）
        #   SLIDE 滑动关节（沿轴平移，如夹爪开合）
        self.n_joint          = self.model.njnt # 关节数量
        self.joint_names      = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_idx)
                                 for joint_idx in range(self.n_joint)]
        self.joint_types      = self.model.jnt_type # 关节类型
        self.joint_ranges     = self.model.jnt_range # 关节范围
        self.joint_mins       = self.joint_ranges[:,0]
        self.joint_maxs       = self.joint_ranges[:,1]

        # 自由关节
        self.free_joint_idxs  = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_FREE)[0].astype(np.int32)
        self.free_joint_names = [self.joint_names[joint_idx] for joint_idx in self.free_joint_idxs]
        self.n_free_joint     = len(self.free_joint_idxs)

        # 旋转关节
        self.rev_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.rev_joint_idxs]
        self.n_rev_joint      = len(self.rev_joint_idxs)
        self.rev_joint_mins   = self.joint_ranges[self.rev_joint_idxs,0]
        self.rev_joint_maxs   = self.joint_ranges[self.rev_joint_idxs,1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins
        
        # 滑动关节
        self.pri_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.pri_joint_idxs]
        self.n_pri_joint      = len(self.pri_joint_idxs)
        self.pri_joint_mins   = self.joint_ranges[self.pri_joint_idxs,0]
        self.pri_joint_maxs   = self.joint_ranges[self.pri_joint_idxs,1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins

        # 旋转关节 + 滑动关节信息
        self.n_rev_pri_joint      = self.n_rev_joint + self.n_pri_joint
        self.rev_pri_joint_idxs   = np.concatenate([self.rev_joint_idxs,self.pri_joint_idxs])
        self.rev_pri_joint_names  = self.rev_joint_names + self.pri_joint_names
        self.rev_pri_joint_mins   = np.concatenate([self.rev_joint_mins,self.pri_joint_mins])
        self.rev_pri_joint_maxs   = np.concatenate([self.rev_joint_maxs,self.pri_joint_maxs])
        self.rev_pri_joint_ranges = self.rev_pri_joint_maxs - self.rev_pri_joint_mins
        
        # 控制（执行器）：actuator 就是“马达”，我们发给它控制信号 ctrl，它给关节施加力/力矩。
        # ctrl_ranges 是每个执行器允许的控制值范围，超出会被截断。
        self.n_ctrl           = self.model.nu # 执行器（或控制）数量
        self.ctrl_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx)
                                 for ctrl_idx in range(self.n_ctrl)]
        self.ctrl_ranges      = self.model.actuator_ctrlrange # 控制范围
        self.ctrl_mins        = self.ctrl_ranges[:,0]
        self.ctrl_maxs        = self.ctrl_ranges[:,1]
        self.ctrl_gears       = self.model.actuator_gear[:,0] # 传动比

        # 相机：场景里定义的固定相机（比如俯视相机、腕部相机）。这里给每个相机建一个
        # MjvCamera 观察对象，并记录它的视场角 fovy（决定“看得多宽”）和视口大小。
        self.n_cam            = self.model.ncam
        self.cam_names        = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_CAMERA,cam_idx)
                                 for cam_idx in range(self.n_cam)]
        self.cams             = []
        self.cam_fovs         = []
        self.cam_viewports    = []
        for cam_idx in range(self.n_cam):
            cam_name = self.cam_names[cam_idx]
            cam      = mujoco.MjvCamera()
            cam.fixedcamid = self.model.cam(cam_name).id
            cam.type       = mujoco.mjtCamera.mjCAMERA_FIXED
            cam_fov        = self.model.cam_fovy[cam_idx]
            viewport       = mujoco.MjrRect(0,0,800,600) # SVGA?
            # 追加
            self.cams.append(cam)
            self.cam_fovs.append(cam_fov)
            self.cam_viewports.append(viewport)

        # 与控制（执行器）关联的 qpos 和 qvel 索引
        # 作用：建立“第几个执行器 -> 它控制的关节在 qpos/qvel 大向量里的下标”的对应。
        # 有了它，就能方便地读出“被控关节的当前角度/速度”，做位置/速度控制时很常用。
        """
        # 用法
        self.env.data.qpos[self.env.ctrl_qpos_idxs] # 关节位置
        self.env.data.qvel[self.env.ctrl_qvel_idxs] # 关节速度
        """
        self.ctrl_qpos_idxs = []
        self.ctrl_qpos_names = []
        self.ctrl_qpos_mins = []
        self.ctrl_qpos_maxs = []
        self.ctrl_qvel_idxs = []
        self.ctrl_types = []
        for ctrl_idx in range(self.n_ctrl):
            # 与执行器关联的传动（关节）索引，这里假设只关联了一个关节
            joint_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid[0]
            # 与控制关联的关节位置
            self.ctrl_qpos_idxs.append(self.model.jnt_qposadr[joint_idx])
            self.ctrl_qpos_names.append(self.joint_names[joint_idx])
            self.ctrl_qpos_mins.append(self.joint_ranges[joint_idx,0])
            self.ctrl_qpos_maxs.append(self.joint_ranges[joint_idx,1])
            # 与控制关联的关节速度
            self.ctrl_qvel_idxs.append(self.model.jnt_dofadr[joint_idx])
            # 检查类型
            trntype = self.model.actuator_trntype[ctrl_idx]
            if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                self.ctrl_types.append('JOINT')
            elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
                self.ctrl_types.append('TENDON')
            else:
                self.ctrl_types.append('UNKNOWN(trntype=%d)'%(trntype))
                
        # 传感器：场景里定义的各种传感器（如力/位置/距离传感器），读数存在 data.sensordata
        self.n_sensor         = self.model.nsensor
        self.sensor_names     = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SENSOR,sensor_idx)
                                 for sensor_idx in range(self.n_sensor)]

        # 站点（site）：刚体上的“虚拟参考点”，不参与物理碰撞，常用来标记末端、传感器位置等
        self.n_site           = self.model.nsite
        self.site_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SITE,site_idx)
                                 for site_idx in range(self.n_site)]
        
    def print_info(self):
        """
            打印模型信息
        """
        print ("")
        print ("-----------------------------------------------------------------------------")
        print ("name:[%s] dt:[%.3f] HZ:[%d]"%(self.name,self.dt,self.HZ))
        print (" n_qpos:[%d] n_qvel:[%d] n_qacc:[%d] n_ctrl:[%d]"%(self.n_qpos,self.n_qvel,self.n_qacc,self.n_ctrl))
        print (" integrator:[%s]"%(self.integrator_name))

        print ("")
        print ("n_body:[%d]"%(self.n_body))
        for body_idx,body_name in enumerate(self.body_names):
            body_mass = self.body_masses[body_idx]
            print (" [%d/%d] [%s] mass:[%.2f]kg"%(body_idx,self.n_body,body_name,body_mass))
        print ("body_total_mass:[%.2f]kg"%(self.body_total_mass))
        
        print ("")
        print ("n_geom:[%d]"%(self.n_geom))
        print ("geom_names:%s"%(self.geom_names))

        print ("")
        print ("n_mesh:[%d]"%(self.n_mesh))
        print ("mesh_names:%s"%(self.mesh_names))

        print ("")
        print ("n_joint:[%d]"%(self.n_joint))
        for joint_idx,joint_name in enumerate(self.joint_names):
            print (" [%d/%d] [%s] axis:%s"%
                   (joint_idx,self.n_joint,joint_name,self.model.joint(joint_idx).axis))
        # print ("joint_types:[%s]"%(self.joint_types))
        # print ("joint_ranges:[%s]"%(self.joint_ranges))

        print ("")
        print ("n_dof:[%d] (=雅可比矩阵的行数)"%(self.n_dof))
        for dof_idx,dof_name in enumerate(self.dof_names):
            joint_name= self.joint_names[self.model.dof_jntid[dof_idx]]
            body_name= self.body_names[self.model.dof_bodyid[dof_idx]]
            print (" [%d/%d] [%s] 所属关节:[%s] 刚体:[%s]"%
                   (dof_idx,self.n_dof,dof_name,joint_name,body_name))
        
        print ("\n自由关节信息。n_free_joint:[%d]"%(self.n_free_joint))
        for idx,free_joint_name in enumerate(self.free_joint_names):
            body_name_attached = self.body_names[self.model.joint(self.free_joint_idxs[idx]).bodyid[0]]
            print (" [%d/%d] [%s] body_name_attached:[%s]"%
                   (idx,self.n_free_joint,free_joint_name,body_name_attached))
            
        print ("\n旋转关节信息。n_rev_joint:[%d]"%(self.n_rev_joint))
        for idx,rev_joint_name in enumerate(self.rev_joint_names):
            print (" [%d/%d] [%s] range:[%.3f]~[%.3f]"%
                   (idx,self.n_rev_joint,rev_joint_name,self.rev_joint_mins[idx],self.rev_joint_maxs[idx]))

        print ("\n滑动关节信息。n_pri_joint:[%d]"%(self.n_pri_joint))
        for idx,pri_joint_name in enumerate(self.pri_joint_names):
            print (" [%d/%d] [%s] range:[%.3f]~[%.3f]"%
                   (idx,self.n_pri_joint,pri_joint_name,self.pri_joint_mins[idx],self.pri_joint_maxs[idx]))
            
        print ("\n控制信息。n_ctrl:[%d]"%(self.n_ctrl))
        for idx,ctrl_name in enumerate(self.ctrl_names):
            print (" [%d/%d] [%s] range:[%.3f]~[%.3f] gear:[%.2f] type:[%s]"%
                   (idx,self.n_ctrl,ctrl_name,self.ctrl_mins[idx],self.ctrl_maxs[idx],
                    self.ctrl_gears[idx],self.ctrl_types[idx]))
            
        print ("\n相机信息。n_cam:[%d]"%(self.n_cam))
        for idx,cam_name in enumerate(self.cam_names):
            print (" [%d/%d] [%s] fov:[%.1f]"%
                   (idx,self.n_cam,cam_name,self.cam_fovs[idx]))
            
        print ("")
        print ("n_sensor:[%d]"%(self.n_sensor))
        print ("sensor_names:%s"%(self.sensor_names))
        print ("n_site:[%d]"%(self.n_site))
        print ("site_names:%s"%(self.site_names))
        print ("-----------------------------------------------------------------------------")
        
    def print_body_joint_info(self):
        """
            打印刚体和关节信息（包含更多细节）
        """
        from termcolor import colored
        # 汇总运动链信息
        JOINT_TYPE_MAP = {
            mujoco.mjtJoint.mjJNT_FREE: 'free',
            mujoco.mjtJoint.mjJNT_HINGE: 'revolute',
            mujoco.mjtJoint.mjJNT_SLIDE: 'prismatic',
        }
        for body_idx in range(self.n_body):
            # 解析刚体信息
            body_name = self.body_names[body_idx] # 刚体名称
            body = self.model.body(body_name) # mujoco 刚体对象
            parent_body_name = self.body_names[body.parentid[0]]
            p_body_offset,quat_body_offset = body.pos,body.quat # 刚体偏移量
            T_body_offset = pr2t(p_body_offset,quat2r(quat_body_offset)) # [4x4]
            print ("[%2d/%d] body_name:[%s] parent_body_name:[%s]"%
                (body_idx,self.n_body,colored(body_name,'green'),colored(parent_body_name,'green')))
            print (" body p_offset:[%.2f,%.2f,%.2f] quat_offset:[%.2f,%.2f,%.2f,%.2f]"%
                (p_body_offset[0],p_body_offset[1],p_body_offset[2],
                    quat_body_offset[0],quat_body_offset[1],quat_body_offset[2],quat_body_offset[3]))
            # 解析关节信息
            n_joint = body.jntnum # 关联的关节数量
            if n_joint == 0: # 固定关节
                print (" n_joint:[0] (%s)"%(colored('该刚体没有关节','blue')))
            elif n_joint == 1: # 一个可动关节（旋转或滑动）
                joint = self.model.joint(body.jntadr[0]) # 关联的关节
                joint_name = joint.name
                joint_type = JOINT_TYPE_MAP[joint.type[0]]
                p_joint_offset,joint_axis = joint.pos,joint.axis
                print (" joint_name:[%s] n_joint:%s type:[%s]"%
                    (colored(joint_name,'green'),n_joint,colored('%s'%(joint_type),'green')))
                print (" joint p_offset:[%.2f,%.2f,%.2f] axis:[%.1f,%.1f,%.1f]"%
                    (p_joint_offset[0],p_joint_offset[1],p_joint_offset[2],
                        joint_axis[0],joint_axis[1],joint_axis[2]))
            else: # 复合关节（不支持）
                print (" n_joint:%s (%s)"%
                    (n_joint,colored('复合关节','red')))
            print ("")
        
    def reset(self,step=True):
        """
            复位仿真：把所有关节位置/速度等动态状态恢复到初始值，并把各种计数器/计时器清零。
            相当于“重新开始”，每次开始一段新仿真前通常都会先调用它。
            参数 step：复位后是否立刻推进一步物理，让状态稳定下来。
        """
        time.sleep(1e-3) # 加一点延时？
        mujoco.mj_resetData(self.model,self.data) # 把 data 恢复到模型定义的初始状态

        if step:
            mujoco.mj_step(self.model,self.data)   # 推进一步物理，让状态进入有效初值
            # mujoco.mj_forward(self.model,self.data) # 前向 <= 这一步是否必要？

        # 重置计数器（tick）
        self.tick        = 0
        self.render_tick = 0
        # 重置墙钟时间
        self.init_sim_time    = self.data.time
        self.init_wall_time   = time.time()
        self.accum_wall_time  = 0.0  # 累计墙钟时间
        self.last_wall_update = time.time()
        # 其他
        self.xyz_left_double_click = None
        self.xyz_right_double_click = None
        # 打印
        if self.verbose: print ("env:[%s] 已重置"%(self.name))
        
    def init_viewer(
            self,
            title             = None,
            fullscreen        = False,
            width             = 1400,
            height            = 1000,
            hide_menu         = True,
            fontscale         = mujoco.mjtFontScale.mjFONTSCALE_200.value,
            azimuth           = 170, # None,
            distance          = 5.0, # None,
            elevation         = -20, # None,
            lookat            = [0.01,0.11,0.5], # None,
            transparent       = None,
            contactpoint      = None,
            contactwidth      = None,
            contactheight     = None,
            contactrgba       = None,
            joint             = None,
            jointlength       = None,
            jointwidth        = None,
            jointrgba         = None,
            geomgroup_0       = None, # 地面、天空
            geomgroup_1       = None, # 碰撞体
            geomgroup_2       = None, # 视觉体
            geomgroup_3       = None,
            geomgroup_4       = None,
            geomgroup_5       = None,
            update            = False,
            maxgeom           = 50000,
            perturbation      = True,
            black_sky         = False,
            convex_hull       = None,
            n_fig             = 0,
            use_rgb_overlay   = False,
            loc_rgb_overlay   = 'top right',
            pre_render        = False,
        ):
        """
        使用给定参数初始化 MuJoCo 查看器。

        参数:
            title (str): 查看器窗口标题。
            fullscreen (bool): 是否使用全屏模式。
            width (int): 查看器窗口宽度。
            height (int): 查看器窗口高度。
            hide_menu (bool): 是否隐藏查看器菜单。
            fontscale: 字体缩放因子。
            azimuth (float): 相机初始方位角。
            distance (float): 相机初始距离。
            elevation (float): 相机初始俯仰角。
            lookat (list or np.array): 相机初始注视点位置。
            transparent, contactpoint, contactwidth, contactheight, contactrgba:
                接触点可视化相关参数。
            joint, jointlength, jointwidth, jointrgba:
                关节可视化相关参数。
            geomgroup_0 ~ geomgroup_5: 几何体分组的可见性标志。
            update (bool): 是否立即更新查看器。
            maxgeom (int): 几何体的最大数量。
            perturbation (bool): 是否允许扰动。
            black_sky (bool): 是否渲染黑色天空盒。
            convex_hull: 凸包可视化标志。
            n_fig (int): 用于叠加绘图的图表数量。
            use_rgb_overlay (bool): 是否使用 RGB 叠加图像。
            loc_rgb_overlay (str): 第一个 RGB 叠加图像的位置。
            pre_render (bool): 是否执行一次初始渲染。

        返回:
            None
        """
        # init_viewer：打开可视化窗口。内部会创建一个 MuJoCoMinimalViewer 实例（前面那个类），
        # 然后通过 set_viewer 设置相机视角和各种显示开关（接触点、关节、几何体分组等）。
        self.use_mujoco_viewer = True
        if title is None: title = self.name

        # 全屏（这会覆盖 'width' 和 'height'）
        w_monitor,h_monitor = get_monitor_size()
        if fullscreen:
            width,height = w_monitor,h_monitor
            
        if width <= 1.0 and height <= 1.0:
            width = int(width*w_monitor)
            height = int(height*h_monitor)

        time.sleep(1e-3)
        self.viewer = MuJoCoMinimalViewer(
            self.model,
            self.data,
            mode              = 'window',
            title             = title,
            width             = width,
            height            = height,
            hide_menus        = hide_menu,
            maxgeom           = maxgeom,
            perturbation      = perturbation,
            n_fig             = n_fig,
            use_rgb_overlay   = use_rgb_overlay,
            loc_rgb_overlay   = loc_rgb_overlay,
        )
        self.viewer.ctx = mujoco.MjrContext(self.model,fontscale)

        # 设置查看器
        self.set_viewer(
            azimuth       = azimuth,
            distance      = distance,
            elevation     = elevation,
            lookat        = lookat,
            transparent   = transparent,
            contactpoint  = contactpoint,
            contactwidth  = contactwidth,
            contactheight = contactheight,
            contactrgba   = contactrgba,
            joint         = joint,
            jointlength   = jointlength,
            jointwidth    = jointwidth,
            jointrgba     = jointrgba,
            geomgroup_0   = geomgroup_0,
            geomgroup_1   = geomgroup_1,
            geomgroup_2   = geomgroup_2,
            geomgroup_3   = geomgroup_3,
            geomgroup_4   = geomgroup_4,
            geomgroup_5   = geomgroup_5,
            black_sky     = black_sky,
            convex_hull   = convex_hull,
            update        = update,
        )
        if pre_render: self.render()
        # 打印
        if self.verbose: print ("env:[%s] 初始化查看器"%(self.name))
        
    def set_viewer(
            self,
            azimuth       = None,
            distance      = None,
            elevation     = None,
            lookat        = None,
            transparent   = None,
            contactpoint  = None,
            contactwidth  = None,
            contactheight = None,
            contactrgba   = None,
            joint         = None,
            jointlength   = None,
            jointwidth    = None,
            jointrgba     = None,
            geomgroup_0   = None,
            geomgroup_1   = None,
            geomgroup_2   = None,
            geomgroup_3   = None,
            geomgroup_4   = None,
            geomgroup_5   = None,
            black_sky     = None,
            convex_hull   = None,
            update        = False,
        ):
        """
        设置或更新查看器的相机和可视化参数。

        参数:
            azimuth (float): 相机方位角。
            distance (float): 相机距离。
            elevation (float): 相机俯仰角。
            lookat (list or np.array): 相机注视点位置。
            transparent (bool): 是否将动态几何体设为透明的标志。
            contactpoint (bool): 是否显示接触点的标志。
            contactwidth (float): 接触点宽度。
            contactheight (float): 接触点高度。
            contactrgba (list): 接触点的 RGBA 颜色。
            joint (bool): 是否进行关节可视化的标志。
            jointlength (float): 关节可视化的长度。
            jointwidth (float): 关节可视化的宽度。
            jointrgba (list): 关节的 RGBA 颜色。
            geomgroup_0 ~ geomgroup_5: 几何体分组的可见性标志。
            black_sky (bool): 是否启用/禁用天空盒的标志。
            convex_hull (bool): 是否进行凸包可视化的标志。
            update (bool): 若为 True，则立即执行一次更新。

        返回:
            None
        """
        # 基本查看器设置（方位角、距离、俯仰角和注视点）
        if azimuth is not None: self.viewer.cam.azimuth = azimuth
        if distance is not None: self.viewer.cam.distance = distance
        if elevation is not None: self.viewer.cam.elevation = elevation
        if lookat is not None: self.viewer.cam.lookat = lookat
        # 将动态几何体设得更透明
        if transparent is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
        # 接触点
        if contactpoint is not None: self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contactpoint
        if contactwidth is not None: self.model.vis.scale.contactwidth = contactwidth
        if contactheight is not None: self.model.vis.scale.contactheight = contactheight
        if contactrgba is not None: self.model.vis.rgba.contactpoint = contactrgba
        # 关节
        if joint is not None: self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = joint
        if jointlength is not None: self.model.vis.scale.jointlength = jointlength
        if jointwidth is not None: self.model.vis.scale.jointwidth = jointwidth
        if jointrgba is not None: self.model.vis.rgba.joint = jointrgba
        # 几何体分组
        if geomgroup_0 is not None: self.viewer.vopt.geomgroup[0] = geomgroup_0
        if geomgroup_1 is not None: self.viewer.vopt.geomgroup[1] = geomgroup_1
        if geomgroup_2 is not None: self.viewer.vopt.geomgroup[2] = geomgroup_2
        if geomgroup_3 is not None: self.viewer.vopt.geomgroup[3] = geomgroup_3
        if geomgroup_4 is not None: self.viewer.vopt.geomgroup[4] = geomgroup_4
        if geomgroup_5 is not None: self.viewer.vopt.geomgroup[5] = geomgroup_5
        # 天空盒
        if black_sky is not None: self.viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = not black_sky
        # 凸包
        if convex_hull is not None: self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = convex_hull
        # 通过渲染来更新设置
        if update:
            mujoco.mj_forward(self.model,self.data) 
            mujoco.mjv_updateScene(
                self.model,self.data,self.viewer.vopt,self.viewer.pert,self.viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,self.viewer.scn)
            mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)
            
    def get_viewer_cam_info(self,verbose=False):
        """
        从查看器中获取当前相机参数。

        参数:
            verbose (bool): 若为 True，则打印相机信息。

        返回:
            tuple: 相机的 (azimuth, distance, elevation, lookat)。
        """
        azimuth   = self.viewer.cam.azimuth
        distance  = self.viewer.cam.distance
        elevation = self.viewer.cam.elevation
        lookat    = self.viewer.cam.lookat.copy()
        if verbose:
            print ("azimuth:[%.2f] distance:[%.2f] elevation:[%.2f] lookat:%s]"%
                   (azimuth,distance,elevation,lookat))
        return azimuth,distance,elevation,lookat
    
    def is_viewer_alive(self):
        """
        检查查看器窗口是否仍处于活动状态。

        返回:
            bool: 若查看器存活则为 True，否则为 False。
        """
        return self.viewer.is_alive
    
    def close_viewer(self):
        """
        关闭并清理查看器资源。

        返回:
            None
        """
        self.use_mujoco_viewer = False
        self.viewer.close()

    def viewer_text_overlay(
            self,
            loc   = 'bottom left',
            text1 = '',
            text2 = '',
        ):
        """
            在查看器上添加文本叠加层
            参数:
                loc: ['top','top right','top left','bottom','bottom right','bottom left']
                text1: 字符串
                text2: 字符串
        """
        self.viewer.add_overlay(loc=loc,text1=text1,text2=text2)

    def viewer_rgb_overlay(
            self,
            rgb = None,
            loc = 'top right',
        ):
        """
            在查看器上添加 RGB 叠加层
            参数:
                loc: ['top','top right','top left','bottom','bottom right','bottom left']
                rgb: RGB 图像
        """
        self.viewer.plot_rgb_overlay(rgb=rgb,loc=loc)

    def render(self):
        """
        将当前仿真状态渲染到查看器。

        返回:
            None
        """
        if self.use_mujoco_viewer:
            self.viewer.render()
        else:
            print ("[%s] 查看器尚未初始化。"%(self.name))
            
    def loop_every(self,HZ=None,tick_every=None):
        """
        根据仿真计数器（tick）或频率判断仿真循环是否应当执行。

        参数:
            HZ (float): 触发循环的频率（单位 Hz）。
            tick_every (int): 每隔多少个仿真计数执行一次。

        返回:
            bool: 若循环应当执行则为 True，否则为 False。
        """
        # tick = int(self.get_sim_time()/self.dt)
        FLAG = False
        if HZ is not None:
            FLAG = (self.tick-1)%(int(1/self.dt/HZ))==0
        if tick_every is not None:
            FLAG = (self.tick-1)%(tick_every)==0
        return FLAG
    
    def step(
            self,
            ctrl             = None,
            ctrl_idxs        = None,
            ctrl_names       = None,
            joint_names      = None,
            nstep            = 1,
            increase_tick    = True,
            step_flag        = True,
        ):
        """
        将仿真推进指定的步数，可选地施加控制输入。

        参数:
            ctrl (np.array): 要施加的控制输入。
            ctrl_idxs (list): 要更新的控制索引。
            ctrl_names (list): 要更新的控制名称。
            joint_names (list): 与控制对应的关节名称。
            nstep (int): 要执行的仿真步数。
            increase_tick (bool): 是否递增仿真计数器。

        返回:
            None
        """
        # step 是仿真主循环的“心跳”：把控制信号写进 data.ctrl，然后调用 mj_step 让
        # MuJoCo 根据物理定律把整个场景的状态往前推进 nstep 小步（更新位置、速度等）。
        if step_flag:
            if ctrl is not None:
                # 控制信号可以发给“全部执行器”，也可以只发给指定的若干个。
                # 下面根据传入的是 ctrl_names（按执行器名）还是 joint_names（按关节名）
                # 来确定要写入 data.ctrl 的下标位置。
                if ctrl_names is not None: # 当显式给定 'ctrl_names' 时
                    ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)
                elif joint_names is not None: # 当显式给定 'joint_names' 时
                    ctrl_idxs = self.get_idxs_step(joint_names=joint_names)
                # 施加控制：把 ctrl 写进 data.ctrl（不指定下标则覆盖全部执行器）
                if ctrl_idxs is None:
                    self.data.ctrl[:] = ctrl
                else:
                    self.data.ctrl[ctrl_idxs] = ctrl
            mujoco.mj_step(self.model,self.data,nstep=nstep)  # 推进物理（核心）

        # 更新墙钟时间（以 'step_flag' 为条件）
        self.increase_wall_time(step_flag=step_flag)

        # 递增计数器
        if increase_tick: self.increase_tick()

    def forward(self,q=None,joint_idxs=None,joint_names=None,increase_tick=True):
        """
        计算系统的前向运动学，可选地更新关节位置。

        参数:
            q (np.array): 要设置的关节位置。
            joint_idxs (list): 要更新的关节索引。
            joint_names (list): 要更新的关节名称。
            increase_tick (bool): 是否递增仿真计数器。

        返回:
            None
        """
        # 正运动学（forward kinematics）：已知每个关节的角度/位置（qpos），
        # 计算出每个刚体/末端在世界坐标里的位置和朝向。
        # mj_forward 与 mj_step 的区别：mj_forward 只“按当前 qpos 算一遍各物体位置”，
        # 不让时间前进、不施加力；常用于“我直接把关节摆到某个角度，看看末端在哪”。
        if q is not None:
            if joint_names is not None: # 如果 'joint_names' 非 None，则会覆盖 'joint_idxs'
                joint_idxs = self.get_idxs_fwd(joint_names=joint_names)
            if joint_idxs is not None:
                self.data.qpos[joint_idxs] = q   # 只设置指定关节的角度
            else: self.data.qpos = q             # 否则一次性设置全部关节
        mujoco.mj_forward(self.model,self.data)  # 根据 qpos 算出所有物体的位姿
        if increase_tick:
            self.increase_tick()

    def increase_wall_time(self,step_flag=True):
        """
        递增累计的墙钟时间。
        """
        current_wall_time = time.time()
        if step_flag:
            # 仅当 'step_flag' 为 True 时才递增墙钟时间
            self.accum_wall_time += current_wall_time - self.last_wall_update
        self.last_wall_update = current_wall_time

    def increase_tick(self):
        """
        递增仿真计数器（tick）。

        返回:
            None
        """
        self.tick = self.tick + 1

    def get_state(self):
        """
        获取当前仿真状态，包括时间、关节位置、速度和执行器状态。

        返回:
            dict: 包含状态信息的字典。

        MuJoCo 中的状态向量为:
            x = (mjData.time, mjData.qpos, mjData.qvel, mjData.act)
        接下来是控制量和施加的力。MuJoCo 中的控制向量为
            u = (mjData.ctrl, mjData.qfrc_applied, mjData.xfrc_applied)
        这些量指定了作用于模型中所定义执行器的控制信号 (mjData.ctrl)，
        或直接施加在关节空间中 (mjData.qfrc_applied)
        或笛卡尔空间中 (mjData.xfrc_applied) 的力与力矩。
        """
        state = {
            'tick':self.tick,
            'time':self.data.time,
            'qpos':self.data.qpos.copy(), # [self.model.nq]
            'qvel':self.data.qvel.copy(), # [self.model.nv]
            'act':self.data.act.copy(),
        }
        return state
    
    def store_state(self):
        """
        保存当前仿真状态以便日后恢复。

        返回:
            None
        """
        state = self.get_state()
        self.state_stored = copy.deepcopy(state) # 深拷贝

    def restore_state(self):
        """
        从此前保存的状态中恢复仿真状态。

        返回:
            None
        """
        state = self.state_stored
        self.set_state(
            qpos = state['qpos'],
            qvel = state['qvel'],
            act  = state['act'],
        )
        mujoco.mj_forward(self.model,self.data)
        
    def set_state(
            self,
            tick = None,
            time = None,
            qpos = None,
            qvel = None,
            act  = None, # 用于模拟肌腱和肌肉
            ctrl = None,
            step = False
        ):
        """
        用给定的值设置仿真状态。

        参数:
            tick (int): 仿真计数。
            time (float): 仿真时间。
            qpos (np.array): 关节位置。
            qvel (np.array): 关节速度。
            act (np.array): 执行器状态。
            ctrl (np.array): 控制信号。
            step (bool): 若为 True，则在设置状态后执行一步仿真。

        返回:
            None
        """
        if tick is not None: self.tick = tick
        if time is not None: self.data.time = time
        if qpos is not None: self.data.qpos = qpos.copy()
        if qvel is not None: self.data.qvel = qvel.copy()
        if act is not None: self.data.act = act.copy()
        if ctrl is not None: self.data.ctrl = ctrl.copy()
        # 前向动力学
        if step:
            mujoco.mj_step(self.model,self.data)
            
    def solve_inverse_dynamics(self,qacc=None):
        """
        求解逆动力学，计算实现给定关节加速度所需的力。

        参数:
            qacc (np.array): 期望的关节加速度。

        返回:
            np.array: 计算得到的逆动力学力。
        """
        if qacc is None:
            qacc = np.zeros(self.n_qacc)
        # 设置期望的 qacc
        self.data.qacc = qacc.copy()
        # 保存状态
        self.store_state()
        # 求解逆动力学
        mujoco.mj_inverse(self.model,self.data)
        # 恢复状态
        self.restore_state()
        # 返回
        """
            输出为 'qfrc_inverse'
            即为了实现观测到的加速度 'mjData.qacc'，必须作用于系统上的力。
        """
        qfrc_inverse = self.data.qfrc_inverse # [n_qacc]
        return qfrc_inverse.copy()
    
    # 说明：下面 set_*_base_body 系列是给“带自由关节(free joint)的基座刚体”用的，
    # 它通过修改 data.qpos（动态状态）来移动整个机器人/物体；
    # 而后面 set_*_body 系列是改 model.body 的“固定偏移”（静态定义），用途不同，别混用。
    def set_p_base_body(self,body_name='base',p=np.array([0,0,0]),forward=True):
        """
        设置基座刚体的位置。

        参数:
            body_name (str): 基座刚体的名称。
            p (np.array): 新位置（三维向量）。

        返回:
            None
        """
        # 找到该刚体的自由关节，再找到它在 qpos 里的起始下标（前 3 个是位置，接下来 4 个是四元数朝向）
        jntadr  = self.model.body(body_name).jntadr[0]
        qposadr = self.model.jnt_qposadr[jntadr]
        self.data.qpos[qposadr:qposadr+3] = p
        if forward:
            mujoco.mj_forward(self.model,self.data)
        
    def set_R_base_body(self,body_name='base',R=rpy2r(np.radians([0,0,0]))):
        """
        设置基座刚体的朝向。

        参数:
            body_name (str): 基座刚体的名称。
            R (np.array): 新的旋转矩阵（3x3）。

        返回:
            None
        """
        jntadr  = self.model.body(body_name).jntadr[0]
        qposadr = self.model.jnt_qposadr[jntadr]
        self.data.qpos[qposadr+3:qposadr+7] = r2quat(R)
        mujoco.mj_forward(self.model,self.data)

    def set_pR_base_body(self,body_name='base',p=np.array([0,0,0]),R=np.eye(3),T=None):
        """
        设置基座刚体的位姿（位置和旋转）。

        参数:
            body_name (str): 基座刚体的名称。
            p (np.array): 新位置（三维向量）。
            R (np.array): 新的旋转矩阵（3x3）。
            T (np.array): 变换矩阵（4x4），若提供则会覆盖 p 和 R。

        返回:
            None
        """
        if T is not None: # 如果 T 非 None，则会覆盖 p 和 R
            p = t2p(T)
            R = t2r(T)
        self.set_p_base_body(body_name=body_name,p=p)
        self.set_R_base_body(body_name=body_name,R=R)

    def set_T_base_body(self,body_name='base',p=np.array([0,0,0]),R=np.eye(3),T=None):
        """
        使用变换矩阵设置基座刚体的位姿。

        参数:
            body_name (str): 基座刚体的名称。
            p (np.array): 位置向量。
            R (np.array): 旋转矩阵。
            T (np.array): 变换矩阵，若提供则会覆盖 p 和 R。

        返回:
            None
        """
        if T is not None: # 如果 T 非 None，则会覆盖 p 和 R
            p = t2p(T)
            R = t2r(T)
        self.set_p_base_body(body_name=body_name,p=p)
        self.set_R_base_body(body_name=body_name,R=R)
        
    def set_p_body(self,body_name='base',p=np.array([0,0,0]),forward=True):
        """
        设置指定刚体的位置。

        参数:
            body_name (str): 刚体的名称。
            p (np.array): 新位置（三维向量）。
            forward (bool): 若为 True，则在设置后执行前向运动学。

        返回:
            None
        """
        self.model.body(body_name).pos = p
        if forward: self.forward(increase_tick=False)

    def set_R_body(self,body_name='base',R=np.eye(3),forward=True):
        """
        设置指定刚体的朝向。

        参数:
            body_name (str): 刚体的名称。
            R (np.array): 新的旋转矩阵（3x3）。
            forward (bool): 若为 True，则在设置后执行前向运动学。

        返回:
            None
        """
        self.model.body(body_name).quat = r2quat(R)
        if forward: self.forward(increase_tick=False)

    def set_pR_body(self,body_name='base',p=np.array([0,0,0]),R=np.eye(3),forward=True):
        """
        同时设置指定刚体的位置和朝向。

        参数:
            body_name (str): 刚体的名称。
            p (np.array): 新位置。
            R (np.array): 新的旋转矩阵。
            forward (bool): 是否在更新后执行前向运动学。

        返回:
            None
        """
        self.model.body(body_name).pos = p
        self.model.body(body_name).quat = r2quat(R)
        if forward: self.forward(increase_tick=False)

    def set_T_body(self,body_name='base',p=np.array([0,0,0]),R=np.eye(3),T=None,forward=True):
        """
        同时设置指定刚体的位置和朝向。

        参数:
            body_name (str): 刚体的名称。
            p (np.array): 新位置。
            R (np.array): 新的旋转矩阵。
            forward (bool): 是否在更新后执行前向运动学。

        返回:
            None
        """
        if T is not None: # 如果 T 非 None，则会覆盖 p 和 R
            p = t2p(T)
            R = t2r(T)
        self.model.body(body_name).pos = p
        self.model.body(body_name).quat = r2quat(R)
        if forward: self.forward(increase_tick=False)
        
    # 说明：mocap（动作捕捉）刚体是一种特殊的“无质量目标点”。常见用法是：让机械臂末端
    # 通过 weld 约束去“追踪”这个 mocap 点，于是我们移动 mocap，就能间接拖动机械臂末端。
    def set_p_mocap(self,mocap_name='',p=np.array([0,0,0])):
        """
        设置动作捕捉（mocap）刚体的位置。

        参数:
            mocap_name (str): mocap 刚体的名称。
            p (np.array): 新位置。

        返回:
            None
        """
        mocap_idx = self.model.body_mocapid[self.body_names.index(mocap_name)]
        self.data.mocap_pos[mocap_idx] = p

    def set_R_mocap(self,mocap_name='',R=np.eye(3)):
        """
        设置动作捕捉（mocap）刚体的朝向。

        参数:
            mocap_name (str): mocap 刚体的名称。
            R (np.array): 新的旋转矩阵。

        返回:
            None
        """
        mocap_idx = self.model.body_mocapid[self.body_names.index(mocap_name)]
        self.data.mocap_quat[mocap_idx] = r2quat(R)

    def set_pR_mocap(self,mocap_name='',p=np.array([0,0,0]),R=np.eye(3)):
        """
        设置动作捕捉（mocap）刚体的位姿。

        参数:
            mocap_name (str): mocap 刚体的名称。
            p (np.array): 新位置。
            R (np.array): 新的旋转矩阵。

        返回:
            None
        """
        self.set_p_mocap(mocap_name=mocap_name,p=p)
        self.set_R_mocap(mocap_name=mocap_name,R=R)
        
    def set_geom_color(
            self,
            body_names_to_color   = None,
            body_names_to_exclude = ['world'],
            body_names_to_exclude_including = [],
            rgba                  = [0.75,0.95,0.15,1.0],
            rgba_list             = None,
        ):
        """
        设置附着在指定刚体上的几何体的颜色。

        参数:
            body_names_to_color (list): 要着色的刚体名称列表。
            body_names_to_exclude (list): 要排除的刚体。
            body_names_to_exclude_including (list): 名称中包含这些子串的刚体将被排除。
            rgba (list): 默认的 RGBA 颜色。
            rgba_list (list): 与每个刚体一一对应的 RGBA 颜色列表。

        返回:
            None
        """
        def should_exclude(x, exclude_list):
            for exclude in exclude_list:
                if exclude in x:
                    return True
            return False

        if body_names_to_color is None: # 默认对所有几何体着色
            body_names_to_color = self.body_names
        for idx,body_name in enumerate(body_names_to_color): # 遍历所有刚体
            if body_name in body_names_to_exclude: # 排除特定刚体
                continue
            if should_exclude(body_name,body_names_to_exclude_including):
                # 排除名称中包含 'body_names_to_exclude_including' 中子串的刚体
                continue
            body_idx = self.body_names.index(body_name)
            geom_idxs = [idx for idx,val in enumerate(self.model.geom_bodyid) if val==body_idx]
            for geom_idx in geom_idxs: # 遍历附着在该刚体上的几何体
                if rgba_list is None:
                    self.model.geom(geom_idx).rgba = rgba
                else:
                    self.model.geom(geom_idx).rgba = rgba_list[idx]
                    
    def set_geom_alpha(self,alpha=1.0,body_names_to_exclude=['world']):
        """
        设置几何体的透明度（alpha）值。

        参数:
            alpha (float): 透明度值（0 到 1）。
            body_names_to_exclude (list): 本次修改要排除的刚体。

        返回:
            None
        """
        for g_idx in range(self.n_geom): # 遍历每个几何体
            geom = self.model.geom(g_idx)
            body_name = self.body_names[geom.bodyid[0]]
            if body_name in body_names_to_exclude: continue # 排除特定刚体
            # 修改几何体的 alpha 值
            self.model.geom(g_idx).rgba[3] = alpha
            
    def get_sim_time(self,init_flag=False):
        """
        获取自初始化以来经过的仿真时间。

        参数:
            init_flag (bool): 若为 True，则重置仿真时间基准。

        返回:
            float: 经过的仿真时间（秒）。
        """
        if init_flag:
            self.init_sim_time = self.data.time
        elapsed_time = self.data.time - self.init_sim_time
        return elapsed_time
    
    def reset_sim_time(self):
        """
        重置仿真时间基准。

        返回:
            None
        """
        self.init_sim_time = self.data.time

    def reset_wall_time(self):
        """
        重置墙钟时间基准。

        返回:
            None
        """
        self.init_wall_time = time.time()

    def get_wall_time(self,init_flag=False):
        """
        获取自上次重置以来经过的墙钟时间。

        参数:
            init_flag (bool): 若为 True，则重置墙钟基准。

        返回:
            float: 经过的墙钟时间（秒）。
        """
        if init_flag:
            self.accum_wall_time = 0.0
            self.last_wall_update = time.time()
        return self.accum_wall_time
    
    def grab_rgbd_img(self):
        """
        从当前查看器捕获一张 RGB-D 图像（颜色和深度）。

        返回:
            tuple: (rgb_img, depth_img)，其中 rgb_img 为 uint8 图像，depth_img 为 float32 图像。
        """
        rgb_img   = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        depth_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_img,depth_img,self.viewer.viewport,self.viewer.ctx)
        rgb_img,depth_img = np.flipud(rgb_img),np.flipud(depth_img) # 上下翻转

        # 重新缩放深度图像：
        # OpenGL 读出的深度是 0~1 的“非线性深度缓冲值”，并不是真实的米。
        # 这里用近裁剪面 near、远裁剪面 far，把它换算回“距相机的真实距离（米）”。
        extent = self.model.stat.extent
        near   = self.model.vis.map.znear * extent
        far    = self.model.vis.map.zfar * extent
        scaled_depth_img = near / (1 - depth_img * (1 - near / far))
        depth_img = scaled_depth_img.squeeze()
        return rgb_img,depth_img
    
    def get_T_viewer(self):
        """
        计算并返回查看器相机当前的变换矩阵。

        返回:
            np.array: 一个 4x4 的变换矩阵。
        """
        cam_lookat    = self.viewer.cam.lookat
        cam_elevation = self.viewer.cam.elevation
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance

        p_lookat = cam_lookat
        R_lookat = rpy2r(np.deg2rad([0,-cam_elevation,cam_azimuth]))
        T_lookat = pr2t(p_lookat,R_lookat)
        T_viewer = T_lookat @ pr2t(np.array([-cam_distance,0,0]),np.eye(3))
        return T_viewer
    
    def get_pcd_from_depth_img(self,depth_img,fovy=45):
        """
        从给定的深度图像生成点云数据。

        点云（point cloud）：一堆三维点 (x,y,z) 的集合，描述物体表面的形状。
        深度图的每个像素记录了“该方向上最近物体离相机多远”，结合相机内参就能把
        每个像素反投影成一个 3D 点，从而得到点云。

        参数:
            depth_img (np.array): 深度图像。
            fovy (float): y 方向的视场角（决定相机焦距/内参）。

        返回:
            tuple: (pcd, xyz_img, xyz_img_world)，分别是世界坐标点云、相机坐标系下的坐标图、世界坐标系下的坐标图。
        """
        # 获取相机位姿
        T_viewer = self.get_T_viewer()

        # 相机内参
        img_height = depth_img.shape[0]
        img_width = depth_img.shape[1]
        focal_scaling = 0.5*img_height/np.tan(fovy*np.pi/360)
        cam_matrix = np.array(((focal_scaling,0,img_width/2),
                            (0,focal_scaling,img_height/2),
                            (0,0,1)))

        # 从深度图像估计 3D 点（此时坐标是“相机坐标系”下的）
        xyz_img = meters2xyz(depth_img,cam_matrix) # [H x W x 3]
        xyz_transpose = np.transpose(xyz_img,(2,0,1)).reshape(3,-1) # [3 x N]
        # 补一行 1，变成齐次坐标 [x,y,z,1]，这样能用 4x4 矩阵 T 一次性做旋转+平移
        xyzone_transpose = np.vstack((xyz_transpose,np.ones((1,xyz_transpose.shape[1])))) # [4 x N]

        # 转换到世界坐标系：左乘相机的位姿矩阵 T_viewer，把“相机系坐标”变成“世界系坐标”
        xyzone_world_transpose = T_viewer @ xyzone_transpose
        xyz_world_transpose = xyzone_world_transpose[:3,:] # [3 x N]
        xyz_world = np.transpose(xyz_world_transpose,(1,0)) # [N x 3]

        xyz_img_world = xyz_world.reshape(depth_img.shape[0],depth_img.shape[1],3)

        return xyz_world,xyz_img,xyz_img_world
    
    def get_egocentric_rgb(
            self,
            p_ego        = None,
            p_trgt       = None,
            rsz_rate     = None,
            fovy         = None,
            restore_view = True,
        ):
        """
        根据给定的自身位置和目标位置捕获一张第一人称（egocentric）RGB 图像。
        “第一人称”常用来模拟装在机械臂腕部的相机：给相机所在点 p_ego 和它要看向的点 p_trgt，
        临时把观察相机搬过去拍一张，拍完再恢复原来的视角（restore_view=True 时）。

        参数:
            p_ego (np.array): 自身相机的位置。
            p_trgt (np.array): 目标位置。
            rsz_rate (float): 捕获图像的缩放比例。
            fovy (float): 视场角。
            restore_view (bool): 捕获后是否恢复原始相机视角。

        返回:
            np.array: 捕获到的 RGB 图像。
        """
        if restore_view:
            # 备份相机信息
            viewer_azimuth,viewer_distance,viewer_elevation,viewer_lookat = self.get_viewer_cam_info()

        if (p_ego is not None) and (p_trgt is not None):
            cam_azimuth,cam_distance,cam_elevation,cam_lookat = compute_view_params(
                camera_pos = p_ego,
                target_pos = p_trgt,
                up_vector  = np.array([0,0,1]),
            )
            self.set_viewer(
                azimuth   = cam_azimuth,
                distance  = cam_distance,
                elevation = cam_elevation,
                lookat    = cam_lookat,
                update    = True,
            )

        # 抓取 RGB 和深度图像
        rgb_img,_ = self.grab_rgbd_img() # 获取 rgb 和深度图像

        # 缩放 rgb_image 和 depth_img（可选）
        if rsz_rate is not None:
            h = int(rgb_img.shape[0]*rsz_rate)
            w = int(rgb_img.shape[1]*rsz_rate)
            rgb_img = cv2.resize(rgb_img,(w,h),interpolation=cv2.INTER_NEAREST)

        # 恢复视角
        if restore_view:
            # 恢复相机信息
            self.set_viewer(
                azimuth   = viewer_azimuth,
                distance  = viewer_distance,
                elevation = viewer_elevation,
                lookat    = viewer_lookat,
                update    = True,
            )
        return rgb_img
    
    def get_egocentric_rgbd_pcd(
            self,
            p_ego            = None,
            p_trgt           = None,
            rsz_rate_for_pcd = None,
            rsz_rate_for_img = None,
            fovy             = None,
            restore_view     = True,
        ):
        """
        捕获第一人称（egocentric）RGB、深度图像并生成点云数据。

        参数:
            p_ego (np.array): 自身相机的位置。
            p_trgt (np.array): 目标位置。
            rsz_rate_for_pcd (float): 用于点云生成的缩放比例。
            rsz_rate_for_img (float): 用于图像的缩放比例。
            fovy (float): 视场角。
            restore_view (bool): 是否恢复原始相机视角。

        返回:
            tuple: (rgb_img, depth_img, pcd, xyz_img, xyz_img_world)

        获取第一人称的 1) RGB 图像，2) 深度图像，3) 点云数据
        返回: (rgb_img,depth_img,pcd,xyz_img,xyz_img_world)
        本函数可能存在问题，因为它无法控制绕视线方向的扭转
        （更多细节见 https://mujoco.readthedocs.io/en/stable/programming/visualization.html
        ）
        """
        if restore_view:
            # 备份相机信息
            viewer_azimuth,viewer_distance,viewer_elevation,viewer_lookat = self.get_viewer_cam_info()

        if (p_ego is not None) and (p_trgt is not None):
            cam_azimuth,cam_distance,cam_elevation,cam_lookat = compute_view_params(
                camera_pos = p_ego,
                target_pos = p_trgt,
                up_vector  = np.array([0,0,1]),
            )
            self.set_viewer(
                azimuth   = cam_azimuth,
                distance  = cam_distance,
                elevation = cam_elevation,
                lookat    = cam_lookat,
                update    = True,
            )
        
        # 抓取 RGB 和深度图像
        rgb_img,depth_img = self.grab_rgbd_img() # 获取 rgb 和深度图像

        # 缩放深度图像以减少点云数量
        if rsz_rate_for_pcd is not None:
            h_rsz         = int(depth_img.shape[0]*rsz_rate_for_pcd)
            w_rsz         = int(depth_img.shape[1]*rsz_rate_for_pcd)
            depth_img_rsz = cv2.resize(depth_img,(w_rsz,h_rsz),interpolation=cv2.INTER_NEAREST)
        else:
            depth_img_rsz = depth_img

        # 获取点云数据
        if fovy is None:
            if len(self.model.cam_fovy)==0: fovy = 45.0 # 如果未定义相机，则使用 45 度（默认值）
            else: fovy = self.model.cam_fovy[0] # 否则使用第一个相机的 fovy
        pcd,xyz_img,xyz_img_world = self.get_pcd_from_depth_img(depth_img_rsz,fovy=fovy) # [N x 3]

        # 缩放 rgb_image 和 depth_img（可选）
        if rsz_rate_for_img is not None:
            h = int(rgb_img.shape[0]*rsz_rate_for_img)
            w = int(rgb_img.shape[1]*rsz_rate_for_img)
            rgb_img   = cv2.resize(rgb_img,(w,h),interpolation=cv2.INTER_NEAREST)
            depth_img = cv2.resize(depth_img,(w,h),interpolation=cv2.INTER_NEAREST)

        # 恢复视角
        if restore_view:
            # 恢复相机信息
            self.set_viewer(
                azimuth   = viewer_azimuth,
                distance  = viewer_distance,
                elevation = viewer_elevation,
                lookat    = viewer_lookat,
                update    = True,
            )
        return rgb_img,depth_img,pcd,xyz_img,xyz_img_world
    
    def grab_image(self,rsz_rate=None,interpolation=cv2.INTER_NEAREST):
        """
        从查看器捕获当前渲染的图像。

        参数:
            rsz_rate (float): 可选的缩放比例。
            interpolation: 缩放时使用的插值方法。

        返回:
            np.array: 捕获到的图像。
        """
        img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)
        mujoco.mjr_readPixels(img, None,self.viewer.viewport,self.viewer.ctx)
        img = np.flipud(img) # 翻转图像
        # 缩放
        if rsz_rate is not None:
            h = int(img.shape[0]*rsz_rate)
            w = int(img.shape[1]*rsz_rate)
            img = cv2.resize(img,(w,h),interpolation=interpolation)
        # 备份
        if img.sum() > 0:
            self.grab_image_backup = img
        if img.sum() == 0: # 改为使用备份
            img = self.grab_image_backup
        return img.copy()

    def get_fixed_cam_rgb(self,cam_name):
        """
            获取固定相机的 RGB 图像。
            “固定相机”指在 XML 里事先定义好、固定在场景中某处的相机（区别于可自由拖动的观察视角）。
            按名字找到该相机，用它的视角渲染一帧，读出像素返回 RGB 图。
        """
        # 解析相机信息
        cam_idx  = self.cam_names.index(cam_name)
        cam      = self.cams[cam_idx]
        cam_fov  = self.cam_fovs[cam_idx]
        viewport = self.cam_viewports[cam_idx]
        # 更新
        mujoco.mjv_updateScene(
            self.model,self.data,self.viewer.vopt,self.viewer.pert,
            cam,mujoco.mjtCatBit.mjCAT_ALL,self.viewer.scn)
        mujoco.mjr_render(viewport,self.viewer.scn,self.viewer.ctx)
        # 抓取 RGBD
        rgb = np.zeros((viewport.height,viewport.width,3),dtype=np.uint8)
        depth_raw = np.zeros((viewport.height,viewport.width),dtype=np.float32)
        mujoco.mjr_readPixels(rgb,depth_raw,viewport,self.viewer.ctx)
        rgb,depth_raw = np.flipud(rgb),np.flipud(depth_raw)
        return rgb
    
    def get_fixed_cam_rgbd_pcd(self,cam_name,downscale_pcd=0.1):
        """
        从固定相机捕获 RGB、深度图像和点云数据。

        参数:
            cam_name (str): 固定相机的名称。
            downscale_pcd (float): 点云的降采样因子。

        返回:
            tuple: 来自固定相机的 (rgb, depth, pcd, T_view)。
        """
        # 解析相机信息
        cam_idx  = self.cam_names.index(cam_name)
        cam      = self.cams[cam_idx]
        cam_fov  = self.cam_fovs[cam_idx]
        viewport = self.cam_viewports[cam_idx]
        # 更新
        mujoco.mjv_updateScene(
            self.model,self.data,self.viewer.vopt,self.viewer.pert,
            cam,mujoco.mjtCatBit.mjCAT_ALL,self.viewer.scn)
        mujoco.mjr_render(viewport,self.viewer.scn,self.viewer.ctx)
        # 抓取 RGBD
        rgb = np.zeros((viewport.height,viewport.width,3),dtype=np.uint8)
        depth_raw = np.zeros((viewport.height,viewport.width),dtype=np.float32)
        mujoco.mjr_readPixels(rgb,depth_raw,viewport,self.viewer.ctx)
        rgb,depth_raw = np.flipud(rgb),np.flipud(depth_raw)
        # 重新缩放深度
        extent = self.model.stat.extent
        near   = self.model.vis.map.znear * extent
        far    = self.model.vis.map.zfar * extent
        depth = near/(1-depth_raw*(1-near/far))
        # 使用缩放后的深度图像获取点云数据
        h_rsz = int(depth.shape[0]*downscale_pcd)
        w_rsz = int(depth.shape[1]*downscale_pcd)
        depth_rsz = cv2.resize(depth,(w_rsz,h_rsz),interpolation=cv2.INTER_NEAREST)
        img_height,img_width = depth_rsz.shape[0],depth_rsz.shape[1]
        focal_scaling = 0.5*img_height/np.tan(cam_fov*np.pi/360)
        cam_matrix = np.array(((focal_scaling,0,img_width/2),
                               (0,focal_scaling,img_height/2),
                               (0,0,1))) # [3 x 3]
        xyz_img = meters2xyz(depth_rsz,cam_matrix) # [H x W x 3]
        xyz_transpose = np.transpose(xyz_img,(2,0,1)).reshape(3,-1) # [3 x N]
        xyzone_transpose = np.vstack((xyz_transpose,np.ones((1,xyz_transpose.shape[1])))) # [4 x N]
        # 将点云转换到世界坐标系
        T_view = self.get_T_cam(cam_name=cam_name)@pr2t(p=np.zeros(3),R=rpy2r(np.deg2rad([-45.,90.,45.])))
        xyzone_world_transpose = T_view @ xyzone_transpose
        xyz_world_transpose = xyzone_world_transpose[:3,:] # [3 x N]
        pcd = np.transpose(xyz_world_transpose,(1,0)) # [N x 3]
        # 返回
        return rgb,depth,pcd,T_view
        
    def get_body_names(self,prefix='',excluding='world'):
        """
        获取以给定前缀开头且排除指定名称的刚体名称列表。

        参数:
            prefix (str): 要匹配的前缀。
            excluding (str): 若名称中包含该子串，则排除该刚体。

        返回:
            list: 过滤后的刚体名称。
        """
        body_names = [x for x in self.body_names if x is not None and x.startswith(prefix) and excluding not in x]
        return body_names
    
    def get_site_names(self,prefix='',excluding='world'):
        """
        获取以给定前缀开头且排除指定名称的站点（site）名称列表。

        参数:
            prefix (str): 要匹配的前缀。
            excluding (str): 要排除的子串。

        返回:
            list: 过滤后的站点名称。
        """
        site_names = [x for x in self.site_names if x is not None and x.startswith(prefix) and excluding not in x]
        return site_names
    
    def get_sensor_names(self,prefix='',excluding='world'):
        """
        获取以给定前缀开头且排除指定名称的传感器名称列表。

        参数:
            prefix (str): 要匹配的前缀。
            excluding (str): 要排除的子串。

        返回:
            list: 过滤后的传感器名称。
        """
        sensor_names = [x for x in self.sensor_names if x is not None and x.startswith(prefix) and excluding not in x]
        return sensor_names
    
    def get_mesh_names(self,including='',excluding='collision'):
        """
        获取网格名称列表。
        """
        if excluding is None:
            mesh_names = [x for x in self.mesh_names if x is not None and including in x]    
        else:
            mesh_names = [x for x in self.mesh_names if x is not None and including in x and excluding not in x]
        return mesh_names
    
    def get_geom_idxs_from_body_name(self,body_name):
        """
            根据刚体名称获取几何体索引
        """
        body_idx = self.body_names.index(body_name)
        geom_idxs = [idx for idx,val in enumerate(self.model.geom_bodyid) if val==body_idx] 
        return geom_idxs

    # ----------------------------------------------------------------------
    # 下面一大批 get_* 方法是“位姿查询”接口，初学者最常用。约定：
    #   p (position)：三维位置向量 [x, y, z]，单位米；
    #   R (rotation)：3x3 旋转矩阵，表示朝向；
    #   T (transform)：4x4 齐次变换矩阵，左上 3x3 是 R、右上 3x1 是 p，一次描述位置+朝向。
    # 这些值都来自 data（动态状态），所以反映的是“当前这一刻”物体在世界坐标里的位姿。
    # ----------------------------------------------------------------------
    def get_p_body(self,body_name):
        """
        获取指定刚体的位置。

        参数:
            body_name (str): 刚体的名称。

        返回:
            np.array: 刚体在世界坐标系下的位置 [x,y,z]。
        """
        return self.data.body(body_name).xpos.copy()

    def get_R_body(self,body_name):
        """
        获取指定刚体的旋转矩阵。

        参数:
            body_name (str): 刚体的名称。

        返回:
            np.array: 3x3 的旋转矩阵。
        """
        return self.data.body(body_name).xmat.reshape([3,3]).copy()

    def get_T_body(self,body_name):
        """
        获取指定刚体的完整变换矩阵（位姿）。

        参数:
            body_name (str): 刚体的名称。

        返回:
            np.array: 4x4 的变换矩阵。
        """
        p_body = self.get_p_body(body_name=body_name)
        R_body = self.get_R_body(body_name=body_name)
        return pr2t(p_body,R_body)

    def get_pR_body(self,body_name):
        """
        同时获取指定刚体的位置和旋转矩阵。

        参数:
            body_name (str): 刚体的名称。

        返回:
            tuple: (位置, 旋转矩阵)
        """
        p = self.get_p_body(body_name)
        R = self.get_R_body(body_name)
        return p,R

    def get_p_joint(self,joint_name):
        """
        获取关节的位置（通过其关联的刚体）。

        参数:
            joint_name (str): 关节的名称。

        返回:
            np.array: 关节位置。
        """
        body_id = self.model.joint(joint_name).bodyid[0] # 第一个刚体 ID
        return self.get_p_body(self.body_names[body_id])

    def get_R_joint(self,joint_name):
        """
        获取关节的旋转矩阵（通过其关联的刚体）。

        参数:
            joint_name (str): 关节的名称。

        返回:
            np.array: 关节旋转矩阵。
        """
        body_id = self.model.joint(joint_name).bodyid[0] # 第一个刚体 ID
        return self.get_R_body(self.body_names[body_id])

    def get_pR_joint(self,joint_name):
        """
        同时获取指定关节的位置和旋转。

        参数:
            joint_name (str): 关节的名称。

        返回:
            tuple: (位置, 旋转矩阵)
        """
        p = self.get_p_joint(joint_name)
        R = self.get_R_joint(joint_name)
        return p,R
    
    def get_p_geom(self,geom_name):
        """
        获取指定几何体的位置。

        参数:
            geom_name (str): 几何体的名称。

        返回:
            np.array: 几何体的位置。
        """
        return self.data.geom(geom_name).xpos

    def get_R_geom(self,geom_name):
        """
        获取指定几何体的旋转矩阵。

        参数:
            geom_name (str): 几何体的名称。

        返回:
            np.array: 3x3 的旋转矩阵。
        """
        return self.data.geom(geom_name).xmat.reshape((3,3))

    def get_pR_geom(self,geom_name):
        """
        同时获取指定几何体的位置和旋转矩阵。

        参数:
            geom_name (str): 几何体的名称。

        返回:
            tuple: (位置, 旋转矩阵)
        """
        p = self.get_p_geom(geom_name)
        R = self.get_R_geom(geom_name)
        return p,R
    
    def get_site_name_of_sensor(self,sensor_name):
        """
        获取与给定传感器关联的站点（site）名称。

        参数:
            sensor_name (str): 传感器的名称。

        返回:
            str: 对应的站点名称。
        """
        sensor_id = self.model.sensor(sensor_name).id # 获取传感器 ID
        sensor_objtype = self.model.sensor_objtype[sensor_id] # 获取所关联的对象类型（即站点）
        sensor_objid = self.model.sensor_objid[sensor_id] # 获取所关联的对象 ID
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # 获取站点名称
        return site_name

    def get_p_sensor(self,sensor_name):
        """
        获取传感器的位置（通过其关联的站点）。

        参数:
            sensor_name (str): 传感器的名称。

        返回:
            np.array: 传感器位置。
        """
        sensor_id = self.model.sensor(sensor_name).id # 获取传感器 ID
        sensor_objtype = self.model.sensor_objtype[sensor_id] # 获取所关联的对象类型（即站点）
        sensor_objid = self.model.sensor_objid[sensor_id] # 获取所关联的对象 ID
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # 获取站点名称
        p = self.data.site(site_name).xpos.copy() # 获取站点的位置
        return p

    def get_p_site(self,site_name):
        """
        获取指定站点（site）的位置。

        参数:
            site_name (str): 站点的名称。

        返回:
            np.array: 站点的位置。
        """
        return self.data.site(site_name).xpos.copy()

    def get_R_site(self,site_name):
        """
        获取指定站点（site）的旋转矩阵。

        参数:
            site_name (str): 站点的名称。

        返回:
            np.array: 3x3 的旋转矩阵。
        """
        return self.data.site(site_name).xmat.reshape(3,3).copy()

    def get_pR_site(self,site_name):
        """
        同时获取指定站点（site）的位置和旋转矩阵。

        参数:
            site_name (str): 站点的名称。

        返回:
            tuple: (位置, 旋转矩阵)
        """
        p_site = self.get_p_site(site_name)
        R_site = self.get_R_site(site_name)
        return p_site,R_site
    
    def get_R_sensor(self,sensor_name):
        """
        获取指定传感器的旋转矩阵。

        参数:
            sensor_name (str): 传感器的名称。

        返回:
            np.array: 传感器的旋转矩阵。
        """
        sensor_id = self.model.sensor(sensor_name).id
        sensor_objtype = self.model.sensor_objtype[sensor_id]
        sensor_objid = self.model.sensor_objid[sensor_id]
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid)
        R = self.data.site(site_name).xmat.reshape([3,3]).copy()
        return R

    def get_pR_sensor(self,sensor_name):
        """
        同时获取指定传感器的位置和旋转。

        参数:
            sensor_name (str): 传感器的名称。

        返回:
            tuple: (位置, 旋转矩阵)
        """
        p = self.get_p_sensor(sensor_name)
        R = self.get_R_sensor(sensor_name)
        return p,R

    def get_T_sensor(self,sensor_name):
        """
        获取指定传感器的变换矩阵（位姿）。

        参数:
            sensor_name (str): 传感器的名称。

        返回:
            np.array: 4x4 的变换矩阵。
        """
        p = self.get_p_sensor(sensor_name)
        R = self.get_R_sensor(sensor_name)
        return pr2t(p,R)

    def get_sensor_value(self,sensor_name):
        """
        获取指定传感器的当前值。

        参数:
            sensor_name (str): 传感器的名称。

        返回:
            传感器值。
        """
        data = self.data.sensor(sensor_name).data
        return data.copy()

    def get_sensor_values(self,sensor_names=None):
        """
        获取多个传感器的值。

        参数:
            sensor_names (list): 传感器名称列表。若为 None，则返回所有传感器的值。

        返回:
            np.array or list: 传感器值。
        """
        if sensor_names is None:
            sensor_names = self.sensor_names
        data = np.array([self.get_sensor_value(sensor_name) for sensor_name in self.sensor_names]).squeeze()
        if self.n_sensor == 1: return [data] # 转为列表
        else: return data.copy()

    def get_p_rf_list(self,sensor_names):
        """
        （别名）获取测距传感器（range finder）检测到的接触位置列表。

        参数:
            sensor_names (list): 传感器名称列表。

        返回:
            list: 接触位置。
        """
        return self.get_p_rf_obs_list(sensor_names)

    def get_p_rf_obs_list(self,sensor_names):
        """
        获取测距传感器（range finder）与障碍物之间的接触位置。

        参数:
            sensor_names (list): 测距传感器名称列表。

        返回:
            list: 观测到的接触位置。
        """
        p_rf_obs_list = []
        for sensor_name in sensor_names: # 遍历所有传感器
            rf_value      = self.get_sensor_value(sensor_name=sensor_name) # 传感器值
            cutoff_val    = self.model.sensor(sensor_name).cutoff[0]
            if cutoff_val == 0: cutoff_val = np.inf
            site_name     = self.get_site_name_of_sensor(sensor_name=sensor_name) # 站点名称
            p_site,R_site = self.get_pR_site(site_name=site_name) # 站点的 p 和 R
            if rf_value >= 0 and rf_value < cutoff_val:
                p_obs = p_site + rf_value*R_site[:,2] # z 轴为射线方向
                p_rf_obs_list.append(p_obs) # 追加
        return p_rf_obs_list # 列表
    
    def get_p_cam(self,cam_name):
        """
        获取指定相机的位置。

        参数:
            cam_name (str): 相机的名称。

        返回:
            np.array: 相机位置。
        """
        return self.data.cam(cam_name).xpos.copy()

    def get_R_cam(self,cam_name):
        """
        获取指定相机的旋转矩阵。

        参数:
            cam_name (str): 相机的名称。

        返回:
            np.array: 3x3 的旋转矩阵。
        """
        return self.data.cam(cam_name).xmat.reshape([3,3]).copy()

    def get_T_cam(self,cam_name):
        """
        获取指定相机的完整变换矩阵（位姿）。

        参数:
            cam_name (str): 相机的名称。

        返回:
            np.array: 4x4 的变换矩阵。
        """
        p_cam = self.get_p_cam(cam_name=cam_name)
        R_cam = self.get_R_cam(cam_name=cam_name)
        return pr2t(p_cam,R_cam)
    
    def plot_T(
            self,
            p           = np.array([0,0,0]),
            R           = np.eye(3),
            T           = None,
            plot_axis   = True,
            axis_len    = 1.0,
            axis_width  = 0.005,
            axis_rgba   = None,
            axis_alpha  = None,
            plot_sphere = False,
            sphere_r    = 0.05,
            sphere_rgba = [1,0,0,0.5],
            label       = None,
            print_xyz   = False,
        ):
        """
        在给定位姿处绘制坐标轴（以及可选的球体和标签）。
        这是最常用的可视化工具：在 3D 场景里画出一个“坐标系小三脚架”——
        三条互相垂直的箭头分别代表 x/y/z 轴（习惯上红=x、绿=y、蓝=z），
        箭头的原点在 p、朝向由 R 决定。用它能直观看出某个物体/末端的位置和朝向。
        注意：所有 plot_* 方法画的都是“仅供观看的装饰”，不参与物理，且每帧需重新调用。

        参数:
            p (np.array): 位置。
            R (np.array): 旋转矩阵。若提供了 T，则会覆盖 p 和 R。
            T (np.array): 4x4 的变换矩阵。
            plot_axis (bool): 是否绘制坐标轴。
            axis_len (float): 每条坐标轴的长度。
            axis_width (float): 坐标轴的粗细。
            axis_rgba (list): 坐标轴的 RGBA 颜色。
            plot_sphere (bool): 是否在 p 处绘制球体标记。
            sphere_r (float): 球体的半径。
            sphere_rgba (list): 球体的 RGBA 颜色。
            label (str): 可选的文本标签。
            print_xyz (bool): 是否打印坐标信息。

        返回:
            None
        """
        if T is not None: # 如果 T 非 None，则会覆盖 p 和 R
            p = t2p(T)
            R = t2r(T)
            
        # 画三条坐标轴：每条轴其实是用一根细长的“圆柱体”来表示的。
        # 下面分别把一个圆柱旋转到 x/y/z 方向，并平移到合适位置，再设成红/绿/蓝色。
        if plot_axis:
            if axis_alpha is None: axis_alpha = 0.9
            if axis_rgba is None:
                rgba_x = [1.0,0.0,0.0,axis_alpha]  # x 轴红色
                rgba_y = [0.0,1.0,0.0,axis_alpha]  # y 轴绿色
                rgba_z = [0.0,0.0,1.0,axis_alpha]  # z 轴蓝色
            else:
                rgba_x = axis_rgba
                rgba_y = axis_rgba
                rgba_z = axis_rgba
            R_x = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([1,0,0]))
            p_x = p+R_x[:,2]*axis_len/2
            if print_xyz: axis_label = 'X-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_x,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_x,
                rgba  = rgba_x,
                label = axis_label,
            )
            R_y = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,1,0]))
            p_y = p + R_y[:,2]*axis_len/2
            if print_xyz: axis_label = 'Y-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_y,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_y,
                rgba  = rgba_y,
                label = axis_label,
            )
            R_z = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,0,1]))
            p_z = p + R_z[:,2]*axis_len/2
            if print_xyz: axis_label = 'Z-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_z,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_z,
                rgba  = rgba_z,
                label = axis_label,
            )

        if plot_sphere:
            self.viewer.add_marker(
                pos   = p,
                size  = [sphere_r,sphere_r,sphere_r],
                rgba  = sphere_rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = '')

        if label is not None:
            self.viewer.add_marker(
                pos   = p,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.01],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )

    def plot_sphere(self,p,r,rgba=[1,1,1,1],label=''):
        """
        在指定位置绘制一个球体标记。

        参数:
            p (np.array): 位置（二维或三维）。
            r (float): 球体的半径。
            rgba (list): RGBA 颜色。
            label (str): 可选的标签。

        返回:
            None
        """
        p = np.asarray(p)
        if len(p) == 2: # 仅给定了 x 和 y（补齐 z=0）
            self.viewer.add_marker(
                pos   = np.append(p,[0]),
                size  = [r,r,r],
                rgba  = rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )
        elif len(p) == 3:
            self.viewer.add_marker(
                pos   = p,
                size  = [r,r,r],
                rgba  = rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )
        
    def plot_spheres(self,p_list,r,rgba=[1,1,1,1],label=''):
        """
        在 p_list 中给定的各位置处绘制多个球体。

        参数:
            p_list (list of np.array): 位置列表。
            r (float): 每个球体的半径。
            rgba (list): RGBA 颜色。
            label (str): 每个球体可选的标签。

        返回:
            None
        """
        for p in p_list:
            self.plot_sphere(p=p,r=r,rgba=rgba,label=label)
                
    def plot_box(
            self,
            p     = np.array([0,0,0]),
            R     = np.eye(3),
            xlen  = 1.0,
            ylen  = 1.0,
            zlen  = 1.0,
            rgba  = [0.5,0.5,0.5,0.5],
            label = '',
        ):
        """
        在指定位姿处绘制一个长方体标记。

        参数:
            p (np.array): 位置。
            R (np.array): 朝向矩阵。
            xlen, ylen, zlen (float): 长方体的尺寸。
            rgba (list): RGBA 颜色。

        返回:
            None
        """
        p = np.asarray(p)
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_BOX,
            size  = [xlen/2,ylen/2,zlen/2],
            rgba  = rgba,
            label = label,
        )
    
    def plot_capsule(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        绘制一个胶囊体标记。

        参数:
            p (np.array): 位置。
            R (np.array): 朝向。
            r (float): 半径。
            h (float): 胶囊体的半长。
            rgba (list): RGBA 颜色。

        返回:
            None
        """
        p = np.asarray(p)
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size  = [r,r,h],
            rgba  = rgba,
            label = label,
        )
        
    def plot_cylinder(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        绘制一个圆柱体标记。

        参数:
            p (np.array): 位置。
            R (np.array): 朝向。
            r (float): 半径。
            h (float): 半高。
            rgba (list): RGBA 颜色。

        返回:
            None
        """
        p = np.asarray(p)
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,h],
            rgba  = rgba,
            label = label,
        )
    
    def plot_ellipsoid(self,p=np.array([0,0,0]),R=np.eye(3),rx=1.0,ry=1.0,rz=1.0,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        绘制一个椭球体标记。

        参数:
            p (np.array): 位置。
            R (np.array): 朝向。
            rx, ry, rz (float): 沿 x、y、z 轴的半径。
            rgba (list): RGBA 颜色。

        返回:
            None
        """
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size  = [rx,ry,rz],
            rgba  = rgba,
            label = label,
        )
        
    def plot_arrow(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        """
        在给定位姿处绘制一个箭头标记。

        参数:
            p (np.array): 位置。
            R (np.array): 朝向。
            r (float): 箭杆的半径。
            h (float): 箭头的长度。
            rgba (list): RGBA 颜色。

        返回:
            None
        """
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,h*2],
            rgba  = rgba,
            label = ''
        )
        
    def plot_line(self,p=np.array([0,0,0]),R=np.eye(3),h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        """
        绘制一个线段标记。

        参数:
            p (np.array): 起始位置。
            R (np.array): 朝向（方向）。
            h (float): 线段的长度。
            rgba (list): RGBA 颜色。

        返回:
            None
        """
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = h,
            rgba  = rgba,
            label = ''
        )
        
    def plot_arrow_fr2to(self,p_fr,p_to,r=1.0,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        绘制一个从点 p_fr 指向点 p_to 的箭头。

        参数:
            p_fr (np.array): 起点。
            p_to (np.array): 终点。
            r (float): 箭杆半径。
            rgba (list): RGBA 颜色。

        返回:
            None
        """
        # 确保 p_fr 和 p_to 为 numpy 数组
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,np.linalg.norm(p_to-p_fr)*2],
            rgba  = rgba,
            label = label,
        )

    def plot_line_fr2to(self,p_fr,p_to,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        绘制一条连接两点的线段。

        参数:
            p_fr (np.array): 起点。
            p_to (np.array): 终点。
            rgba (list): RGBA 颜色。

        返回:
            None
        """
        # 确保 p_fr 和 p_to 为 numpy 数组
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = np.linalg.norm(p_to-p_fr),
            rgba  = rgba,
            label = label,
        )
    
    def plot_cylinder_fr2to(self,p_fr,p_to,r=0.01,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        在两点之间绘制一个圆柱体标记。

        参数:
            p_fr (np.array): 起点。
            p_to (np.array): 终点。
            r (float): 圆柱体半径。
            rgba (list): RGBA 颜色。

        返回:
            None
        """
        # 确保 p_fr 和 p_to 为 numpy 数组
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = (p_fr+p_to)/2,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,np.linalg.norm(p_to-p_fr)/2],
            rgba  = rgba,
            label = label,
        )
        
    def plot_traj(
            self,
            traj, # [L x 3] 表示 (x,y,z) 序列，或 [L x 2] 表示 (x,y) 序列
            rgba          = [1,0,0,1],
            plot_line     = False,
            plot_cylinder = True,
            plot_sphere   = False,
            cylinder_r    = 0.01,
            sphere_r      = 0.025,
        ):
        """
        绘制由一系列点构成的轨迹。

        参数:
            traj (np.array): 形状为 [L x 3] 或 [L x 2] 的数组，表示轨迹。
            rgba (list): 绘图所用的 RGBA 颜色。
            plot_line (bool): 是否绘制连接各点的线段。
            plot_cylinder (bool): 是否在各点之间绘制圆柱体。
            plot_sphere (bool): 是否在各点处绘制球体。
            cylinder_r (float): 圆柱体的半径。
            sphere_r (float): 球体的半径。

        返回:
            None
        """
        L = traj.shape[0]
        colors = None
        for idx in range(L-1):
            p_fr = traj[idx,:]
            p_to = traj[idx+1,:]
            if len(p_fr) == 2: p_fr = np.append(p_fr,[0])
            if len(p_to) == 2: p_to = np.append(p_to,[0])
            if plot_line:
                self.plot_line_fr2to(p_fr=p_fr,p_to=p_to,rgba=rgba)
            if plot_cylinder:
                self.plot_cylinder_fr2to(p_fr=p_fr,p_to=p_to,r=cylinder_r,rgba=rgba)
        if plot_sphere:
            for idx in range(L):
                p = traj[idx,:]
                self.plot_sphere(p=p,r=sphere_r,rgba=rgba)
        
    def plot_text(self,p,label=''):
        """
        在指定位置绘制一个文本标签。

        参数:
            p (np.array): 文本的位置。
            label (str): 要显示的文本。

        返回:
            None
        """
        p = np.asarray(p)
        self.viewer.add_marker(
            pos   = p,
            size  = [0.0001,0.0001,0.0001],
            rgba  = [1,1,1,0.01],
            type  = mujoco.mjtGeom.mjGEOM_SPHERE,
            label = label,
        )

    def plot_time(
            self,
            loc = 'bottom left',
        ):
        """
        在查看器上叠加显示当前仿真计数（tick）、仿真时间和墙钟时间。

        参数:
            loc (str): 叠加层的位置。

        返回:
            None
        """
        self.viewer.add_overlay(text1='tick',text2='%d'%(self.tick),loc=loc)
        self.viewer.add_overlay(text1='sim time',text2='%.2fsec'%(self.get_sim_time()),loc=loc)
        self.viewer.add_overlay(text1='wall time',text2='%.2fsec'%(self.get_wall_time()),loc=loc)
        
    def plot_sensor_T(
            self,
            sensor_name,
            plot_axis   = True,
            axis_len    = 0.1,
            axis_width  = 0.005,
            axis_rgba   = None,
            label       = None,
        ):
        """
        绘制某个传感器的坐标系。

        参数:
            sensor_name (str): 传感器的名称。
            plot_axis (bool): 是否绘制坐标轴。
            axis_len (float): 每条坐标轴的长度。
            axis_width (float): 坐标轴的宽度。
            axis_rgba (list): 坐标轴的 RGBA 颜色。
            label (str): 可选的标签。

        返回:
            None
        """
        p_sensor,R_sensor = self.get_pR_sensor(sensor_name=sensor_name)
        self.plot_T(
            p_sensor,
            R_sensor,
            plot_axis   = plot_axis,
            axis_len    = axis_len,
            axis_width  = axis_width,
            axis_rgba   = axis_rgba,
            plot_sphere = False,
            label       = label,
        )
        
    def plot_sensors_T(
            self,
            sensor_names,
            plot_axis   = True,
            axis_len    = 0.1,
            axis_width  = 0.005,
            axis_rgba   = None,
            plot_name   = False,
        ):
        """
        绘制多个传感器的坐标系。

        参数:
            sensor_names (list): 传感器名称列表。
            plot_axis (bool): 是否绘制坐标轴。
            axis_len (float): 坐标轴长度。
            axis_width (float): 坐标轴宽度。
            axis_rgba (list): RGBA 颜色。
            plot_name (bool): 是否显示传感器名称。

        返回:
            None
        """
        for sensor_idx,sensor_name in enumerate(sensor_names):
            if plot_name:
                label = '[%d] %s'%(sensor_idx,sensor_name)
            else:
                label = ''
            self.plot_sensor_T(
                sensor_name = sensor_name,
                plot_axis   = plot_axis,
                axis_len    = axis_len,
                axis_width  = axis_width,
                axis_rgba   = axis_rgba,
                label       = label,
             )
        
    def plot_sensors(
            self,
            loc = 'bottom right',
        ):
        """
        以文本形式在查看器上叠加显示传感器值。

        参数:
            loc (str): 叠加层的位置。

        返回:
            None
        """
        sensor_values = self.get_sensor_values() # 打印传感器值
        for sensor_idx,sensor_name in enumerate(self.sensor_names):
            self.viewer.add_overlay(
                text1 = '%s'%(sensor_name),
                text2 = '%.2f'%(sensor_values[sensor_idx]),
                loc   = loc,
            )

    def plot_body_T(
            self,
            body_name,
            plot_axis   = True,
            axis_len    = 0.1,
            axis_width  = 0.005,
            axis_rgba   = None,
            plot_sphere = False,
            sphere_r    = 0.05,
            sphere_rgba = [1,0,0,0.5],
            label       = None,
        ):
        """
        绘制指定刚体的坐标系。

        参数:
            body_name (str): 刚体的名称。
            plot_axis (bool): 是否绘制坐标轴。
            axis_len (float): 坐标轴的长度。
            axis_width (float): 坐标轴的宽度。
            axis_rgba (list): RGBA 颜色。
            plot_sphere (bool): 是否绘制球体标记。
            sphere_r (float): 球体半径。
            sphere_rgba (list): 球体颜色。
            label (str): 可选的标签。

        返回:
            None
        """
        p,R = self.get_pR_body(body_name=body_name)
        self.plot_T(
            p,
            R,
            plot_axis   = plot_axis,
            axis_len    = axis_len,
            axis_width  = axis_width,
            axis_rgba   = axis_rgba,
            plot_sphere = plot_sphere,
            sphere_r    = sphere_r,
            sphere_rgba = sphere_rgba,
            label       = label,
        )

    def plot_body_sphere(
            self,
            body_name,
            r     = 0.05,
            rgba  = (1,0,0,0.5),
            label = None,
        ):
        """
        绘制指定刚体的坐标系。

        参数:
            body_name (str): 刚体的名称。
            plot_axis (bool): 是否绘制坐标轴。
            axis_len (float): 坐标轴的长度。
            axis_width (float): 坐标轴的宽度。
            axis_rgba (list): RGBA 颜色。
            plot_sphere (bool): 是否绘制球体标记。
            sphere_r (float): 球体半径。
            sphere_rgba (list): 球体颜色。
            label (str): 可选的标签。

        返回:
            None
        """
        p,R = self.get_pR_body(body_name=body_name)
        self.plot_T(
            p,
            R,
            plot_axis   = False,
            axis_len    = None,
            axis_width  = None,
            axis_rgba   = None,
            plot_sphere = True,
            sphere_r    = r,
            sphere_rgba = rgba,
            label       = label,
        )
        
    def plot_joint_T(
            self,
            joint_name,
            plot_axis  = True,
            axis_len   = 1.0,
            axis_width = 0.01,
            axis_rgba  = None,
            label      = None,
        ):
        """
        绘制指定关节的坐标系。

        参数:
            joint_name (str): 关节的名称。
            plot_axis (bool): 是否绘制坐标轴。
            axis_len (float): 坐标轴的长度。
            axis_width (float): 坐标轴的宽度。
            axis_rgba (list): RGBA 颜色。
            label (str): 可选的标签。

        返回:
            None
        """
        p,R = self.get_pR_joint(joint_name=joint_name)
        self.plot_T(
            p,
            R,
            plot_axis  = plot_axis,
            axis_len   = axis_len,
            axis_width = axis_width,
            axis_rgba  = axis_rgba,
            label      = label,
        )
        
    def plot_bodies_T(
            self,
            body_names            = None,
            body_names_to_exclude = [],
            body_names_to_exclude_including = [],
            plot_axis             = True,
            axis_len              = 0.05,
            axis_width            = 0.005,
            rate                  = 1.0,
            plot_name             = False,
        ):
        """
        绘制多个刚体的坐标系。

        参数:
            body_names (list): 要绘制的刚体名称列表（若为 None，则绘制所有刚体）。
            body_names_to_exclude (list): 要排除的刚体名称。
            body_names_to_exclude_including (list): 名称中包含这些子串的刚体将被排除。
            plot_axis (bool): 是否绘制坐标轴。
            axis_len (float): 坐标轴长度。
            axis_width (float): 坐标轴宽度。
            rate (float): 缩放因子。
            plot_name (bool): 是否显示刚体名称。

        返回:
            None
        """
        def should_exclude(x, exclude_list):
            for exclude in exclude_list:
                if exclude in x:
                    return True
            return False

        if body_names is None:
            body_names = self.body_names

        for body_idx,body_name in enumerate(body_names):
            if body_name in body_names_to_exclude: continue
            if body_name is None: continue

            if should_exclude(body_name,body_names_to_exclude_including):
                # 排除名称中包含 'body_names_to_exclude_including' 中子串的刚体
                continue
            
            if plot_name:
                label = '[%d] %s'%(body_idx,body_name)
            else:
                label = ''
            self.plot_body_T(
                body_name  = body_name,
                plot_axis  = plot_axis,
                axis_len   = rate*axis_len,
                axis_width = rate*axis_width,
                label      = label,
            )
            
    def plot_links_between_bodies(
            self,
            parent_body_names_to_exclude = ['world'],
            body_names_to_exclude        = [],
            pbne                         = None,
            bne                          = None,
            r                            = 0.005,
            rgba                         = (0.0,0.0,0.0,0.5),
        ):
        """
        绘制连接父刚体和子刚体的可视化连杆（例如圆柱体）。

        参数:
            parent_body_names_to_exclude (list): 要排除的父刚体名称。
            body_names_to_exclude (list): 要排除的子刚体名称。
            pbne, bne: 备选的排除列表。
            r (float): 连接圆柱体的半径。
            rgba (tuple): 连杆的颜色。

        返回:
            None
        """
        if pbne is not None: parent_body_names_to_exclude = pbne
        if bne is not None: body_names_to_exclude = bne
        for body_idx,body_name in enumerate(self.body_names):
            parent_body_name = self.parent_body_names[body_idx]
            if parent_body_name in parent_body_names_to_exclude: continue
            if body_name in body_names_to_exclude: continue
            if body_name is None: continue
            
            self.plot_cylinder_fr2to(
                p_fr = self.get_p_body(body_name=parent_body_name),
                p_to = self.get_p_body(body_name=body_name),
                r    = r,
                rgba = rgba,
            )

    def plot_joint_axis(
            self,
            axis_len    = 0.1,
            axis_r      = 0.01,
            joint_names = None,
            alpha       = 0.2,
            rate        = 1.0,
            print_name  = False,
        ):
        """
        绘制旋转关节的轴。

        参数:
            axis_len (float): 关节轴的长度。
            axis_r (float): 轴标记的半径。
            joint_names (list): 要绘制的关节名称列表。
            alpha (float): 透明度因子。
            rate (float): 缩放因子。
            print_name (bool): 是否打印关节名称。

        返回:
            None
        """
        rev_joint_idxs  = self.rev_joint_idxs
        rev_joint_names = self.rev_joint_names

        if joint_names is not None:
            idxs = get_idxs(self.rev_joint_names,joint_names)
            rev_joint_idxs_to_use  = rev_joint_idxs[idxs]
            rev_joint_names_to_use = [rev_joint_names[i] for i in idxs]
        else:
            rev_joint_idxs_to_use  = rev_joint_idxs
            rev_joint_names_to_use = rev_joint_names

        for rev_joint_idx,rev_joint_name in zip(rev_joint_idxs_to_use,rev_joint_names_to_use):
            axis_joint      = self.model.jnt_axis[rev_joint_idx]
            p_joint,R_joint = self.get_pR_joint(joint_name=rev_joint_name)
            axis_world      = R_joint@axis_joint
            axis_rgba       = np.append(np.eye(3)[:,np.argmax(np.abs(axis_joint))],alpha)
            self.plot_arrow_fr2to(
                p_fr = p_joint,
                p_to = p_joint+rate*axis_len*axis_world,
                r    = rate*axis_r,
                rgba = axis_rgba
            )
            if print_name:
                self.plot_text(p=p_joint,label=rev_joint_name)
                
    def get_contact_body_names(self):
        """
        获取每个接触所涉及的刚体名称。

        返回:
            list: 每个接触对应的 [body1, body2] 对组成的列表。
        """
        contact_body_names = []
        for c_idx in range(self.data.ncon):
            contact = self.data.contact[c_idx]
            contact_body1 = self.body_names[self.model.geom_bodyid[contact.geom1]]
            contact_body2 = self.body_names[self.model.geom_bodyid[contact.geom2]]
            contact_body_names.append([contact_body1,contact_body2])
        return contact_body_names
    
    def get_contact_info(self,must_include_prefix=None,must_exclude_prefix=None):
        """
        获取详细的接触信息，包括位置、力以及所涉及的几何体和刚体。
        “接触（contact）”指两个几何体碰到一起的事件，比如夹爪碰到方块、物体放到桌面上。
        MuJoCo 每一步会算出当前所有接触点，存在 data.contact 里；本函数把它们整理成
        易用的列表：每个接触的位置、接触力、以及参与碰撞的两个几何体/刚体的名字。

        参数:
            must_include_prefix (str): 仅包含其中某个几何体名称以该前缀开头的接触。
            must_exclude_prefix (str): 排除几何体名称以该前缀开头的接触。

        返回:
            tuple: (p_contacts, f_contacts, geom1s, geom2s, body1s, body2s)
        """
        p_contacts = []
        f_contacts = []
        geom1s = []
        geom2s = []
        body1s = []
        body2s = []
        for c_idx in range(self.data.ncon):
            contact   = self.data.contact[c_idx]
            # 接触位置和坐标系朝向
            p_contact = contact.pos # 接触位置
            R_frame   = contact.frame.reshape(( 3,3))
            # 接触力
            f_contact_local = np.zeros(6,dtype=np.float64)
            mujoco.mj_contactForce(self.model,self.data,0,f_contact_local)
            f_contact = R_frame @ f_contact_local[:3] # 全局坐标系下
            # 发生接触的几何体
            contact_geom1 = self.geom_names[contact.geom1]
            contact_geom2 = self.geom_names[contact.geom2]
            contact_body1 = self.body_names[self.model.geom_bodyid[contact.geom1]]
            contact_body2 = self.body_names[self.model.geom_bodyid[contact.geom2]]
            # 追加
            if must_include_prefix is not None:
                if (contact_geom1[:len(must_include_prefix)] == must_include_prefix) or \
                (contact_geom2[:len(must_include_prefix)] == must_include_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
                    body1s.append(contact_body1)
                    body2s.append(contact_body2)
            elif must_exclude_prefix is not None:
                if (contact_geom1[:len(must_exclude_prefix)] != must_exclude_prefix) and \
                    (contact_geom2[:len(must_exclude_prefix)] != must_exclude_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
                    body1s.append(contact_body1)
                    body2s.append(contact_body2)
            else:
                p_contacts.append(p_contact)
                f_contacts.append(f_contact)
                geom1s.append(contact_geom1)
                geom2s.append(contact_geom2)
                body1s.append(contact_body1)
                body2s.append(contact_body2)
        return p_contacts,f_contacts,geom1s,geom2s,body1s,body2s

    def print_contact_info(self,must_include_prefix=None):
        """
        打印满足指定条件的接触信息。

        参数:
            must_include_prefix (str): 过滤条件，仅包含几何体名称以该前缀开头的接触。

        返回:
            None
        """
        # 获取接触信息
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        for (p_contact,f_contact,geom1,geom2,body1,body2) in zip(p_contacts,f_contacts,geom1s,geom2s,body1s,body2s):
            print ("Tick:[%d] 刚体接触:[%s]-[%s]"%(self.tick,body1,body2))

    def plot_arrow_contact(self,p,uv,r_arrow=0.03,h_arrow=0.3,rgba=[1,0,0,1],label=''):
        """
        在给定接触点处绘制一个表示接触力的箭头。

        参数:
            p (np.array): 接触位置。
            uv (np.array): 表示力方向的单位向量。
            r_arrow (float): 箭头的半径。
            h_arrow (float): 箭头的长度。
            rgba (list): RGBA 颜色。
            label (str): 可选的标签。

        返回:
            None
        """
        p_a = np.copy(np.array([0,0,1]))
        p_b = np.copy(uv)
        p_a_norm = np.linalg.norm(p_a)
        p_b_norm = np.linalg.norm(p_b)
        if p_a_norm > 1e-9: p_a = p_a/p_a_norm
        if p_b_norm > 1e-9: p_b = p_b/p_b_norm
        v = np.cross(p_a,p_b)
        S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        if np.linalg.norm(v) == 0:
            R = np.eye(3,3)
        else:
            R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))

        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r_arrow,r_arrow,h_arrow],
            rgba  = rgba,
            label = label
        )

    def plot_joints(
            self,
            joint_names      = None,
            plot_axis        = True,
            axis_len         = 0.1,
            axis_width       = 0.01,
            axis_rgba        = None,
            plot_joint_names = False,
        ):
        """
        绘制多个关节的坐标系。

        参数:
            joint_names (list): 关节名称列表。若为 None，则绘制所有关节。
            plot_axis (bool): 是否显示坐标轴。
            axis_len (float): 坐标轴长度。
            axis_width (float): 坐标轴宽度。
            axis_rgba (list): RGBA 颜色。
            plot_joint_names (bool): 是否打印关节名称。

        返回:
            None
        """
        if joint_names is None:
            joint_names = self.joint_names
        for joint_name in joint_names:
            if joint_name is not None:
                if plot_joint_names:
                    label = joint_name
                else:
                    label = None
                self.plot_joint_T(
                    joint_name,
                    plot_axis  = plot_axis,
                    axis_len   = axis_len,
                    axis_width = axis_width,
                    axis_rgba  = axis_rgba,
                    label      = label,
                )

    def plot_contact_info(
            self,
            must_include_prefix = None,
            plot_arrow          = True,
            r_arrow             = 0.005,
            h_arrow             = 0.1,
            rate                = 1.0,
            plot_sphere         = False,
            r_sphere            = 0.02,
            rgba_contact        = [1,0,0,1],
            print_contact_body  = False,
            print_contact_geom  = False,
            verbose             = False
        ):
        """
        可视化接触力，并可选地显示接触标签。

        参数:
            must_include_prefix (str): 接触的过滤条件。
            plot_arrow (bool): 是否为接触力绘制箭头。
            r_arrow (float): 箭头半径。
            h_arrow (float): 箭头长度。
            rate (float): 缩放因子。
            plot_sphere (bool): 是否在接触点处绘制球体。
            r_sphere (float): 球体半径。
            rgba_contact (list): 接触标记的 RGBA 颜色。
            print_contact_body (bool): 是否显示发生接触的刚体名称。
            print_contact_geom (bool): 是否显示发生接触的几何体名称。
            verbose (bool): 若为 True，则同时将接触信息打印到控制台。

        返回:
            None
        """
        # 获取接触信息
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        # 渲染接触信息
        for (p_contact,f_contact,geom1,geom2,body1,body2) in zip(p_contacts,f_contacts,geom1s,geom2s,body1s,body2s):
            f_norm = np.linalg.norm(f_contact)
            f_uv   = f_contact / (f_norm+1e-8)
            # h_arrow = 0.3 # f_norm*0.05
            if plot_arrow:
                self.plot_arrow_contact(
                    p       = p_contact,
                    uv      = f_uv,
                    r_arrow = rate*r_arrow,
                    h_arrow = rate*h_arrow,
                    rgba    = rgba_contact,
                    label   = '',
                )
                self.plot_arrow_contact(
                    p       = p_contact,
                    uv      = -f_uv,
                    r_arrow = rate*r_arrow,
                    h_arrow = rate*h_arrow,
                    rgba    = rgba_contact,
                    label   = '',
                )
            if plot_sphere:
                # contact_label = '[%s]-[%s]'%(body1,body2)
                contact_label = ''
                self.plot_sphere(p=p_contact,r=r_sphere,rgba=rgba_contact,label=contact_label)
            if print_contact_body:
                label = '[%s]-[%s]'%(body1,body2)
            elif print_contact_geom:
                label = '[%s]-[%s]'%(geom1,geom2)
            else:
                label = ''
        # 打印
        if verbose:
            self.print_contact_info(must_include_prefix=must_include_prefix)
            
    def plot_xy_heading(
            self,
            xy,
            heading,
            r             = 0.01,
            arrow_len     = 0.1,
            rgba          = (1,0,0,1),
            plot_sphere   = False,
            plot_arrow    = True,
        ):
        """
        绘制一个 2D 点，并配以表示其朝向的箭头。

        参数:
            xy (np.array): (x, y) 位置。
            heading (float): 朝向角（弧度）。
            r (float): 点标记的半径。
            arrow_len (float): 朝向箭头的长度。
            rgba (tuple): RGBA 颜色。
            plot_sphere (bool): 是否在该点处绘制球体。
            plot_arrow (bool): 是否绘制朝向箭头。

        返回:
            None
        """
        dir_vec = np.array([np.cos(heading),np.sin(heading)])
        if plot_sphere:
            self.plot_sphere(p=np.append(xy,[0]),r=r,rgba=rgba)
        if plot_arrow:
            self.plot_arrow_fr2to(
                p_fr = np.append(xy,[0]),
                p_to = np.append(xy+arrow_len*dir_vec,[0]),
                r    = r,
                rgba = rgba,
            )
                    
    def plot_xy_heading_traj(
            self,
            xy_traj,
            heading_traj,
            r             = 0.01,
            arrow_len     = 0.1,
            rgba          = None,
            cmap_name     = 'gist_rainbow',
            alpha         = 0.5,
            plot_sphere   = False,
            plot_arrow    = True,
            plot_cylinder = False,
        ):
        """
        在 XY 平面上绘制一条轨迹，并配以相应的朝向箭头。

        参数:
            xy_traj (np.array): (x, y) 位置序列。
            heading_traj (np.array): 朝向角序列。
            r (float): 标记半径。
            arrow_len (float): 朝向箭头的长度。
            rgba (list): RGBA 颜色；若为 None，则使用颜色映射表。
            cmap_name (str): 要使用的颜色映射表名称。
            alpha (float): 颜色映射的透明度。
            plot_sphere (bool): 是否在各点处绘制球体。
            plot_arrow (bool): 是否为朝向绘制箭头。
            plot_cylinder (bool): 是否用圆柱体连接各点。

        返回:
            None
        """
        L = len(xy_traj)
        colors = get_colors(n_color=L,cmap_name=cmap_name,alpha=alpha)
        for idx in range(L):
            xy_i,heading_i = xy_traj[idx],heading_traj[idx]
            if rgba is None:
                rgba = colors[idx]
            dir_vec_i = np.array([np.cos(heading_i),np.sin(heading_i)])
            if plot_sphere:
                self.plot_sphere(p=np.append(xy_i,[0]),r=r,rgba=rgba)
            if plot_arrow:
                self.plot_arrow_fr2to(
                    p_fr = np.append(xy_i,[0]),
                    p_to = np.append(xy_i+arrow_len*dir_vec_i,[0]),
                    r    = r,
                    rgba = rgba,
                )
            if plot_cylinder:
                if idx > 1:
                    xy_prev = xy_traj[idx-1]
                    self.plot_cylinder_fr2to(
                        p_fr = np.append(xy_prev,[0]),
                        p_to = np.append(xy_i,[0]),
                        r    = r,
                        rgba = rgba,
                    )
            
    # ----------------------------------------------------------------------
    # 重要：同一个关节在不同场合有不同的“下标体系”，初学者很容易混淆，这里说明：
    #   - idxs_fwd（qpos 下标）：用于读写位置 data.qpos，配合 forward()。
    #   - idxs_jac（qvel/dof 下标）：用于雅可比矩阵的列、速度 data.qvel。
    #   - idxs_step（执行器/控制下标）：用于发控制信号 data.ctrl，配合 step()。
    # 为什么不一样？因为自由关节朝向用四元数占 qpos 4 个位、却只占 qvel/dof 3 个位，
    # 所以位置维度(nq)和速度维度(nv)不一致；而控制下标又对应的是“执行器”而非关节本身。
    # 下面三个 get_idxs_* 就是按关节名分别取出这三类下标。
    # ----------------------------------------------------------------------
    def get_idxs_fwd(self,joint_names):
        """
        根据关节名称获取用于前向运动学的关节索引（qpos 中的下标）。

        参数:
            joint_names (list): 关节名称列表。

        返回:
            list: 与各关节对应的索引。

        示例:
            env.forward(q=q,joint_idxs=idxs_fwd) # <= 此处
        """
        return [self.model.joint(jname).qposadr[0] for jname in joint_names]

    def get_idxs_jac(self,joint_names):
        """
        根据关节名称获取用于雅可比计算的关节索引。

        参数:
            joint_names (list): 关节名称列表。

        返回:
            list: 与各关节对应的索引。
        """
        return [self.model.joint(jname).dofadr[0] for jname in joint_names]

    def get_idxs_step(self,joint_names):
        """
        根据关节名称获取在仿真步进时施加控制所用的索引。

        参数:
            joint_names (list): 关节名称列表。

        返回:
            list: 控制索引。
        """
        return [self.ctrl_qpos_names.index(jname) for jname in joint_names]

    def get_qpos(self):
        """
        获取当前关节位置。

        返回:
            np.array: 关节位置。
        """
        return self.data.qpos.copy() # [n_qpos]

    def get_qvel(self):
        """
        获取当前关节速度。

        返回:
            np.array: 关节速度。
        """
        return self.data.qvel.copy() # [n_qvel]

    def get_qacc(self):
        """
        获取当前关节加速度。

        返回:
            np.array: 关节加速度。
        """
        return self.data.qacc.copy() # [n_qacc]

    def get_qpos_joint(self,joint_name):
        """
        获取某个特定关节的位置。

        参数:
            joint_name (str): 关节的名称。

        返回:
            np.array: 关节位置。
        """
        addr = self.model.joint(joint_name).qposadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        qpos = self.data.qpos[addr:addr+L]
        return qpos
    
    def get_qvel_joint(self,joint_name):
        """
        获取某个特定关节的速度。

        参数:
            joint_name (str): 关节的名称。

        返回:
            np.array: 关节速度。
        """
        addr = self.model.joint(joint_name).dofadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        if L > 1: L = 6
        qvel = self.data.qvel[addr:addr+L]
        return qvel

    def get_qpos_joints(self,joint_names):
        """
        获取多个关节的位置。

        参数:
            joint_names (list): 关节名称列表。

        返回:
            np.array: 关节位置。
        """
        return np.array([self.get_qpos_joint(joint_name) for joint_name in joint_names]).squeeze()

    def get_qvel_joints(self,joint_names):
        """
        获取某个特定关节的速度。

        参数:
            joint_name (str): 关节的名称。

        返回:
            np.array: 关节速度。
        """
        return np.array([self.get_qvel_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def get_q_couple(
        self,
        q_raw,
        coupled_joint_idxs_list    = None,
        coupled_joint_names_list   = None,
        coupled_joint_weights_list = None,
        ):
        """
        根据原始关节位置和耦合定义计算耦合后的关节位置。
        “耦合（couple）”指有些关节并非独立，而是会按固定比例联动（常见于多指灵巧手：
        一根手指里几个指节由一根肌腱带动，按权重一起弯曲）。本函数把这些联动组里的关节
        按权重重新分配，得到一组“符合联动关系”的关节角度。

        参数:
            q_raw (np.array): 原始关节位置向量。
            coupled_joint_idxs_list (list): 每个耦合组的关节索引列表（列表的列表）。
            coupled_joint_names_list (list): 使用关节名称的另一种指定方式。
            coupled_joint_weights_list (list): 每个耦合组的权重列表。

        返回:
            np.array: 应用耦合后修改的关节位置向量。

        用法?
            耦合关节位置
            示例:
            # 应用关节位置耦合
            coupled_joint_idxs_list = [
                [22,23],[24,25,26],[27,28,29],[30,31,32],[33,34,35],
                [45,46],[47,48,49],[50,51,52],[53,54,55],[56,57,58]]
            coupled_joint_weights_list = [
                [1,1],[1,3,2],[1,3,2],[1,3,2],[1,3,2],
                [1,1],[1,3,2],[1,3,2],[1,3,2],[1,3,2]]
            q_couple = env.get_q_couple(
                q_raw=env.data.qpos,
                coupled_joint_idxs_list=coupled_joint_idxs_list,
                coupled_joint_weights_list=coupled_joint_weights_list)
        """
        q_couple = q_raw.copy()
        if coupled_joint_idxs_list is not None:
            for i in range(len(coupled_joint_idxs_list)): # 遍历每个耦合组
                coupled_joint_idxs    = coupled_joint_idxs_list[i]
                coupled_joint_weights = coupled_joint_weights_list[i]
                joint_sum = 0
                for j in range(len(coupled_joint_idxs)):
                    joint_sum += q_raw[coupled_joint_idxs[j]]
                joint_sum /= np.sum(coupled_joint_weights)
                for k in range(len(coupled_joint_idxs)):
                    q_couple[coupled_joint_idxs[k]] = joint_sum*coupled_joint_weights[k] # 分配耦合后的关节位置
        if coupled_joint_names_list is not None:
            for i in range(len(coupled_joint_names_list)): # 遍历每个耦合组
                coupled_joint_names   = coupled_joint_names_list[i]
                coupled_joint_idxs    = get_idxs(self.joint_names,coupled_joint_names)
                coupled_joint_weights = coupled_joint_weights_list[i]
                joint_sum = 0
                for j in range(len(coupled_joint_idxs)):
                    joint_sum += q_raw[coupled_joint_idxs[j]]
                joint_sum /= np.sum(coupled_joint_weights)
                for k in range(len(coupled_joint_idxs)):
                    q_couple[coupled_joint_idxs[k]] = joint_sum*coupled_joint_weights[k] # 分配耦合后的关节位置
        return q_couple

    def get_ctrl(self,ctrl_names):
        """
        获取指定执行器的控制值。

        参数:
            ctrl_names (list): 控制名称列表。

        返回:
            np.array: 控制值。
        """
        idxs = get_idxs(self.ctrl_names,ctrl_names)
        return np.array([self.data.ctrl[idx] for idx in idxs]).squeeze()
    
        
    def set_qpos_joints(self,joint_names,qpos):
        """
        设置指定关节的关节位置并更新前向运动学。

        参数:
            joint_names (list): 关节名称。
            qpos (np.array): 关节位置。

        返回:
            None
        """
        joint_idxs = self.get_idxs_fwd(joint_names)
        self.data.qpos[joint_idxs] = qpos
        mujoco.mj_forward(self.model,self.data)

    def set_ctrl(self,ctrl_names,ctrl,nstep=1):
        """
        为指定执行器设置控制输入并执行若干步仿真。

        参数:
            ctrl_names (list): 控制名称。
            ctrl (np.array): 控制值。
            nstep (int): 要执行的仿真步数。

        返回:
            None
        """
        ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)
        self.data.ctrl[ctrl_idxs] = ctrl
        mujoco.mj_step(self.model,self.data,nstep=nstep)

    def viewer_pause(self):
        """
        暂停查看器的渲染循环。

        返回:
            None
        """
        self.viewer._paused = True

    def viewer_resume(self):
        """
        恢复查看器的渲染循环。

        返回:
            None
        """
        self.viewer._paused = False

    def get_viewer_mouse_xy(self):
        """
        获取查看器中当前鼠标的 (x, y) 坐标。

        返回:
            np.array: 鼠标坐标。
        """
        viewer_mouse_xy = np.array([self.viewer._last_mouse_x,self.viewer._last_mouse_y])
        return viewer_mouse_xy
    
    def get_xyz_left_double_click(self,verbose=False,fovy=45):
        """
            获取鼠标左键双击处对应的 3D 世界坐标 (x,y,z)。
            原理：鼠标点的是屏幕上的 2D 像素，但我们想知道它打到 3D 场景里的哪个点。
            做法是用当前视角渲染一张带深度的图，得到“每个像素对应的世界坐标”，
            再按鼠标像素位置取出那个点，就把 2D 点击“反投影”成了 3D 坐标。
            常用于交互式拾取目标位置（比如双击桌面某处，让机械臂去那里）。
            :return self.xyz_left_double_click,flag_click:
        """
        flag_click = False
        if self.viewer._left_double_click_pressed: # 左键双击
            viewer_mouse_xy = self.get_viewer_mouse_xy()  # 鼠标当前像素坐标
            # 渲染带深度的画面，xyz_img_world[行,列] 即该像素对应的世界坐标
            _,_,_,_,xyz_img_world = self.get_egocentric_rgbd_pcd(fovy=fovy)
            self.xyz_left_double_click = xyz_img_world[int(viewer_mouse_xy[1]),int(viewer_mouse_xy[0])]
            self.viewer._left_double_click_pressed = False
            flag_click = True
            if verbose:
                print ("左键双击:(%.3f,%.3f,%.3f)"%
                       (self.xyz_left_double_click[0],self.xyz_left_double_click[1],self.xyz_left_double_click[2]))
        return self.xyz_left_double_click,flag_click
    
    def is_left_double_clicked(self):
        """
        检查是否发生了左键双击事件。

        返回:
            bool: 若检测到则为 True，否则为 False。
        """
        if self.viewer._left_double_click_pressed: # 左键双击
            viewer_mouse_xy = self.get_viewer_mouse_xy()
            _,_,_,_,xyz_img_world = self.get_egocentric_rgbd_pcd()
            self.xyz_left_double_click = xyz_img_world[int(viewer_mouse_xy[1]),int(viewer_mouse_xy[0])]
            self.viewer._left_double_click_pressed = False # 切换标志
            return True
        else:
            return False
    
    def get_xyz_right_double_click(self,verbose=False,fovy=45):
        """
        获取与右键双击事件对应的 3D 世界坐标。

        参数:
            verbose (bool): 若为 True，则打印被点击的坐标。
            fovy (float): 用于投影的视场角。

        返回:
            tuple: (xyz, flag_click)
        """
        flag_click = False
        if self.viewer._right_double_click_pressed: # 右键双击
            viewer_mouse_xy = self.get_viewer_mouse_xy()
            _,_,_,_,xyz_img_world = self.get_egocentric_rgbd_pcd(fovy=fovy)
            self.xyz_right_double_click = xyz_img_world[int(viewer_mouse_xy[1]),int(viewer_mouse_xy[0])]
            self.viewer._right_double_click_pressed = False
            flag_click = True
            if verbose:
                print ("右键双击:(%.3f,%.3f,%.3f)"%
                       (self.xyz_right_double_click[0],self.xyz_right_double_click[1],self.xyz_right_double_click[2]))
        return self.xyz_right_double_click,flag_click
    
    def is_right_double_clicked(self):
        """
        检查是否发生了右键双击事件。

        返回:
            bool: 若检测到则为 True，否则为 False。
        """
        if self.viewer._right_double_click_pressed: # 右键双击
            viewer_mouse_xy = self.get_viewer_mouse_xy()
            _,_,_,_,xyz_img_world = self.get_egocentric_rgbd_pcd()
            self.xyz_right_double_click = xyz_img_world[int(viewer_mouse_xy[1]),int(viewer_mouse_xy[0])]
            self.viewer._right_double_click_pressed = False # 切换标志
            return True
        else:
            return False
        
    def get_body_name_closest(self,xyz,body_names=None,verbose=False):
        """
        确定哪个刚体距离给定的 3D 点最近。

        参数:
            xyz (np.array): 查询的 3D 点。
            body_names (list): 要考虑的刚体名称列表（若为 None，则考虑所有刚体）。
            verbose (bool): 若为 True，则打印所选的刚体。

        返回:
            tuple: (body_name_closest, p_body_closest)
        """
        if body_names is None:
            body_names = self.body_names
        dists = np.zeros(len(body_names))
        p_body_list = []
        for body_idx,body_name in enumerate(body_names):
            p_body = self.get_p_body(body_name=body_name)
            dist = np.linalg.norm(p_body-xyz)
            dists[body_idx] = dist # 追加
            p_body_list.append(p_body) # 追加
        idx_min = np.argmin(dists)
        body_name_closest = body_names[idx_min]
        p_body_closest = p_body_list[idx_min]
        if verbose:
            print ("[%s] 已选中"%(body_name_closest))
        return body_name_closest,p_body_closest
    
    # ----------------------------------------------------------------------
    # 逆运动学（Inverse Kinematics, IK）相关。先解释几个概念：
    #   - 正运动学：已知关节角度 -> 求末端在哪（前面 forward 干的事）。
    #   - 逆运动学：反过来，已知“想让末端到达的目标位姿”-> 求各关节该转到多少度。
    #     这通常没有简单公式，常用“迭代逼近”的办法一点点逼近目标。
    #   - 雅可比矩阵 J：描述“关节微小变化”如何引起“末端微小移动/转动”的对应关系，
    #     即 (末端速度) = J × (关节速度)。它是从关节空间到笛卡尔空间的“放大镜/换算表”。
    # 迭代 IK 的套路：算当前末端与目标的误差 err，再用 J 把 err 换算成应该调整的关节增量 dq，
    # 反复执行直到误差足够小。下面几个方法就是这套流程的零件。
    # ----------------------------------------------------------------------
    def get_J_body(self,body_name):
        """
        计算指定刚体的雅可比矩阵（位置部分 J_p 和旋转部分 J_R）。
        J 的列数等于自由度数；J_p 把关节速度映射成末端的线速度，J_R 映射成角速度。

        参数:
            body_name (str): 刚体的名称。

        返回:
            tuple: (J_p, J_R, J_full)，其中 J_full 是位置和旋转部分上下堆叠后的 6×nv 雅可比矩阵。
        """
        J_p = np.zeros((3,self.n_dof)) # nv: 自由度数
        J_R = np.zeros((3,self.n_dof))
        mujoco.mj_jacBody(self.model,self.data,J_p,J_R,self.data.body(body_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_J_geom(self,geom_name):
        """
        计算指定几何体的雅可比矩阵。

        参数:
            geom_name (str): 几何体的名称。

        返回:
            tuple: (J_p, J_R, J_full)
        """
        J_p = np.zeros((3,self.n_dof)) # nv: 自由度数
        J_R = np.zeros((3,self.n_dof))
        mujoco.mj_jacGeom(self.model,self.data,J_p,J_R,self.data.geom(geom_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_ik_ingredients(
            self,
            body_name = None,
            geom_name = None,
            p_trgt    = None,
            R_trgt    = None,
            IK_P      = True,
            IK_R      = True,
        ):
        """
        计算逆运动学所需的“原料”：雅可比矩阵 J 和误差向量 err。
        这是 IK 一次迭代的第一步：拿当前末端位姿和目标位姿做对比，算出还差多少（err），
        同时取出对应的雅可比 J。之后把它们交给 damped_ls 求出关节该怎么调整。
        IK_P / IK_R 控制是否考虑“位置误差 / 朝向误差”（有时只关心末端到没到点、不关心朝向）。

        参数:
            body_name (str): 刚体的名称（若提供）。
            geom_name (str): 几何体的名称（若提供）。
            p_trgt (np.array): 目标位置。
            R_trgt (np.array): 目标旋转矩阵。
            IK_P (bool): 是否包含位置误差。
            IK_R (bool): 是否包含朝向误差。

        返回:
            tuple: (J, err)，其中 J 是雅可比矩阵，err 是误差向量。
        """

        # 没给目标位置/朝向，就关掉对应的误差项
        if p_trgt is None: IK_P = False
        if R_trgt is None: IK_R = False

        if body_name is not None:
            J_p,J_R,J_full = self.get_J_body(body_name=body_name)
            p_curr,R_curr = self.get_pR_body(body_name=body_name)
        if geom_name is not None:
            J_p,J_R,J_full = self.get_J_geom(geom_name=geom_name)
            p_curr,R_curr = self.get_pR_geom(geom_name=geom_name)
        if (body_name is not None) and (geom_name is not None):
            print ("[get_ik_ingredients] body_name:[%s] geom_name:[%s] 不能同时非 None!"%(body_name,geom_name))
        # 根据要不要位置/朝向，拼出对应的 J 和 err：
        # 位置误差就是目标位置减当前位置；朝向误差则换算成一个“需要绕哪个轴转多少”的向量 w_err。
        if (IK_P and IK_R):
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)   # 当前朝向到目标朝向的相对旋转
            w_err = R_curr @ r2w(R_err)              # 把相对旋转转成世界系下的旋转误差向量
            J     = J_full
            err   = np.concatenate((p_err,w_err))
        elif (IK_P and not IK_R):
            p_err = (p_trgt-p_curr)
            J     = J_p
            err   = p_err
        elif (not IK_P and IK_R):
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_R
            err   = w_err
        else:
            J   = None
            err = None
        return J,err
    
    def damped_ls(self,J,err,eps=1e-6,stepsize=1.0,th=5*np.pi/180.0):
        """
        使用阻尼最小二乘法（Damped Least Squares）由误差 err 求出关节增量 dq。
        直观理解：我们想找一组关节调整量 dq，使末端朝目标移动（即 J·dq ≈ err）。
        直接求逆在某些姿态（“奇异位形”）会数值爆炸，于是加一个小阻尼项 eps 让解更稳定、
        代价是稍微保守一点。最后再用 trim_scale 限制单步幅度，避免一步迈太大跳过头。

        参数:
            J (np.array): 雅可比矩阵。
            err (np.array): 误差向量（当前末端与目标的差距）。
            eps (float): 阻尼因子（越大越稳但越慢）。
            stepsize (float): 步长乘子。
            th (float): 单步关节增量的上限阈值（弧度）。

        返回:
            np.array: 计算得到的关节增量 (dq)。
        """
        # 解带阻尼的法方程：dq = (JᵀJ + eps·I)⁻¹ Jᵀ err
        dq = stepsize*np.linalg.solve(a=(J.T@J)+eps*np.eye(J.shape[1]),b=J.T@err)
        dq = trim_scale(x=dq,th=th)   # 限幅，防止单步动作过大
        return dq
    
    def check_key_pressed(self,char=None):
        """
        用于检查某个键是否被按下的高层函数。

        参数:
            char (str): 要检查的单个字符。
        返回:
            bool: 若该键被按下则为 True，否则为 False。
        """
        if self.viewer._is_key_pressed:
            if self.get_key_pressed() == char:
                self.viewer._is_key_pressed = False
                return True
            else:
                return False
        else:
            return False
        
    def get_key_pressed(self):
        """
        获取最后被按下的键。

        返回:
            str: 被按下的键。
        """
        return self.viewer._key_pressed

    def open_interactive_viewer(self):
        """
        为仿真启动一个交互式查看器。

        返回:
            None
        """
        from mujoco import viewer
        viewer.launch(self.model)

    def compensate_gravity(self,root_body_names):
        """
            重力补偿：给机器人额外施加一组力，正好抵消重力，让它“失重”般悬停，
            不会因为自重而往下塌。常用于示教/拖动场景，让人能轻松拖动机械臂摆姿势。
            root_body_names：要补偿的若干子树根刚体（会对其下整条运动链补偿）。
        """
        qfrc_applied = self.data.qfrc_applied
        qfrc_applied[:] = 0.0  # 不要累加此前调用的结果。
        jac = np.empty((3,self.model.nv))
        for root_body_name in root_body_names:
            subtree_id = self.model.body(root_body_name).id
            total_mass = self.model.body_subtreemass[subtree_id]  # 这条子树的总质量
            # 子树质心相对各关节的雅可比：把“支撑住质心所需的力”换算成各关节上的力矩
            mujoco.mj_jacSubtreeCom(self.model,self.data,jac,subtree_id)
            # 施加 -重力×质量 对应的关节力，恰好抵消重力
            qfrc_applied[:] -= self.model.opt.gravity * total_mass @ jac
            
    def set_rangefinder_rgba(self,rgba=(1,1,0,0.1)):
        """
        设置测距仪（rangefinder）可视化的 RGBA 颜色。

        参数:
            rgba (tuple): RGBA 格式的颜色。

        返回:
            None
        """
        self.model.vis.rgba.rangefinder = np.array(rgba,dtype=np.float32)

    def tic(self):
        """
        启动一个用于性能测量的计时器。

        返回:
            None
        """
        self.tt.tic()

    def toc(self):
        """
        返回自上次调用 tic() 以来经过的时间。

        返回:
            float: 经过的时间（秒）。
        """
        return self.tt.toc()

    def sync_sim_wall_time(self):
        """
        将仿真时间与墙钟时间同步。

        返回:
            None
        """
        time_diff = self.get_sim_time() - self.get_wall_time()
        if time_diff > 0: time.sleep(time_diff)

    def get_key_pressed_list(self):
        """
        获取已被按下的键的列表。

        返回:
            list: 已按下的键的列表。
        """
        return list(self.viewer._key_pressed_set)

    def get_key_repeated_list(self):
        """
        获取已被持续（重复）按下的键的列表。

        返回:
            list: 重复按下的键的列表。
        """
        return list(self.viewer._key_repeated_set)

    def pop_key_pressed_list(self,key=None):
        """
        从已按下的键列表中弹出最后一个键。

        返回:
            str: 最后被按下的键。
        """
        if key is not None:
            self.viewer._key_pressed_set.discard(key)

    def is_key_pressed_once(self,key=None,key_list=None):
        """
        检查某个键这一帧是否“刚被按下”（按一次只触发一次）。
        与 is_key_pressed_repeat 的区别：本方法判断后会把该键从集合里移除，
        所以按住不放也只会返回一次 True，适合“按一下切换一个状态”的场景。

        参数:
            key (str): 要检查的键。

        返回:
            bool: 若该键被按下则为 True，否则为 False。
        """
        if key is not None:
            if key in self.get_key_pressed_list():
                self.pop_key_pressed_list(key=key)
                return True
            else:
                return False
        elif key_list is not None:
            for key in key_list:
                if key in self.get_key_pressed_list():
                    self.pop_key_pressed_list(key=key)
                    return True
            return False
        else:
            return False
        
    def is_key_pressed_repeat(self,key=None,key_list=None):
        """
        检查某个（些）键当前是否“正被按住”（按住期间会持续返回 True）。
        适合“按住持续移动”的场景，比如按住方向键让机械臂一直缓慢移动。

        参数:
            key (str, 可选): 要检查的单个键。
            key_list (list of str, 可选): 要检查的键列表。
                如果同时提供 key 和 key_list，则只使用 'key'。

        返回:
            bool: 若该键（或 key_list 中的任意键）被按下则为 True，否则为 False。
        """
        if key is not None:
            return key in self.get_key_pressed_list()+self.get_key_repeated_list()
        elif key_list is not None:
            for key in key_list:
                if key in self.get_key_pressed_list()+self.get_key_repeated_list():
                    return True
            return False
        else:
            return False
    