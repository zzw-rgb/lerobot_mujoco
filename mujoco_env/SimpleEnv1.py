import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

class SimpleEnv:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                seed = None):
        """
        参数:
            xml_path: str, xml 文件的路径
            action_type: str, 动作空间类型, 'eef_pose'、'delta_joint_angle' 或 'joint_angle'
            state_type: str, 状态空间类型, 'joint_angle' 或 'ee_pose'
            seed: int, 随机数生成器的种子
        """
        # 加载 xml 文件
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        self.joint_names = ['joint1',
                    'joint2',
                    'joint3',
                    'joint4',
                    'joint5',
                    'joint6',
                    'joint7',]
        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        '''
        初始化查看器
        '''
        self.env.reset()
        self.env.init_viewer(
            azimuth           = 0,     # 从机械臂正后方看向桌面/杯子(操作视角)
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    def reset(self, seed = None):
        '''
        重置环境
        将机器人移动到初始位置, 根据种子设置物体位置
        '''
        if seed is not None: np.random.seed(seed)
        q_init = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Franka Panda home 位姿(7 关节)
        q_zero,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = 'tcp_link',
            q_init       = q_init, # 从零位姿开始求解 ik
            p_trgt       = np.array([0.3,0.0,0.95]),
            R_trgt       = rpy2r(np.deg2rad([180,0,0])),  # 末端竖直朝下
        )
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)

        # 设置物体位置
        obj_names = self.env.get_body_names(prefix='body_obj_')
        n_obj = len(obj_names)
        obj_xyzs = sample_xyzs(
            n_obj,
            x_range   = [+0.28,+0.4],  # 近端 0.24 franka 够不到, 收窄到 0.28 保证全可达
            y_range   = [-0.2,+0.2],
            z_range   = [0.85,0.85],
            min_dist  = 0.2,
            xy_margin = 0.0
        )
        for obj_idx in range(n_obj):
            self.env.set_p_base_body(body_name=obj_names[obj_idx],p=obj_xyzs[obj_idx,:])
            self.env.set_R_base_body(body_name=obj_names[obj_idx],R=np.eye(3,3))
        self.env.forward(increase_tick=False)

        # 设置机器人的初始位姿
        self.last_q = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, np.array([255.0])])  # 底层 MuJoCo 控制量：255=张开
        self.gripper_command = 0.0  # 对外统一语义：0=张开，1=闭合
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        # 清零残留速度/加速度：避免机器人带着上一回合的速度进入 settle 而摆动，实现丝滑初始化
        self.env.data.qvel[:] = 0.0
        self.env.data.qacc[:] = 0.0
        for _ in range(100):
            self.step_env()
        # 物体在 settle 之后已落定，此处重新捕获初始位姿，记录真实落点（而非穿模前的生成高度）
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        # settle 之后重新捕获末端起始位姿：遥操作的位置/姿态增量以此为基准，避免第一步跳变
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        print("初始化完成")
        self.success_counter = 0   # 成功判定的连续保持计数器，每轮 reset 清零
        self.gripper_state = False
        self.past_chars = []

    def step(self, action):
        '''
        在环境中执行一步步进
        参数:
            action: 形状为 (7,) 的 np.array, 要执行的动作
        返回:
            state: np.array, 执行动作后的环境状态
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]

        '''
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'tcp_link',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')
        
        # 策略/数据集始终使用 0=张开、1=闭合；只在这里转成 MuJoCo 的 255~0。
        self.gripper_command = float(np.clip(action[-1], 0.0, 1.0))
        gripper_ctrl = np.array([255.0 * (1.0 - self.gripper_command)])
        self.compute_q = q
        q = np.concatenate([q, gripper_ctrl])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        '''
        从环境中抓取图像
        返回:
            rgb_agent: np.array, 来自智能体视角的 rgb 图像
            rgb_ego: np.array, 来自第一人称视角的 rgb 图像
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='egocentric')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False, idx=0, total=None):
        '''
        渲染环境
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
        # tcp 的 z 轴竖直朝下, 直接用其姿态让绿色光柱竖直
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
        # 在画面上显示采集进度(已保存轮数 / 目标轮数)
        progress = ("Saved %d/%d"%(idx,total)) if total is not None else ("Episode %d"%idx)
        self.env.plot_T(p=np.array([0.1,0.0,1.0]), label=progress, plot_axis=False, plot_sphere=False)
        self.env.render()

    def get_joint_state(self):
        '''
        获取机器人的关节状态
        返回:
            q: np.array, 机器人的关节角 + 夹爪状态 (0 表示张开, 1 表示闭合)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        # 夹住物体时手指无法完全合拢，不能用指缝大小反推开合命令。
        return np.concatenate([qpos, [self.gripper_command]], dtype=np.float32)
    
    def teleop_robot(self):
        '''
        使用键盘遥操作机器人
        返回:
            action: np.array, 要执行的动作
            done: bool, 若用户想要重置遥操作则为 True

        按键:
            ---------     -----------------------
               w       ->        向后
            s  a  d        左     向前     右
            ---------      -----------------------
            在 x, y 平面内

            ---------
            R: 向上移动
            F: 向下移动
            ---------
            在 z 轴方向

            ---------
            Q: 向左倾斜
            E: 向右倾斜
            UP: 向上看
            Down: 向下看
            Right: 向右转
            Left: 向左转
            ---------
            用于旋转

            ---------
            z: 重置
            SPACEBAR: 夹爪张开/闭合
            ---------


        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([-0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0,0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0,-0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action, False
    
    def get_delta_q(self):
        '''
        获取机器人的关节角增量
        返回:
            delta: np.array, 机器人的关节角增量 + 夹爪状态 (0 表示张开, 1 表示闭合)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        return np.concatenate([delta, [self.gripper_command]], dtype=np.float32)

    def is_finish_pressed(self):
        # 检测回车键：手动确认“本轮完成、保存并进入下一轮”
        return self.env.is_key_pressed_once(key=glfw.KEY_ENTER)

    def check_success(self):
        '''
        判断“把杯子稳稳放到盘子上”是否完成。需同时满足并【连续保持约 1.25 秒】：
          杯子在盘子正上方 + 已落到盘面高度 + 夹爪已张开松手 + 末端已抬起 + 杯子基本静止。
        用连续帧计数避免“杯子刚一碰到盘子就草草判成功”。
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        mug_dofadr = self.env.model.body('body_obj_mug_5').dofadr[0]
        mug_speed = np.linalg.norm(self.env.data.qvel[mug_dofadr:mug_dofadr+3])
        placed = (np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.09
                  and abs(p_mug[2] - p_plate[2]) < 0.09
                  and self.env.get_qpos_joint('finger_joint1') > 0.03
                  and self.env.get_p_body('tcp_link')[2] > 0.9
                  and mug_speed < 0.01)
        if placed:
            self.success_counter = getattr(self, 'success_counter', 0) + 1
        else:
            self.success_counter = 0
        return self.success_counter >= 25

    def get_obj_pose(self):
        '''
        返回:
            p_mug: np.array, 杯子的位置
            p_plate: np.array, 盘子的位置
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        return p_mug, p_plate
    
    def set_obj_pose(self, p_mug, p_plate):
        '''
        设置物体位姿
        参数:
            p_mug: np.array, 杯子的位置
            p_plate: np.array, 盘子的位置
        '''
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=p_mug)
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        获取机器人的末端执行器位姿 + 夹爪状态
        '''
        p, R = self.env.get_pR_body(body_name='tcp_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)
