"""环境与代码一致性自检。

逐项验证仿真场景、策略配置和（可选）本地数据集是否满足代码假设：

1. 两个场景 XML 可被 MuJoCo 编译（同时校验全部 include/mesh/texture 可用）；
2. 执行器数量与顺序：8 个执行器，前 7 个按序驱动 joint1..joint7，
   第 8 个是 tendon 夹爪且 ctrlrange 为 [0,255]——SimpleEnv 的
   ``data.ctrl[:] = [7 关节角, 255*(1-夹爪命令)]`` 依赖这一布局；
3. 代码引用的 joint/body/camera 名称存在；杯子/盘子带自由关节；
4. timestep 能整除出 20Hz 采样（采集与部署都按每策略步固定物理步数运行）；
5. IL/VLA 配置文件的已知约束（Diffusion 的 GroupNorm 与预训练权重互斥、
   crop_shape 不超过图像尺寸、horizon 可被 U-Net 下采样整除、
   π0 要求 n_action_steps == chunk_size、ACT 时间集成要求 n_action_steps == 1）；
6. 本地数据集存在时核对 state/action 维度（8 维）与 fps（20）。

运行命令：

    conda activate lerobot
    python verify_setup.py

全部通过打印 ALL CHECKS PASSED 并以 0 退出；否则列出失败项并以 1 退出。
提示：完整场景编译约需 1~2 GB 空闲内存，内存过低会误报第 1 项。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"[{'OK ' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILURES.append(msg)


def verify_scene(xml_rel: str, expect_mug6: bool) -> None:
    import mujoco

    xml_path = PROJECT_ROOT / xml_rel
    print(f"\n===== 场景：{xml_rel} =====")
    if not xml_path.is_file():
        check(False, f"场景文件存在: {xml_path}")
        return
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:
        check(False, f"MuJoCo 编译通过（{exc}）")
        return
    check(True, "MuJoCo 编译通过（含全部 include/mesh/texture）")

    def names(obj_type, count):
        return [mujoco.mj_id2name(model, obj_type, i) for i in range(count)]

    joints = names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    bodies = names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    cams = names(mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)

    # 控制向量布局
    check(model.nu == 8, f"执行器数量 == 8（实际 {model.nu}）")
    if model.nu == 8:
        order_ok = True
        for i in range(7):
            ok = (
                model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                and joints[model.actuator_trnid[i][0]] == f"joint{i + 1}"
            )
            order_ok = order_ok and ok
        check(order_ok, "ctrl[0..6] 依次驱动 joint1..joint7")
        check(
            model.actuator_trntype[7] == mujoco.mjtTrn.mjTRN_TENDON,
            "ctrl[7] 为 tendon 夹爪执行器",
        )
        cr = model.actuator_ctrlrange[7]
        check(abs(cr[0]) < 1e-9 and abs(cr[1] - 255.0) < 1e-9, f"夹爪 ctrlrange == [0,255]（实际 {cr}）")
        gain = float(model.actuator_gainprm[7][0])
        kp = -float(model.actuator_biasprm[7][1])
        check(gain > 0 and kp > 0, f"夹爪位置伺服参数符号正确（gain={gain:.4f}, kp={kp:.1f}）")
        if gain > 0 and kp > 0:
            open_target = gain * 255.0 / kp
            # split tendon = 0.5*finger1 + 0.5*finger2，全开时约 0.04
            check(
                0.03 <= open_target <= 0.09,
                f"ctrl=255 对应张开目标 {open_target:.4f}（应≈0.04，255=张开 的映射成立）",
            )
        print(
            f"      夹爪 kp={kp:.0f}, kv={-float(model.actuator_biasprm[7][2]):.0f}, "
            f"forcerange={model.actuator_forcerange[7]}（抓取太用力挤飞/太松掉落时调这里）"
        )
        cover_ok = True
        for i in range(7):
            jr = model.jnt_range[model.actuator_trnid[i][0]]
            ar = model.actuator_ctrlrange[i]
            cover_ok = cover_ok and ar[0] <= jr[0] + 1e-6 and ar[1] >= jr[1] - 1e-6
        check(cover_ok, "7 个手臂执行器 ctrlrange 覆盖关节 range（IK 裁剪后的命令始终可执行）")

    # 名称引用
    for j in [f"joint{i}" for i in range(1, 8)] + ["finger_joint1"]:
        check(j in joints, f"joint 存在: {j}")
    if "finger_joint1" in joints:
        fr = model.jnt_range[joints.index("finger_joint1")]
        check(fr[1] >= 0.035, f"finger_joint1 range={fr}（check_success 的 0.03 张开阈值有效）")
    need_bodies = ["tcp_link", "link0", "body_obj_mug_5", "body_obj_plate_11"]
    if expect_mug6:
        need_bodies.append("body_obj_mug_6")
    for b in need_bodies:
        check(b in bodies, f"body 存在: {b}")
    for b in [x for x in need_bodies if x.startswith("body_obj_")]:
        if b in bodies:
            bid = bodies.index(b)
            jadr = model.body_jntadr[bid]
            check(
                jadr >= 0 and model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE,
                f"{b} 带自由关节（set_p_base_body/check_success 依赖）",
            )
    for c in ["agentview", "egocentric", "sideview", "topview"]:
        check(c in cams, f"camera 存在: {c}")

    # 时序与渲染缓冲
    hz = 1.0 / model.opt.timestep
    check(
        abs(hz - round(hz)) < 1e-9 and round(hz) % 20 == 0,
        f"timestep={model.opt.timestep} → HZ={hz:.0f}，可整除 20Hz 采样",
    )
    check(
        model.vis.global_.offwidth >= 400 and model.vis.global_.offheight >= 300,
        f"离屏渲染缓冲 {model.vis.global_.offwidth}x{model.vis.global_.offheight} >= 400x300",
    )


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def verify_configs() -> None:
    print("\n===== 策略配置 =====")
    act_path = PROJECT_ROOT / "config/il/act_franka.yaml"
    dif_path = PROJECT_ROOT / "config/il/diffusion_franka.yaml"
    pi0_path = PROJECT_ROOT / "config/vla/pi0_franka.yaml"

    if act_path.is_file():
        policy = load_yaml(act_path).get("policy", {})
        if policy.get("temporal_ensemble_coeff") is not None:
            check(policy.get("n_action_steps") == 1, "ACT：启用时间集成时 n_action_steps == 1")
        check(
            int(policy.get("chunk_size", 100)) >= int(policy.get("n_action_steps", 1)),
            "ACT：chunk_size >= n_action_steps",
        )
    if dif_path.is_file():
        policy = load_yaml(dif_path).get("policy", {})
        both = bool(policy.get("use_group_norm", True)) and bool(policy.get("pretrained_backbone_weights"))
        check(not both, "Diffusion：use_group_norm 与 pretrained_backbone_weights 未同时开启（否则构建即报错）")
        crop = policy.get("crop_shape")
        if crop:
            check(int(crop[0]) <= 256 and int(crop[1]) <= 256, f"Diffusion：crop_shape={crop} 不超过 256x256 图像")
        horizon = int(policy.get("horizon", 16))
        down_dims = policy.get("down_dims", [512, 1024, 2048])
        check(horizon % (2 ** len(down_dims)) == 0, f"Diffusion：horizon={horizon} 可被 U-Net 下采样 {2**len(down_dims)} 整除")
        n_obs = int(policy.get("n_obs_steps", 1))
        n_act = int(policy.get("n_action_steps", 1))
        check(n_act <= horizon - n_obs + 1, f"Diffusion：n_action_steps={n_act} <= horizon-n_obs_steps+1={horizon-n_obs+1}")
    if pi0_path.is_file():
        policy = load_yaml(pi0_path).get("policy", {})
        if "chunk_size" in policy or "n_action_steps" in policy:
            check(
                policy.get("n_action_steps") == policy.get("chunk_size"),
                "π0：n_action_steps == chunk_size（锁定版本的实现要求）",
            )


def verify_dataset(root_rel: str) -> None:
    info_path = PROJECT_ROOT / root_rel / "meta" / "info.json"
    print(f"\n===== 数据集：{root_rel} =====")
    if not info_path.is_file():
        print("（不存在，跳过——尚未采集属于正常情况）")
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    state = tuple(features.get("observation.state", {}).get("shape", ()))
    action = tuple(features.get("action", {}).get("shape", ()))
    check(state == (8,), f"observation.state 形状 == (8,)（实际 {state}）")
    check(action == (8,), f"action 形状 == (8,)（实际 {action}）")
    check(int(info.get("fps", 0)) == 20, f"fps == 20（实际 {info.get('fps')}）")
    print(f"      episodes={info.get('total_episodes')}, frames={info.get('total_frames')}")


def main() -> None:
    verify_scene("asset/example_scene_y.xml", expect_mug6=False)
    verify_scene("asset/example_scene_y2.xml", expect_mug6=True)
    verify_configs()
    verify_dataset("demo_data")
    verify_dataset("demo_data_language")

    print("\n===== 结论 =====")
    if FAILURES:
        print(f"{len(FAILURES)} 项未通过：")
        for item in FAILURES:
            print(" -", item)
        sys.exit(1)
    print("ALL CHECKS PASSED：场景、配置与数据集均满足代码假设。")


if __name__ == "__main__":
    main()
