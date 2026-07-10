"""在导入 MuJoCo 前选择跨平台离屏 OpenGL 后端。"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence


GL_BACKENDS = {"auto", "egl", "osmesa", "wgl", "cgl", "glfw"}


def option_from_argv(argv: Sequence[str], name: str, default: str) -> str:
    """读取 ``--name=value`` 或 ``--name value``，供 argparse 运行前使用。"""
    prefix = f"{name}="
    for index, item in enumerate(argv):
        if item.startswith(prefix):
            return item[len(prefix) :]
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
    return default


def resolve_gl_backend(requested: str = "auto") -> str:
    requested = str(requested).strip().lower()
    if requested not in GL_BACKENDS:
        raise ValueError(f"Unsupported MuJoCo GL backend: {requested!r}; choose from {sorted(GL_BACKENDS)}")
    if requested != "auto":
        return requested
    if sys.platform.startswith("win"):
        return "wgl"
    if sys.platform == "darwin":
        return "cgl"
    return "egl"


def configure_headless_gl(requested: str = "auto", egl_device_id: int | None = None) -> str:
    """设置 MuJoCo 离屏渲染环境变量并返回实际后端。

    必须在首次导入 ``mujoco``、``OpenGL`` 或环境封装之前调用。
    """
    backend = resolve_gl_backend(requested)
    os.environ["MUJOCO_GL"] = backend

    # PyOpenGL 的软件/EGL 平台也必须与 MuJoCo 一致。WGL/CGL 由系统默认
    # 平台加载器处理；清掉旧 shell 遗留的 egl/osmesa 值，避免跨平台冲突。
    if backend in {"egl", "osmesa"}:
        os.environ["PYOPENGL_PLATFORM"] = backend
    elif os.environ.get("PYOPENGL_PLATFORM", "").lower() in {"egl", "osmesa"}:
        os.environ.pop("PYOPENGL_PLATFORM", None)

    if backend == "egl" and egl_device_id is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(egl_device_id))

    return backend


def render_diagnostics() -> str:
    return (
        f"platform={sys.platform}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}, "
        f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')}"
    )
