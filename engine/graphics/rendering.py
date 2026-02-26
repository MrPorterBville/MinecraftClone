import math

import pyglet
from pyglet import gl
from pyglet.math import Mat4, Vec3


def setup_gl() -> None:
    gl.glClearColor(0.52, 0.80, 0.92, 1.0)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


def set_3d(window: pyglet.window.Window, rotation: tuple[float, float], position: tuple[float, float, float]) -> None:
    width, height = window.get_framebuffer_size()
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glViewport(0, 0, width, height)

    window.projection = Mat4.perspective_projection(width / float(height), z_near=0.1, z_far=200.0, fov=65.0)

    yaw_deg, pitch_deg = rotation
    px, py, pz = position
    yaw = Mat4.from_rotation(math.radians(yaw_deg), Vec3(0.0, 1.0, 0.0))
    pitch = Mat4.from_rotation(math.radians(-pitch_deg), Vec3(1.0, 0.0, 0.0))
    translate = Mat4.from_translation(Vec3(-px, -py, -pz))
    window.view = pitch @ yaw @ translate


def set_2d(window: pyglet.window.Window) -> None:
    width, height = window.get_framebuffer_size()
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glViewport(0, 0, width, height)
    window.projection = Mat4.orthogonal_projection(0.0, float(width), 0.0, float(height), -1.0, 1.0)
    window.view = Mat4()
