import pyglet
from pyglet import gl


def setup_gl() -> None:
    gl.glClearColor(0.52, 0.80, 0.92, 1.0)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


def set_3d(window: pyglet.window.Window, rotation: tuple[float, float], position: tuple[float, float, float]) -> None:
    width, height = window.get_framebuffer_size()
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.gluPerspective(65.0, width / float(height), 0.1, 200.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    x, y = rotation
    gl.glRotatef(y, 0, 1, 0)
    gl.glRotatef(-x, 0, 0, 1)
    px, py, pz = position
    gl.glTranslatef(-px, -py, -pz)


def set_2d(window: pyglet.window.Window) -> None:
    width, height = window.get_framebuffer_size()
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0, width, 0, height, -1, 1)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
