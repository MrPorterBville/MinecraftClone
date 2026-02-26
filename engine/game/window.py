import math

import pyglet
from pyglet import gl
from pyglet.window import key, mouse

from engine.blocks import SOLID_BLOCKS
from engine.constants import GRAVITY, JUMP_SPEED, PLAYER_HEIGHT, TERMINAL_VELOCITY, TICKS_PER_SECOND, WALK_SPEED
from engine.gameplay.inventory import Inventory
from engine.graphics.rendering import set_2d, set_3d
from engine.world.world import World


class GameWindow(pyglet.window.Window):
    def __init__(self):
        super().__init__(width=1280, height=720, caption="Python Minecraft Clone", resizable=True)
        self.exclusive = False
        self.world = World(seed=90125, flat_height=8)
        self.world.generate_terrain()

        self.inventory = Inventory()

        spawn_y = float(self.world.height_at(0, 0) + 2)
        self.position = (0.0, spawn_y, 0.0)
        self.rotation = (0.0, -25.0)
        self.dy = 0.0

        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SECOND)

        self.label = pyglet.text.Label("", x=10, y=self.height - 10, anchor_x="left", anchor_y="top", color=(255, 255, 255, 255))
        self.crosshair = pyglet.shapes.Line(self.width // 2 - 8, self.height // 2, self.width // 2 + 8, self.height // 2, color=(255, 255, 255), batch=None)
        self.crosshair2 = pyglet.shapes.Line(self.width // 2, self.height // 2 - 8, self.width // 2, self.height // 2 + 8, color=(255, 255, 255), batch=None)

    def set_exclusive_mouse(self, exclusive: bool) -> None:
        super().set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    def get_sight_vector(self) -> tuple[float, float, float]:
        yaw, pitch = self.rotation
        m = math.cos(math.radians(pitch))
        dy = math.sin(math.radians(pitch))
        dx = math.cos(math.radians(yaw - 90)) * m
        dz = math.sin(math.radians(yaw - 90)) * m
        return dx, dy, dz

    def get_motion_vector(self) -> tuple[float, float, float]:
        forward = int(self.keys[key.W]) - int(self.keys[key.S])
        right = int(self.keys[key.D]) - int(self.keys[key.A])
        if not forward and not right:
            return 0.0, 0.0, 0.0

        sx, _, sz = self.get_sight_vector()
        forward_x, forward_z = sx, sz
        forward_len = math.sqrt(forward_x * forward_x + forward_z * forward_z)
        if forward_len > 0:
            forward_x /= forward_len
            forward_z /= forward_len

        right_x = -forward_z
        right_z = forward_x

        dx = forward * forward_x + right * right_x
        dz = forward * forward_z + right * right_z

        mag = math.sqrt(dx * dx + dz * dz)
        return dx / mag, 0.0, dz / mag

    def update(self, dt: float) -> None:
        d = dt * WALK_SPEED
        dx, dy, dz = self.get_motion_vector()
        dx, dy, dz = dx * d, dy * d, dz * d

        self.dy -= dt * GRAVITY
        self.dy = max(self.dy, -TERMINAL_VELOCITY)
        dy += self.dy * dt

        x, y, z = self.position
        x, y, z = self.collide((x + dx, y + dy, z + dz), PLAYER_HEIGHT)
        self.position = (x, y, z)

        held = self.inventory.selected_block()
        held_count = self.inventory.items.get(held, 0)
        self.label.text = f"XYZ: ({x:.1f}, {y:.1f}, {z:.1f})  Held: {held} ({held_count})"

    def collide(self, position: tuple[float, float, float], height: int) -> tuple[float, float, float]:
        pad = 0.25
        p = list(position)
        np = self.world.normalize(position)
        for face in ((0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1)):
            for i in range(3):
                if not face[i]:
                    continue
                d = (p[i] - np[i]) * face[i]
                if d < pad:
                    continue
                for dy in range(height):
                    op = list(np)
                    op[1] -= dy
                    op[i] += face[i]
                    if self.world.blocks.get(tuple(op)) in SOLID_BLOCKS:
                        p[i] -= (d - pad) * face[i]
                        if face == (0, -1, 0) or face == (0, 1, 0):
                            self.dy = 0
                        break
        return tuple(p)

    def on_mouse_press(self, x, y, button, modifiers):
        if not self.exclusive:
            self.set_exclusive_mouse(True)
            return

        vector = self.get_sight_vector()
        block, previous = self.world.hit_test(self.position, vector)
        if button == mouse.RIGHT and previous:
            held = self.inventory.selected_block()
            if self.inventory.remove(held, 1):
                self.world.add_block(previous, held)
        elif button == mouse.LEFT and block:
            removed = self.world.remove_block(block)
            if removed:
                self.inventory.add(removed, 1)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.exclusive:
            sensitivity = 0.15
            yaw, pitch = self.rotation
            yaw += dx * sensitivity
            pitch = max(-90, min(90, pitch + dy * sensitivity))
            self.rotation = (yaw, pitch)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE and self.dy == 0:
            self.dy = JUMP_SPEED
        elif symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif key._1 <= symbol <= key._9:
            self.inventory.selected = symbol - key._1

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.label.y = height - 10
        self.crosshair.x = width // 2 - 8
        self.crosshair.x2 = width // 2 + 8
        self.crosshair.y = self.crosshair.y2 = height // 2
        self.crosshair2.y = height // 2 - 8
        self.crosshair2.y2 = height // 2 + 8
        self.crosshair2.x = self.crosshair2.x2 = width // 2

    def on_draw(self):
        self.clear()
        set_3d(self, self.rotation, self.position)
        gl.glDisable(gl.GL_CULL_FACE)
        self.world.batch.draw()

        set_2d(self)
        self.label.draw()
        self.crosshair.draw()
        self.crosshair2.draw()
