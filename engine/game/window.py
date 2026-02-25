import math
import random

import pyglet
from pyglet import gl
from pyglet.window import key, mouse

from engine.constants import FLY_SPEED, GRAVITY, JUMP_SPEED, PLAYER_HEIGHT, SOLID_BLOCKS, TERMINAL_VELOCITY, TICKS_PER_SECOND, WALK_SPEED
from engine.entities.mob import Mob
from engine.gameplay.inventory import Inventory
from engine.graphics.rendering import set_2d, set_3d
from engine.world.world import World


class GameWindow(pyglet.window.Window):
    def __init__(self):
        super().__init__(width=1280, height=720, caption="Python Minecraft Clone", resizable=True)
        self.exclusive = False
        self.world = World(seed=90125)
        self.world.generate_terrain()

        self.inventory = Inventory()
        self.mobs = self._spawn_mobs(10)

        self.position = (0.0, 24.0, 0.0)
        self.rotation = (0.0, 0.0)
        self.strafe = [0, 0]
        self.dy = 0.0
        self.flying = False

        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SECOND)

        self.label = pyglet.text.Label("", x=10, y=self.height - 10, anchor_x="left", anchor_y="top", color=(255, 255, 255, 255))
        self.crosshair = pyglet.shapes.Line(self.width // 2 - 8, self.height // 2, self.width // 2 + 8, self.height // 2, color=(255, 255, 255), batch=None)
        self.crosshair2 = pyglet.shapes.Line(self.width // 2, self.height // 2 - 8, self.width // 2, self.height // 2 + 8, color=(255, 255, 255), batch=None)

    def _spawn_mobs(self, count: int) -> list[Mob]:
        mobs: list[Mob] = []
        rnd = random.Random(self.world.seed ^ 0xCAFE)
        for _ in range(count):
            x = rnd.randint(-40, 40)
            z = rnd.randint(-40, 40)
            y = self.world.height_at(x, z) + 1
            mobs.append(Mob(float(x), float(y), float(z), color=(rnd.uniform(0.5, 0.9), rnd.uniform(0.2, 0.8), rnd.uniform(0.2, 0.8))))
        return mobs

    def set_exclusive_mouse(self, exclusive: bool) -> None:
        super().set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    def get_sight_vector(self) -> tuple[float, float, float]:
        x_rot, y_rot = self.rotation
        m = math.cos(math.radians(y_rot))
        dy = math.sin(math.radians(y_rot))
        dx = math.cos(math.radians(x_rot - 90)) * m
        dz = math.sin(math.radians(x_rot - 90)) * m
        return dx, dy, dz

    def get_motion_vector(self) -> tuple[float, float, float]:
        if any(self.strafe):
            x_rot = math.radians(self.rotation[0])
            strafe = math.atan2(*self.strafe)
            if self.flying:
                m = math.cos(math.radians(self.rotation[1]))
                dy = math.sin(math.radians(self.rotation[1]))
                if self.strafe[1]:
                    dy = 0.0
                    m = 1
                if self.strafe[0] > 0:
                    dy *= -1
                dx = math.cos(x_rot + strafe) * m
                dz = math.sin(x_rot + strafe) * m
            else:
                dy = 0.0
                dx = math.cos(x_rot + strafe)
                dz = math.sin(x_rot + strafe)
        else:
            dy = 0.0
            dx = 0.0
            dz = 0.0
        return dx, dy, dz

    def update(self, dt: float) -> None:
        for mob in self.mobs:
            mob.update(dt, self.world)

        speed = FLY_SPEED if self.flying else WALK_SPEED
        d = dt * speed
        dx, dy, dz = self.get_motion_vector()
        dx, dy, dz = dx * d, dy * d, dz * d

        if not self.flying:
            self.dy -= dt * GRAVITY
            self.dy = max(self.dy, -TERMINAL_VELOCITY)
            dy += self.dy * dt

        x, y, z = self.position
        x, y, z = self.collide((x + dx, y + dy, z + dz), PLAYER_HEIGHT)
        self.position = (x, y, z)

        self.label.text = (
            f"XYZ: ({x:.1f}, {y:.1f}, {z:.1f})  Seed: {self.world.seed}  "
            f"Held: {self.inventory.selected_block()} ({self.inventory.items.get(self.inventory.selected_block(), 0)})"
        )

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
            if removed and removed != "water":
                self.inventory.add(removed, 1)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.exclusive:
            m = 0.15
            x_rot, y_rot = self.rotation
            x_rot += dx * m
            y_rot = max(-90, min(90, y_rot + dy * m))
            self.rotation = (x_rot, y_rot)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.W:
            self.strafe[0] -= 1
        elif symbol == key.S:
            self.strafe[0] += 1
        elif symbol == key.A:
            self.strafe[1] -= 1
        elif symbol == key.D:
            self.strafe[1] += 1
        elif symbol == key.SPACE:
            if self.flying:
                self.dy = 8
            elif self.dy == 0:
                self.dy = JUMP_SPEED
        elif symbol == key.TAB:
            self.flying = not self.flying
        elif symbol == key.E:
            crafted = self.inventory.craft_first_available()
            if crafted:
                print(f"Crafted: {crafted}")
            else:
                print("No craftable recipes")
        elif symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif key._1 <= symbol <= key._9:
            self.inventory.selected = symbol - key._1

    def on_key_release(self, symbol, modifiers):
        if symbol == key.W:
            self.strafe[0] += 1
        elif symbol == key.S:
            self.strafe[0] -= 1
        elif symbol == key.A:
            self.strafe[1] += 1
        elif symbol == key.D:
            self.strafe[1] -= 1

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.label.y = height - 10
        self.crosshair.x = width // 2 - 8
        self.crosshair.x2 = width // 2 + 8
        self.crosshair.y = self.crosshair.y2 = height // 2
        self.crosshair2.y = height // 2 - 8
        self.crosshair2.y2 = height // 2 + 8
        self.crosshair2.x = self.crosshair2.x2 = width // 2

    def draw_mobs(self):
        for mob in self.mobs:
            gl.glPushMatrix()
            gl.glTranslatef(mob.x, mob.y, mob.z)
            gl.glColor3f(*mob.color)
            n = 0.4
            pyglet.graphics.draw(24, gl.GL_QUADS, (
                "v3f",
                [
                    -n, -n, n, n, -n, n, n, n, n, -n, n, n,
                    -n, -n, -n, -n, n, -n, n, n, -n, n, -n, -n,
                    -n, n, -n, -n, n, n, n, n, n, n, n, -n,
                    -n, -n, -n, n, -n, -n, n, -n, n, -n, -n, n,
                    n, -n, -n, n, n, -n, n, n, n, n, -n, n,
                    -n, -n, -n, -n, -n, n, -n, n, n, -n, n, -n,
                ],
            ))
            gl.glPopMatrix()

    def on_draw(self):
        self.clear()
        set_3d(self, self.rotation, self.position)
        gl.glEnable(gl.GL_CULL_FACE)
        self.world.batch.draw()
        self.draw_mobs()

        set_2d(self)
        self.label.draw()
        self.crosshair.draw()
        self.crosshair2.draw()
