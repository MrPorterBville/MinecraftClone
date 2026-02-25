import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pyglet
from pyglet import gl
from pyglet.window import key, mouse


Vec3 = Tuple[int, int, int]

TICKS_PER_SECOND = 60
SECTOR_SIZE = 16
WALK_SPEED = 6.0
FLY_SPEED = 16.0
GRAVITY = 20.0
MAX_JUMP_HEIGHT = 1.2
JUMP_SPEED = math.sqrt(2 * GRAVITY * MAX_JUMP_HEIGHT)
TERMINAL_VELOCITY = 50
PLAYER_HEIGHT = 2


BLOCK_COLORS = {
    "grass": (0.20, 0.66, 0.20),
    "dirt": (0.50, 0.35, 0.20),
    "stone": (0.50, 0.50, 0.52),
    "wood": (0.55, 0.39, 0.20),
    "leaf": (0.12, 0.52, 0.12),
    "sand": (0.82, 0.76, 0.52),
    "water": (0.18, 0.40, 0.74),
    "crafting_table": (0.58, 0.40, 0.22),
}

SOLID_BLOCKS = {"grass", "dirt", "stone", "wood", "leaf", "sand", "crafting_table"}

CRAFTING_RECIPES = {
    (("wood", 1),): ("crafting_table", 1),
    (("wood", 2),): ("stone", 2),
    (("stone", 2),): ("sand", 2),
}


@dataclass
class Mob:
    x: float
    y: float
    z: float
    color: Tuple[float, float, float] = (0.8, 0.2, 0.2)
    direction: float = field(default_factory=lambda: random.uniform(0, math.tau))
    speed: float = 1.6

    def update(self, dt: float, world: "World") -> None:
        if random.random() < 0.02:
            self.direction += random.uniform(-0.8, 0.8)
        nx = self.x + math.cos(self.direction) * self.speed * dt
        nz = self.z + math.sin(self.direction) * self.speed * dt
        ny = int(round(self.y - 1))
        if world.is_walkable(int(nx), ny, int(nz)) and world.is_walkable(int(nx), int(self.y), int(nz)):
            self.x, self.z = nx, nz
        else:
            self.direction += math.pi * 0.7


class Inventory:
    def __init__(self) -> None:
        self.items: Dict[str, int] = {"grass": 16, "dirt": 12, "stone": 6, "wood": 8}
        self.hotbar: List[str] = ["grass", "dirt", "stone", "wood", "leaf", "sand", "water", "crafting_table", "grass"]
        self.selected = 0

    def add(self, block: str, count: int = 1) -> None:
        self.items[block] = self.items.get(block, 0) + count

    def remove(self, block: str, count: int = 1) -> bool:
        if self.items.get(block, 0) < count:
            return False
        self.items[block] -= count
        return True

    def selected_block(self) -> str:
        return self.hotbar[self.selected]

    def can_craft(self, recipe: Tuple[Tuple[str, int], ...]) -> bool:
        return all(self.items.get(name, 0) >= qty for name, qty in recipe)

    def craft_first_available(self) -> Optional[str]:
        for recipe, result in CRAFTING_RECIPES.items():
            if self.can_craft(recipe):
                for name, qty in recipe:
                    self.remove(name, qty)
                self.add(result[0], result[1])
                return result[0]
        return None


class World:
    def __init__(self, seed: int = 1337) -> None:
        self.seed = seed
        self.blocks: Dict[Vec3, str] = {}
        self._shown: Dict[Vec3, pyglet.graphics.vertexdomain.VertexList] = {}
        self.batch = pyglet.graphics.Batch()
        self.group = pyglet.graphics.Group()

    @staticmethod
    def normalize(position: Tuple[float, float, float]) -> Vec3:
        x, y, z = position
        return int(round(x)), int(round(y)), int(round(z))

    @staticmethod
    def sectorize(position: Vec3) -> Tuple[int, int, int]:
        x, y, z = position
        return (x // SECTOR_SIZE, y // SECTOR_SIZE, z // SECTOR_SIZE)

    def height_at(self, x: int, z: int) -> int:
        rnd = random.Random((x * 928371 + z * 523421 + self.seed * 6113) & 0xFFFFFFFF)
        base = 9 + int(4 * math.sin(x * 0.11) + 4 * math.cos(z * 0.08))
        detail = rnd.randint(-2, 2)
        return max(2, min(24, base + detail))

    def generate_terrain(self, radius: int = 56) -> None:
        for x in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                h = self.height_at(x, z)
                for y in range(-4, h + 1):
                    if y == h:
                        block = "sand" if h < 7 else "grass"
                    elif y > h - 3:
                        block = "dirt"
                    else:
                        block = "stone"
                    self.add_block((x, y, z), block, immediate=False)
                if h < 6:
                    for y in range(h + 1, 7):
                        self.add_block((x, y, z), "water", immediate=False)
                if h > 8 and random.Random((x, z, self.seed)).random() < 0.025:
                    self._generate_tree(x, h + 1, z)
        self.rebuild_visible()

    def _generate_tree(self, x: int, y: int, z: int) -> None:
        trunk_h = random.Random((x, y, z, self.seed)).randint(3, 5)
        for dy in range(trunk_h):
            self.add_block((x, y + dy, z), "wood", immediate=False)
        for lx in range(-2, 3):
            for lz in range(-2, 3):
                for ly in range(trunk_h - 2, trunk_h + 1):
                    if abs(lx) + abs(lz) + abs(ly - trunk_h) <= 4:
                        self.add_block((x + lx, y + ly, z + lz), "leaf", immediate=False)

    def add_block(self, position: Vec3, block: str, immediate: bool = True) -> None:
        self.blocks[position] = block
        if immediate:
            self.refresh_neighbors(position)

    def remove_block(self, position: Vec3, immediate: bool = True) -> Optional[str]:
        old = self.blocks.pop(position, None)
        if old and immediate:
            self.refresh_neighbors(position)
        return old

    def is_exposed(self, position: Vec3) -> bool:
        x, y, z = position
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = (x + dx, y + dy, z + dz)
            if self.blocks.get(n) not in SOLID_BLOCKS:
                return True
        return False

    def refresh_neighbors(self, position: Vec3) -> None:
        x, y, z = position
        for p in [(x, y, z), (x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]:
            self.show_or_hide(p)

    def rebuild_visible(self) -> None:
        for v in list(self._shown.values()):
            v.delete()
        self._shown.clear()
        for pos in self.blocks:
            self.show_or_hide(pos)

    def show_or_hide(self, position: Vec3) -> None:
        if position in self._shown:
            self._shown[position].delete()
            del self._shown[position]
        block = self.blocks.get(position)
        if block is None:
            return
        if not self.is_exposed(position):
            return
        self._shown[position] = self._make_cube(position, block)

    @staticmethod
    def _face(x: float, y: float, z: float, n: float, axis: int) -> List[float]:
        if axis == 0:
            return [x + n, y - n, z - n, x + n, y + n, z - n, x + n, y + n, z + n, x + n, y - n, z + n]
        if axis == 1:
            return [x - n, y + n, z - n, x + n, y + n, z - n, x + n, y + n, z + n, x - n, y + n, z + n]
        return [x - n, y - n, z + n, x + n, y - n, z + n, x + n, y + n, z + n, x - n, y + n, z + n]

    def _jittered_color(self, pos: Vec3, block: str, shade: float = 1.0) -> Tuple[int, ...]:
        base = BLOCK_COLORS[block]
        rnd = random.Random(hash((pos, block, self.seed)) & 0xFFFFFFFF)
        jitter = rnd.uniform(-0.08, 0.08)
        r = max(0.0, min(1.0, (base[0] + jitter) * shade))
        g = max(0.0, min(1.0, (base[1] + jitter) * shade))
        b = max(0.0, min(1.0, (base[2] + jitter) * shade))
        c = (int(r * 255), int(g * 255), int(b * 255))
        return c * 4

    def _make_cube(self, position: Vec3, block: str) -> pyglet.graphics.vertexdomain.VertexList:
        x, y, z = position
        n = 0.5
        vertices: List[float] = []
        colors: List[int] = []
        faces = [
            self._face(x, y, z, n, 0),
            [x - n, y - n, z - n, x - n, y - n, z + n, x - n, y + n, z + n, x - n, y + n, z - n],
            self._face(x, y, z, n, 1),
            [x - n, y - n, z - n, x + n, y - n, z - n, x + n, y - n, z + n, x - n, y - n, z + n],
            self._face(x, y, z, n, 2),
            [x - n, y - n, z - n, x - n, y + n, z - n, x + n, y + n, z - n, x + n, y - n, z - n],
        ]
        shades = [0.86, 0.86, 1.0, 0.55, 0.72, 0.72]
        for face, shade in zip(faces, shades):
            vertices.extend(face)
            colors.extend(self._jittered_color(position, block, shade))
        return self.batch.add(24, gl.GL_QUADS, self.group, ("v3f/static", vertices), ("c3B/static", colors))

    def hit_test(self, position: Tuple[float, float, float], vector: Tuple[float, float, float], max_distance: int = 8):
        m = 12
        x, y, z = position
        dx, dy, dz = vector
        previous = None
        for _ in range(max_distance * m):
            key_pos = self.normalize((x, y, z))
            if key_pos != previous and key_pos in self.blocks:
                return key_pos, previous
            previous = key_pos
            x, y, z = x + dx / m, y + dy / m, z + dz / m
        return None, None

    def is_walkable(self, x: int, y: int, z: int) -> bool:
        return self.blocks.get((x, y, z)) not in SOLID_BLOCKS


class Window(pyglet.window.Window):
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

    def _spawn_mobs(self, count: int) -> List[Mob]:
        mobs: List[Mob] = []
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

    def get_sight_vector(self) -> Tuple[float, float, float]:
        x_rot, y_rot = self.rotation
        m = math.cos(math.radians(y_rot))
        dy = math.sin(math.radians(y_rot))
        dx = math.cos(math.radians(x_rot - 90)) * m
        dz = math.sin(math.radians(x_rot - 90)) * m
        return dx, dy, dz

    def get_motion_vector(self) -> Tuple[float, float, float]:
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

    def collide(self, position: Tuple[float, float, float], height: int) -> Tuple[float, float, float]:
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

    def set_3d(self):
        width, height = self.get_framebuffer_size()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(65.0, width / float(height), 0.1, 200.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        x, y = self.rotation
        gl.glRotatef(y, 0, 1, 0)
        gl.glRotatef(-x, 0, 0, 1)
        px, py, pz = self.position
        gl.glTranslatef(-px, -py, -pz)

    def set_2d(self):
        width, height = self.get_framebuffer_size()
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def draw_mobs(self):
        for mob in self.mobs:
            gl.glPushMatrix()
            gl.glTranslatef(mob.x, mob.y, mob.z)
            gl.glColor3f(*mob.color)
            n = 0.4
            pyglet.graphics.draw(24, gl.GL_QUADS, (
                'v3f', [
                    -n, -n, n, n, -n, n, n, n, n, -n, n, n,
                    -n, -n, -n, -n, n, -n, n, n, -n, n, -n, -n,
                    -n, n, -n, -n, n, n, n, n, n, n, n, -n,
                    -n, -n, -n, n, -n, -n, n, -n, n, -n, -n, n,
                    n, -n, -n, n, n, -n, n, n, n, n, -n, n,
                    -n, -n, -n, -n, -n, n, -n, n, n, -n, n, -n,
                ]
            ))
            gl.glPopMatrix()

    def on_draw(self):
        self.clear()
        self.set_3d()
        gl.glEnable(gl.GL_CULL_FACE)
        self.world.batch.draw()
        self.draw_mobs()

        self.set_2d()
        self.label.draw()
        self.crosshair.draw()
        self.crosshair2.draw()


def setup_gl() -> None:
    gl.glClearColor(0.52, 0.80, 0.92, 1.0)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


if __name__ == "__main__":
    window = Window()
    setup_gl()
    pyglet.app.run()
