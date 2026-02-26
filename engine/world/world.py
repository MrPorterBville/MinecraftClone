import pyglet
from typing import Protocol

from engine.blocks import SOLID_BLOCKS
from engine.constants import SECTOR_SIZE, Vec3
from engine.graphics.block_renderer import BlockRenderer
from engine.world.terrain import TerrainGenerator


class Deletable(Protocol):
    def delete(self) -> None: ...


class World:
    def __init__(self, seed: int = 1337, flat_height: int = 8) -> None:
        self.seed = seed
        self.blocks: dict[Vec3, str] = {}
        self._shown: dict[Vec3, Deletable] = {}
        self.batch = pyglet.graphics.Batch()
        self.terrain = TerrainGenerator(seed, flat_height=flat_height)
        self.renderer = BlockRenderer(seed)
        self.group = pyglet.graphics.ShaderGroup(program=self.renderer.shader)

    @staticmethod
    def normalize(position: tuple[float, float, float]) -> Vec3:
        x, y, z = position
        return int(round(x)), int(round(y)), int(round(z))

    @staticmethod
    def sectorize(position: Vec3) -> tuple[int, int, int]:
        x, y, z = position
        return (x // SECTOR_SIZE, y // SECTOR_SIZE, z // SECTOR_SIZE)

    def height_at(self, x: int, z: int) -> int:
        return self.terrain.height_at(x, z)

    def generate_terrain(self, radius: int = 56) -> None:
        for x in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                h = self.height_at(x, z)
                for y in range(-4, h + 1):
                    if y == h:
                        block = "grass"
                    elif y >= h - 2:
                        block = "dirt"
                    else:
                        block = "stone"
                    self.add_block((x, y, z), block, immediate=False)
        self.rebuild_visible()

    def add_block(self, position: Vec3, block: str, immediate: bool = True) -> None:
        self.blocks[position] = block
        if immediate:
            self.refresh_neighbors(position)

    def remove_block(self, position: Vec3, immediate: bool = True) -> str | None:
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
        if block is None or not self.is_exposed(position):
            return
        self._shown[position] = self.renderer.make_cube(position, block, self.batch, self.group)

    def hit_test(self, position: tuple[float, float, float], vector: tuple[float, float, float], max_distance: int = 8):
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
