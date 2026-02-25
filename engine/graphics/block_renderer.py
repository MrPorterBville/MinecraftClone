import random

import pyglet
from pyglet import gl

from engine.constants import BLOCK_COLORS, Vec3


class BlockRenderer:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    @staticmethod
    def _face(x: float, y: float, z: float, n: float, axis: int) -> list[float]:
        if axis == 0:
            return [x + n, y - n, z - n, x + n, y + n, z - n, x + n, y + n, z + n, x + n, y - n, z + n]
        if axis == 1:
            return [x - n, y + n, z - n, x + n, y + n, z - n, x + n, y + n, z + n, x - n, y + n, z + n]
        return [x - n, y - n, z + n, x + n, y - n, z + n, x + n, y + n, z + n, x - n, y + n, z + n]

    def _jittered_color(self, pos: Vec3, block: str, shade: float = 1.0) -> tuple[int, ...]:
        base = BLOCK_COLORS[block]
        rnd = random.Random(hash((pos, block, self.seed)) & 0xFFFFFFFF)
        jitter = rnd.uniform(-0.08, 0.08)
        r = max(0.0, min(1.0, (base[0] + jitter) * shade))
        g = max(0.0, min(1.0, (base[1] + jitter) * shade))
        b = max(0.0, min(1.0, (base[2] + jitter) * shade))
        c = (int(r * 255), int(g * 255), int(b * 255))
        return c * 4

    def make_cube(
        self,
        position: Vec3,
        block: str,
        batch: pyglet.graphics.Batch,
        group: pyglet.graphics.Group,
    ) -> pyglet.graphics.vertexdomain.VertexList:
        x, y, z = position
        n = 0.5
        vertices: list[float] = []
        colors: list[int] = []
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
        return batch.add(24, gl.GL_QUADS, group, ("v3f/static", vertices), ("c3B/static", colors))
