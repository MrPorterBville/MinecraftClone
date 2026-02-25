import math
import random


class TerrainGenerator:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def height_at(self, x: int, z: int) -> int:
        rnd = random.Random((x * 928371 + z * 523421 + self.seed * 6113) & 0xFFFFFFFF)
        base = 9 + int(4 * math.sin(x * 0.11) + 4 * math.cos(z * 0.08))
        detail = rnd.randint(-2, 2)
        return max(2, min(24, base + detail))

    def should_place_tree(self, x: int, z: int, height: int) -> bool:
        return height > 8 and random.Random((x, z, self.seed)).random() < 0.025

    def tree_height(self, x: int, y: int, z: int) -> int:
        return random.Random((x, y, z, self.seed)).randint(3, 5)
