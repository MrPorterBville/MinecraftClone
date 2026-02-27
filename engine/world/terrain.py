import math
import random


class TerrainGenerator:
    def __init__(
        self,
        seed: int,
        flat_height: int = 8,
        scale: float = 64.0,
        amplitude: float = 10.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> None:
        self.seed = seed
        self.base_height = flat_height
        self.scale = scale
        self.amplitude = amplitude
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

        rng = random.Random(seed)
        permutation = list(range(256))
        rng.shuffle(permutation)
        self._perm = permutation + permutation

    @staticmethod
    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    @staticmethod
    def _grad(hash_value: int, x: float, y: float) -> float:
        h = hash_value & 7
        u = x if h < 4 else y
        v = y if h < 4 else x
        return ((u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v))

    def _perlin(self, x: float, y: float) -> float:
        xi = math.floor(x) & 255
        yi = math.floor(y) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)

        u = self._fade(xf)
        v = self._fade(yf)

        aa = self._perm[self._perm[xi] + yi]
        ab = self._perm[self._perm[xi] + yi + 1]
        ba = self._perm[self._perm[xi + 1] + yi]
        bb = self._perm[self._perm[xi + 1] + yi + 1]

        x1 = self._lerp(self._grad(aa, xf, yf), self._grad(ba, xf - 1.0, yf), u)
        x2 = self._lerp(self._grad(ab, xf, yf - 1.0), self._grad(bb, xf - 1.0, yf - 1.0), u)
        return self._lerp(x1, x2, v)

    def height_at(self, x: int, z: int) -> int:
        frequency = 1.0 / self.scale
        amplitude = 1.0
        noise_sum = 0.0
        max_amplitude = 0.0

        for _ in range(self.octaves):
            noise_sum += self._perlin(x * frequency, z * frequency) * amplitude
            max_amplitude += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity

        normalized = noise_sum / max_amplitude if max_amplitude else 0.0
        height = self.base_height + normalized * self.amplitude
        return int(round(height))
