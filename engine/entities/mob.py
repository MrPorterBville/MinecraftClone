import math
import random
from dataclasses import dataclass, field


@dataclass
class Mob:
    x: float
    y: float
    z: float
    color: tuple[float, float, float] = (0.8, 0.2, 0.2)
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
