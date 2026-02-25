import math

Vec3 = tuple[int, int, int]

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
