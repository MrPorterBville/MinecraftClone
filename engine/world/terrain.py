class TerrainGenerator:
    def __init__(self, seed: int, flat_height: int = 8) -> None:
        self.seed = seed
        self.flat_height = flat_height

    def height_at(self, x: int, z: int) -> int:
        return self.flat_height
