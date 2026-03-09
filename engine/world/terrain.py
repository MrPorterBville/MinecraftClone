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
        try:
            import minecraftclone_native  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Native terrain kernel is required but not installed. "
                "Build/install it from engine/world/_terrain_native with: "
                "`maturin develop --release`."
            ) from exc

        self._native = minecraftclone_native.TerrainKernel(
            seed,
            flat_height,
            scale,
            amplitude,
            octaves,
            persistence,
            lacunarity,
            permutation,
        )

    def biome_noise_at(self, x: int, z: int) -> float:
        return float(self._native.biome_noise_at(x, z))

    @staticmethod
    def biome_from_noise(noise: float) -> str:
        if noise < -0.35:
            return "desert"
        if noise < 0.2:
            return "plains"
        if noise < 0.55:
            return "forest"
        return "mountains"


    def river_noise_at(self, x: int, z: int) -> float:
        base = self.biome_noise_at(x, z)
        detail = self.biome_noise_at(x + 9876, z - 5432)
        return 0.7 * base + 0.3 * detail

    def cave_noise_at(self, x: int, y: int, z: int) -> float:
        return float(self._native.cave_noise_at(x, y, z))

    def material_noise_at(self, x: int, y: int, z: int) -> float:
        return float(self._native.material_noise_at(x, y, z))

    def cave_gate_noise_at(self, x: int, y: int, z: int) -> float:
        return float(self._native.cave_gate_noise_at(x, y, z))

    def height_at(self, x: int, z: int, depth: float, scale: float) -> int:
        return int(self._native.height_at(x, z, depth, scale))

    def biome_at(self, x: int, z: int) -> str:
        return self.biome_from_noise(self.biome_noise_at(x, z))

    def is_cave_at(
        self,
        x: int,
        y: int,
        z: int,
        surface_y: int,
        cave_min_y: int,
        cave_max_y: int,
        bedrock_y: int,
        cave_bedrock_buffer: int,
        cave_near_surface_depth: int,
        cave_tunnel_primary_y_scale: float,
        cave_tunnel_secondary_y_scale: float,
        cave_tunnel_gate_y_scale: float,
        cave_tunnel_primary_band: float,
        cave_tunnel_secondary_band: float,
        cave_combined_band: float,
        cave_near_surface_band_scale: float,
        cave_tunnel_gate_threshold: float,
        cave_warp_strength_xz: float,
        cave_warp_strength_y: float,
        cave_level_variation_scale: float,
        cave_level_variation_band_boost: float,
        cave_edge_fade_y: float,
        cave_min_connected_neighbors: int,
        cave_max_local_open: int,
        cave_family2_warp_strength_xz: float,
        cave_family2_warp_strength_y: float,
        cave_family2_primary_band_scale: float,
        cave_family2_secondary_band_scale: float,
        cave_family2_gate_threshold: float,
    ) -> bool:
        return bool(
            self._native.is_cave_at(
                x,
                y,
                z,
                surface_y,
                cave_min_y,
                cave_max_y,
                bedrock_y,
                cave_bedrock_buffer,
                cave_near_surface_depth,
                cave_tunnel_primary_y_scale,
                cave_tunnel_secondary_y_scale,
                cave_tunnel_gate_y_scale,
                cave_tunnel_primary_band,
                cave_tunnel_secondary_band,
                cave_combined_band,
                cave_near_surface_band_scale,
                cave_tunnel_gate_threshold,
                cave_warp_strength_xz,
                cave_warp_strength_y,
                cave_level_variation_scale,
                cave_level_variation_band_boost,
                cave_edge_fade_y,
                cave_min_connected_neighbors,
                cave_max_local_open,
                cave_family2_warp_strength_xz,
                cave_family2_warp_strength_y,
                cave_family2_primary_band_scale,
                cave_family2_secondary_band_scale,
                cave_family2_gate_threshold,
            )
        )

    def generate_classic_cave_mask(
        self,
        chunk_x: int,
        chunk_y: int,
        chunk_z: int,
        chunk_size: int,
        chunk_height: int,
        world_height: int,
        surface_heights: list[int],
        bedrock_y: int,
        cave_bedrock_buffer: int,
        cave_min_y: int,
        cave_max_y: int,
        cave_range: int,
        cave_system_chance: int,
        cave_vertical_scale: float,
        cave_branch_chance: int,
        cave_initial_pitch_range: float,
        cave_pitch_change_strength: float,
        cave_vertical_drift_strength: float,
    ) -> bytes:
        return bytes(
            self._native.generate_classic_cave_mask(
                chunk_x,
                chunk_y,
                chunk_z,
                chunk_size,
                chunk_height,
                world_height,
                surface_heights,
                bedrock_y,
                cave_bedrock_buffer,
                cave_min_y,
                cave_max_y,
                cave_range,
                cave_system_chance,
                cave_vertical_scale,
                cave_branch_chance,
                cave_initial_pitch_range,
                cave_pitch_change_strength,
                cave_vertical_drift_strength,
            )
        )

    def generate_ore_mask(
        self,
        chunk_x: int,
        chunk_y: int,
        chunk_z: int,
        chunk_size: int,
        chunk_height: int,
        world_height: int,
        vein_sizes: list[int],
        veins_per_chunk: list[int],
        min_ys: list[int],
        max_ys: list[int],
        discard_on_air_chances: list[float],
    ) -> list[int]:
        return list(
            self._native.generate_ore_mask(
                chunk_x,
                chunk_y,
                chunk_z,
                chunk_size,
                chunk_height,
                world_height,
                vein_sizes,
                veins_per_chunk,
                min_ys,
                max_ys,
                discard_on_air_chances,
            )
        )
