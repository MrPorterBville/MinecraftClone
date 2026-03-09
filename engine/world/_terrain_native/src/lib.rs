use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::{HashMap, HashSet};

type MeshBuildOutput = (
    Vec<f32>,
    Vec<f32>,
    HashMap<String, Vec<f32>>,
    HashMap<String, Vec<f32>>,
    HashMap<String, Vec<f32>>,
);

#[pyclass]
struct TerrainKernel {
    world_seed: i64,
    base_height: f64,
    scale: f64,
    amplitude: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    perm: Vec<i32>,
}

struct CaveParams {
    cave_min_y: i32,
    cave_max_y: i32,
    bedrock_y: i32,
    cave_bedrock_buffer: i32,
    cave_near_surface_depth: i32,
    cave_tunnel_primary_y_scale: f64,
    cave_tunnel_secondary_y_scale: f64,
    cave_tunnel_gate_y_scale: f64,
    cave_tunnel_primary_band: f64,
    cave_tunnel_secondary_band: f64,
    cave_combined_band: f64,
    cave_near_surface_band_scale: f64,
    cave_tunnel_gate_threshold: f64,
    cave_warp_strength_xz: f64,
    cave_warp_strength_y: f64,
    cave_level_variation_scale: f64,
    cave_level_variation_band_boost: f64,
    cave_edge_fade_y: f64,
    cave_family2_warp_strength_xz: f64,
    cave_family2_warp_strength_y: f64,
    cave_family2_primary_band_scale: f64,
    cave_family2_secondary_band_scale: f64,
    cave_family2_gate_threshold: f64,
}

struct JavaRandom {
    seed: u64,
}

impl JavaRandom {
    const MULTIPLIER: u64 = 0x5DEECE66D;
    const ADDEND: u64 = 0xB;
    const MASK: u64 = (1_u64 << 48) - 1;

    fn new(seed: i64) -> Self {
        let mut r = Self { seed: 0 };
        r.set_seed(seed);
        r
    }

    fn set_seed(&mut self, seed: i64) {
        self.seed = ((seed as u64) ^ Self::MULTIPLIER) & Self::MASK;
    }

    fn next_bits(&mut self, bits: u32) -> i32 {
        self.seed = (self.seed.wrapping_mul(Self::MULTIPLIER).wrapping_add(Self::ADDEND)) & Self::MASK;
        (self.seed >> (48 - bits)) as i32
    }

    fn next_int(&mut self, bound: i32) -> i32 {
        if bound <= 0 {
            return 0;
        }
        if (bound & -bound) == bound {
            return (((bound as i64) * (self.next_bits(31) as i64)) >> 31) as i32;
        }
        loop {
            let bits = self.next_bits(31);
            let val = bits % bound;
            if bits - val + (bound - 1) >= 0 {
                return val;
            }
        }
    }

    fn next_long(&mut self) -> i64 {
        ((self.next_bits(32) as i64) << 32) + (self.next_bits(32) as i64)
    }

    fn next_float(&mut self) -> f32 {
        (self.next_bits(24) as f32) / ((1_u32 << 24) as f32)
    }
}

#[pymethods]
impl TerrainKernel {
    #[new]
    fn new(
        seed: i64,
        flat_height: i64,
        scale: f64,
        amplitude: f64,
        octaves: usize,
        persistence: f64,
        lacunarity: f64,
        permutation: Vec<u8>,
    ) -> PyResult<Self> {
        if permutation.len() != 256 {
            return Err(PyValueError::new_err("permutation must contain exactly 256 entries"));
        }
        let mut perm: Vec<i32> = permutation.iter().map(|v| i32::from(*v)).collect();
        let copy = perm.clone();
        perm.extend(copy);
        Ok(Self {
            world_seed: seed,
            base_height: flat_height as f64,
            scale,
            amplitude,
            octaves,
            persistence,
            lacunarity,
            perm,
        })
    }

    #[staticmethod]
    fn biome_from_noise(noise: f64) -> &'static str {
        if noise < 0.0 {
            "desert"
        } else {
            "plains"
        }
    }

    fn biome_noise_at(&self, x: i32, z: i32) -> f64 {
        self.perlin2((x as f64) * 0.01, (z as f64) * 0.01)
    }

    fn cave_noise_at(&self, x: i32, y: i32, z: i32) -> f64 {
        let primary = self.fbm3(
            x as f64,
            y as f64,
            z as f64,
            1.0 / 24.0,
            2,
            0.52,
            2.02,
        );
        let detail = self.fbm3(
            x as f64 + 113.0,
            y as f64 + 37.0,
            z as f64 - 71.0,
            1.0 / 12.0,
            1,
            0.5,
            2.0,
        );
        primary * 0.72 + detail * 0.28
    }

    fn material_noise_at(&self, x: i32, y: i32, z: i32) -> f64 {
        self.fbm3(
            x as f64 - 211.0,
            y as f64 + 157.0,
            z as f64 + 89.0,
            1.0 / 24.0,
            2,
            0.55,
            2.0,
        )
    }

    fn cave_gate_noise_at(&self, x: i32, y: i32, z: i32) -> f64 {
        self.perlin3(
            (x as f64 + 47.0) / 19.0,
            (y as f64 - 19.0) / 19.0,
            (z as f64 + 83.0) / 19.0,
        )
    }

    fn height_at(&self, x: i32, z: i32, depth: f64, scale: f64) -> i64 {
        let biome_scale = scale.max(0.001);
        let biome_depth = depth;

        let biome_wavelength = self.scale * (0.9 + biome_scale * 8.0);
        let mut frequency = 1.0 / biome_wavelength.max(1.0);
        let mut amplitude = 1.0;
        let mut noise_sum = 0.0;
        let mut max_amplitude = 0.0;

        for _ in 0..self.octaves {
            noise_sum += self.perlin2((x as f64) * frequency, (z as f64) * frequency) * amplitude;
            max_amplitude += amplitude;
            amplitude *= self.persistence;
            frequency *= self.lacunarity;
        }

        let normalized = if max_amplitude > 0.0 {
            noise_sum / max_amplitude
        } else {
            0.0
        };
        let biome_amplitude = self
            .amplitude
            * (0.8 + biome_scale * 20.0 + biome_depth * 6.0).max(0.5);
        (self.base_height + normalized * biome_amplitude).round() as i64
    }

    #[allow(clippy::too_many_arguments)]
    fn is_cave_at(
        &self,
        x: i32,
        y: i32,
        z: i32,
        surface_y: i32,
        cave_min_y: i32,
        cave_max_y: i32,
        bedrock_y: i32,
        cave_bedrock_buffer: i32,
        cave_near_surface_depth: i32,
        cave_tunnel_primary_y_scale: f64,
        cave_tunnel_secondary_y_scale: f64,
        cave_tunnel_gate_y_scale: f64,
        cave_tunnel_primary_band: f64,
        cave_tunnel_secondary_band: f64,
        cave_combined_band: f64,
        cave_near_surface_band_scale: f64,
        cave_tunnel_gate_threshold: f64,
        cave_warp_strength_xz: f64,
        cave_warp_strength_y: f64,
        cave_level_variation_scale: f64,
        cave_level_variation_band_boost: f64,
        cave_edge_fade_y: f64,
        cave_min_connected_neighbors: i32,
        cave_max_local_open: i32,
        cave_family2_warp_strength_xz: f64,
        cave_family2_warp_strength_y: f64,
        cave_family2_primary_band_scale: f64,
        cave_family2_secondary_band_scale: f64,
        cave_family2_gate_threshold: f64,
    ) -> bool {
        let base = CaveParams {
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
            cave_family2_warp_strength_xz,
            cave_family2_warp_strength_y,
            cave_family2_primary_band_scale,
            cave_family2_secondary_band_scale,
            cave_family2_gate_threshold,
        };
        self.is_cave_eval(
            x,
            y,
            z,
            surface_y,
            &base,
            cave_min_connected_neighbors,
            cave_max_local_open,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_classic_cave_mask(
        &self,
        chunk_x: i32,
        chunk_y: i32,
        chunk_z: i32,
        chunk_size: i32,
        chunk_height: i32,
        world_height: i32,
        surface_heights: Vec<i32>,
        bedrock_y: i32,
        cave_bedrock_buffer: i32,
        cave_min_y: i32,
        cave_max_y: i32,
        cave_range: i32,
        cave_system_chance: i32,
        cave_vertical_scale: f64,
        cave_branch_chance: i32,
        cave_initial_pitch_range: f64,
        cave_pitch_change_strength: f64,
        cave_vertical_drift_strength: f64,
    ) -> PyResult<Vec<u8>> {
        if chunk_size <= 0 || chunk_height <= 0 || world_height <= 0 {
            return Err(PyValueError::new_err("chunk_size, chunk_height, and world_height must be > 0"));
        }
        let expected = (chunk_size as usize) * (chunk_size as usize);
        if surface_heights.len() != expected {
            return Err(PyValueError::new_err("surface_heights length must be chunk_size * chunk_size"));
        }
        if cave_max_y <= cave_min_y {
            return Err(PyValueError::new_err("cave_max_y must be greater than cave_min_y"));
        }
        Ok(self.classic_cave_mask(
            chunk_x,
            chunk_y,
            chunk_z,
            chunk_size,
            chunk_height,
            world_height,
            &surface_heights,
            bedrock_y,
            cave_bedrock_buffer,
            cave_min_y,
            cave_max_y,
            cave_range.max(1),
            cave_system_chance.max(1),
            cave_vertical_scale.max(0.1),
            cave_branch_chance.max(1),
            cave_initial_pitch_range.max(0.0),
            cave_pitch_change_strength.max(0.0),
            cave_vertical_drift_strength.max(0.0),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_ore_mask(
        &self,
        chunk_x: i32,
        chunk_y: i32,
        chunk_z: i32,
        chunk_size: i32,
        chunk_height: i32,
        world_height: i32,
        vein_sizes: Vec<i32>,
        veins_per_chunk: Vec<i32>,
        min_ys: Vec<i32>,
        max_ys: Vec<i32>,
        discard_on_air_chances: Vec<f64>,
    ) -> PyResult<Vec<i16>> {
        if chunk_size <= 0 || chunk_height <= 0 || world_height <= 0 {
            return Err(PyValueError::new_err("chunk_size, chunk_height, and world_height must be > 0"));
        }
        let n = vein_sizes.len();
        if veins_per_chunk.len() != n
            || min_ys.len() != n
            || max_ys.len() != n
            || discard_on_air_chances.len() != n
        {
            return Err(PyValueError::new_err(
                "ore rule arrays must have the same length",
            ));
        }
        Ok(self.ore_mask(
            chunk_x,
            chunk_y,
            chunk_z,
            chunk_size,
            chunk_height,
            world_height,
            &vein_sizes,
            &veins_per_chunk,
            &min_ys,
            &max_ys,
            &discard_on_air_chances,
        ))
    }
}

impl TerrainKernel {
    #[inline]
    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    #[inline]
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }

    #[inline]
    fn grad(hash_value: i32, x: f64, y: f64) -> f64 {
        let h = hash_value & 7;
        let u = if h < 4 { x } else { y };
        let v = if h < 4 { y } else { x };
        let u_term = if (h & 1) == 0 { u } else { -u };
        let v_term = if (h & 2) == 0 { v } else { -v };
        u_term + v_term
    }

    #[inline]
    fn grad3(hash_value: i32, x: f64, y: f64, z: f64) -> f64 {
        let h = hash_value & 15;
        let u = if h < 8 { x } else { y };
        let v = if h < 4 {
            y
        } else if h == 12 || h == 14 {
            x
        } else {
            z
        };
        let u_term = if (h & 1) == 0 { u } else { -u };
        let v_term = if (h & 2) == 0 { v } else { -v };
        u_term + v_term
    }

    fn perlin2(&self, x: f64, y: f64) -> f64 {
        let xi = ((x.floor() as i32) & 255) as usize;
        let yi = (y.floor() as i32) & 255;
        let xf = x - x.floor();
        let yf = y - y.floor();

        let u = Self::fade(xf);
        let v = Self::fade(yf);

        let aa = self.perm[(self.perm[xi] + yi) as usize];
        let ab = self.perm[(self.perm[xi] + yi + 1) as usize];
        let ba = self.perm[(self.perm[xi + 1] + yi) as usize];
        let bb = self.perm[(self.perm[xi + 1] + yi + 1) as usize];

        let x1 = Self::lerp(Self::grad(aa, xf, yf), Self::grad(ba, xf - 1.0, yf), u);
        let x2 = Self::lerp(
            Self::grad(ab, xf, yf - 1.0),
            Self::grad(bb, xf - 1.0, yf - 1.0),
            u,
        );
        Self::lerp(x1, x2, v)
    }

    fn perlin3(&self, x: f64, y: f64, z: f64) -> f64 {
        let xi = ((x.floor() as i32) & 255) as usize;
        let yi = (y.floor() as i32) & 255;
        let zi = (z.floor() as i32) & 255;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();

        let u = Self::fade(xf);
        let v = Self::fade(yf);
        let w = Self::fade(zf);

        let a = self.perm[xi] + yi;
        let aa = self.perm[a as usize] + zi;
        let ab = self.perm[(a + 1) as usize] + zi;
        let b = self.perm[xi + 1] + yi;
        let ba = self.perm[b as usize] + zi;
        let bb = self.perm[(b + 1) as usize] + zi;

        let x1 = Self::lerp(
            Self::grad3(self.perm[aa as usize], xf, yf, zf),
            Self::grad3(self.perm[ba as usize], xf - 1.0, yf, zf),
            u,
        );
        let x2 = Self::lerp(
            Self::grad3(self.perm[ab as usize], xf, yf - 1.0, zf),
            Self::grad3(self.perm[bb as usize], xf - 1.0, yf - 1.0, zf),
            u,
        );
        let y1 = Self::lerp(x1, x2, v);

        let x3 = Self::lerp(
            Self::grad3(self.perm[(aa + 1) as usize], xf, yf, zf - 1.0),
            Self::grad3(self.perm[(ba + 1) as usize], xf - 1.0, yf, zf - 1.0),
            u,
        );
        let x4 = Self::lerp(
            Self::grad3(self.perm[(ab + 1) as usize], xf, yf - 1.0, zf - 1.0),
            Self::grad3(self.perm[(bb + 1) as usize], xf - 1.0, yf - 1.0, zf - 1.0),
            u,
        );
        let y2 = Self::lerp(x3, x4, v);

        Self::lerp(y1, y2, w)
    }

    fn fbm3(
        &self,
        x: f64,
        y: f64,
        z: f64,
        base_frequency: f64,
        octaves: usize,
        persistence: f64,
        lacunarity: f64,
    ) -> f64 {
        let mut frequency = base_frequency;
        let mut amplitude = 1.0;
        let mut total = 0.0;
        let mut max_amplitude = 0.0;
        for _ in 0..octaves {
            total += self.perlin3(x * frequency, y * frequency, z * frequency) * amplitude;
            max_amplitude += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        if max_amplitude > 0.0 {
            total / max_amplitude
        } else {
            0.0
        }
    }

    fn is_cave_raw_eval(&self, x: i32, y: i32, z: i32, surface_y: i32, p: &CaveParams) -> bool {
        if y <= p.cave_min_y || y >= p.cave_max_y {
            return false;
        }
        if y <= p.bedrock_y + p.cave_bedrock_buffer {
            return false;
        }

        let warp_seed_y = (f64::from(y) * 0.85) as i32;
        let warp_x = self.cave_gate_noise_at(x + 311, warp_seed_y, z - 173) * p.cave_warp_strength_xz;
        let warp_y = self.cave_gate_noise_at(x - 191, warp_seed_y, z + 241) * p.cave_warp_strength_y;
        let warp_z = self.cave_gate_noise_at(x + 83, warp_seed_y, z + 127) * p.cave_warp_strength_xz;

        let wx = x + warp_x as i32;
        let wy = f64::from(y) + warp_y;
        let wz = z + warp_z as i32;
        let depth_from_surface = surface_y - y;

        let level_noise = self
            .cave_gate_noise_at(
                (f64::from(wx) * p.cave_level_variation_scale) as i32,
                (wy * p.cave_level_variation_scale) as i32,
                (f64::from(wz) * p.cave_level_variation_scale) as i32,
            )
            .abs();
        let mut primary_band = p.cave_tunnel_primary_band + p.cave_level_variation_band_boost * level_noise;
        let mut secondary_band =
            p.cave_tunnel_secondary_band + (p.cave_level_variation_band_boost * 0.7) * level_noise;
        let mut combined_band = p.cave_combined_band + (p.cave_level_variation_band_boost * 0.8) * level_noise;

        let lower_edge = y - (p.cave_min_y.max(p.bedrock_y + p.cave_bedrock_buffer + 1));
        let upper_edge = p.cave_max_y - y;
        let edge_dist = 0.0_f64.max(f64::from(lower_edge.min(upper_edge)));
        let edge_factor = (edge_dist / p.cave_edge_fade_y.max(1e-6)).clamp(0.0, 1.0);
        let scale = 0.55 + 0.45 * edge_factor;
        primary_band *= scale;
        secondary_band *= scale;
        combined_band *= scale;
        if depth_from_surface <= p.cave_near_surface_depth {
            primary_band *= p.cave_near_surface_band_scale;
            secondary_band *= p.cave_near_surface_band_scale;
            combined_band *= p.cave_near_surface_band_scale;
        }

        let family_hits = |sx: i32, sy: f64, sz: i32, p_lim: f64, s_lim: f64, c_lim: f64, g_lim: f64| {
            let f_primary = self
                .cave_noise_at(sx, (sy * p.cave_tunnel_primary_y_scale) as i32, sz)
                .abs();
            let f_secondary = self
                .material_noise_at(sx + 67, (sy * p.cave_tunnel_secondary_y_scale) as i32, sz - 41)
                .abs();
            if f_primary > p_lim || f_secondary > s_lim || (f_primary + f_secondary) > c_lim {
                return false;
            }
            let f_gate = self
                .cave_gate_noise_at(sx, (sy * p.cave_tunnel_gate_y_scale) as i32, sz)
                .abs();
            f_gate > g_lim
        };

        if family_hits(
            wx,
            wy,
            wz,
            primary_band,
            secondary_band,
            combined_band,
            p.cave_tunnel_gate_threshold,
        ) {
            return true;
        }

        let warp2_x = self.cave_gate_noise_at(x - 503, warp_seed_y, z + 359) * p.cave_family2_warp_strength_xz;
        let warp2_y = self.cave_gate_noise_at(x + 421, warp_seed_y, z - 287) * p.cave_family2_warp_strength_y;
        let warp2_z = self.cave_gate_noise_at(x - 173, warp_seed_y, z - 449) * p.cave_family2_warp_strength_xz;
        let wx2 = x + warp2_x as i32;
        let wy2 = f64::from(y) + warp2_y;
        let wz2 = z + warp2_z as i32;
        family_hits(
            wx2,
            wy2,
            wz2,
            primary_band * p.cave_family2_primary_band_scale,
            secondary_band * p.cave_family2_secondary_band_scale,
            combined_band * ((p.cave_family2_primary_band_scale + p.cave_family2_secondary_band_scale) * 0.5),
            p.cave_family2_gate_threshold,
        )
    }

    fn is_cave_eval(
        &self,
        x: i32,
        y: i32,
        z: i32,
        surface_y: i32,
        p: &CaveParams,
        min_connected_neighbors: i32,
        max_local_open: i32,
    ) -> bool {
        if !self.is_cave_raw_eval(x, y, z, surface_y, p) {
            return false;
        }
        if min_connected_neighbors > 0 {
            let mut connected = 0;
            for (dx, dy, dz) in [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)] {
                if self.is_cave_raw_eval(x + dx, y + dy, z + dz, surface_y, p) {
                    connected += 1;
                    if connected >= min_connected_neighbors {
                        break;
                    }
                }
            }
            if connected < min_connected_neighbors {
                return false;
            }
        }
        if max_local_open > 0 {
            let mut local_open = 0;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        if self.is_cave_raw_eval(x + dx, y + dy, z + dz, surface_y, p) {
                            local_open += 1;
                            if local_open > max_local_open {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }

    #[inline]
    fn chunk_mask_index(lx: i32, ly: i32, lz: i32, chunk_size: i32, chunk_height: i32) -> usize {
        ((lx * chunk_height + ly) * chunk_size + lz) as usize
    }

    #[inline]
    fn surface_index(lx: i32, lz: i32, chunk_size: i32) -> usize {
        (lx * chunk_size + lz) as usize
    }

    #[allow(clippy::too_many_arguments)]
    fn carve_ellipsoid_into_chunk(
        &self,
        mask: &mut [u8],
        chunk_x: i32,
        chunk_y: i32,
        chunk_z: i32,
        chunk_size: i32,
        chunk_height: i32,
        world_height: i32,
        surface_heights: &[i32],
        bedrock_y: i32,
        cave_bedrock_buffer: i32,
        center_x: f64,
        center_y: f64,
        center_z: f64,
        radius_xz: f64,
        radius_y: f64,
    ) {
        let world_x0 = chunk_x * chunk_size;
        let world_z0 = chunk_z * chunk_size;
        let world_y0 = chunk_y * chunk_height;
        let world_y1 = (world_y0 + chunk_height - 1).min(world_height - 1);

        let min_x = (center_x - radius_xz).floor() as i32;
        let max_x = (center_x + radius_xz).ceil() as i32;
        let min_y = (center_y - radius_y).floor() as i32;
        let max_y = (center_y + radius_y).ceil() as i32;
        let min_z = (center_z - radius_xz).floor() as i32;
        let max_z = (center_z + radius_xz).ceil() as i32;

        let start_x = min_x.max(world_x0);
        let end_x = max_x.min(world_x0 + chunk_size - 1);
        let start_z = min_z.max(world_z0);
        let end_z = max_z.min(world_z0 + chunk_size - 1);
        let start_y = min_y.max(world_y0).max(bedrock_y + cave_bedrock_buffer + 1);
        let end_y = max_y.min(world_y1);

        if start_x > end_x || start_y > end_y || start_z > end_z {
            return;
        }

        let inv_rx = 1.0 / radius_xz.max(1e-6);
        let inv_ry = 1.0 / radius_y.max(1e-6);
        for wx in start_x..=end_x {
            let lx = wx - world_x0;
            let dx = ((wx as f64 + 0.5) - center_x) * inv_rx;
            let dx2 = dx * dx;
            if dx2 >= 1.0 {
                continue;
            }
            for wz in start_z..=end_z {
                let lz = wz - world_z0;
                let dz = ((wz as f64 + 0.5) - center_z) * inv_rx;
                let dz2 = dz * dz;
                let xz2 = dx2 + dz2;
                if xz2 >= 1.0 {
                    continue;
                }

                let surface_y = surface_heights[Self::surface_index(lx, lz, chunk_size)];
                for wy in start_y..=end_y {
                    if wy >= surface_y {
                        continue;
                    }
                    let dy = ((wy as f64 + 0.5) - center_y) * inv_ry;
                    if xz2 + dy * dy < 1.0 {
                        let ly = wy - world_y0;
                        let idx = Self::chunk_mask_index(lx, ly, lz, chunk_size, chunk_height);
                        mask[idx] = 1;
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_tunnel(
        &self,
        target_chunk_x: i32,
        target_chunk_y: i32,
        target_chunk_z: i32,
        chunk_size: i32,
        chunk_height: i32,
        world_height: i32,
        surface_heights: &[i32],
        bedrock_y: i32,
        cave_bedrock_buffer: i32,
        rng: &mut JavaRandom,
        mask: &mut [u8],
        mut x: f64,
        mut y: f64,
        mut z: f64,
        width: f64,
        mut yaw: f64,
        mut pitch: f64,
        mut step: i32,
        max_steps: i32,
        vertical_scale: f64,
        pitch_change_strength: f64,
        vertical_drift_strength: f64,
    ) {
        let mut yaw_change = 0.0_f64;
        let mut pitch_change = 0.0_f64;
        let mut vertical_drift = 0.0_f64;
        let split_step = rng.next_int(max_steps / 2 + 1) + max_steps / 4;

        while step < max_steps {
            let t = (step as f64) / (max_steps as f64);
            let radius_xz = 1.5 + (std::f64::consts::PI * t).sin() * width;
            let radius_y = radius_xz * vertical_scale;

            let cos_pitch = pitch.cos();
            x += yaw.cos() * cos_pitch;
            vertical_drift *= 0.88;
            vertical_drift += (rng.next_float() as f64 - 0.5) * vertical_drift_strength;
            y += pitch.sin() + vertical_drift;
            z += yaw.sin() * cos_pitch;

            pitch *= 0.78;
            pitch += pitch_change * 0.05;
            yaw += yaw_change * 0.05;
            pitch_change *= 0.8;
            yaw_change *= 0.5;
            pitch_change += (rng.next_float() as f64 - rng.next_float() as f64)
                * (rng.next_float() as f64)
                * (2.0 + pitch_change_strength);
            yaw_change += (rng.next_float() as f64 - rng.next_float() as f64) * (rng.next_float() as f64) * 4.0;

            if step == split_step && width > 1.0 {
                let branch_pitch = pitch / 3.0;
                let next_step = step;
                let branch_width = width * (0.75 + (rng.next_float() as f64) * 0.3);
                self.add_tunnel(
                    target_chunk_x,
                    target_chunk_y,
                    target_chunk_z,
                    chunk_size,
                    chunk_height,
                    world_height,
                    surface_heights,
                    bedrock_y,
                    cave_bedrock_buffer,
                    rng,
                    mask,
                    x,
                    y,
                    z,
                    branch_width,
                    yaw - std::f64::consts::FRAC_PI_2,
                    branch_pitch,
                    next_step,
                    max_steps,
                    vertical_scale,
                    pitch_change_strength,
                    vertical_drift_strength,
                );
                self.add_tunnel(
                    target_chunk_x,
                    target_chunk_y,
                    target_chunk_z,
                    chunk_size,
                    chunk_height,
                    world_height,
                    surface_heights,
                    bedrock_y,
                    cave_bedrock_buffer,
                    rng,
                    mask,
                    x,
                    y,
                    z,
                    branch_width,
                    yaw + std::f64::consts::FRAC_PI_2,
                    branch_pitch,
                    next_step,
                    max_steps,
                    vertical_scale,
                    pitch_change_strength,
                    vertical_drift_strength,
                );
                return;
            }

            if rng.next_int(4) != 0 {
                self.carve_ellipsoid_into_chunk(
                    mask,
                    target_chunk_x,
                    target_chunk_y,
                    target_chunk_z,
                    chunk_size,
                    chunk_height,
                    world_height,
                    surface_heights,
                    bedrock_y,
                    cave_bedrock_buffer,
                    x,
                    y,
                    z,
                    radius_xz,
                    radius_y,
                );
            }
            step += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn carve_from_source_chunk(
        &self,
        src_chunk_x: i32,
        src_chunk_z: i32,
        target_chunk_x: i32,
        target_chunk_y: i32,
        target_chunk_z: i32,
        chunk_size: i32,
        chunk_height: i32,
        world_height: i32,
        surface_heights: &[i32],
        bedrock_y: i32,
        cave_bedrock_buffer: i32,
        cave_min_y: i32,
        cave_max_y: i32,
        cave_system_chance: i32,
        cave_vertical_scale: f64,
        cave_branch_chance: i32,
        cave_initial_pitch_range: f64,
        cave_pitch_change_strength: f64,
        cave_vertical_drift_strength: f64,
        x_seed: i64,
        z_seed: i64,
        mask: &mut [u8],
    ) {
        let seed = (src_chunk_x as i64)
            .wrapping_mul(x_seed)
            ^ (src_chunk_z as i64).wrapping_mul(z_seed)
            ^ self.world_seed;
        let mut rng = JavaRandom::new(seed);

        let n0 = rng.next_int(40) + 1;
        let n1 = rng.next_int(n0) + 1;
        let mut systems = rng.next_int(n1);
        if rng.next_int(cave_system_chance) != 0 {
            systems = 0;
        }

        let y_span = (cave_max_y - cave_min_y).max(8);
        for _ in 0..systems {
            let mut x = (src_chunk_x * chunk_size + rng.next_int(chunk_size)) as f64;
            let y_pick = rng.next_int(y_span);
            let mut y = (cave_min_y + y_pick).min(cave_max_y - 1) as f64;
            let mut z = (src_chunk_z * chunk_size + rng.next_int(chunk_size)) as f64;

            let mut tunnels = 1 + rng.next_int(2);
            if rng.next_int(cave_branch_chance) == 0 {
                let large_width = 1.0 + rng.next_float() as f64 * 3.0;
                let yaw = rng.next_float() as f64 * std::f64::consts::TAU;
                let pitch = (rng.next_float() as f64 - 0.5) * cave_initial_pitch_range;
                let max_steps = 48 + rng.next_int(48);
                self.add_tunnel(
                    target_chunk_x,
                    target_chunk_y,
                    target_chunk_z,
                    chunk_size,
                    chunk_height,
                    world_height,
                    surface_heights,
                    bedrock_y,
                    cave_bedrock_buffer,
                    &mut rng,
                    mask,
                    x,
                    y,
                    z,
                    large_width,
                    yaw,
                    pitch,
                    0,
                    max_steps,
                    cave_vertical_scale,
                    cave_pitch_change_strength,
                    cave_vertical_drift_strength,
                );
                tunnels += rng.next_int(3);
            }

            for _ in 0..tunnels {
                x += (rng.next_float() as f64 - 0.5) * 6.0;
                y += (rng.next_float() as f64 - 0.5) * 3.0;
                z += (rng.next_float() as f64 - 0.5) * 6.0;
                let width = 0.8 + rng.next_float() as f64 * 1.6;
                let yaw = rng.next_float() as f64 * std::f64::consts::TAU;
                let pitch = (rng.next_float() as f64 - 0.5) * (cave_initial_pitch_range * 0.9);
                let max_steps = 36 + rng.next_int(52);
                self.add_tunnel(
                    target_chunk_x,
                    target_chunk_y,
                    target_chunk_z,
                    chunk_size,
                    chunk_height,
                    world_height,
                    surface_heights,
                    bedrock_y,
                    cave_bedrock_buffer,
                    &mut rng,
                    mask,
                    x,
                    y,
                    z,
                    width,
                    yaw,
                    pitch,
                    0,
                    max_steps,
                    cave_vertical_scale,
                    cave_pitch_change_strength,
                    cave_vertical_drift_strength,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn classic_cave_mask(
        &self,
        chunk_x: i32,
        chunk_y: i32,
        chunk_z: i32,
        chunk_size: i32,
        chunk_height: i32,
        world_height: i32,
        surface_heights: &[i32],
        bedrock_y: i32,
        cave_bedrock_buffer: i32,
        cave_min_y: i32,
        cave_max_y: i32,
        cave_range: i32,
        cave_system_chance: i32,
        cave_vertical_scale: f64,
        cave_branch_chance: i32,
        cave_initial_pitch_range: f64,
        cave_pitch_change_strength: f64,
        cave_vertical_drift_strength: f64,
    ) -> Vec<u8> {
        let mask_len = (chunk_size as usize) * (chunk_height as usize) * (chunk_size as usize);
        let mut mask = vec![0_u8; mask_len];

        let mut seed_rng = JavaRandom::new(self.world_seed);
        let x_seed = seed_rng.next_long();
        let z_seed = seed_rng.next_long();

        for src_cx in (chunk_x - cave_range)..=(chunk_x + cave_range) {
            for src_cz in (chunk_z - cave_range)..=(chunk_z + cave_range) {
                self.carve_from_source_chunk(
                    src_cx,
                    src_cz,
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
                    cave_system_chance,
                    cave_vertical_scale,
                    cave_branch_chance,
                    cave_initial_pitch_range,
                    cave_pitch_change_strength,
                    cave_vertical_drift_strength,
                    x_seed,
                    z_seed,
                    &mut mask,
                );
            }
        }
        mask
    }

    #[allow(clippy::too_many_arguments)]
    fn ore_mask(
        &self,
        chunk_x: i32,
        chunk_y: i32,
        chunk_z: i32,
        chunk_size: i32,
        chunk_height: i32,
        world_height: i32,
        vein_sizes: &[i32],
        veins_per_chunk: &[i32],
        min_ys: &[i32],
        max_ys: &[i32],
        discard_on_air_chances: &[f64],
    ) -> Vec<i16> {
        let len = (chunk_size as usize) * (chunk_height as usize) * (chunk_size as usize);
        let mut mask = vec![-1_i16; len];
        let x0 = chunk_x * chunk_size;
        let y0 = chunk_y * chunk_height;
        let z0 = chunk_z * chunk_size;
        let y1 = (y0 + chunk_height - 1).min(world_height - 1);

        for rule_idx in 0..vein_sizes.len() {
            let vein_size = vein_sizes[rule_idx].max(1);
            let tries = veins_per_chunk[rule_idx].max(0);
            let min_y = min_ys[rule_idx].max(0).min(world_height - 1);
            let max_y = max_ys[rule_idx].max(0).min(world_height - 1);
            if tries <= 0 || max_y < min_y {
                continue;
            }
            let discard_chance = discard_on_air_chances[rule_idx].clamp(0.0, 1.0);
            for attempt in 0..tries {
                let seed_base = (self.world_seed as u64)
                    ^ ((chunk_x as i64 as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    ^ ((chunk_y as i64 as u64).wrapping_mul(0xBF58476D1CE4E5B9))
                    ^ ((chunk_z as i64 as u64).wrapping_mul(0x94D049BB133111EB))
                    ^ ((rule_idx as u64).wrapping_mul(0x369DEA0F31A53F85))
                    ^ (attempt as u64).wrapping_mul(0xDB4F0B9175AE2165);
                let mut rng = JavaRandom::new(Self::splitmix64(seed_base) as i64);
                let start_x = x0 + rng.next_int(chunk_size);
                let start_z = z0 + rng.next_int(chunk_size);
                let y_span = (max_y - min_y + 1).max(1);
                let start_y = min_y + rng.next_int(y_span);

                let angle = rng.next_float() as f64 * std::f64::consts::TAU;
                let dx = (angle.sin() * (vein_size as f64) / 8.0).max(0.1);
                let dz = (angle.cos() * (vein_size as f64) / 8.0).max(0.1);
                let x1v = start_x as f64 + dx;
                let x2v = start_x as f64 - dx;
                let z1v = start_z as f64 + dz;
                let z2v = start_z as f64 - dz;
                let y1v = start_y as f64 + (rng.next_int(3) - 1) as f64;
                let y2v = start_y as f64 + (rng.next_int(3) - 1) as f64;

                for step in 0..vein_size {
                    let t = if vein_size > 1 {
                        (step as f64) / ((vein_size - 1) as f64)
                    } else {
                        0.0
                    };
                    let cx = x1v + (x2v - x1v) * t;
                    let cy = y1v + (y2v - y1v) * t;
                    let cz = z1v + (z2v - z1v) * t;
                    let blob_r = (((std::f64::consts::PI * t).sin() + 1.0) * (rng.next_float() as f64) + 1.0)
                        * (vein_size as f64)
                        / 32.0;
                    let rx = blob_r + 0.6;
                    let ry = blob_r + 0.4;
                    let rz = blob_r + 0.6;

                    let min_x = (cx - rx).floor() as i32;
                    let max_x = (cx + rx).ceil() as i32;
                    let min_yb = (cy - ry).floor() as i32;
                    let max_yb = (cy + ry).ceil() as i32;
                    let min_z = (cz - rz).floor() as i32;
                    let max_z = (cz + rz).ceil() as i32;

                    for wx in min_x..=max_x {
                        if wx < x0 || wx >= x0 + chunk_size {
                            continue;
                        }
                        let nx = ((wx as f64 + 0.5) - cx) / rx.max(1e-6);
                        let nx2 = nx * nx;
                        if nx2 >= 1.0 {
                            continue;
                        }
                        for wz in min_z..=max_z {
                            if wz < z0 || wz >= z0 + chunk_size {
                                continue;
                            }
                            let nz = ((wz as f64 + 0.5) - cz) / rz.max(1e-6);
                            let nz2 = nz * nz;
                            if nx2 + nz2 >= 1.0 {
                                continue;
                            }
                            for wy in min_yb..=max_yb {
                                if wy < y0 || wy > y1 || wy < min_y || wy > max_y {
                                    continue;
                                }
                                let ny = ((wy as f64 + 0.5) - cy) / ry.max(1e-6);
                                if nx2 + nz2 + ny * ny >= 1.0 {
                                    continue;
                                }
                                if discard_chance > 0.0 && (rng.next_float() as f64) < discard_chance {
                                    continue;
                                }
                                let lx = wx - x0;
                                let ly = wy - y0;
                                let lz = wz - z0;
                                let idx = Self::chunk_mask_index(lx, ly, lz, chunk_size, chunk_height);
                                mask[idx] = rule_idx as i16;
                            }
                        }
                    }
                }
            }
        }
        mask
    }
}

#[inline]
fn face_cell(face_index: usize, x: i32, y: i32, z: i32) -> (i32, i32, i32) {
    match face_index {
        0 => (x + 1, z, y), // +X
        1 => (x, z, y),     // -X
        2 => (y + 1, x, z), // +Y
        3 => (y, x, z),     // -Y
        4 => (z + 1, x, y), // +Z
        _ => (z, x, y),     // -Z
    }
}

#[inline]
fn merged_face_quad(face_index: usize, plane: i32, u: i32, v: i32, w: i32, h: i32) -> [f32; 12] {
    let half = 0.5f32;
    let p = plane as f32;
    let uf = u as f32;
    let vf = v as f32;
    let wf = w as f32;
    let hf = h as f32;
    match face_index {
        0 => {
            let x = p - half;
            let z0 = uf - half;
            let z1 = uf + wf - half;
            let y0 = vf - half;
            let y1 = vf + hf - half;
            [x, y0, z0, x, y1, z0, x, y1, z1, x, y0, z1]
        }
        1 => {
            let x = p - half;
            let z0 = uf - half;
            let z1 = uf + wf - half;
            let y0 = vf - half;
            let y1 = vf + hf - half;
            [x, y0, z0, x, y0, z1, x, y1, z1, x, y1, z0]
        }
        2 => {
            let y = p - half;
            let x0 = uf - half;
            let x1 = uf + wf - half;
            let z0 = vf - half;
            let z1 = vf + hf - half;
            [x0, y, z0, x0, y, z1, x1, y, z1, x1, y, z0]
        }
        3 => {
            let y = p - half;
            let x0 = uf - half;
            let x1 = uf + wf - half;
            let z0 = vf - half;
            let z1 = vf + hf - half;
            [x0, y, z0, x1, y, z0, x1, y, z1, x0, y, z1]
        }
        4 => {
            let z = p - half;
            let x0 = uf - half;
            let x1 = uf + wf - half;
            let y0 = vf - half;
            let y1 = vf + hf - half;
            [x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z]
        }
        _ => {
            let z = p - half;
            let x0 = uf - half;
            let x1 = uf + wf - half;
            let y0 = vf - half;
            let y1 = vf + hf - half;
            [x0, y0, z, x0, y1, z, x1, y1, z, x1, y0, z]
        }
    }
}

#[inline]
fn quad_to_triangles(face: &[f32; 12]) -> Vec<f32> {
    let mut out = Vec::with_capacity(18);
    out.extend_from_slice(&face[0..9]);
    out.extend_from_slice(&face[0..3]);
    out.extend_from_slice(&face[6..12]);
    out
}

#[inline]
fn face_uv(face_index: usize) -> [f32; 12] {
    match face_index {
        0 => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        1 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        2 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        3 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        4 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        _ => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    }
}

#[inline]
fn face_uv_tiled(face_index: usize, repeat_u: f32, repeat_v: f32) -> Vec<f32> {
    let uv = face_uv(face_index);
    let mut tiled = [0.0f32; 12];
    for i in (0..12).step_by(3) {
        tiled[i] = uv[i] * repeat_u;
        tiled[i + 1] = uv[i + 1] * repeat_v;
        tiled[i + 2] = uv[i + 2];
    }
    quad_to_triangles(&tiled)
}

#[allow(clippy::too_many_arguments)]
fn append_face(
    color_vertices: &mut Vec<f32>,
    color_values: &mut Vec<f32>,
    textured_vertices: &mut HashMap<String, Vec<f32>>,
    textured_texcoords: &mut HashMap<String, Vec<f32>>,
    textured_colors: &mut HashMap<String, Vec<f32>>,
    block: &str,
    face_index: usize,
    face: &[f32; 12],
    shade: f32,
    repeat_u: f32,
    repeat_v: f32,
    block_face_textures: &HashMap<String, Vec<Option<String>>>,
    block_colors: &HashMap<String, (f32, f32, f32)>,
) {
    let tri_face = quad_to_triangles(face);
    let texture_name = block_face_textures
        .get(block)
        .and_then(|textures| textures.get(face_index))
        .and_then(|texture| texture.clone());

    if let Some(tex) = texture_name {
        textured_vertices
            .entry(tex.clone())
            .or_default()
            .extend_from_slice(&tri_face);
        textured_texcoords
            .entry(tex.clone())
            .or_default()
            .extend_from_slice(&face_uv_tiled(face_index, repeat_u, repeat_v));
        textured_colors
            .entry(tex)
            .or_default()
            .extend_from_slice(&[0.0f32; 24]);
        return;
    }

    let (br, bg, bb) = block_colors
        .get(block)
        .copied()
        .unwrap_or((1.0, 0.0, 1.0));
    let r = (br * shade).clamp(0.0, 1.0);
    let g = (bg * shade).clamp(0.0, 1.0);
    let b = (bb * shade).clamp(0.0, 1.0);

    color_vertices.extend_from_slice(&tri_face);
    for _ in 0..6 {
        color_values.extend_from_slice(&[r, g, b, 1.0]);
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn build_chunk_mesh_data_native(
    positions: Vec<(i32, i32, i32)>,
    blocks: HashMap<(i32, i32, i32), String>,
    solid_blocks: HashSet<String>,
    block_face_textures: HashMap<String, Vec<Option<String>>>,
    block_colors: HashMap<String, (f32, f32, f32)>,
    use_texture_array: bool,
) -> PyResult<MeshBuildOutput> {
    let mut color_vertices: Vec<f32> = Vec::new();
    let mut color_values: Vec<f32> = Vec::new();
    let mut textured_vertices: HashMap<String, Vec<f32>> = HashMap::new();
    let mut textured_texcoords: HashMap<String, Vec<f32>> = HashMap::new();
    let mut textured_colors: HashMap<String, Vec<f32>> = HashMap::new();

    let shades: [f32; 6] = [0.86, 0.86, 1.0, 0.55, 0.72, 0.72];
    let face_offsets: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    let mut face_planes: Vec<HashMap<i32, HashMap<(i32, i32), String>>> =
        vec![HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new()];

    for (x, y, z) in positions {
        let Some(block) = blocks.get(&(x, y, z)) else {
            continue;
        };
        if block == "water" && blocks.contains_key(&(x, y + 1, z)) {
            continue;
        }
        for (face_index, (dx, dy, dz)) in face_offsets.iter().enumerate() {
            if let Some(neighbor) = blocks.get(&(x + dx, y + dy, z + dz)) {
                if solid_blocks.contains(neighbor) || (block == "water" && neighbor == "water") {
                    continue;
                }
            }
            let (plane, u, v) = face_cell(face_index, x, y, z);
            face_planes[face_index]
                .entry(plane)
                .or_default()
                .insert((u, v), block.clone());
        }
    }

    for (face_index, planes) in face_planes.iter().enumerate() {
        for (plane, cells) in planes {
            let mut visited: HashSet<(i32, i32)> = HashSet::new();
            let mut keys: Vec<(i32, i32)> = cells.keys().copied().collect();
            keys.sort_by_key(|(u, v)| (*v, *u));

            for (u0, v0) in keys {
                if visited.contains(&(u0, v0)) {
                    continue;
                }
                let Some(block) = cells.get(&(u0, v0)) else {
                    continue;
                };

                let mut width = 1;
                loop {
                    let key = (u0 + width, v0);
                    if visited.contains(&key) {
                        break;
                    }
                    if cells.get(&key) != Some(block) {
                        break;
                    }
                    width += 1;
                }

                let mut height = 1;
                loop {
                    let next_v = v0 + height;
                    let mut row_ok = true;
                    for du in 0..width {
                        let key = (u0 + du, next_v);
                        if visited.contains(&key) || cells.get(&key) != Some(block) {
                            row_ok = false;
                            break;
                        }
                    }
                    if !row_ok {
                        break;
                    }
                    height += 1;
                }

                for dv in 0..height {
                    for du in 0..width {
                        visited.insert((u0 + du, v0 + dv));
                    }
                }

                let merged_face = merged_face_quad(face_index, *plane, u0, v0, width, height);
                let texture_name = block_face_textures
                    .get(block)
                    .and_then(|textures| textures.get(face_index))
                    .and_then(|texture| texture.as_ref());

                if texture_name.is_some() && (width > 1 || height > 1) && !use_texture_array {
                    for dv in 0..height {
                        for du in 0..width {
                            let tiled_face =
                                merged_face_quad(face_index, *plane, u0 + du, v0 + dv, 1, 1);
                            append_face(
                                &mut color_vertices,
                                &mut color_values,
                                &mut textured_vertices,
                                &mut textured_texcoords,
                                &mut textured_colors,
                                block,
                                face_index,
                                &tiled_face,
                                shades[face_index],
                                1.0,
                                1.0,
                                &block_face_textures,
                                &block_colors,
                            );
                        }
                    }
                } else {
                    append_face(
                        &mut color_vertices,
                        &mut color_values,
                        &mut textured_vertices,
                        &mut textured_texcoords,
                        &mut textured_colors,
                        block,
                        face_index,
                        &merged_face,
                        shades[face_index],
                        width as f32,
                        height as f32,
                        &block_face_textures,
                        &block_colors,
                    );
                }
            }
        }
    }

    Ok((
        color_vertices,
        color_values,
        textured_vertices,
        textured_texcoords,
        textured_colors,
    ))
}

#[pymodule]
fn minecraftclone_native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TerrainKernel>()?;
    module.add_function(wrap_pyfunction!(build_chunk_mesh_data_native, module)?)?;
    Ok(())
}
