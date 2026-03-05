import math
import time
import json
import heapq
import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from collections import OrderedDict
from queue import Empty, Queue
from typing import Protocol

import pyglet

from engine.blocks import SOLID_BLOCKS, get_block_definition
from engine.constants import SECTOR_SIZE, Vec3
from engine.debug.profiler import RuntimeProfiler
from engine.graphics.block_renderer import BlockRenderer, ChunkMeshData
from engine.world.terrain import TerrainGenerator

_WORLD_GEN_DEFAULTS: dict[str, int | float | bool] = {
    "WORLD_HEIGHT": 256,
    "CHUNK_HEIGHT": 256,
    "VERTICAL_ACTIVE_BELOW_PLAYER": 0,
    "BEDROCK_Y": 0,
    "VISIBLE_RADIUS_CHUNKS": 4,
    "ACTIVE_RADIUS_CHUNKS": 5,
    "UNLOAD_RADIUS_CHUNKS": 6,
    "MAX_LOADED_CHUNKS": 140,
    "CHUNK_CACHE_MAX_ENTRIES": 256,
    "CHUNKS_REQUESTED_PER_UPDATE": 2,
    "CHUNKS_APPLIED_PER_UPDATE": 1,
    "CHUNK_APPLY_BUDGET_SECONDS": 0.0015,
    "CHUNKS_UNLOADED_PER_UPDATE": 1,
    "CHUNK_UNLOAD_BUDGET_SECONDS": 0.0015,
    "MAX_CAP_UNLOADS_PER_UPDATE": 2,
    "CAP_UNLOAD_BUDGET_SECONDS": 0.0030,
    "CHUNK_REBUILDS_PER_UPDATE": 1,
    "CHUNK_REBUILD_BUDGET_SECONDS": 0.0008,
    "CHUNK_DIRTY_SCAN_LIMIT": 12,
    "CHUNK_REBUILD_REQUEUE_COOLDOWN_SECONDS": 0.18,
    "NEIGHBOR_DIRTY_UPDATES_PER_EVENT": 4,
    "MAX_INFLIGHT_MESH_BUILDS": 6,
    "CHUNK_MESH_UPLOADS_PER_UPDATE": 1,
    "CHUNK_MESH_UPLOAD_BUDGET_SECONDS": 0.0015,
    "MIN_LOADED_CHUNKS": 25,
    "MAX_INFLIGHT_REQUESTS": 10,
    "CHUNK_WORKERS": 2,
    "CHUNK_MESH_WORKERS": 2,
    "CHUNK_FRUSTUM_HALF_FOV_DEGREES": 65.0,
    "CHUNK_FRUSTUM_MARGIN_DEGREES": 20.0,
    "CHUNK_FRUSTUM_NEAR_RADIUS_CHUNKS": 1,
    "USE_CHUNK_FRUSTUM_CULLING": False,
    "CAVE_MIN_Y": 5,
    "CAVE_MAX_Y": 64,
    "CAVE_BEDROCK_BUFFER": 2,
    "CAVE_NEAR_SURFACE_DEPTH": 2,
    "CAVE_TUNNEL_PRIMARY_Y_SCALE": 1.10,
    "CAVE_TUNNEL_SECONDARY_Y_SCALE": 1.20,
    "CAVE_TUNNEL_GATE_Y_SCALE": 1.25,
    "CAVE_TUNNEL_PRIMARY_BAND": 0.09,
    "CAVE_TUNNEL_SECONDARY_BAND": 0.10,
    "CAVE_COMBINED_BAND": 0.165,
    "CAVE_NEAR_SURFACE_BAND_SCALE": 0.45,
    "CAVE_TUNNEL_GATE_THRESHOLD": 0.24,
    "CAVE_WARP_STRENGTH_XZ": 10.0,
    "CAVE_WARP_STRENGTH_Y": 4.0,
    "CAVE_LEVEL_VARIATION_SCALE": 0.30,
    "CAVE_LEVEL_VARIATION_BAND_BOOST": 0.01,
    "CAVE_EDGE_FADE_Y": 10.0,
    "CAVE_MIN_CONNECTED_NEIGHBORS": 2,
    "CAVE_MAX_LOCAL_OPEN": 8,
    "CAVE_FAMILY2_WARP_STRENGTH_XZ": 9.0,
    "CAVE_FAMILY2_WARP_STRENGTH_Y": 3.0,
    "CAVE_FAMILY2_PRIMARY_BAND_SCALE": 0.78,
    "CAVE_FAMILY2_SECONDARY_BAND_SCALE": 0.78,
    "CAVE_FAMILY2_GATE_THRESHOLD": 1.0,
    "CAVE_CLASSIC_RANGE": 8,
    "CAVE_CLASSIC_SYSTEM_CHANCE": 17,
    "CAVE_CLASSIC_VERTICAL_SCALE": 1.0,
    "CAVE_CLASSIC_BRANCH_CHANCE": 5,
    "CAVE_CLASSIC_INITIAL_PITCH_RANGE": 0.55,
    "CAVE_CLASSIC_PITCH_CHANGE_STRENGTH": 2.0,
    "CAVE_CLASSIC_VERTICAL_DRIFT_STRENGTH": 0.18,
    "BIOME_BLEND_NOISE_MIN": -0.25,
    "BIOME_BLEND_NOISE_MAX": 0.25,
    "TOP_LAYER_DEPTH": 2,
    "DESERT_SANDSTONE_HIGH_NOISE_THRESHOLD": 0.35,
    "DESERT_SANDSTONE_LOW_Y_MAX": 22,
    "DESERT_SANDSTONE_LOW_NOISE_THRESHOLD": -0.45,
    "PLAINS_DIRT_NOISE_THRESHOLD": 0.58,
    "PLAINS_DIRT_MIN_Y": 18,
    "PLAINS_SANDSTONE_NOISE_THRESHOLD": -0.65,
    "PLAINS_SANDSTONE_MAX_Y": 20,
}
_WORLD_GEN_SECTIONS: dict[str, tuple[str, ...]] = {
    "world": (
        "WORLD_HEIGHT",
        "CHUNK_HEIGHT",
        "VERTICAL_ACTIVE_BELOW_PLAYER",
        "BEDROCK_Y",
    ),
    "streaming": (
        "VISIBLE_RADIUS_CHUNKS",
        "ACTIVE_RADIUS_CHUNKS",
        "UNLOAD_RADIUS_CHUNKS",
        "MAX_LOADED_CHUNKS",
        "CHUNK_CACHE_MAX_ENTRIES",
        "CHUNKS_REQUESTED_PER_UPDATE",
        "CHUNKS_APPLIED_PER_UPDATE",
        "CHUNK_APPLY_BUDGET_SECONDS",
        "CHUNKS_UNLOADED_PER_UPDATE",
        "CHUNK_UNLOAD_BUDGET_SECONDS",
        "MAX_CAP_UNLOADS_PER_UPDATE",
        "CAP_UNLOAD_BUDGET_SECONDS",
        "CHUNK_REBUILDS_PER_UPDATE",
        "CHUNK_REBUILD_BUDGET_SECONDS",
        "CHUNK_DIRTY_SCAN_LIMIT",
        "CHUNK_REBUILD_REQUEUE_COOLDOWN_SECONDS",
        "NEIGHBOR_DIRTY_UPDATES_PER_EVENT",
        "MAX_INFLIGHT_MESH_BUILDS",
        "CHUNK_MESH_UPLOADS_PER_UPDATE",
        "CHUNK_MESH_UPLOAD_BUDGET_SECONDS",
        "MIN_LOADED_CHUNKS",
        "MAX_INFLIGHT_REQUESTS",
        "CHUNK_WORKERS",
        "CHUNK_MESH_WORKERS",
        "CHUNK_FRUSTUM_HALF_FOV_DEGREES",
        "CHUNK_FRUSTUM_MARGIN_DEGREES",
        "CHUNK_FRUSTUM_NEAR_RADIUS_CHUNKS",
        "USE_CHUNK_FRUSTUM_CULLING",
    ),
    "caves": (
        "CAVE_MIN_Y",
        "CAVE_MAX_Y",
        "CAVE_BEDROCK_BUFFER",
        "CAVE_NEAR_SURFACE_DEPTH",
        "CAVE_TUNNEL_PRIMARY_Y_SCALE",
        "CAVE_TUNNEL_SECONDARY_Y_SCALE",
        "CAVE_TUNNEL_GATE_Y_SCALE",
        "CAVE_TUNNEL_PRIMARY_BAND",
        "CAVE_TUNNEL_SECONDARY_BAND",
        "CAVE_COMBINED_BAND",
        "CAVE_NEAR_SURFACE_BAND_SCALE",
        "CAVE_TUNNEL_GATE_THRESHOLD",
        "CAVE_WARP_STRENGTH_XZ",
        "CAVE_WARP_STRENGTH_Y",
        "CAVE_LEVEL_VARIATION_SCALE",
        "CAVE_LEVEL_VARIATION_BAND_BOOST",
        "CAVE_EDGE_FADE_Y",
        "CAVE_MIN_CONNECTED_NEIGHBORS",
        "CAVE_MAX_LOCAL_OPEN",
        "CAVE_FAMILY2_WARP_STRENGTH_XZ",
        "CAVE_FAMILY2_WARP_STRENGTH_Y",
        "CAVE_FAMILY2_PRIMARY_BAND_SCALE",
        "CAVE_FAMILY2_SECONDARY_BAND_SCALE",
        "CAVE_FAMILY2_GATE_THRESHOLD",
        "CAVE_CLASSIC_RANGE",
        "CAVE_CLASSIC_SYSTEM_CHANCE",
        "CAVE_CLASSIC_VERTICAL_SCALE",
        "CAVE_CLASSIC_BRANCH_CHANCE",
        "CAVE_CLASSIC_INITIAL_PITCH_RANGE",
        "CAVE_CLASSIC_PITCH_CHANGE_STRENGTH",
        "CAVE_CLASSIC_VERTICAL_DRIFT_STRENGTH",
    ),
    "biomes": (
        "BIOME_BLEND_NOISE_MIN",
        "BIOME_BLEND_NOISE_MAX",
        "TOP_LAYER_DEPTH",
    ),
    "materials": (
        "DESERT_SANDSTONE_HIGH_NOISE_THRESHOLD",
        "DESERT_SANDSTONE_LOW_Y_MAX",
        "DESERT_SANDSTONE_LOW_NOISE_THRESHOLD",
        "PLAINS_DIRT_NOISE_THRESHOLD",
        "PLAINS_DIRT_MIN_Y",
        "PLAINS_SANDSTONE_NOISE_THRESHOLD",
        "PLAINS_SANDSTONE_MAX_Y",
    ),
}
_WORLD_GEN_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "world_gen_settings.json")
_ORE_RULES_PATH = os.path.join(os.path.dirname(__file__), "ore_rules.json")

_DEFAULT_ORE_RULES: list[dict[str, object]] = [
    {
        "name": "coal",
        "block": "coal_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 18,
        "vein_size": 14,
        "min_y": 0,
        "max_y": 128,
        "discard_on_air_chance": 0.20,
    },
    {
        "name": "iron",
        "block": "iron_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 10,
        "vein_size": 9,
        "min_y": -32,
        "max_y": 80,
        "discard_on_air_chance": 0.25,
    },
    {
        "name": "gold",
        "block": "gold_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 3,
        "vein_size": 8,
        "min_y": -64,
        "max_y": 32,
        "discard_on_air_chance": 0.35,
    },
    {
        "name": "lapis",
        "block": "lapis_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 3,
        "vein_size": 7,
        "min_y": -32,
        "max_y": 32,
        "discard_on_air_chance": 0.25,
    },
    {
        "name": "redstone",
        "block": "redstone_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 6,
        "vein_size": 8,
        "min_y": -64,
        "max_y": 16,
        "discard_on_air_chance": 0.30,
    },
    {
        "name": "diamond",
        "block": "diamond_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 3,
        "vein_size": 7,
        "min_y": -64,
        "max_y": 16,
        "discard_on_air_chance": 0.40,
    },
]


def _load_world_gen_settings() -> dict[str, int | float | bool]:
    settings = dict(_WORLD_GEN_DEFAULTS)
    if not os.path.exists(_WORLD_GEN_SETTINGS_PATH):
        return settings
    with open(_WORLD_GEN_SETTINGS_PATH, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"World settings file must contain a JSON object: {_WORLD_GEN_SETTINGS_PATH}")

    # Backward compatible: accept either old flat keys or grouped sections.
    for key in settings:
        if key in loaded:
            settings[key] = loaded[key]

    for section_name, section_keys in _WORLD_GEN_SECTIONS.items():
        section = loaded.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in section_keys:
            if key in section:
                settings[key] = section[key]
    return settings


def _load_ore_rules() -> list[dict[str, object]]:
    rules_obj: object
    if os.path.exists(_ORE_RULES_PATH):
        with open(_ORE_RULES_PATH, "r", encoding="utf-8") as f:
            rules_obj = json.load(f)
    else:
        rules_obj = _DEFAULT_ORE_RULES

    if not isinstance(rules_obj, list):
        raise RuntimeError(f"Ore rules file must contain a JSON array: {_ORE_RULES_PATH}")

    normalized: list[dict[str, object]] = []
    for i, raw in enumerate(rules_obj):
        if not isinstance(raw, dict):
            continue
        block = str(raw.get("block", "")).strip().lower()
        if not block:
            continue
        enabled = bool(raw.get("enabled", True))
        replace_raw = raw.get("replace", ["stone"])
        replace_list = [str(v).strip().lower() for v in replace_raw] if isinstance(replace_raw, list) else ["stone"]
        replace = tuple(v for v in replace_list if v)
        if not replace:
            replace = ("stone",)
        normalized.append(
            {
                "name": str(raw.get("name", f"rule_{i}")),
                "block": block,
                "enabled": enabled,
                "replace": replace,
                "veins_per_chunk": int(raw.get("veins_per_chunk", 0)),
                "vein_size": int(raw.get("vein_size", 1)),
                "min_y": int(raw.get("min_y", 0)),
                "max_y": int(raw.get("max_y", 0)),
                "discard_on_air_chance": float(raw.get("discard_on_air_chance", 0.0)),
            }
        )
    return normalized


class Deletable(Protocol):
    def delete(self) -> None: ...


class World:
    _WORLD_SETTINGS = _load_world_gen_settings()
    _ORE_RULES = _load_ore_rules()
    CHUNK_SIZE = SECTOR_SIZE
    WORLD_HEIGHT = int(_WORLD_SETTINGS["WORLD_HEIGHT"])
    CHUNK_HEIGHT = int(_WORLD_SETTINGS["CHUNK_HEIGHT"])
    VERTICAL_CHUNKS = max(1, WORLD_HEIGHT // max(1, CHUNK_HEIGHT))
    VERTICAL_ACTIVE_BELOW_PLAYER = int(_WORLD_SETTINGS["VERTICAL_ACTIVE_BELOW_PLAYER"])
    BEDROCK_Y = int(_WORLD_SETTINGS["BEDROCK_Y"])
    VISIBLE_RADIUS_CHUNKS = int(_WORLD_SETTINGS["VISIBLE_RADIUS_CHUNKS"])
    ACTIVE_RADIUS_CHUNKS = int(_WORLD_SETTINGS["ACTIVE_RADIUS_CHUNKS"])
    UNLOAD_RADIUS_CHUNKS = int(_WORLD_SETTINGS["UNLOAD_RADIUS_CHUNKS"])
    MAX_LOADED_CHUNKS = int(_WORLD_SETTINGS["MAX_LOADED_CHUNKS"])
    CHUNK_CACHE_MAX_ENTRIES = int(_WORLD_SETTINGS["CHUNK_CACHE_MAX_ENTRIES"])
    CHUNKS_REQUESTED_PER_UPDATE = int(_WORLD_SETTINGS["CHUNKS_REQUESTED_PER_UPDATE"])
    CHUNKS_APPLIED_PER_UPDATE = int(_WORLD_SETTINGS["CHUNKS_APPLIED_PER_UPDATE"])
    CHUNK_APPLY_BUDGET_SECONDS = float(_WORLD_SETTINGS["CHUNK_APPLY_BUDGET_SECONDS"])
    CHUNKS_UNLOADED_PER_UPDATE = int(_WORLD_SETTINGS["CHUNKS_UNLOADED_PER_UPDATE"])
    CHUNK_UNLOAD_BUDGET_SECONDS = float(_WORLD_SETTINGS["CHUNK_UNLOAD_BUDGET_SECONDS"])
    MAX_CAP_UNLOADS_PER_UPDATE = int(_WORLD_SETTINGS["MAX_CAP_UNLOADS_PER_UPDATE"])
    CAP_UNLOAD_BUDGET_SECONDS = float(_WORLD_SETTINGS["CAP_UNLOAD_BUDGET_SECONDS"])
    CHUNK_REBUILDS_PER_UPDATE = int(_WORLD_SETTINGS["CHUNK_REBUILDS_PER_UPDATE"])
    CHUNK_REBUILD_BUDGET_SECONDS = float(_WORLD_SETTINGS["CHUNK_REBUILD_BUDGET_SECONDS"])
    CHUNK_DIRTY_SCAN_LIMIT = int(_WORLD_SETTINGS["CHUNK_DIRTY_SCAN_LIMIT"])
    CHUNK_REBUILD_REQUEUE_COOLDOWN_SECONDS = float(_WORLD_SETTINGS["CHUNK_REBUILD_REQUEUE_COOLDOWN_SECONDS"])
    # Keep chunk borders consistent as chunks load/unload; stale seam meshes can
    # look like overlapping geometry if neighbors are never dirtied.
    NEIGHBOR_DIRTY_UPDATES_PER_EVENT = int(_WORLD_SETTINGS["NEIGHBOR_DIRTY_UPDATES_PER_EVENT"])
    MAX_INFLIGHT_MESH_BUILDS = int(_WORLD_SETTINGS["MAX_INFLIGHT_MESH_BUILDS"])
    CHUNK_MESH_UPLOADS_PER_UPDATE = int(_WORLD_SETTINGS["CHUNK_MESH_UPLOADS_PER_UPDATE"])
    CHUNK_MESH_UPLOAD_BUDGET_SECONDS = float(_WORLD_SETTINGS["CHUNK_MESH_UPLOAD_BUDGET_SECONDS"])
    MIN_LOADED_CHUNKS = int(_WORLD_SETTINGS["MIN_LOADED_CHUNKS"])
    MAX_INFLIGHT_REQUESTS = int(_WORLD_SETTINGS["MAX_INFLIGHT_REQUESTS"])
    CHUNK_WORKERS = int(_WORLD_SETTINGS["CHUNK_WORKERS"])
    CHUNK_MESH_WORKERS = int(_WORLD_SETTINGS["CHUNK_MESH_WORKERS"])
    CHUNK_FRUSTUM_HALF_FOV_DEGREES = float(_WORLD_SETTINGS["CHUNK_FRUSTUM_HALF_FOV_DEGREES"])
    CHUNK_FRUSTUM_MARGIN_DEGREES = float(_WORLD_SETTINGS["CHUNK_FRUSTUM_MARGIN_DEGREES"])
    CHUNK_FRUSTUM_NEAR_RADIUS_CHUNKS = int(_WORLD_SETTINGS["CHUNK_FRUSTUM_NEAR_RADIUS_CHUNKS"])
    USE_CHUNK_FRUSTUM_CULLING = bool(_WORLD_SETTINGS["USE_CHUNK_FRUSTUM_CULLING"])
    CAVE_MIN_Y = int(_WORLD_SETTINGS["CAVE_MIN_Y"])
    CAVE_MAX_Y = int(_WORLD_SETTINGS["CAVE_MAX_Y"])
    CAVE_BEDROCK_BUFFER = int(_WORLD_SETTINGS["CAVE_BEDROCK_BUFFER"])
    CAVE_NEAR_SURFACE_DEPTH = int(_WORLD_SETTINGS["CAVE_NEAR_SURFACE_DEPTH"])
    CAVE_TUNNEL_PRIMARY_Y_SCALE = float(_WORLD_SETTINGS["CAVE_TUNNEL_PRIMARY_Y_SCALE"])
    CAVE_TUNNEL_SECONDARY_Y_SCALE = float(_WORLD_SETTINGS["CAVE_TUNNEL_SECONDARY_Y_SCALE"])
    CAVE_TUNNEL_GATE_Y_SCALE = float(_WORLD_SETTINGS["CAVE_TUNNEL_GATE_Y_SCALE"])
    CAVE_TUNNEL_PRIMARY_BAND = float(_WORLD_SETTINGS["CAVE_TUNNEL_PRIMARY_BAND"])
    CAVE_TUNNEL_SECONDARY_BAND = float(_WORLD_SETTINGS["CAVE_TUNNEL_SECONDARY_BAND"])
    CAVE_COMBINED_BAND = float(_WORLD_SETTINGS["CAVE_COMBINED_BAND"])
    CAVE_NEAR_SURFACE_BAND_SCALE = float(_WORLD_SETTINGS["CAVE_NEAR_SURFACE_BAND_SCALE"])
    CAVE_TUNNEL_GATE_THRESHOLD = float(_WORLD_SETTINGS["CAVE_TUNNEL_GATE_THRESHOLD"])
    CAVE_WARP_STRENGTH_XZ = float(_WORLD_SETTINGS["CAVE_WARP_STRENGTH_XZ"])
    CAVE_WARP_STRENGTH_Y = float(_WORLD_SETTINGS["CAVE_WARP_STRENGTH_Y"])
    CAVE_LEVEL_VARIATION_SCALE = float(_WORLD_SETTINGS["CAVE_LEVEL_VARIATION_SCALE"])
    CAVE_LEVEL_VARIATION_BAND_BOOST = float(_WORLD_SETTINGS["CAVE_LEVEL_VARIATION_BAND_BOOST"])
    CAVE_EDGE_FADE_Y = float(_WORLD_SETTINGS["CAVE_EDGE_FADE_Y"])
    CAVE_MIN_CONNECTED_NEIGHBORS = int(_WORLD_SETTINGS["CAVE_MIN_CONNECTED_NEIGHBORS"])
    CAVE_MAX_LOCAL_OPEN = int(_WORLD_SETTINGS["CAVE_MAX_LOCAL_OPEN"])
    CAVE_FAMILY2_WARP_STRENGTH_XZ = float(_WORLD_SETTINGS["CAVE_FAMILY2_WARP_STRENGTH_XZ"])
    CAVE_FAMILY2_WARP_STRENGTH_Y = float(_WORLD_SETTINGS["CAVE_FAMILY2_WARP_STRENGTH_Y"])
    CAVE_FAMILY2_PRIMARY_BAND_SCALE = float(_WORLD_SETTINGS["CAVE_FAMILY2_PRIMARY_BAND_SCALE"])
    CAVE_FAMILY2_SECONDARY_BAND_SCALE = float(_WORLD_SETTINGS["CAVE_FAMILY2_SECONDARY_BAND_SCALE"])
    CAVE_FAMILY2_GATE_THRESHOLD = float(_WORLD_SETTINGS["CAVE_FAMILY2_GATE_THRESHOLD"])
    CAVE_CLASSIC_RANGE = int(_WORLD_SETTINGS["CAVE_CLASSIC_RANGE"])
    CAVE_CLASSIC_SYSTEM_CHANCE = int(_WORLD_SETTINGS["CAVE_CLASSIC_SYSTEM_CHANCE"])
    CAVE_CLASSIC_VERTICAL_SCALE = float(_WORLD_SETTINGS["CAVE_CLASSIC_VERTICAL_SCALE"])
    CAVE_CLASSIC_BRANCH_CHANCE = int(_WORLD_SETTINGS["CAVE_CLASSIC_BRANCH_CHANCE"])
    CAVE_CLASSIC_INITIAL_PITCH_RANGE = float(_WORLD_SETTINGS["CAVE_CLASSIC_INITIAL_PITCH_RANGE"])
    CAVE_CLASSIC_PITCH_CHANGE_STRENGTH = float(_WORLD_SETTINGS["CAVE_CLASSIC_PITCH_CHANGE_STRENGTH"])
    CAVE_CLASSIC_VERTICAL_DRIFT_STRENGTH = float(_WORLD_SETTINGS["CAVE_CLASSIC_VERTICAL_DRIFT_STRENGTH"])
    BIOME_BLEND_NOISE_MIN = float(_WORLD_SETTINGS["BIOME_BLEND_NOISE_MIN"])
    BIOME_BLEND_NOISE_MAX = float(_WORLD_SETTINGS["BIOME_BLEND_NOISE_MAX"])
    TOP_LAYER_DEPTH = int(_WORLD_SETTINGS["TOP_LAYER_DEPTH"])
    DESERT_SANDSTONE_HIGH_NOISE_THRESHOLD = float(_WORLD_SETTINGS["DESERT_SANDSTONE_HIGH_NOISE_THRESHOLD"])
    DESERT_SANDSTONE_LOW_Y_MAX = int(_WORLD_SETTINGS["DESERT_SANDSTONE_LOW_Y_MAX"])
    DESERT_SANDSTONE_LOW_NOISE_THRESHOLD = float(_WORLD_SETTINGS["DESERT_SANDSTONE_LOW_NOISE_THRESHOLD"])
    PLAINS_DIRT_NOISE_THRESHOLD = float(_WORLD_SETTINGS["PLAINS_DIRT_NOISE_THRESHOLD"])
    PLAINS_DIRT_MIN_Y = int(_WORLD_SETTINGS["PLAINS_DIRT_MIN_Y"])
    PLAINS_SANDSTONE_NOISE_THRESHOLD = float(_WORLD_SETTINGS["PLAINS_SANDSTONE_NOISE_THRESHOLD"])
    PLAINS_SANDSTONE_MAX_Y = int(_WORLD_SETTINGS["PLAINS_SANDSTONE_MAX_Y"])
    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, "..", "biomes", "biomes.json")
    json_path = os.path.normpath(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        biomes_data = json.load(f)

    _FACE_NEIGHBORS = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    _NO_OVERRIDE = object()

    def __init__(
        self,
        seed: int = 1337,
        flat_height: int = 64,
        profiler: RuntimeProfiler | None = None,
        use_texture_array: bool = False,
    ) -> None:
        self.seed = seed
        self.profiler = profiler
        self.blocks: dict[Vec3, str] = {}
        self._chunk_blocks: dict[tuple[int, int, int], set[Vec3]] = {}
        self._chunk_meshes: dict[tuple[int, int, int], Deletable] = {}
        self._dirty_chunks: set[tuple[int, int, int]] = set()
        self._loaded_chunks: set[tuple[int, int, int]] = set()
        self._modified_blocks: dict[Vec3, str | None] = {}
        self._chunk_modified_positions: dict[tuple[int, int, int], set[Vec3]] = {}
        self._center_chunk: tuple[int, int, int] | None = None
        self._desired_chunks: set[tuple[int, int, int]] = set()
        self._visible_chunks: set[tuple[int, int, int]] = set()
        self._requested_chunks: set[tuple[int, int, int]] = set()
        self._chunk_futures: dict[tuple[int, int, int], Future[list[tuple[Vec3, str]]]] = {}
        self._completed_chunks: Queue[tuple[int, int, int]] = Queue()
        self._executor = ThreadPoolExecutor(max_workers=self.CHUNK_WORKERS, thread_name_prefix="chunkgen")

        self._mesh_executor = ThreadPoolExecutor(max_workers=self.CHUNK_MESH_WORKERS, thread_name_prefix="meshbuild")
        self._mesh_futures: dict[tuple[int, int, int], Future[ChunkMeshData]] = {}
        self._mesh_completed: Queue[tuple[int, int, int, int]] = Queue()
        self._mesh_chunk_versions: dict[tuple[int, int, int], int] = {}
        self._mesh_inflight_versions: dict[tuple[int, int, int], int] = {}
        self._chunk_mesh_requeue_after: dict[tuple[int, int, int], float] = {}
        self._chunk_cache: OrderedDict[tuple[int, int, int], list[tuple[Vec3, str]]] = OrderedDict()

        self.batch = pyglet.graphics.Batch()
        self.terrain = TerrainGenerator(seed, flat_height=flat_height)
        self.renderer = BlockRenderer(seed, use_texture_array=use_texture_array)
        self.group = pyglet.graphics.ShaderGroup(program=self.renderer.shader)
        self.hide_surface_layer = False
        self.xray_hide_depth = 1
        self._ore_rules = [dict(rule) for rule in self._ORE_RULES]


    def _profile(self, name: str):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.section(name)

    @staticmethod
    def normalize(position: tuple[float, float, float]) -> Vec3:
        x, y, z = position
        return int(round(x)), int(round(y)), int(round(z))

    @staticmethod
    def sectorize(position: Vec3) -> tuple[int, int, int]:
        x, y, z = position
        return (x // SECTOR_SIZE, y // SECTOR_SIZE, z // SECTOR_SIZE)

    @staticmethod
    def chunk_coords(x: float, z: float) -> tuple[int, int]:
        return math.floor(x / World.CHUNK_SIZE), math.floor(z / World.CHUNK_SIZE)

    @staticmethod
    def vertical_chunk(y: float) -> int:
        return max(0, min(World.VERTICAL_CHUNKS - 1, math.floor(y / World.CHUNK_HEIGHT)))

    @staticmethod
    def chunk_key(x: float, y: float, z: float) -> tuple[int, int, int]:
        cx, cz = World.chunk_coords(x, z)
        cy = World.vertical_chunk(y)
        return cx, cy, cz

    def _target_vertical_chunks(self, y: float) -> set[int]:
        player_cy = self.vertical_chunk(y)
        target = {self.VERTICAL_CHUNKS - 1}
        for d in range(self.VERTICAL_ACTIVE_BELOW_PLAYER + 1):
            target.add(max(0, player_cy - d))
        return target

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, x: float) -> float:
        if edge0 == edge1:
            return 0.0
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)

    def _chunk_neighbors(
        self,
        chunk: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        cx, cy, cz = chunk
        up = (cx, cy + 1, cz) if cy + 1 < self.VERTICAL_CHUNKS else (cx, cy, cz)
        down = (cx, cy - 1, cz) if cy - 1 >= 0 else (cx, cy, cz)
        return (cx - 1, cy, cz), (cx + 1, cy, cz), (cx, cy, cz - 1), (cx, cy, cz + 1), up, down

    def _chunk_in_frustum(
        self,
        chunk: tuple[int, int],
        center_chunk: tuple[int, int],
        forward_x: float,
        forward_z: float,
    ) -> bool:
        cx, cz = chunk
        pcx, pcz = center_chunk
        dx = cx - pcx
        dz = cz - pcz
        dist2 = dx * dx + dz * dz
        if dist2 <= self.CHUNK_FRUSTUM_NEAR_RADIUS_CHUNKS * self.CHUNK_FRUSTUM_NEAR_RADIUS_CHUNKS:
            return True

        dist = math.sqrt(dist2)
        dir_x = dx / dist
        dir_z = dz / dist
        dot = dir_x * forward_x + dir_z * forward_z

        # Expand by chunk half-diagonal in chunk-space so edge chunks are not
        # clipped when only their corners are on screen.
        edge_inflate_deg = math.degrees(math.atan2(0.75, dist))
        max_angle_deg = self.CHUNK_FRUSTUM_HALF_FOV_DEGREES + self.CHUNK_FRUSTUM_MARGIN_DEGREES + edge_inflate_deg
        return dot >= math.cos(math.radians(max_angle_deg))

    def _mark_chunk_dirty(self, chunk: tuple[int, int, int]) -> None:
        if chunk in self._loaded_chunks:
            self._mesh_chunk_versions[chunk] = self._mesh_chunk_versions.get(chunk, 0) + 1
            self._dirty_chunks.add(chunk)
            # Coalesce rebuilds: if the queued job has not started yet, cancel it so we can
            # schedule a single newer build later instead of keeping stale backlog.
            future = self._mesh_futures.get(chunk)
            if future is not None and future.cancel():
                self._mesh_futures.pop(chunk, None)
                self._mesh_inflight_versions.pop(chunk, None)

    def _cancel_chunk_mesh_build(self, chunk: tuple[int, int, int]) -> None:
        future = self._mesh_futures.pop(chunk, None)
        if future is not None:
            future.cancel()
        self._mesh_inflight_versions.pop(chunk, None)

    def _chunk_mesh_snapshot(self, chunk: tuple[int, int, int]) -> tuple[set[Vec3], dict[Vec3, str]]:
        positions = set(self._chunk_blocks.get(chunk, set()))
        if not positions:
            return set(), {}

        cx, cy, cz = chunk
        x0 = cx * self.CHUNK_SIZE
        z0 = cz * self.CHUNK_SIZE
        y0 = cy * self.CHUNK_HEIGHT
        x1 = x0 + self.CHUNK_SIZE
        z1 = z0 + self.CHUNK_SIZE
        y1 = min(self.WORLD_HEIGHT, y0 + self.CHUNK_HEIGHT)

        block_sample: dict[Vec3, str] = {}
        for p in positions:
            block = self.blocks.get(p)
            if block is not None:
                block_sample[p] = block

        # Border visibility depends on adjacent chunks; only sample border neighbors
        # to keep snapshot work off the main-thread hot path.
        for x, y, z in positions:
            if x == x0:
                p = (x - 1, y, z)
                block = self.blocks.get(p)
                if block is not None:
                    block_sample[p] = block
            if x == x1 - 1:
                p = (x + 1, y, z)
                block = self.blocks.get(p)
                if block is not None:
                    block_sample[p] = block
            if z == z0:
                p = (x, y, z - 1)
                block = self.blocks.get(p)
                if block is not None:
                    block_sample[p] = block
            if z == z1 - 1:
                p = (x, y, z + 1)
                block = self.blocks.get(p)
                if block is not None:
                    block_sample[p] = block
            if y == y0:
                p = (x, y - 1, z)
                block = self.blocks.get(p)
                if block is not None:
                    block_sample[p] = block
            if y == y1 - 1:
                p = (x, y + 1, z)
                block = self.blocks.get(p)
                if block is not None:
                    block_sample[p] = block

        if self.hide_surface_layer:
            top_by_column: dict[tuple[int, int], int] = {}
            for x, y, z in positions:
                key = (x, z)
                if y > top_by_column.get(key, -10_000):
                    top_by_column[key] = y
            hidden_positions = {
                (x, y, z)
                for x, y, z in positions
                if y == top_by_column.get((x, z))
            }
            if hidden_positions:
                # Keep hidden blocks in block_sample so neighboring faces are still
                # culled as if those blocks existed; only remove them from positions
                # so they are not emitted into the mesh.
                positions = positions - hidden_positions
        return positions, block_sample

    def _on_chunk_mesh_built(self, chunk: tuple[int, int, int], version: int, future: Future[ChunkMeshData]) -> None:
        if future.cancelled():
            return
        self._mesh_completed.put((chunk[0], chunk[1], chunk[2], version))

    def _queue_chunk_mesh_build(self, chunk: tuple[int, int, int]) -> bool:
        if chunk not in self._loaded_chunks:
            return False
        if chunk not in self._visible_chunks:
            return False
        if chunk in self._mesh_futures:
            return False

        positions, block_sample = self._chunk_mesh_snapshot(chunk)
        if not positions:
            old = self._chunk_meshes.pop(chunk, None)
            if old is not None:
                old.delete()
            return True

        version = self._mesh_chunk_versions.get(chunk, 0)
        if version == 0:
            version = 1
            self._mesh_chunk_versions[chunk] = version
        future = self._mesh_executor.submit(self.renderer.build_chunk_mesh_data, positions, block_sample, SOLID_BLOCKS)
        self._mesh_futures[chunk] = future
        self._mesh_inflight_versions[chunk] = version
        self._chunk_mesh_requeue_after[chunk] = time.perf_counter() + self.CHUNK_REBUILD_REQUEUE_COOLDOWN_SECONDS
        future.add_done_callback(lambda f, c=chunk, v=version: self._on_chunk_mesh_built(c, v, f))
        return True

    def _mark_neighbors_dirty_for_border_change(self, chunk: tuple[int, int, int]) -> None:
        neighbors = [n for n in self._chunk_neighbors(chunk) if n != chunk and n in self._loaded_chunks and n in self._chunk_meshes]
        if not neighbors:
            return
        if self._center_chunk is not None:
            ccx, ccy, ccz = self._center_chunk
            neighbors.sort(
                key=lambda c: (c[0] - ccx) * (c[0] - ccx) + (c[2] - ccz) * (c[2] - ccz) + abs(c[1] - ccy)
            )
        for neighbor in neighbors[: self.NEIGHBOR_DIRTY_UPDATES_PER_EVENT]:
            self._mark_chunk_dirty(neighbor)

    def _upload_chunk_mesh(self, chunk: tuple[int, int, int], mesh_data: ChunkMeshData) -> None:
        old = self._chunk_meshes.pop(chunk, None)
        if old is not None:
            old.delete()
        if chunk not in self._loaded_chunks:
            return
        mesh = self.renderer.upload_chunk_mesh(mesh_data, self.batch, self.group)
        if mesh.parts:
            self._chunk_meshes[chunk] = mesh

    def _process_dirty_chunks(self, limit: int, budget_seconds: float) -> None:
        if limit <= 0 or not self._dirty_chunks:
            return

        available_slots = max(0, self.MAX_INFLIGHT_MESH_BUILDS - len(self._mesh_futures))
        if available_slots <= 0:
            return
        effective_limit = min(limit, available_slots)

        candidate_chunks = self._dirty_chunks.intersection(self._loaded_chunks).intersection(self._visible_chunks)
        if not candidate_chunks:
            return

        if self._center_chunk is not None:
            ccx, ccy, ccz = self._center_chunk
            chunk_dist = lambda c: (c[0] - ccx) * (c[0] - ccx) + (c[2] - ccz) * (c[2] - ccz) + abs(c[1] - ccy)
            if self.CHUNK_DIRTY_SCAN_LIMIT > 0:
                chunks = heapq.nsmallest(self.CHUNK_DIRTY_SCAN_LIMIT, candidate_chunks, key=chunk_dist)
            else:
                chunks = sorted(candidate_chunks, key=chunk_dist)
        else:
            chunks = list(candidate_chunks)
            if self.CHUNK_DIRTY_SCAN_LIMIT > 0 and len(chunks) > self.CHUNK_DIRTY_SCAN_LIMIT:
                chunks = chunks[: self.CHUNK_DIRTY_SCAN_LIMIT]

        start = time.perf_counter()
        queued = 0
        for chunk in chunks:
            now = time.perf_counter()
            if queued >= effective_limit:
                break
            if budget_seconds > 0 and (now - start) >= budget_seconds:
                break
            if now < self._chunk_mesh_requeue_after.get(chunk, 0.0):
                continue
            if self._queue_chunk_mesh_build(chunk):
                self._dirty_chunks.discard(chunk)
                queued += 1

    def _drain_completed_meshes(self, limit: int, budget_seconds: float) -> None:
        if limit <= 0:
            return
        start = time.perf_counter()
        applied = 0
        while applied < limit:
            if budget_seconds > 0 and (time.perf_counter() - start) >= budget_seconds:
                break
            try:
                cx, cy, cz, version = self._mesh_completed.get_nowait()
            except Empty:
                break

            chunk = (cx, cy, cz)
            future = self._mesh_futures.pop(chunk, None)
            inflight_version = self._mesh_inflight_versions.pop(chunk, None)
            if future is None or inflight_version is None:
                continue
            if future.cancelled():
                continue
            if chunk not in self._loaded_chunks:
                continue
            if chunk not in self._visible_chunks:
                self._dirty_chunks.add(chunk)
                continue
            latest_version = self._mesh_chunk_versions.get(chunk, 0)
            if version != inflight_version or version != latest_version:
                continue

            try:
                mesh_data = future.result()
            except Exception:
                self._mark_chunk_dirty(chunk)
                continue

            with self._profile("world.mesh.upload.single"):
                self._upload_chunk_mesh(chunk, mesh_data)
            applied += 1

    def _affected_chunks_for_position(self, position: Vec3) -> set[tuple[int, int, int]]:
        x, y, z = position
        cx, cy, cz = self.chunk_key(x, y, z)
        chunks = {(cx, cy, cz)}
        lx = x % self.CHUNK_SIZE
        ly = y % self.CHUNK_HEIGHT
        lz = z % self.CHUNK_SIZE
        if lx == 0:
            chunks.add((cx - 1, cy, cz))
        elif lx == self.CHUNK_SIZE - 1:
            chunks.add((cx + 1, cy, cz))
        if lz == 0:
            chunks.add((cx, cy, cz - 1))
        elif lz == self.CHUNK_SIZE - 1:
            chunks.add((cx, cy, cz + 1))
        if ly == 0 and cy > 0:
            chunks.add((cx, cy - 1, cz))
        elif ly == self.CHUNK_HEIGHT - 1 and cy < self.VERTICAL_CHUNKS - 1:
            chunks.add((cx, cy + 1, cz))
        return chunks

    def _rebuild_chunks_now(self, chunks: set[tuple[int, int, int]]) -> None:
        for chunk in chunks:
            if chunk not in self._loaded_chunks:
                continue
            self._cancel_chunk_mesh_build(chunk)
            positions, block_sample = self._chunk_mesh_snapshot(chunk)
            if not positions:
                old = self._chunk_meshes.pop(chunk, None)
                if old is not None:
                    old.delete()
                continue
            mesh_data = self.renderer.build_chunk_mesh_data(positions, block_sample, SOLID_BLOCKS)
            self._upload_chunk_mesh(chunk, mesh_data)
            self._dirty_chunks.discard(chunk)

    def height_at(self, x: int, z: int, biome: str, biome_noise: float | None = None) -> int:
        biome_json = self.biomes_data.get(biome, {})
        depth = float(biome_json.get("depth", 0.1))
        scale = float(biome_json.get("scale", 0.05))

        desert = self.biomes_data.get("desert")
        plains = self.biomes_data.get("plains")
        if desert is not None and plains is not None:
            # Blend biome terrain parameters around biome borders so there are
            # no abrupt terrain-step cliffs when biome IDs switch.
            if biome_noise is None:
                biome_noise = self.terrain.biome_noise_at(x, z)
            blend = self._smoothstep(self.BIOME_BLEND_NOISE_MIN, self.BIOME_BLEND_NOISE_MAX, biome_noise)
            desert_depth = float(desert.get("depth", depth))
            desert_scale = float(desert.get("scale", scale))
            plains_depth = float(plains.get("depth", depth))
            plains_scale = float(plains.get("scale", scale))
            depth = self._lerp(desert_depth, plains_depth, blend)
            scale = self._lerp(desert_scale, plains_scale, blend)

        return max(1, min(self.WORLD_HEIGHT - 1, self.terrain.height_at(x, z, depth, scale)))

    def _base_block_at(
        self,
        x: int,
        y: int,
        z: int,
        h: int,
        biome: str,
        cave_cache: dict[tuple[int, int, int, int], bool] | None = None,
    ) -> str | None:

        if y < self.BEDROCK_Y or y > h:
            return None
        if y == self.BEDROCK_Y:
            return "bedrock"
        if self._is_cave(x, y, z, h, cave_cache=cave_cache):
            return None

        if biome == "desert":
            if y == h:
                return "sand"
            if y >= h - self.TOP_LAYER_DEPTH:
                return "sandstone"
        else:
            if y == h:
                return "grass"
            if y >= h - self.TOP_LAYER_DEPTH:
                return "dirt"

        return self._underground_block_at(x, y, z, biome)

    def _is_cave(
        self,
        x: int,
        y: int,
        z: int,
        surface_y: int,
        cave_cache: dict[tuple[int, int, int, int], bool] | None = None,
    ) -> bool:
        if cave_cache is not None:
            key = (x, y, z, surface_y)
            cached = cave_cache.get(key)
            if cached is not None:
                return cached

        result = self.terrain.is_cave_at(
            x,
            y,
            z,
            surface_y,
            self.CAVE_MIN_Y,
            self.CAVE_MAX_Y,
            self.BEDROCK_Y,
            self.CAVE_BEDROCK_BUFFER,
            self.CAVE_NEAR_SURFACE_DEPTH,
            self.CAVE_TUNNEL_PRIMARY_Y_SCALE,
            self.CAVE_TUNNEL_SECONDARY_Y_SCALE,
            self.CAVE_TUNNEL_GATE_Y_SCALE,
            self.CAVE_TUNNEL_PRIMARY_BAND,
            self.CAVE_TUNNEL_SECONDARY_BAND,
            self.CAVE_COMBINED_BAND,
            self.CAVE_NEAR_SURFACE_BAND_SCALE,
            self.CAVE_TUNNEL_GATE_THRESHOLD,
            self.CAVE_WARP_STRENGTH_XZ,
            self.CAVE_WARP_STRENGTH_Y,
            self.CAVE_LEVEL_VARIATION_SCALE,
            self.CAVE_LEVEL_VARIATION_BAND_BOOST,
            self.CAVE_EDGE_FADE_Y,
            self.CAVE_MIN_CONNECTED_NEIGHBORS,
            self.CAVE_MAX_LOCAL_OPEN,
            self.CAVE_FAMILY2_WARP_STRENGTH_XZ,
            self.CAVE_FAMILY2_WARP_STRENGTH_Y,
            self.CAVE_FAMILY2_PRIMARY_BAND_SCALE,
            self.CAVE_FAMILY2_SECONDARY_BAND_SCALE,
            self.CAVE_FAMILY2_GATE_THRESHOLD,
        )
        if cave_cache is not None:
            cave_cache[key] = result
        return result

    def _underground_block_at(self, x: int, y: int, z: int, biome: str) -> str:
        noise = self.terrain.material_noise_at(x, y, z)

        if biome == "desert":
            if noise > self.DESERT_SANDSTONE_HIGH_NOISE_THRESHOLD:
                return "sandstone"
            if y < self.DESERT_SANDSTONE_LOW_Y_MAX and noise < self.DESERT_SANDSTONE_LOW_NOISE_THRESHOLD:
                return "sandstone"
            return "stone"

        if noise > self.PLAINS_DIRT_NOISE_THRESHOLD and y > self.PLAINS_DIRT_MIN_Y:
            return "dirt"
        if noise < self.PLAINS_SANDSTONE_NOISE_THRESHOLD and y < self.PLAINS_SANDSTONE_MAX_Y:
            return "sandstone"
        return "stone"

    def _generate_chunk_data(self, chunk: tuple[int, int, int]) -> list[tuple[Vec3, str]]:
        cx, cy, cz = chunk
        x0 = cx * self.CHUNK_SIZE
        z0 = cz * self.CHUNK_SIZE
        x1 = x0 + self.CHUNK_SIZE
        z1 = z0 + self.CHUNK_SIZE
        y0 = cy * self.CHUNK_HEIGHT
        y1 = min(self.WORLD_HEIGHT - 1, y0 + self.CHUNK_HEIGHT - 1)

        data: list[tuple[Vec3, str]] = []
        surface_heights = [0] * (self.CHUNK_SIZE * self.CHUNK_SIZE)
        column_meta: list[tuple[int, str]] = [(0, "plains")] * (self.CHUNK_SIZE * self.CHUNK_SIZE)

        block_map: dict[Vec3, str] = {}
        for x in range(x0, x1):
            lx = x - x0
            for z in range(z0, z1):
                lz = z - z0
                idx = lx * self.CHUNK_SIZE + lz
                biome_noise = self.terrain.biome_noise_at(x, z)
                biome_id = self.terrain.biome_from_noise(biome_noise)
                h = self.height_at(x, z, biome_id, biome_noise=biome_noise)
                surface_heights[idx] = h
                column_meta[idx] = (h, biome_id)

        cave_mask = self.terrain.generate_classic_cave_mask(
            cx,
            cy,
            cz,
            self.CHUNK_SIZE,
            self.CHUNK_HEIGHT,
            self.WORLD_HEIGHT,
            surface_heights,
            self.BEDROCK_Y,
            self.CAVE_BEDROCK_BUFFER,
            self.CAVE_MIN_Y,
            self.CAVE_MAX_Y,
            self.CAVE_CLASSIC_RANGE,
            self.CAVE_CLASSIC_SYSTEM_CHANCE,
            self.CAVE_CLASSIC_VERTICAL_SCALE,
            self.CAVE_CLASSIC_BRANCH_CHANCE,
            self.CAVE_CLASSIC_INITIAL_PITCH_RANGE,
            self.CAVE_CLASSIC_PITCH_CHANGE_STRENGTH,
            self.CAVE_CLASSIC_VERTICAL_DRIFT_STRENGTH,
        )

        for x in range(x0, x1):
            lx = x - x0
            for z in range(z0, z1):
                lz = z - z0
                idx = lx * self.CHUNK_SIZE + lz
                h, biome_id = column_meta[idx]
                col_start = max(self.BEDROCK_Y, y0)
                col_end = min(h, y1)
                if col_end < col_start:
                    continue
                for y in range(col_start, col_end + 1):
                    if y == self.BEDROCK_Y:
                        block_map[(x, y, z)] = "bedrock"
                        continue

                    ly = y - y0
                    cave_idx = (lx * self.CHUNK_HEIGHT + ly) * self.CHUNK_SIZE + lz
                    if cave_mask[cave_idx] != 0:
                        continue

                    if biome_id == "desert":
                        if y == h:
                            block = "sand"
                        elif y >= h - self.TOP_LAYER_DEPTH:
                            block = "sandstone"
                        else:
                            block = self._underground_block_at(x, y, z, biome_id)
                    else:
                        if y == h:
                            block = "grass"
                        elif y >= h - self.TOP_LAYER_DEPTH:
                            block = "dirt"
                        else:
                            block = self._underground_block_at(x, y, z, biome_id)
                    block_map[(x, y, z)] = block

        active_rules = [r for r in self._ore_rules if bool(r.get("enabled", True))]
        if active_rules:
            vein_sizes = [max(1, int(r.get("vein_size", 1))) for r in active_rules]
            veins_per_chunk = [max(0, int(r.get("veins_per_chunk", 0))) for r in active_rules]
            min_ys = [int(r.get("min_y", 0)) for r in active_rules]
            max_ys = [int(r.get("max_y", self.WORLD_HEIGHT - 1)) for r in active_rules]
            discard = [float(r.get("discard_on_air_chance", 0.0)) for r in active_rules]
            ore_mask = self.terrain.generate_ore_mask(
                cx,
                cy,
                cz,
                self.CHUNK_SIZE,
                self.CHUNK_HEIGHT,
                self.WORLD_HEIGHT,
                vein_sizes,
                veins_per_chunk,
                min_ys,
                max_ys,
                discard,
            )
            replace_sets = [set(r.get("replace", ("stone",))) for r in active_rules]
            ore_blocks = [str(r.get("block", "")).strip().lower() for r in active_rules]
            for x in range(x0, x1):
                lx = x - x0
                for z in range(z0, z1):
                    lz = z - z0
                    for y in range(y0, y1 + 1):
                        ly = y - y0
                        idx = (lx * self.CHUNK_HEIGHT + ly) * self.CHUNK_SIZE + lz
                        rule_idx = ore_mask[idx]
                        if rule_idx < 0 or rule_idx >= len(ore_blocks):
                            continue
                        pos = (x, y, z)
                        current = block_map.get(pos)
                        if current is None:
                            continue
                        if current in replace_sets[rule_idx]:
                            block_map[pos] = ore_blocks[rule_idx]

        data.extend(block_map.items())
        return data

    def biome_at(self, x: int, z: int):
        return self.terrain.biome_at(x,z)

    def _on_chunk_generated(self, chunk: tuple[int, int, int], future: Future[list[tuple[Vec3, str]]]) -> None:
        if future.cancelled():
            return
        self._completed_chunks.put(chunk)

    def _queue_chunk_request(self, chunk: tuple[int, int, int]) -> None:
        if chunk in self._loaded_chunks or chunk in self._requested_chunks:
            return
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            self._chunk_cache.move_to_end(chunk)
            self._apply_chunk_data(chunk, cached)
            return
        future = self._executor.submit(self._generate_chunk_data, chunk)
        self._chunk_futures[chunk] = future
        self._requested_chunks.add(chunk)
        future.add_done_callback(lambda f, c=chunk: self._on_chunk_generated(c, f))

    def _apply_chunk_data(self, chunk: tuple[int, int, int], base_data: list[tuple[Vec3, str]]) -> None:
        if chunk in self._loaded_chunks:
            return

        block_map: dict[Vec3, str] = dict(base_data)
        for pos in self._chunk_modified_positions.get(chunk, set()):
            modified = self._modified_blocks.get(pos, self._NO_OVERRIDE)
            if modified is self._NO_OVERRIDE:
                continue
            if modified is None:
                block_map.pop(pos, None)
            else:
                block_map[pos] = modified

        positions = set(block_map.keys())
        for pos, block in block_map.items():
            self.blocks[pos] = block

        self._chunk_blocks[chunk] = positions
        self._loaded_chunks.add(chunk)
        self._mark_chunk_dirty(chunk)
        self._mark_neighbors_dirty_for_border_change(chunk)

    def _remove_chunk(self, chunk: tuple[int, int, int]) -> None:
        if chunk not in self._loaded_chunks:
            return

        self._cancel_chunk_mesh_build(chunk)
        positions = self._chunk_blocks.pop(chunk, set())
        cached_data: list[tuple[Vec3, str]] = []
        for pos in positions:
            block = self.blocks.pop(pos, None)
            if block is not None:
                cached_data.append((pos, block))

        mesh = self._chunk_meshes.pop(chunk, None)
        if mesh is not None:
            mesh.delete()

        self._loaded_chunks.remove(chunk)
        self._dirty_chunks.discard(chunk)
        self._mesh_chunk_versions.pop(chunk, None)
        self._mesh_inflight_versions.pop(chunk, None)
        self._chunk_mesh_requeue_after.pop(chunk, None)
        if cached_data:
            self._chunk_cache[chunk] = cached_data
            self._chunk_cache.move_to_end(chunk)
            while len(self._chunk_cache) > self.CHUNK_CACHE_MAX_ENTRIES:
                self._chunk_cache.popitem(last=False)
        self._mark_neighbors_dirty_for_border_change(chunk)

    def _unload_chunks_with_budget(
        self,
        candidates: list[tuple[int, int, int]],
        limit: int,
        budget_seconds: float,
    ) -> int:
        if limit <= 0 or not candidates:
            return 0
        start = time.perf_counter()
        unloaded = 0
        for chunk in candidates:
            if unloaded >= limit:
                break
            if budget_seconds > 0 and (time.perf_counter() - start) >= budget_seconds:
                break
            if chunk not in self._loaded_chunks:
                continue
            self._remove_chunk(chunk)
            unloaded += 1
        return unloaded

    def _cancel_queued_chunk(self, chunk: tuple[int, int, int]) -> None:
        future = self._chunk_futures.pop(chunk, None)
        if future is not None:
            future.cancel()
        self._requested_chunks.discard(chunk)

    def _drain_completed_chunks(self, budget_seconds: float) -> None:
        start = time.perf_counter()
        applied = 0
        while applied < self.CHUNKS_APPLIED_PER_UPDATE:
            if budget_seconds > 0 and (time.perf_counter() - start) >= budget_seconds:
                break
            try:
                chunk = self._completed_chunks.get_nowait()
            except Empty:
                break

            future = self._chunk_futures.pop(chunk, None)
            self._requested_chunks.discard(chunk)
            if future is None:
                continue
            if chunk not in self._desired_chunks or chunk in self._loaded_chunks:
                continue
            if future.cancelled():
                continue

            try:
                data = future.result()
            except Exception:
                continue

            with self._profile("world.chunk.apply.single"):
                self._apply_chunk_data(chunk, data)
            applied += 1

    def update_visible_chunks(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float] | None = None,
        allow_optional_work: bool = True,
    ) -> None:
        px, py, pz = position
        pcx, pcz = self.chunk_coords(px, pz)
        pcy = self.vertical_chunk(py)
        vertical_layers = self._target_vertical_chunks(py)
        # Direction-agnostic chunk streaming: keep the full visible square loaded
        # regardless of where the player is facing unless explicitly enabled.
        use_frustum = self.USE_CHUNK_FRUSTUM_CULLING
        if use_frustum and rotation is not None:
            yaw, _ = rotation
            forward_x = math.cos(math.radians(yaw - 90.0))
            forward_z = math.sin(math.radians(yaw - 90.0))
        else:
            forward_x = 0.0
            forward_z = 0.0

        visible2d: set[tuple[int, int]] = set()
        with self._profile("world.visible.compute_desired"):
            for dcx in range(-self.VISIBLE_RADIUS_CHUNKS, self.VISIBLE_RADIUS_CHUNKS + 1):
                for dcz in range(-self.VISIBLE_RADIUS_CHUNKS, self.VISIBLE_RADIUS_CHUNKS + 1):
                    chunk2d = (pcx + dcx, pcz + dcz)
                    if use_frustum and not self._chunk_in_frustum(chunk2d, (pcx, pcz), forward_x, forward_z):
                        continue
                    visible2d.add(chunk2d)

        visible: set[tuple[int, int, int]] = set()
        for cx, cz in visible2d:
            for cy in vertical_layers:
                visible.add((cx, cy, cz))

        desired2d: set[tuple[int, int]] = set()
        for dcx in range(-self.ACTIVE_RADIUS_CHUNKS, self.ACTIVE_RADIUS_CHUNKS + 1):
            for dcz in range(-self.ACTIVE_RADIUS_CHUNKS, self.ACTIVE_RADIUS_CHUNKS + 1):
                desired2d.add((pcx + dcx, pcz + dcz))
        desired: set[tuple[int, int, int]] = set()
        for cx, cz in desired2d:
            for cy in vertical_layers:
                desired.add((cx, cy, cz))
        self._desired_chunks = desired

        entered_visible = visible - self._visible_chunks
        exited_visible = self._visible_chunks - visible
        self._visible_chunks = visible

        for chunk in entered_visible:
            if chunk in self._loaded_chunks:
                self._mark_chunk_dirty(chunk)

        # Keep meshes resident even when chunks leave the current view direction.
        # This avoids "offloading" when rotating away and back.
        _ = exited_visible
        keep_loaded: set[tuple[int, int, int]] = set()
        unload_radius = max(self.UNLOAD_RADIUS_CHUNKS, self.ACTIVE_RADIUS_CHUNKS)
        for dcx in range(-unload_radius, unload_radius + 1):
            for dcz in range(-unload_radius, unload_radius + 1):
                for cy in vertical_layers:
                    keep_loaded.add((pcx + dcx, cy, pcz + dcz))

        with self._profile("world.visible.cancel_outside_requests"):
            for chunk in sorted(self._requested_chunks - desired):
                self._cancel_queued_chunk(chunk)

        with self._profile("world.visible.queue_requests"):
            pending_loads = desired - self._loaded_chunks - self._requested_chunks
            if len(self._loaded_chunks) >= self.MAX_LOADED_CHUNKS:
                # Near/over cap: only pull chunks that are immediately visible to avoid
                # active-ring churn against the cap.
                pending_loads = pending_loads.intersection(self._visible_chunks)
            available_slots = max(0, self.MAX_INFLIGHT_REQUESTS - len(self._requested_chunks))
            to_request = min(self.CHUNKS_REQUESTED_PER_UPDATE, available_slots)
            nearest_pending = heapq.nsmallest(
                to_request,
                pending_loads,
                key=lambda c: (c[0] - pcx) * (c[0] - pcx) + (c[2] - pcz) * (c[2] - pcz) + abs(c[1] - pcy),
            )
            for chunk in nearest_pending:
                self._queue_chunk_request(chunk)

        with self._profile("world.visible.apply_completed_chunks"):
            self._drain_completed_chunks(self.CHUNK_APPLY_BUDGET_SECONDS)

        with self._profile("world.visible.unload_chunks"):
            unloaded = 0
            if allow_optional_work and len(self._loaded_chunks) > self.MIN_LOADED_CHUNKS:
                unload_candidates = heapq.nlargest(
                    self.CHUNKS_UNLOADED_PER_UPDATE,
                    self._loaded_chunks - keep_loaded,
                    key=lambda c: (c[0] - pcx) * (c[0] - pcx) + (c[2] - pcz) * (c[2] - pcz) + abs(c[1] - pcy),
                )
                unloaded = self._unload_chunks_with_budget(
                    unload_candidates,
                    self.CHUNKS_UNLOADED_PER_UPDATE,
                    self.CHUNK_UNLOAD_BUDGET_SECONDS,
                )

            if len(self._loaded_chunks) > self.MAX_LOADED_CHUNKS:
                # Keep visible chunks stable; evict farthest non-visible chunks first.
                to_remove_for_cap = len(self._loaded_chunks) - self.MAX_LOADED_CHUNKS
                cap_candidates = heapq.nlargest(
                    to_remove_for_cap,
                    self._loaded_chunks - self._visible_chunks,
                    key=lambda c: (c[0] - pcx) * (c[0] - pcx) + (c[2] - pcz) * (c[2] - pcz) + abs(c[1] - pcy),
                )
                # Spread hard-cap eviction over multiple frames to avoid long stalls.
                self._unload_chunks_with_budget(
                    cap_candidates,
                    min(to_remove_for_cap, self.MAX_CAP_UNLOADS_PER_UPDATE),
                    self.CAP_UNLOAD_BUDGET_SECONDS,
                )

        self._center_chunk = (pcx, pcy, pcz)
        with self._profile("world.visible.upload_completed_meshes"):
            self._drain_completed_meshes(self.CHUNK_MESH_UPLOADS_PER_UPDATE, self.CHUNK_MESH_UPLOAD_BUDGET_SECONDS)
        if allow_optional_work:
            with self._profile("world.visible.queue_mesh_rebuilds"):
                self._process_dirty_chunks(self.CHUNK_REBUILDS_PER_UPDATE, self.CHUNK_REBUILD_BUDGET_SECONDS)

    def prime_chunks(self, position: tuple[float, float, float], radius_chunks: int = 1) -> None:
        px, py, pz = position
        pcx, pcz = self.chunk_coords(px, pz)
        vertical_layers = self._target_vertical_chunks(py)
        for dcx in range(-radius_chunks, radius_chunks + 1):
            for dcz in range(-radius_chunks, radius_chunks + 1):
                for cy in vertical_layers:
                    chunk = (pcx + dcx, cy, pcz + dcz)
                    data = self._generate_chunk_data(chunk)
                    self._apply_chunk_data(chunk, data)
        self._process_dirty_chunks(len(self._dirty_chunks), 0.0)
        self._drain_completed_meshes(len(self._loaded_chunks), 0.0)

    def ensure_chunk_loaded(self, position: tuple[float, float, float]) -> bool:
        px, py, pz = position
        chunk = self.chunk_key(px, py, pz)
        if chunk in self._loaded_chunks:
            return True

        future = self._chunk_futures.get(chunk)
        if future is not None:
            return False

        self._queue_chunk_request(chunk)
        return False

    def loading_chunks_in_radius(self, position: tuple[float, float, float], radius: int) -> set[tuple[int, int, int]]:
        px, py, pz = position
        pcx, pcz = self.chunk_coords(px, pz)
        layers = self._target_vertical_chunks(py)
        chunks: set[tuple[int, int, int]] = set()
        for dcx in range(-radius, radius + 1):
            for dcz in range(-radius, radius + 1):
                for cy in layers:
                    chunks.add((pcx + dcx, cy, pcz + dcz))
        return chunks

    def diagnostics_snapshot(self) -> dict[str, int]:
        return {
            "loaded_chunks": len(self._loaded_chunks),
            "visible_chunks": len(self._visible_chunks),
            "active_chunks": len(self._desired_chunks),
            "requested_chunks": len(self._requested_chunks),
            "dirty_chunks": len(self._dirty_chunks),
            "chunk_futures": len(self._chunk_futures),
            "mesh_futures": len(self._mesh_futures),
            "mesh_ready_queue": self._mesh_completed.qsize(),
            "chunk_ready_queue": self._completed_chunks.qsize(),
            "chunk_meshes": len(self._chunk_meshes),
        }

    def is_over_loaded_cap(self) -> bool:
        return len(self._loaded_chunks) > self.MAX_LOADED_CHUNKS

    def are_chunks_rendered(self, chunks: set[tuple[int, int, int]]) -> bool:
        for chunk in chunks:
            if chunk not in self._loaded_chunks:
                return False
            if chunk not in self._chunk_meshes:
                # Empty chunks intentionally have no mesh; they are still ready.
                if self._chunk_blocks.get(chunk):
                    return False
            if chunk in self._dirty_chunks:
                return False
            if chunk in self._mesh_futures:
                return False
        return True

    def are_chunks_generated(self, chunks: set[tuple[int, int, int]]) -> bool:
        for chunk in chunks:
            if chunk not in self._loaded_chunks:
                return False
            if chunk in self._requested_chunks:
                return False
            if chunk in self._chunk_futures:
                return False
        return True

    def add_block(self, position: Vec3, block: str, immediate: bool = True) -> None:
        x, y, z = position
        if y < self.BEDROCK_Y or y >= self.WORLD_HEIGHT:
            return
        self.blocks[position] = block
        self._modified_blocks[position] = block
        chunk = self.chunk_key(x, y, z)
        self._chunk_modified_positions.setdefault(chunk, set()).add(position)
        self._chunk_blocks.setdefault(chunk, set()).add(position)
        affected_chunks = self._affected_chunks_for_position(position)
        if immediate:
            self._rebuild_chunks_now(affected_chunks)
        else:
            for c in affected_chunks:
                self._mark_chunk_dirty(c)

    def remove_block(self, position: Vec3, immediate: bool = True) -> str | None:
        old = self.blocks.get(position)
        if old is None:
            return None

        definition = get_block_definition(old)
        if definition is not None and not definition.breakable:
            return None

        self.blocks.pop(position, None)
        self._modified_blocks[position] = None
        x, _, z = position
        chunk = self.chunk_key(x, position[1], z)
        self._chunk_modified_positions.setdefault(chunk, set()).add(position)
        chunk_positions = self._chunk_blocks.get(chunk)
        if chunk_positions is not None:
            chunk_positions.discard(position)

        affected_chunks = self._affected_chunks_for_position(position)
        if old and immediate:
            self._rebuild_chunks_now(affected_chunks)
        else:
            for c in affected_chunks:
                self._mark_chunk_dirty(c)
        return old

    def is_exposed(self, position: Vec3) -> bool:
        return bool(self.visible_faces(position))

    def visible_faces(self, position: Vec3) -> list[int]:
        x, y, z = position
        visible: list[int] = []
        for face_index, (dx, dy, dz) in enumerate(self._FACE_NEIGHBORS):
            n = (x + dx, y + dy, z + dz)
            if self.blocks.get(n) not in SOLID_BLOCKS:
                visible.append(face_index)
        return visible

    def refresh_neighbors(self, position: Vec3) -> None:
        x, y, z = position
        for p in ((x, y, z), (x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)):
            for chunk in self._affected_chunks_for_position(p):
                self._mark_chunk_dirty(chunk)
        self._process_dirty_chunks(len(self._dirty_chunks), 0.0)
        self._drain_completed_meshes(len(self._loaded_chunks), 0.0)

    def rebuild_visible(self) -> None:
        for mesh in list(self._chunk_meshes.values()):
            mesh.delete()
        self._chunk_meshes.clear()
        for chunk in self._loaded_chunks:
            self._mark_chunk_dirty(chunk)
        self._process_dirty_chunks(len(self._dirty_chunks), 0.0)
        self._drain_completed_meshes(len(self._loaded_chunks), 0.0)

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

    def shutdown(self) -> None:
        for mesh in list(self._chunk_meshes.values()):
            mesh.delete()
        self._chunk_meshes.clear()
        for chunk in list(self._requested_chunks):
            self._cancel_queued_chunk(chunk)
        for chunk in list(self._mesh_futures.keys()):
            self._cancel_chunk_mesh_build(chunk)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._mesh_executor.shutdown(wait=False, cancel_futures=True)
