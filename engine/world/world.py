import math
import time
import heapq
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


class Deletable(Protocol):
    def delete(self) -> None: ...


class World:
    CHUNK_SIZE = SECTOR_SIZE
    WORLD_HEIGHT = 256
    BEDROCK_Y = 0
    VISIBLE_RADIUS_CHUNKS = 4
    ACTIVE_RADIUS_CHUNKS = 5
    UNLOAD_RADIUS_CHUNKS = 6
    MAX_LOADED_CHUNKS = 140
    CHUNK_CACHE_MAX_ENTRIES = 256
    CHUNKS_REQUESTED_PER_UPDATE = 2
    CHUNKS_APPLIED_PER_UPDATE = 1
    CHUNK_APPLY_BUDGET_SECONDS = 0.0015
    CHUNKS_UNLOADED_PER_UPDATE = 1
    CHUNK_UNLOAD_BUDGET_SECONDS = 0.0015
    MAX_CAP_UNLOADS_PER_UPDATE = 2
    CAP_UNLOAD_BUDGET_SECONDS = 0.0030
    CHUNK_REBUILDS_PER_UPDATE = 1
    CHUNK_REBUILD_BUDGET_SECONDS = 0.0008
    CHUNK_DIRTY_SCAN_LIMIT = 12
    CHUNK_REBUILD_REQUEUE_COOLDOWN_SECONDS = 0.18
    # Keep chunk borders consistent as chunks load/unload; stale seam meshes can
    # look like overlapping geometry if neighbors are never dirtied.
    NEIGHBOR_DIRTY_UPDATES_PER_EVENT = 4
    MAX_INFLIGHT_MESH_BUILDS = 6
    CHUNK_MESH_UPLOADS_PER_UPDATE = 1
    CHUNK_MESH_UPLOAD_BUDGET_SECONDS = 0.0015
    MIN_LOADED_CHUNKS = 25
    MAX_INFLIGHT_REQUESTS = 10
    CHUNK_WORKERS = 2
    CHUNK_MESH_WORKERS = 2
    CHUNK_FRUSTUM_HALF_FOV_DEGREES = 65.0
    CHUNK_FRUSTUM_MARGIN_DEGREES = 20.0
    CHUNK_FRUSTUM_NEAR_RADIUS_CHUNKS = 1

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
        self._chunk_blocks: dict[tuple[int, int], set[Vec3]] = {}
        self._chunk_meshes: dict[tuple[int, int], Deletable] = {}
        self._dirty_chunks: set[tuple[int, int]] = set()
        self._loaded_chunks: set[tuple[int, int]] = set()
        self._modified_blocks: dict[Vec3, str | None] = {}
        self._chunk_modified_positions: dict[tuple[int, int], set[Vec3]] = {}
        self._center_chunk: tuple[int, int] | None = None
        self._desired_chunks: set[tuple[int, int]] = set()
        self._visible_chunks: set[tuple[int, int]] = set()
        self._requested_chunks: set[tuple[int, int]] = set()
        self._chunk_futures: dict[tuple[int, int], Future[list[tuple[Vec3, str]]]] = {}
        self._completed_chunks: Queue[tuple[int, int]] = Queue()
        self._executor = ThreadPoolExecutor(max_workers=self.CHUNK_WORKERS, thread_name_prefix="chunkgen")

        self._mesh_executor = ThreadPoolExecutor(max_workers=self.CHUNK_MESH_WORKERS, thread_name_prefix="meshbuild")
        self._mesh_futures: dict[tuple[int, int], Future[ChunkMeshData]] = {}
        self._mesh_completed: Queue[tuple[int, int, int]] = Queue()
        self._mesh_chunk_versions: dict[tuple[int, int], int] = {}
        self._mesh_inflight_versions: dict[tuple[int, int], int] = {}
        self._chunk_mesh_requeue_after: dict[tuple[int, int], float] = {}
        self._chunk_cache: OrderedDict[tuple[int, int], list[tuple[Vec3, str]]] = OrderedDict()

        self.batch = pyglet.graphics.Batch()
        self.terrain = TerrainGenerator(seed, flat_height=flat_height)
        self.renderer = BlockRenderer(seed, use_texture_array=use_texture_array)
        self.group = pyglet.graphics.ShaderGroup(program=self.renderer.shader)

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

    def _chunk_neighbors(self, chunk: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        cx, cz = chunk
        return (cx - 1, cz), (cx + 1, cz), (cx, cz - 1), (cx, cz + 1)

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

    def _mark_chunk_dirty(self, chunk: tuple[int, int]) -> None:
        if chunk in self._loaded_chunks:
            self._mesh_chunk_versions[chunk] = self._mesh_chunk_versions.get(chunk, 0) + 1
            self._dirty_chunks.add(chunk)
            # Coalesce rebuilds: if the queued job has not started yet, cancel it so we can
            # schedule a single newer build later instead of keeping stale backlog.
            future = self._mesh_futures.get(chunk)
            if future is not None and future.cancel():
                self._mesh_futures.pop(chunk, None)
                self._mesh_inflight_versions.pop(chunk, None)

    def _cancel_chunk_mesh_build(self, chunk: tuple[int, int]) -> None:
        future = self._mesh_futures.pop(chunk, None)
        if future is not None:
            future.cancel()
        self._mesh_inflight_versions.pop(chunk, None)

    def _chunk_mesh_snapshot(self, chunk: tuple[int, int]) -> tuple[set[Vec3], dict[Vec3, str]]:
        positions = set(self._chunk_blocks.get(chunk, set()))
        if not positions:
            return set(), {}

        cx, cz = chunk
        x0 = cx * self.CHUNK_SIZE
        z0 = cz * self.CHUNK_SIZE
        x1 = x0 + self.CHUNK_SIZE
        z1 = z0 + self.CHUNK_SIZE

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
        return positions, block_sample

    def _on_chunk_mesh_built(self, chunk: tuple[int, int], version: int, future: Future[ChunkMeshData]) -> None:
        if future.cancelled():
            return
        self._mesh_completed.put((chunk[0], chunk[1], version))

    def _queue_chunk_mesh_build(self, chunk: tuple[int, int]) -> bool:
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

    def _mark_neighbors_dirty_for_border_change(self, chunk: tuple[int, int]) -> None:
        neighbors = [n for n in self._chunk_neighbors(chunk) if n in self._loaded_chunks and n in self._chunk_meshes]
        if not neighbors:
            return
        if self._center_chunk is not None:
            ccx, ccz = self._center_chunk
            neighbors.sort(key=lambda c: (c[0] - ccx) * (c[0] - ccx) + (c[1] - ccz) * (c[1] - ccz))
        for neighbor in neighbors[: self.NEIGHBOR_DIRTY_UPDATES_PER_EVENT]:
            self._mark_chunk_dirty(neighbor)

    def _upload_chunk_mesh(self, chunk: tuple[int, int], mesh_data: ChunkMeshData) -> None:
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
            ccx, ccz = self._center_chunk
            chunk_dist = lambda c: (c[0] - ccx) * (c[0] - ccx) + (c[1] - ccz) * (c[1] - ccz)
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
                cx, cz, version = self._mesh_completed.get_nowait()
            except Empty:
                break

            chunk = (cx, cz)
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

    def _affected_chunks_for_position(self, position: Vec3) -> set[tuple[int, int]]:
        x, _, z = position
        cx, cz = self.chunk_coords(x, z)
        chunks = {(cx, cz)}
        lx = x % self.CHUNK_SIZE
        lz = z % self.CHUNK_SIZE
        if lx == 0:
            chunks.add((cx - 1, cz))
        elif lx == self.CHUNK_SIZE - 1:
            chunks.add((cx + 1, cz))
        if lz == 0:
            chunks.add((cx, cz - 1))
        elif lz == self.CHUNK_SIZE - 1:
            chunks.add((cx, cz + 1))
        return chunks

    def _rebuild_chunks_now(self, chunks: set[tuple[int, int]]) -> None:
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

    def height_at(self, x: int, z: int) -> int:
        return max(1, min(self.WORLD_HEIGHT - 1, self.terrain.height_at(x, z)))

    def _base_block_at(self, x: int, y: int, z: int, h: int) -> str | None:
        if y < self.BEDROCK_Y or y > h:
            return None
        if y == self.BEDROCK_Y:
            return "bedrock"
        if y == h:
            return "grass"
        if y >= h - 2:
            return "dirt"
        return "stone"

    def _generate_chunk_data(self, chunk: tuple[int, int]) -> list[tuple[Vec3, str]]:
        cx, cz = chunk
        x0 = cx * self.CHUNK_SIZE
        z0 = cz * self.CHUNK_SIZE
        x1 = x0 + self.CHUNK_SIZE
        z1 = z0 + self.CHUNK_SIZE

        data: list[tuple[Vec3, str]] = []
        for x in range(x0, x1):
            for z in range(z0, z1):
                h = self.height_at(x, z)
                for y in range(self.BEDROCK_Y, h + 1):
                    block = self._base_block_at(x, y, z, h)
                    if block is not None:
                        data.append(((x, y, z), block))
        return data

    def _on_chunk_generated(self, chunk: tuple[int, int], future: Future[list[tuple[Vec3, str]]]) -> None:
        if future.cancelled():
            return
        self._completed_chunks.put(chunk)

    def _queue_chunk_request(self, chunk: tuple[int, int]) -> None:
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

    def _apply_chunk_data(self, chunk: tuple[int, int], base_data: list[tuple[Vec3, str]]) -> None:
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

    def _remove_chunk(self, chunk: tuple[int, int]) -> None:
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
        candidates: list[tuple[int, int]],
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

    def _cancel_queued_chunk(self, chunk: tuple[int, int]) -> None:
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
        px, _, pz = position
        pcx, pcz = self.chunk_coords(px, pz)
        use_frustum = rotation is not None
        if use_frustum:
            yaw, _ = rotation
            forward_x = math.cos(math.radians(yaw - 90.0))
            forward_z = math.sin(math.radians(yaw - 90.0))
        else:
            forward_x = 0.0
            forward_z = 0.0

        visible: set[tuple[int, int]] = set()
        with self._profile("world.visible.compute_desired"):
            for dcx in range(-self.VISIBLE_RADIUS_CHUNKS, self.VISIBLE_RADIUS_CHUNKS + 1):
                for dcz in range(-self.VISIBLE_RADIUS_CHUNKS, self.VISIBLE_RADIUS_CHUNKS + 1):
                    chunk = (pcx + dcx, pcz + dcz)
                    if use_frustum and not self._chunk_in_frustum(chunk, (pcx, pcz), forward_x, forward_z):
                        continue
                    visible.add(chunk)

        desired: set[tuple[int, int]] = set()
        for dcx in range(-self.ACTIVE_RADIUS_CHUNKS, self.ACTIVE_RADIUS_CHUNKS + 1):
            for dcz in range(-self.ACTIVE_RADIUS_CHUNKS, self.ACTIVE_RADIUS_CHUNKS + 1):
                desired.add((pcx + dcx, pcz + dcz))
        self._desired_chunks = desired

        entered_visible = visible - self._visible_chunks
        exited_visible = self._visible_chunks - visible
        self._visible_chunks = visible

        for chunk in entered_visible:
            if chunk in self._loaded_chunks:
                self._mark_chunk_dirty(chunk)

        for chunk in exited_visible:
            self._cancel_chunk_mesh_build(chunk)
            mesh = self._chunk_meshes.pop(chunk, None)
            if mesh is not None:
                mesh.delete()
        keep_loaded: set[tuple[int, int]] = set()
        unload_radius = max(self.UNLOAD_RADIUS_CHUNKS, self.ACTIVE_RADIUS_CHUNKS)
        for dcx in range(-unload_radius, unload_radius + 1):
            for dcz in range(-unload_radius, unload_radius + 1):
                keep_loaded.add((pcx + dcx, pcz + dcz))

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
                key=lambda c: (c[0] - pcx) * (c[0] - pcx) + (c[1] - pcz) * (c[1] - pcz),
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
                    key=lambda c: (c[0] - pcx) * (c[0] - pcx) + (c[1] - pcz) * (c[1] - pcz),
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
                    key=lambda c: (c[0] - pcx) * (c[0] - pcx) + (c[1] - pcz) * (c[1] - pcz),
                )
                # Spread hard-cap eviction over multiple frames to avoid long stalls.
                self._unload_chunks_with_budget(
                    cap_candidates,
                    min(to_remove_for_cap, self.MAX_CAP_UNLOADS_PER_UPDATE),
                    self.CAP_UNLOAD_BUDGET_SECONDS,
                )

        self._center_chunk = (pcx, pcz)
        with self._profile("world.visible.upload_completed_meshes"):
            self._drain_completed_meshes(self.CHUNK_MESH_UPLOADS_PER_UPDATE, self.CHUNK_MESH_UPLOAD_BUDGET_SECONDS)
        if allow_optional_work:
            with self._profile("world.visible.queue_mesh_rebuilds"):
                self._process_dirty_chunks(self.CHUNK_REBUILDS_PER_UPDATE, self.CHUNK_REBUILD_BUDGET_SECONDS)

    def prime_chunks(self, position: tuple[float, float, float], radius_chunks: int = 1) -> None:
        px, _, pz = position
        pcx, pcz = self.chunk_coords(px, pz)
        for dcx in range(-radius_chunks, radius_chunks + 1):
            for dcz in range(-radius_chunks, radius_chunks + 1):
                chunk = (pcx + dcx, pcz + dcz)
                data = self._generate_chunk_data(chunk)
                self._apply_chunk_data(chunk, data)
        self._process_dirty_chunks(len(self._dirty_chunks), 0.0)
        self._drain_completed_meshes(len(self._loaded_chunks), 0.0)

    def ensure_chunk_loaded(self, position: tuple[float, float, float]) -> bool:
        px, _, pz = position
        chunk = self.chunk_coords(px, pz)
        if chunk in self._loaded_chunks:
            return True

        future = self._chunk_futures.get(chunk)
        if future is not None:
            return False

        self._queue_chunk_request(chunk)
        return False

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

    def are_chunks_rendered(self, chunks: set[tuple[int, int]]) -> bool:
        for chunk in chunks:
            if chunk not in self._loaded_chunks:
                return False
            if chunk not in self._chunk_meshes:
                return False
            if chunk in self._dirty_chunks:
                return False
            if chunk in self._mesh_futures:
                return False
        return True

    def add_block(self, position: Vec3, block: str, immediate: bool = True) -> None:
        x, y, z = position
        if y < self.BEDROCK_Y or y >= self.WORLD_HEIGHT:
            return
        self.blocks[position] = block
        self._modified_blocks[position] = block
        chunk = self.chunk_coords(x, z)
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
        chunk = self.chunk_coords(x, z)
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
