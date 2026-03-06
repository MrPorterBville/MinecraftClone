import math
import time

import pyglet
from pyglet import gl
from pyglet.window import key, mouse

from engine.blocks import SOLID_BLOCKS, get_block_texture_for_face
from engine.blocks.registry import TEXTURES_DIR
from engine.constants import GRAVITY, JUMP_SPEED, PLAYER_HEIGHT, TERMINAL_VELOCITY, TICKS_PER_SECOND, WALK_SPEED
from engine.debug.profiler import RuntimeProfiler
from engine.gameplay.inventory import Inventory
from engine.graphics.rendering import set_2d, set_3d
from engine.world.world import World


class GameWindow(pyglet.window.Window):
    CHUNK_STREAM_UPDATE_INTERVAL_SECONDS = 1.0 / 20.0
    CHUNK_STREAM_UPDATE_INTERVAL_OVER_CAP_SECONDS = 1.0 / 8.0
    FRAME_CATCHUP_THRESHOLD_MS = 20.0
    LOADING_REQUIRED_RADIUS_CHUNKS = 1

    def __init__(self, seed: int = 90125, use_texture_array: bool = False):
        super().__init__(width=1280, height=720, caption="Python Minecraft Clone", resizable=True)
        self.exclusive = False
        self.profiler = RuntimeProfiler(enabled=True, slow_frame_ms=25.0, max_slow_frames=500)
        self.profiler.clear_previous_reports("profiling")
        self.world = World(seed=seed, flat_height=64, profiler=self.profiler, use_texture_array=use_texture_array)

        self.inventory = Inventory()
        self.inventory_open = False
        self._held_inventory_slot: tuple[str, int] | None = None
        self._mouse_x = self.width // 2
        self._mouse_y = self.height // 2

        startBiome = self.world.biome_at(0, 0)
        spawn_y = float(self.world.height_at(0, 0, startBiome) + 2)
        self.position = (0.0, spawn_y, 0.0)
        self.world.prime_chunks(self.position, radius_chunks=self.LOADING_REQUIRED_RADIUS_CHUNKS)
        self.world.update_visible_chunks(self.position)
        self.rotation = (0.0, -25.0)
        self.dy = 0.0
        self._chunk_stream_timer = 0.0
        self._last_stream_chunk = self.world.chunk_coords(self.position[0], self.position[2])
        self._pending_view_chunk_refresh = False
        self._last_update_ms = 0.0

        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SECOND)

        self.ui_batch = pyglet.graphics.Batch()
        self.label = pyglet.text.Label(
            "",
            x=10,
            y=self.height - 10,
            anchor_x="left",
            anchor_y="top",
            color=(255, 255, 255, 255),
            batch=self.ui_batch,
        )
        self.crosshair = pyglet.shapes.Line(
            self.width // 2 - 8,
            self.height // 2,
            self.width // 2 + 8,
            self.height // 2,
            color=(255, 255, 255),
            batch=self.ui_batch,
        )
        self.crosshair2 = pyglet.shapes.Line(
            self.width // 2,
            self.height // 2 - 8,
            self.width // 2,
            self.height // 2 + 8,
            color=(255, 255, 255),
            batch=self.ui_batch,
        )

        self._hotbar_slot_size = 48
        self._hotbar_slot_gap = 6
        self._hotbar_y = 24
        self._hotbar_backgrounds: list[pyglet.shapes.Rectangle] = []
        self._hotbar_borders: list[pyglet.shapes.BorderedRectangle] = []
        self._hotbar_icon_sprites: list[pyglet.sprite.Sprite | None] = []
        self._hotbar_icon_blocks: list[str | None] = []
        self._hotbar_texture_cache: dict[str, pyglet.image.AbstractImage] = {}
        self._hotbar_counts: list[pyglet.text.Label] = []
        self._build_hotbar_ui()

        self._inventory_batch = pyglet.graphics.Batch()
        self._inv_slot_size = 52
        self._inv_slot_gap = 8
        self._inv_hotbar_gap = 22
        self._inv_panel_padding = 20
        self._inventory_panel = pyglet.shapes.Rectangle(0, 0, 10, 10, color=(20, 20, 20), batch=self._inventory_batch)
        self._inventory_panel.opacity = 220
        self._inventory_panel_border = pyglet.shapes.BorderedRectangle(
            0,
            0,
            10,
            10,
            border=4,
            color=(20, 20, 20),
            border_color=(140, 140, 140),
            batch=self._inventory_batch,
        )
        self._inventory_panel_border.opacity = 255
        self._inventory_backgrounds: list[pyglet.shapes.Rectangle] = []
        self._inventory_borders: list[pyglet.shapes.BorderedRectangle] = []
        self._inventory_icon_sprites: list[pyglet.sprite.Sprite | None] = []
        self._inventory_icon_blocks: list[str | None] = []
        self._inventory_counts: list[pyglet.text.Label] = []
        self._inventory_title = pyglet.text.Label(
            "Inventory",
            x=0,
            y=0,
            anchor_x="left",
            anchor_y="bottom",
            color=(255, 255, 255, 255),
            batch=self._inventory_batch,
        )
        self._build_inventory_ui()

        self._show_atlas = False

        self._loading = True
        self._loading_required_chunks = self.world.loading_chunks_in_radius(
            self.position,
            self.LOADING_REQUIRED_RADIUS_CHUNKS,
        )
        self._loading_texture = pyglet.image.TileableTexture.create_for_image(
            pyglet.image.load(str(TEXTURES_DIR / "dirt.png"))
        )
        self._loading_label = pyglet.text.Label(
            "Generating World",
            x=self.width // 2,
            y=self.height // 2,
            anchor_x="center",
            anchor_y="center",
            font_size=28,
            color=(255, 255, 255, 255),
        )
        self._atlas_label = pyglet.text.Label(
            "Atlas View (I to close)",
            x=10,
            y=self.height - 10,
            anchor_x="left",
            anchor_y="top",
            color=(255, 255, 255, 255),
        )

    def set_exclusive_mouse(self, exclusive: bool) -> None:
        super().set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    def _set_inventory_open(self, open_inventory: bool) -> None:
        if self.inventory_open == open_inventory:
            return
        self.inventory_open = open_inventory
        if open_inventory:
            self.set_exclusive_mouse(False)
            return

        if self._held_inventory_slot is not None:
            block, count = self._held_inventory_slot
            remaining = self.inventory.add(block, count)
            if remaining > 0:
                for i in range(self.inventory.TOTAL_SIZE):
                    if self.inventory.ui_slot(i) is None:
                        self.inventory.set_ui_slot(i, (block, remaining))
                        remaining = 0
                        break
            self._held_inventory_slot = None
        self.set_exclusive_mouse(True)

    def get_sight_vector(self) -> tuple[float, float, float]:
        yaw, pitch = self.rotation
        m = math.cos(math.radians(pitch))
        dy = math.sin(math.radians(pitch))
        dx = math.cos(math.radians(yaw - 90)) * m
        dz = math.sin(math.radians(yaw - 90)) * m
        return dx, dy, dz

    def get_motion_vector(self) -> tuple[float, float, float]:
        if self.inventory_open:
            return 0.0, 0.0, 0.0

        forward = int(self.keys[key.W]) - int(self.keys[key.S])
        right = int(self.keys[key.D]) - int(self.keys[key.A])
        if not forward and not right:
            return 0.0, 0.0, 0.0

        sx, _, sz = self.get_sight_vector()
        forward_x, forward_z = sx, sz
        forward_len = math.sqrt(forward_x * forward_x + forward_z * forward_z)
        if forward_len > 0:
            forward_x /= forward_len
            forward_z /= forward_len

        right_x = -forward_z
        right_z = forward_x

        dx = forward * forward_x + right * right_x
        dz = forward * forward_z + right * right_z

        mag = math.sqrt(dx * dx + dz * dz)
        return dx / mag, 0.0, dz / mag

    def update(self, dt: float) -> None:
        frame_start = time.perf_counter()
        dt = min(dt, 0.25)
        steps = max(1, int(math.ceil(dt / (1.0 / 60.0))))
        step_dt = dt / steps
        x, y, z = self.position
        chunk_x, chunk_z = self.world.chunk_coords(x, z)
        self.profiler.begin_frame(
            "update",
            {
                "chunk": [chunk_x, chunk_z],
                "steps": steps,
            },
        )

        try:
            if self._loading:
                with self.profiler.section("update.visible_chunks"):
                    self.world.update_visible_chunks(self.position)
                if self.world.are_chunks_generated(self._loading_required_chunks):
                    self._loading = False
                return

            for _ in range(steps):
                with self.profiler.section("update.ensure_chunk_loaded"):
                    self.world.ensure_chunk_loaded(self.position)
                d = step_dt * WALK_SPEED
                with self.profiler.section("update.motion_vector"):
                    dx, dy, dz = self.get_motion_vector()
                dx, dy, dz = dx * d, dy * d, dz * d

                with self.profiler.section("update.physics"):
                    self.dy -= step_dt * GRAVITY
                    self.dy = max(self.dy, -TERMINAL_VELOCITY)
                    dy += self.dy * step_dt

                x, y, z = self.position
                with self.profiler.section("update.collide"):
                    x, y, z = self.collide((x + dx, y + dy, z + dz), PLAYER_HEIGHT)
                self.position = (x, y, z)
            self._chunk_stream_timer += dt
            current_chunk = self.world.chunk_coords(self.position[0], self.position[2])
            over_loaded_cap = self.world.is_over_loaded_cap()
            stream_interval = (
                self.CHUNK_STREAM_UPDATE_INTERVAL_OVER_CAP_SECONDS
                if over_loaded_cap
                else self.CHUNK_STREAM_UPDATE_INTERVAL_SECONDS
            )
            should_update_chunks = (
                current_chunk != self._last_stream_chunk
                or self._chunk_stream_timer >= stream_interval
                or self._pending_view_chunk_refresh
            )
            if should_update_chunks:
                allow_optional_work = over_loaded_cap or self._last_update_ms <= self.FRAME_CATCHUP_THRESHOLD_MS
                with self.profiler.section("update.visible_chunks"):
                    self.world.update_visible_chunks(
                        self.position,
                        rotation=self.rotation,
                        allow_optional_work=allow_optional_work,
                    )
                self._chunk_stream_timer = 0.0
                self._last_stream_chunk = current_chunk
                self._pending_view_chunk_refresh = False

            held = self.inventory.selected_block()
            held_count = self.inventory.selected_count()
            self.label.text = f"XYZ: ({x:.1f}, {y:.1f}, {z:.1f})  Held: {held} ({held_count})"
        finally:
            self._last_update_ms = (time.perf_counter() - frame_start) * 1000.0
            self.profiler.end_frame(extra_context=self.world.diagnostics_snapshot())

    def collide(self, position: tuple[float, float, float], height: int) -> tuple[float, float, float]:
        pad = 0.25
        p = list(position)
        np = self.world.normalize(position)
        for face in ((0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1)):
            for i in range(3):
                if not face[i]:
                    continue
                d = (p[i] - np[i]) * face[i]
                if d < pad:
                    continue
                for dy in range(height):
                    op = list(np)
                    op[1] -= dy
                    op[i] += face[i]
                    if self.world.blocks.get(tuple(op)) in SOLID_BLOCKS:
                        p[i] -= (d - pad) * face[i]
                        if face == (0, -1, 0) or face == (0, 1, 0):
                            self.dy = 0
                        break
        return tuple(p)

    def _player_occupied_blocks(self) -> set[tuple[int, int, int]]:
        px, py, pz = self.world.normalize(self.position)
        return {(px, py - dy, pz) for dy in range(PLAYER_HEIGHT)}

    def _inventory_slot_at(self, x: float, y: float) -> int | None:
        for i, bg in enumerate(self._inventory_backgrounds):
            if bg.x <= x <= bg.x + bg.width and bg.y <= y <= bg.y + bg.height:
                return i
        return None

    def _handle_inventory_click(self, x: float, y: float, button: int) -> None:
        if button != mouse.LEFT:
            return

        index = self._inventory_slot_at(x, y)
        if index is None:
            return

        slot = self.inventory.ui_slot(index)
        held = self._held_inventory_slot
        if held is None:
            if slot is None:
                return
            self._held_inventory_slot = slot
            self.inventory.set_ui_slot(index, None)
            return

        if slot is None:
            self.inventory.set_ui_slot(index, held)
            self._held_inventory_slot = None
            return

        if slot[0] == held[0] and slot[1] < self.inventory.STACK_SIZE:
            transfer = min(self.inventory.STACK_SIZE - slot[1], held[1])
            self.inventory.set_ui_slot(index, (slot[0], slot[1] + transfer))
            left = held[1] - transfer
            self._held_inventory_slot = None if left == 0 else (held[0], left)
            return

        self.inventory.set_ui_slot(index, held)
        self._held_inventory_slot = slot

    def on_mouse_press(self, x, y, button, modifiers):
        if self._loading:
            return

        self._mouse_x = x
        self._mouse_y = y

        if self.inventory_open:
            self._handle_inventory_click(x, y, button)
            return

        if not self.exclusive:
            self.set_exclusive_mouse(True)
            return

        vector = self.get_sight_vector()
        block, previous = self.world.hit_test(self.position, vector)
        if button == mouse.RIGHT and previous:
            if previous in self._player_occupied_blocks():
                return
            held = self.inventory.selected_block()
            if held and self.inventory.remove_selected(1):
                self.world.add_block(previous, held)
        elif button == mouse.LEFT and block:
            removed = self.world.remove_block(block)
            if removed:
                self.inventory.add(removed, 1)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self._loading or self.inventory_open:
            return
        if scroll_y > 0:
            self.inventory.selected = (self.inventory.selected - 1) % self.inventory.HOTBAR_SIZE
        elif scroll_y < 0:
            self.inventory.selected = (self.inventory.selected + 1) % self.inventory.HOTBAR_SIZE

    def on_mouse_motion(self, x, y, dx, dy):
        if self._loading:
            return

        self._mouse_x = x
        self._mouse_y = y

        if self.exclusive and not self.inventory_open:
            sensitivity = 0.15
            yaw, pitch = self.rotation
            yaw += dx * sensitivity
            pitch = max(-90, min(90, pitch + dy * sensitivity))
            self.rotation = (yaw, pitch)
            if dx:
                self._pending_view_chunk_refresh = True

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self._mouse_x = x
        self._mouse_y = y

    def on_key_press(self, symbol, modifiers):
        if symbol == key.I:
            self._show_atlas = not self._show_atlas
            return
        if self._loading:
            return

        if symbol == key.E:
            self._set_inventory_open(not self.inventory_open)
            return

        if symbol == key.C and not self.inventory_open:
            self.world.hide_surface_layer = not self.world.hide_surface_layer
            self.world.rebuild_visible()
            return

        if symbol == key.ESCAPE:
            if self.inventory_open:
                self._set_inventory_open(False)
            else:
                self.set_exclusive_mouse(False)
            return

        if self.inventory_open:
            return

        if symbol == key.SPACE and self.dy == 0:
            self.dy = JUMP_SPEED
        elif key._1 <= symbol < key._1 + self.inventory.HOTBAR_SIZE:
            self.inventory.selected = symbol - key._1

    @staticmethod
    def _chunks_in_radius(center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
        cx, cz = center
        chunks: set[tuple[int, int]] = set()
        for dcx in range(-radius, radius + 1):
            for dcz in range(-radius, radius + 1):
                chunks.add((cx + dcx, cz + dcz))
        return chunks

    @staticmethod
    def _shade_color(color: tuple[float, float, float], shade: float) -> tuple[int, int, int]:
        r = max(0, min(255, int(color[0] * shade * 255)))
        g = max(0, min(255, int(color[1] * shade * 255)))
        b = max(0, min(255, int(color[2] * shade * 255)))
        return r, g, b

    def _hotbar_start_x(self) -> int:
        slot_count = self.inventory.HOTBAR_SIZE
        total_width = slot_count * self._hotbar_slot_size + (slot_count - 1) * self._hotbar_slot_gap
        return (self.width - total_width) // 2

    def _build_hotbar_ui(self) -> None:
        start_x = self._hotbar_start_x()
        slot_size = self._hotbar_slot_size
        slot_gap = self._hotbar_slot_gap
        y = self._hotbar_y

        for i in range(self.inventory.HOTBAR_SIZE):
            x = start_x + i * (slot_size + slot_gap)
            bg = pyglet.shapes.Rectangle(x, y, slot_size, slot_size, color=(30, 30, 30), batch=self.ui_batch)
            bg.opacity = 180
            self._hotbar_backgrounds.append(bg)

            border = pyglet.shapes.BorderedRectangle(
                x,
                y,
                slot_size,
                slot_size,
                border=5,
                color=(0, 0, 0),
                border_color=(120, 120, 120),
                batch=self.ui_batch,
            )
            border.opacity = 0
            self._hotbar_borders.append(border)

            self._hotbar_icon_sprites.append(None)
            self._hotbar_icon_blocks.append(None)

            count = pyglet.text.Label(
                "",
                x=x + slot_size - 5,
                y=y + 4,
                anchor_x="right",
                anchor_y="bottom",
                color=(255, 255, 255, 255),
                font_size=10,
                batch=self.ui_batch,
            )
            self._hotbar_counts.append(count)

    def _inventory_start(self) -> tuple[int, int, int, int]:
        cols = self.inventory.HOTBAR_SIZE
        rows = self.inventory.MAIN_ROWS + 1
        total_width = cols * self._inv_slot_size + (cols - 1) * self._inv_slot_gap
        total_height = (
            rows * self._inv_slot_size
            + (self.inventory.MAIN_ROWS - 1) * self._inv_slot_gap
            + self._inv_hotbar_gap
        )
        start_x = (self.width - total_width) // 2
        start_y = (self.height - total_height) // 2
        return start_x, start_y, total_width, total_height

    def _inventory_slot_position(self, index: int) -> tuple[int, int]:
        start_x, start_y, _, _ = self._inventory_start()
        if index < self.inventory.MAIN_SIZE:
            row = index // self.inventory.MAIN_COLS
            col = index % self.inventory.MAIN_COLS
            y = (
                start_y
                + self._inv_slot_size
                + self._inv_hotbar_gap
                + (self.inventory.MAIN_ROWS - 1 - row) * (self._inv_slot_size + self._inv_slot_gap)
            )
            x = start_x + col * (self._inv_slot_size + self._inv_slot_gap)
            return x, y

        hotbar_index = index - self.inventory.MAIN_SIZE
        x = start_x + hotbar_index * (self._inv_slot_size + self._inv_slot_gap)
        y = start_y
        return x, y

    def _build_inventory_ui(self) -> None:
        for i in range(self.inventory.TOTAL_SIZE):
            x, y = self._inventory_slot_position(i)
            bg = pyglet.shapes.Rectangle(x, y, self._inv_slot_size, self._inv_slot_size, color=(35, 35, 35), batch=self._inventory_batch)
            bg.opacity = 220
            self._inventory_backgrounds.append(bg)

            border = pyglet.shapes.BorderedRectangle(
                x,
                y,
                self._inv_slot_size,
                self._inv_slot_size,
                border=4,
                color=(0, 0, 0),
                border_color=(130, 130, 130),
                batch=self._inventory_batch,
            )
            border.opacity = 255
            self._inventory_borders.append(border)

            self._inventory_icon_sprites.append(None)
            self._inventory_icon_blocks.append(None)

            count = pyglet.text.Label(
                "",
                x=x + self._inv_slot_size - 5,
                y=y + 4,
                anchor_x="right",
                anchor_y="bottom",
                color=(255, 255, 255, 255),
                font_size=10,
                batch=self._inventory_batch,
            )
            self._inventory_counts.append(count)

        self._update_inventory_layout()

    def _update_hotbar_layout(self) -> None:
        slot_count = self.inventory.HOTBAR_SIZE
        start_x = self._hotbar_start_x()
        slot_size = self._hotbar_slot_size
        slot_gap = self._hotbar_slot_gap
        y = self._hotbar_y
        for i in range(slot_count):
            x = start_x + i * (slot_size + slot_gap)
            self._hotbar_backgrounds[i].x = x
            self._hotbar_backgrounds[i].y = y

            self._hotbar_borders[i].x = x
            self._hotbar_borders[i].y = y

            self._hotbar_counts[i].x = x + slot_size - 5
            self._hotbar_counts[i].y = y + 4

    def _update_inventory_layout(self) -> None:
        start_x, start_y, total_width, total_height = self._inventory_start()
        panel_x = start_x - self._inv_panel_padding
        panel_y = start_y - self._inv_panel_padding
        panel_width = total_width + self._inv_panel_padding * 2
        panel_height = total_height + self._inv_panel_padding * 2

        self._inventory_panel.x = panel_x
        self._inventory_panel.y = panel_y
        self._inventory_panel.width = panel_width
        self._inventory_panel.height = panel_height

        self._inventory_panel_border.x = panel_x
        self._inventory_panel_border.y = panel_y
        self._inventory_panel_border.width = panel_width
        self._inventory_panel_border.height = panel_height

        self._inventory_title.x = panel_x + 10
        self._inventory_title.y = panel_y + panel_height - 22

        for i in range(self.inventory.TOTAL_SIZE):
            x, y = self._inventory_slot_position(i)
            self._inventory_backgrounds[i].x = x
            self._inventory_backgrounds[i].y = y
            self._inventory_borders[i].x = x
            self._inventory_borders[i].y = y
            self._inventory_counts[i].x = x + self._inv_slot_size - 5
            self._inventory_counts[i].y = y + 4

    def _slot_texture_image(self, block: str) -> pyglet.image.AbstractImage | None:
        texture_name = get_block_texture_for_face(block, 2) or get_block_texture_for_face(block, 4)
        if texture_name is None:
            return None
        cached = self._hotbar_texture_cache.get(texture_name)
        if cached is not None:
            return cached
        path = TEXTURES_DIR / texture_name
        if not path.is_file():
            return None
        image = pyglet.image.load(str(path))
        self._hotbar_texture_cache[texture_name] = image
        return image

    def _update_hotbar_ui(self) -> None:
        for i in range(self.inventory.HOTBAR_SIZE):
            is_selected = i == self.inventory.selected
            self._hotbar_borders[i].border_color = (255, 255, 255) if is_selected else (120, 120, 120)
            self._hotbar_borders[i].opacity = 255 if is_selected else 200
            self._hotbar_backgrounds[i].opacity = 230 if is_selected else 180

            slot = self.inventory.slot(i)
            sprite = self._hotbar_icon_sprites[i]
            if slot is None:
                if sprite is not None:
                    sprite.visible = False
                self._hotbar_icon_blocks[i] = None
                self._hotbar_counts[i].text = ""
                continue

            block, count = slot
            self._hotbar_counts[i].text = str(count)

            image = self._slot_texture_image(block)
            if image is None:
                if sprite is not None:
                    sprite.visible = False
                self._hotbar_icon_blocks[i] = block
                continue

            if sprite is None:
                sprite = pyglet.sprite.Sprite(image, x=0, y=0, batch=self.ui_batch)
                self._hotbar_icon_sprites[i] = sprite
            elif self._hotbar_icon_blocks[i] != block:
                sprite.image = image

            bg = self._hotbar_backgrounds[i]
            target_size = bg.width - 14
            sprite.scale_x = target_size / float(sprite.image.width)
            sprite.scale_y = target_size / float(sprite.image.height)
            sprite.x = bg.x + (bg.width - sprite.width) / 2
            sprite.y = bg.y + (bg.height - sprite.height) / 2
            sprite.opacity = 255
            sprite.visible = True
            self._hotbar_icon_blocks[i] = block

    def _update_inventory_ui(self) -> None:
        for i in range(self.inventory.TOTAL_SIZE):
            border_color = (255, 255, 255) if i >= self.inventory.MAIN_SIZE and (i - self.inventory.MAIN_SIZE) == self.inventory.selected else (130, 130, 130)
            self._inventory_borders[i].border_color = border_color

            slot = self.inventory.ui_slot(i)
            sprite = self._inventory_icon_sprites[i]
            if slot is None:
                if sprite is not None:
                    sprite.visible = False
                self._inventory_icon_blocks[i] = None
                self._inventory_counts[i].text = ""
                continue

            block, count = slot
            self._inventory_counts[i].text = str(count)

            image = self._slot_texture_image(block)
            if image is None:
                if sprite is not None:
                    sprite.visible = False
                self._inventory_icon_blocks[i] = block
                continue

            if sprite is None:
                sprite = pyglet.sprite.Sprite(image, x=0, y=0, batch=self._inventory_batch)
                self._inventory_icon_sprites[i] = sprite
            elif self._inventory_icon_blocks[i] != block:
                sprite.image = image

            bg = self._inventory_backgrounds[i]
            target_size = bg.width - 14
            sprite.scale_x = target_size / float(sprite.image.width)
            sprite.scale_y = target_size / float(sprite.image.height)
            sprite.x = bg.x + (bg.width - sprite.width) / 2
            sprite.y = bg.y + (bg.height - sprite.height) / 2
            sprite.opacity = 255
            sprite.visible = True
            self._inventory_icon_blocks[i] = block

    def _draw_held_inventory_slot(self) -> None:
        if self._held_inventory_slot is None:
            return

        block, count = self._held_inventory_slot
        image = self._slot_texture_image(block)
        if image is None:
            return

        sprite = pyglet.sprite.Sprite(image, x=0, y=0)
        target_size = self._inv_slot_size - 14
        sprite.scale_x = target_size / float(sprite.image.width)
        sprite.scale_y = target_size / float(sprite.image.height)
        sprite.x = self._mouse_x - sprite.width / 2
        sprite.y = self._mouse_y - sprite.height / 2
        sprite.opacity = 220
        sprite.draw()

        count_label = pyglet.text.Label(
            str(count),
            x=int(self._mouse_x + self._inv_slot_size * 0.28),
            y=int(self._mouse_y - self._inv_slot_size * 0.34),
            anchor_x="right",
            anchor_y="bottom",
            color=(255, 255, 255, 255),
            font_size=10,
        )
        count_label.draw()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.label.y = height - 10
        self.crosshair.x = width // 2 - 8
        self.crosshair.x2 = width // 2 + 8
        self.crosshair.y = self.crosshair.y2 = height // 2
        self.crosshair2.y = height // 2 - 8
        self.crosshair2.y2 = height // 2 + 8
        self.crosshair2.x = self.crosshair2.x2 = width // 2
        self._update_hotbar_layout()
        self._update_inventory_layout()
        self._loading_label.x = width // 2
        self._loading_label.y = height // 2
        self._atlas_label.y = height - 10

    def on_draw(self):
        self.profiler.begin_frame("draw")
        try:
            if self._show_atlas:
                with self.profiler.section("draw.clear"):
                    self.clear()
                with self.profiler.section("draw.atlas"):
                    set_2d(self)
                    gl.glDisable(gl.GL_CULL_FACE)
                    atlas_texture = self.world.renderer._atlas_texture
                    if atlas_texture is not None:
                        atlas_texture.blit(0, 0, 0, width=self.width, height=self.height)
                    self._atlas_label.draw()
                return

            if self._loading:
                with self.profiler.section("draw.clear"):
                    self.clear()
                with self.profiler.section("draw.loading"):
                    set_2d(self)
                    self._loading_label.draw()
                return

            with self.profiler.section("draw.clear"):
                self.clear()

            with self.profiler.section("draw.set_3d"):
                set_3d(self, self.rotation, self.position)
            with self.profiler.section("draw.world_batch"):
                self.world.batch.draw()

            with self.profiler.section("draw.set_2d"):
                set_2d(self)
            with self.profiler.section("draw.ui_update"):
                self._update_hotbar_ui()
                self.crosshair.opacity = 0 if self.inventory_open else 255
                self.crosshair2.opacity = 0 if self.inventory_open else 255
            with self.profiler.section("draw.ui_batch"):
                self.ui_batch.draw()

            if self.inventory_open:
                with self.profiler.section("draw.inventory_ui"):
                    self._update_inventory_ui()
                    self._inventory_batch.draw()
                    self._draw_held_inventory_slot()

        finally:
            self.profiler.end_frame(extra_context=self.world.diagnostics_snapshot())

    def on_close(self):
        report_paths = self.profiler.write_report()
        if report_paths is not None:
            txt_path, json_path = report_paths
            print(f"[profiler] wrote lag report: {txt_path}")
            print(f"[profiler] wrote lag report: {json_path}")
        self.world.shutdown()
        super().on_close()
