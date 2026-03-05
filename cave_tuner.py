import argparse
import json
from pathlib import Path

import pyglet
from pyglet import shapes
from pyglet import gl
from pyglet.window import key

from engine.world.terrain import TerrainGenerator


class CavePreviewWindow(pyglet.window.Window):
    SETTINGS_PATH = Path(__file__).resolve().parent / "engine" / "world" / "world_gen_settings.json"
    BIOMES_PATH = Path(__file__).resolve().parent / "engine" / "biomes" / "biomes.json"

    # Cave-only preview resolution (no chunk/block meshing).
    XZ_SPAN = 96
    XZ_STEP = 1
    Y_STEP = 1
    TUNING_PREVIEW_MULTIPLIER = 1
    CLASSIC_CAVES_PRESET: dict[str, int | float] = {
        "CAVE_MIN_Y": 5,
        "CAVE_MAX_Y": 64,
        "CAVE_TUNNEL_PRIMARY_Y_SCALE": 1.1,
        "CAVE_TUNNEL_SECONDARY_Y_SCALE": 1.2,
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
    }

    def __init__(self, seed: int = 90125):
        super().__init__(width=1360, height=820, caption="Cave Tuner Preview", resizable=True)
        self.seed = seed
        self.center_x = 0
        self.center_z = 0
        self.slice_z = 0
        self.slice_y = 40
        self._status = "Ready"

        with open(self.SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.settings = dict(data.get("caves", {}))

        with open(self.BIOMES_PATH, "r", encoding="utf-8") as f:
            self.biomes_data = json.load(f)

        self.terrain = TerrainGenerator(seed, flat_height=64)

        self._slice_image: pyglet.image.ImageData | None = None
        self._hslice_image: pyglet.image.ImageData | None = None

        self._help_label = pyglet.text.Label(
            "",
            x=14,
            y=self.height - 12,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            width=self.width - 28,
            font_size=10,
            color=(235, 235, 235, 255),
        )
        self._rebuild_preview()

    @staticmethod
    def _blit_crisp(image: pyglet.image.ImageData, x: float, y: float, width: float, height: float) -> None:
        tex = image.get_texture()
        gl.glBindTexture(tex.target, tex.id)
        gl.glTexParameteri(tex.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(tex.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        tex.blit(x, y, width=width, height=height)

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, x: float) -> float:
        if edge0 == edge1:
            return 0.0
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _height_at(self, x: int, z: int) -> int:
        biome_noise = self.terrain.biome_noise_at(x, z)
        biome = self.terrain.biome_from_noise(biome_noise)
        biome_json = self.biomes_data.get(biome, {})
        depth = float(biome_json.get("depth", 0.1))
        scale = float(biome_json.get("scale", 0.05))

        desert = self.biomes_data.get("desert")
        plains = self.biomes_data.get("plains")
        if desert is not None and plains is not None:
            blend = self._smoothstep(-0.25, 0.25, biome_noise)
            depth = self._lerp(float(desert.get("depth", depth)), float(plains.get("depth", depth)), blend)
            scale = self._lerp(float(desert.get("scale", scale)), float(plains.get("scale", scale)), blend)
        return int(self.terrain.height_at(x, z, depth, scale))

    def _is_cave(self, x: int, y: int, z: int, surface_y: int) -> bool:
        return self.terrain.is_cave_at(
            x,
            y,
            z,
            surface_y,
            int(self.settings.get("CAVE_MIN_Y", 3)),
            int(self.settings.get("CAVE_MAX_Y", 126)),
            0,
            int(self.settings.get("CAVE_BEDROCK_BUFFER", 2)),
            int(self.settings.get("CAVE_NEAR_SURFACE_DEPTH", 2)),
            float(self.settings.get("CAVE_TUNNEL_PRIMARY_Y_SCALE", 1.05)),
            float(self.settings.get("CAVE_TUNNEL_SECONDARY_Y_SCALE", 1.20)),
            float(self.settings.get("CAVE_TUNNEL_GATE_Y_SCALE", 1.35)),
            float(self.settings.get("CAVE_TUNNEL_PRIMARY_BAND", 0.08)),
            float(self.settings.get("CAVE_TUNNEL_SECONDARY_BAND", 0.09)),
            float(self.settings.get("CAVE_COMBINED_BAND", 0.15)),
            float(self.settings.get("CAVE_NEAR_SURFACE_BAND_SCALE", 0.55)),
            float(self.settings.get("CAVE_TUNNEL_GATE_THRESHOLD", 0.16)),
            float(self.settings.get("CAVE_WARP_STRENGTH_XZ", 14.0)),
            float(self.settings.get("CAVE_WARP_STRENGTH_Y", 10.0)),
            float(self.settings.get("CAVE_LEVEL_VARIATION_SCALE", 0.35)),
            float(self.settings.get("CAVE_LEVEL_VARIATION_BAND_BOOST", 0.02)),
            float(self.settings.get("CAVE_EDGE_FADE_Y", 12.0)),
            int(self.settings.get("CAVE_MIN_CONNECTED_NEIGHBORS", 2)),
            int(self.settings.get("CAVE_MAX_LOCAL_OPEN", 11)),
            float(self.settings.get("CAVE_FAMILY2_WARP_STRENGTH_XZ", 11.0)),
            float(self.settings.get("CAVE_FAMILY2_WARP_STRENGTH_Y", 8.0)),
            float(self.settings.get("CAVE_FAMILY2_PRIMARY_BAND_SCALE", 0.78)),
            float(self.settings.get("CAVE_FAMILY2_SECONDARY_BAND_SCALE", 0.78)),
            float(self.settings.get("CAVE_FAMILY2_GATE_THRESHOLD", 0.18)),
        )

    def _rebuild_vertical_slice(self, sample_multiplier: int = 1) -> None:
        sample_multiplier = max(1, int(sample_multiplier))
        x_step = self.XZ_STEP * sample_multiplier
        y_step = self.Y_STEP * sample_multiplier
        cave_min_y = int(self.settings.get("CAVE_MIN_Y", 3))
        cave_max_y = int(self.settings.get("CAVE_MAX_Y", 126))
        self.slice_y = max(cave_min_y, min(cave_max_y - 1, self.slice_y))
        x_count = max(1, self.XZ_SPAN // x_step)
        y_count = max(1, (cave_max_y - cave_min_y) // y_step)
        x0 = self.center_x - (self.XZ_SPAN // 2)
        slice_pixels = bytearray(x_count * y_count * 3)
        wz = self.slice_z

        for ix in range(x_count):
            wx = x0 + ix * x_step
            surface = self._height_at(wx, wz)
            for y in range(cave_min_y, cave_max_y, y_step):
                if not self._is_cave(wx, y, wz, surface):
                    continue
                iy = (y - cave_min_y) // y_step
                idx = ((y_count - 1 - iy) * x_count + ix) * 3
                slice_pixels[idx: idx + 3] = bytes((238, 238, 238))

        self._slice_image = pyglet.image.ImageData(x_count, y_count, "RGB", bytes(slice_pixels), pitch=x_count * 3)

    def _rebuild_horizontal_slice(self, sample_multiplier: int = 1) -> None:
        sample_multiplier = max(1, int(sample_multiplier))
        x_step = self.XZ_STEP * sample_multiplier
        cave_min_y = int(self.settings.get("CAVE_MIN_Y", 3))
        cave_max_y = int(self.settings.get("CAVE_MAX_Y", 126))
        self.slice_y = max(cave_min_y, min(cave_max_y - 1, self.slice_y))
        x_count = max(1, self.XZ_SPAN // x_step)
        z_count = max(1, self.XZ_SPAN // x_step)
        x0 = self.center_x - (self.XZ_SPAN // 2)
        z0 = self.center_z - (self.XZ_SPAN // 2)
        hslice_pixels = bytearray(x_count * z_count * 3)

        for ix in range(x_count):
            wx = x0 + ix * x_step
            for iz in range(z_count):
                wz = z0 + iz * x_step
                surface = self._height_at(wx, wz)
                idx = ((z_count - 1 - iz) * x_count + ix) * 3
                if self._is_cave(wx, self.slice_y, wz, surface):
                    hslice_pixels[idx: idx + 3] = bytes((240, 240, 240))
                else:
                    hslice_pixels[idx: idx + 3] = bytes((14, 14, 14))

        self._hslice_image = pyglet.image.ImageData(x_count, z_count, "RGB", bytes(hslice_pixels), pitch=x_count * 3)

    def _rebuild_preview(
        self,
        sample_multiplier: int = 1,
        rebuild_vertical: bool = True,
        rebuild_horizontal: bool = True,
    ) -> None:
        if rebuild_vertical:
            self._rebuild_vertical_slice(sample_multiplier=sample_multiplier)
        if rebuild_horizontal:
            self._rebuild_horizontal_slice(sample_multiplier=sample_multiplier)
        mode = "full" if sample_multiplier == 1 else f"fast x{sample_multiplier}"
        which = (
            "both"
            if (rebuild_vertical and rebuild_horizontal)
            else "vertical"
            if rebuild_vertical
            else "horizontal"
        )
        self._status = f"Rebuilt {which} ({mode}) x={self.center_x} z={self.center_z} slice_z={self.slice_z} slice_y={self.slice_y}"

    def _save(self) -> None:
        with open(self.SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("caves", {}).update(self.settings)
        with open(self.SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        self._status = "Saved cave settings"

    def _adjust(self, key_name: str, delta: float, min_v: float, max_v: float) -> None:
        cur = float(self.settings.get(key_name, 0.0))
        new_val = max(min_v, min(max_v, cur + delta))
        self.settings[key_name] = int(new_val) if key_name in {"CAVE_MAX_Y", "CAVE_MIN_CONNECTED_NEIGHBORS", "CAVE_MAX_LOCAL_OPEN"} else round(new_val, 4)
        self._status = f"{key_name}={self.settings[key_name]}"
        self._rebuild_preview(sample_multiplier=self.TUNING_PREVIEW_MULTIPLIER)

    def _apply_classic_caves_preset(self) -> None:
        self.settings.update(self.CLASSIC_CAVES_PRESET)
        self._status = "Applied Classic Caves preset (pre-Caves & Cliffs style)"
        self._rebuild_preview(sample_multiplier=1)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.R:
            self._rebuild_preview()
            return
        if symbol == key.F5:
            self._save()
            return
        if symbol == key.F6:
            self._apply_classic_caves_preset()
            return
        if symbol == key.UP:
            self.slice_z += self.XZ_STEP
            self._rebuild_preview(
                sample_multiplier=self.TUNING_PREVIEW_MULTIPLIER,
                rebuild_vertical=True,
                rebuild_horizontal=False,
            )
            return
        if symbol == key.DOWN:
            self.slice_z -= self.XZ_STEP
            self._rebuild_preview(
                sample_multiplier=self.TUNING_PREVIEW_MULTIPLIER,
                rebuild_vertical=True,
                rebuild_horizontal=False,
            )
            return
        if symbol == key.PAGEUP:
            self.slice_y += self.Y_STEP
            self._rebuild_preview(
                sample_multiplier=self.TUNING_PREVIEW_MULTIPLIER,
                rebuild_vertical=False,
                rebuild_horizontal=True,
            )
            return
        if symbol == key.PAGEDOWN:
            self.slice_y -= self.Y_STEP
            self._rebuild_preview(
                sample_multiplier=self.TUNING_PREVIEW_MULTIPLIER,
                rebuild_vertical=False,
                rebuild_horizontal=True,
            )
            return
        if symbol == key.LEFT:
            self.center_x -= self.XZ_STEP * 2
            self._rebuild_preview(sample_multiplier=self.TUNING_PREVIEW_MULTIPLIER)
            return
        if symbol == key.RIGHT:
            self.center_x += self.XZ_STEP * 2
            self._rebuild_preview(sample_multiplier=self.TUNING_PREVIEW_MULTIPLIER)
            return

        if symbol == key.J:
            self._adjust("CAVE_TUNNEL_PRIMARY_BAND", -0.005, 0.01, 0.4)
            return
        if symbol == key.K:
            self._adjust("CAVE_TUNNEL_PRIMARY_BAND", 0.005, 0.01, 0.4)
            return
        if symbol == key.N:
            self._adjust("CAVE_TUNNEL_SECONDARY_BAND", -0.005, 0.01, 0.4)
            return
        if symbol == key.M:
            self._adjust("CAVE_TUNNEL_SECONDARY_BAND", 0.005, 0.01, 0.4)
            return
        if symbol == key.U:
            self._adjust("CAVE_COMBINED_BAND", -0.005, 0.03, 0.8)
            return
        if symbol == key.I:
            self._adjust("CAVE_COMBINED_BAND", 0.005, 0.03, 0.8)
            return
        if symbol == key.O:
            self._adjust("CAVE_TUNNEL_GATE_THRESHOLD", -0.01, 0.0, 1.0)
            return
        if symbol == key.P:
            self._adjust("CAVE_TUNNEL_GATE_THRESHOLD", 0.01, 0.0, 1.0)
            return
        if symbol == key.BRACKETLEFT:
            self._adjust("CAVE_FAMILY2_GATE_THRESHOLD", -0.01, 0.0, 1.0)
            return
        if symbol == key.BRACKETRIGHT:
            self._adjust("CAVE_FAMILY2_GATE_THRESHOLD", 0.01, 0.0, 1.0)
            return
        if symbol == key._9:
            self._adjust("CAVE_WARP_STRENGTH_XZ", -0.5, 1.0, 40.0)
            return
        if symbol == key._0:
            self._adjust("CAVE_WARP_STRENGTH_XZ", 0.5, 1.0, 40.0)
            return
        if symbol == key.MINUS:
            self._adjust("CAVE_WARP_STRENGTH_Y", -0.5, 1.0, 30.0)
            return
        if symbol == key.EQUAL:
            self._adjust("CAVE_WARP_STRENGTH_Y", 0.5, 1.0, 30.0)
            return
        if symbol == key.COMMA:
            self._adjust("CAVE_MAX_Y", -2, 20, 255)
            return
        if symbol == key.PERIOD:
            self._adjust("CAVE_MAX_Y", 2, 20, 255)
            return
        if symbol == key.H:
            self._adjust("CAVE_MIN_CONNECTED_NEIGHBORS", -1, 0, 6)
            return
        if symbol == key.L:
            self._adjust("CAVE_MIN_CONNECTED_NEIGHBORS", 1, 0, 6)
            return
        if symbol == key.SEMICOLON:
            self._adjust("CAVE_MAX_LOCAL_OPEN", -1, 0, 26)
            return
        if symbol == key.APOSTROPHE:
            self._adjust("CAVE_MAX_LOCAL_OPEN", 1, 0, 26)
            return

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self._help_label.y = height - 12
        self._help_label.width = width - 28

    def on_draw(self):
        self.clear()
        pad = 14
        top_h = int(self.height * 0.56)
        panel_y = 18
        panel_w = (self.width - pad * 3) // 2

        rects = []
        for i in range(2):
            x = pad + i * (panel_w + pad)
            rects.append((x, panel_y, panel_w, top_h))
            shapes.BorderedRectangle(
                x,
                panel_y,
                panel_w,
                top_h,
                border=1,
                color=(10, 10, 10),
                border_color=(110, 110, 110),
            ).draw()

        if self._slice_image is not None:
            x, y, w, h = rects[0]
            self._blit_crisp(self._slice_image, x + 2, y + 2, width=w - 4, height=h - 4)
        if self._hslice_image is not None:
            x, y, w, h = rects[1]
            self._blit_crisp(self._hslice_image, x + 2, y + 2, width=w - 4, height=h - 4)

        pyglet.text.Label("Vertical Slice", x=rects[0][0] + 6, y=rects[0][1] + rects[0][3] - 6, anchor_x="left", anchor_y="top").draw()
        pyglet.text.Label("Horizontal Slice", x=rects[1][0] + 6, y=rects[1][1] + rects[1][3] - 6, anchor_x="left", anchor_y="top").draw()

        cave_min_y = int(self.settings.get("CAVE_MIN_Y", 3))
        cave_max_y = int(self.settings.get("CAVE_MAX_Y", 126))
        y_span = max(1, cave_max_y - cave_min_y)
        z_span = max(1, self.XZ_SPAN)

        # Intersection guide in vertical slice: y = slice_y
        vx, vy, vw, vh = rects[0]
        y_ratio = (self.slice_y - cave_min_y) / y_span
        y_px = vy + 2 + (vh - 4) * (1.0 - max(0.0, min(1.0, y_ratio)))
        shapes.Line(vx + 2, y_px, vx + vw - 2, y_px, color=(255, 160, 90), thickness=2).draw()

        # Intersection guide in horizontal slice: z = slice_z
        hx, hy, hw, hh = rects[1]
        z_ratio = (self.slice_z - (self.center_z - self.XZ_SPAN // 2)) / z_span
        z_px = hy + 2 + (hh - 4) * (1.0 - max(0.0, min(1.0, z_ratio)))
        shapes.Line(hx + 2, z_px, hx + hw - 2, z_px, color=(255, 160, 90), thickness=2).draw()

        lines = [
            f"CAVE_TUNNEL_PRIMARY_BAND={float(self.settings.get('CAVE_TUNNEL_PRIMARY_BAND', 0.0)):.4f}  J/K  thinner/wider family-1 tunnels",
            f"CAVE_TUNNEL_SECONDARY_BAND={float(self.settings.get('CAVE_TUNNEL_SECONDARY_BAND', 0.0)):.4f}  N/M  trims broad openings",
            f"CAVE_COMBINED_BAND={float(self.settings.get('CAVE_COMBINED_BAND', 0.0)):.4f}  U/I  lower prevents caverns",
            f"CAVE_TUNNEL_GATE_THRESHOLD={float(self.settings.get('CAVE_TUNNEL_GATE_THRESHOLD', 0.0)):.4f}  O/P  family-1 density",
            f"CAVE_FAMILY2_GATE_THRESHOLD={float(self.settings.get('CAVE_FAMILY2_GATE_THRESHOLD', 0.0)):.4f}  [/]  family-2 density",
            f"CAVE_WARP_STRENGTH_XZ={float(self.settings.get('CAVE_WARP_STRENGTH_XZ', 0.0)):.2f}  9/0  horizontal twisting",
            f"CAVE_WARP_STRENGTH_Y={float(self.settings.get('CAVE_WARP_STRENGTH_Y', 0.0)):.2f}  -/=  vertical movement",
            f"CAVE_MAX_Y={int(self.settings.get('CAVE_MAX_Y', 0))}  ,/.  highest cave elevation",
            f"CAVE_MIN_CONNECTED_NEIGHBORS={int(self.settings.get('CAVE_MIN_CONNECTED_NEIGHBORS', 0))}  H/L  remove isolated cave voxels",
            f"CAVE_MAX_LOCAL_OPEN={int(self.settings.get('CAVE_MAX_LOCAL_OPEN', 0))}  ;/'  suppress large open chambers",
            f"slice_z={self.slice_z}  Up/Down   slice_y={self.slice_y}  PgUp/PgDn",
            "R rebuild, F5 save, F6 apply classic preset",
            f"Status: {self._status}",
        ]
        self._help_label.text = (
            "Cave-only preview. No chunk rendering, no block/material meshing.\n"
            "Only cross-sections are computed. Arrow Left/Right move preview center X.\n"
            + "\n".join(lines)
        )
        self._help_label.draw()


def run(seed: int = 90125) -> None:
    CavePreviewWindow(seed=seed)
    pyglet.app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cave Tuner Preview")
    parser.add_argument("--seed", type=int, default=90125, help="Terrain seed")
    args = parser.parse_args()
    run(seed=args.seed)
