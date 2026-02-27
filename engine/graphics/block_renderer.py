from dataclasses import dataclass
import math
from pathlib import Path
import time

import pyglet
from pyglet import gl

from engine.blocks import get_block_color, get_block_texture_for_face
from engine.blocks.registry import BLOCKS, TEXTURES_DIR
from engine.constants import Vec3


class RenderedCube:
    def __init__(self, parts: list[pyglet.graphics.vertexdomain.VertexList]) -> None:
        self.parts = parts

    def delete(self) -> None:
        for part in self.parts:
            part.delete()


class TextureArrayGroup(pyglet.graphics.Group):
    def __init__(
        self,
        texture_id: int,
        shader_program: pyglet.graphics.shader.ShaderProgram,
        order: int = 0,
        parent: pyglet.graphics.Group | None = None,
    ) -> None:
        super().__init__(order=order, parent=parent)
        self.texture_id = texture_id
        self.shader_program = shader_program

    def set_state(self) -> None:
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.texture_id)

    def __hash__(self) -> int:
        return hash((self.texture_id, self.shader_program, self.order, self.parent))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TextureArrayGroup):
            return False
        return (
            self.texture_id == other.texture_id
            and self.shader_program == other.shader_program
            and self.order == other.order
            and self.parent == other.parent
        )


@dataclass
class ChunkMeshData:
    color_vertices: list[float]
    color_values: list[float]
    textured_vertices: dict[str, list[float]]
    textured_texcoords: dict[str, list[float]]
    textured_colors: dict[str, list[float]]


class BlockRenderer:
    ATLAS_TILE_SIZE = 16
    _TEXTURE_ARRAY_VERTEX_SOURCE = """#version 330 core
in vec3 position;
in vec4 colors;
in vec3 tex_coords;
in float tex_layer;
out vec4 vertex_colors;
out vec2 texture_coords;
flat out int texture_layer;

uniform WindowBlock
{
    mat4 projection;
    mat4 view;
} window;

void main()
{
    gl_Position = window.projection * window.view * vec4(position, 1.0);
    vertex_colors = colors;
    texture_coords = tex_coords.xy;
    texture_layer = int(tex_layer + 0.5);
}
"""
    _TEXTURE_ARRAY_FRAGMENT_SOURCE = """#version 330 core
in vec4 vertex_colors;
in vec2 texture_coords;
flat in int texture_layer;
out vec4 final_colors;

uniform sampler2DArray block_texture_array;

void main()
{
    vec4 texel = texture(block_texture_array, vec3(fract(texture_coords), float(texture_layer)));
    final_colors = texel + vertex_colors;
}
"""

    def __init__(self, seed: int, use_texture_array: bool = False) -> None:
        self.seed = seed
        # Experimental fast textured meshing path.
        # Off by default because some drivers/states may still show UV artifacts.
        self.use_texture_array = use_texture_array
        self.shader = pyglet.graphics.get_default_shader()
        self._atlas_texture: pyglet.image.Texture | None = None
        self._atlas_groups: dict[int, pyglet.graphics.TextureGroup] = {}
        self._texture_groups: dict[tuple[str, int], pyglet.graphics.TextureGroup] = {}
        self._texture_cache: dict[str, pyglet.image.Texture] = {}
        self._atlas_rects: dict[str, tuple[float, float, float, float]] = {}
        self._atlas_face_uv_cache: dict[tuple[str, int], list[float]] = {}
        self._texture_layer_map: dict[str, int] = {}
        self._texture_array_id: int | None = None
        self._texture_array_shader: pyglet.graphics.shader.ShaderProgram | None = None
        self._texture_array_shader_group: pyglet.graphics.ShaderGroup | None = None
        self._texture_array_group: TextureArrayGroup | None = None
        self._build_texture_atlas()
        self._export_texture_atlas()
        if self.use_texture_array:
            self._init_texture_array_pipeline()

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        v = max(1, value)
        return 1 << (v - 1).bit_length()

    def _build_texture_atlas(self) -> None:
        texture_names: set[str] = set()
        for definition in BLOCKS.values():
            for texture_name in (definition.texture, definition.texture_top, definition.texture_bottom, definition.texture_side):
                if texture_name:
                    texture_names.add(texture_name)

        if not texture_names:
            return

        images: dict[str, pyglet.image.AbstractImage] = {}
        for texture_name in sorted(texture_names):
            texture_path = TEXTURES_DIR / texture_name
            if not texture_path.is_file():
                continue
            img = pyglet.image.load(str(texture_path))
            if img.width != self.ATLAS_TILE_SIZE or img.height != self.ATLAS_TILE_SIZE:
                raise ValueError(
                    f"Texture '{texture_name}' must be {self.ATLAS_TILE_SIZE}x{self.ATLAS_TILE_SIZE}, "
                    f"got {img.width}x{img.height}"
                )
            images[texture_name] = img

        if not images:
            return

        names = sorted(images.keys())
        tile_size = self.ATLAS_TILE_SIZE
        cell_w = tile_size
        cell_h = tile_size
        cols = max(1, int(math.ceil(math.sqrt(len(names)))))
        rows = int(math.ceil(len(names) / cols))

        atlas_w = self._next_power_of_two(cols * cell_w)
        atlas_h = self._next_power_of_two(rows * cell_h)
        atlas_texture = pyglet.image.Texture.create(atlas_w, atlas_h)

        for index, texture_name in enumerate(names):
            col = index % cols
            row = index // cols
            cell_x = col * cell_w
            cell_y = row * cell_h
            img = images[texture_name]
            atlas_texture.blit_into(img, cell_x, cell_y, 0)

            x0 = cell_x
            y0 = cell_y
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            self._atlas_rects[texture_name] = (
                x0 / float(atlas_w),
                y0 / float(atlas_h),
                x1 / float(atlas_w),
                y1 / float(atlas_h),
            )

        self._atlas_texture = atlas_texture

        if self._atlas_texture is not None:
            gl.glBindTexture(self._atlas_texture.target, self._atlas_texture.id)
            gl.glTexParameteri(self._atlas_texture.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(self._atlas_texture.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(self._atlas_texture.target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(self._atlas_texture.target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    def _export_texture_atlas(self) -> None:
        if self._atlas_texture is None:
            return
        out_dir = Path("profiling")
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in out_dir.glob("atlas_*.png"):
            try:
                path.unlink()
            except OSError:
                continue
        latest_path = out_dir / "atlas_latest.png"
        if latest_path.exists():
            try:
                latest_path.unlink()
            except OSError:
                pass
        safe_stamp = time.strftime("%Y%m%d_%H%M%S")
        stamped_path = out_dir / f"atlas_{safe_stamp}.png"
        image_data = self._atlas_texture.get_image_data()
        image_data.save(str(latest_path))
        image_data.save(str(stamped_path))

    @staticmethod
    def _face_uv(face_index: int) -> list[float]:
        uv_by_face = {
            0: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            1: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            2: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            3: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            4: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            5: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        }
        return uv_by_face[face_index]

    @staticmethod
    def _face(x: float, y: float, z: float, n: float, axis: int) -> list[float]:
        if axis == 0:
            return [x + n, y - n, z - n, x + n, y + n, z - n, x + n, y + n, z + n, x + n, y - n, z + n]
        if axis == 1:
            return [x - n, y + n, z - n, x + n, y + n, z - n, x + n, y + n, z + n, x - n, y + n, z + n]
        return [x - n, y - n, z + n, x + n, y - n, z + n, x + n, y + n, z + n, x - n, y + n, z + n]

    @staticmethod
    def _quad_to_triangles(face: list[float]) -> list[float]:
        return face[0:9] + face[0:3] + face[6:12]

    def _face_uv_tiled(self, face_index: int, repeat_u: float, repeat_v: float) -> list[float]:
        uv = self._face_uv(face_index)
        tiled_uv: list[float] = []
        for i in range(0, len(uv), 3):
            u = uv[i] * repeat_u
            v = uv[i + 1] * repeat_v
            w = uv[i + 2]
            tiled_uv.extend((u, v, w))
        return self._quad_to_triangles(tiled_uv)

    def _jittered_color(self, pos: Vec3, block: str, shade: float = 1.0, vertex_count: int = 4) -> tuple[float, ...]:
        base = get_block_color(block)
        r = max(0.0, min(1.0, base[0] * shade))
        g = max(0.0, min(1.0, base[1] * shade))
        b = max(0.0, min(1.0, base[2] * shade))
        c = (r, g, b, 1.0)
        return c * vertex_count

    def _atlas_group_for_parent(self, parent_group: pyglet.graphics.Group) -> pyglet.graphics.Group:
        if self._atlas_texture is None:
            return parent_group
        key = id(parent_group)
        group = self._atlas_groups.get(key)
        if group is None:
            group = pyglet.graphics.TextureGroup(self._atlas_texture, parent=parent_group)
            self._atlas_groups[key] = group
        return group

    def _texture_group_for_texture(self, texture_name: str, parent_group: pyglet.graphics.Group) -> pyglet.graphics.TextureGroup | None:
        key = (texture_name, id(parent_group))
        cached_group = self._texture_groups.get(key)
        if cached_group is not None:
            return cached_group

        texture = self._texture_cache.get(texture_name)
        if texture is None:
            texture_path = TEXTURES_DIR / texture_name
            if not texture_path.is_file():
                return None
            texture = pyglet.image.load(str(texture_path)).get_texture()
            gl.glBindTexture(texture.target, texture.id)
            gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(texture.target, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameteri(texture.target, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
            self._texture_cache[texture_name] = texture

        group = pyglet.graphics.TextureGroup(texture, parent=parent_group)
        self._texture_groups[key] = group
        return group

    def _collect_texture_images(self) -> dict[str, pyglet.image.AbstractImage]:
        texture_names: set[str] = set()
        for definition in BLOCKS.values():
            for texture_name in (definition.texture, definition.texture_top, definition.texture_bottom, definition.texture_side):
                if texture_name:
                    texture_names.add(texture_name)

        images: dict[str, pyglet.image.AbstractImage] = {}
        for texture_name in sorted(texture_names):
            texture_path = TEXTURES_DIR / texture_name
            if not texture_path.is_file():
                continue
            image = pyglet.image.load(str(texture_path))
            if image.width != self.ATLAS_TILE_SIZE or image.height != self.ATLAS_TILE_SIZE:
                raise ValueError(
                    f"Texture '{texture_name}' must be {self.ATLAS_TILE_SIZE}x{self.ATLAS_TILE_SIZE}, "
                    f"got {image.width}x{image.height}"
                )
            images[texture_name] = image
        return images

    def _build_texture_array(self) -> None:
        images = self._collect_texture_images()
        if not images:
            return

        self._texture_layer_map = {name: idx for idx, name in enumerate(sorted(images.keys()))}
        layer_count = len(self._texture_layer_map)
        texture_id = gl.GLuint()
        gl.glGenTextures(1, texture_id)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, texture_id.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

        gl.glTexImage3D(
            gl.GL_TEXTURE_2D_ARRAY,
            0,
            gl.GL_RGBA8,
            self.ATLAS_TILE_SIZE,
            self.ATLAS_TILE_SIZE,
            layer_count,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )

        row_pitch = self.ATLAS_TILE_SIZE * 4
        for texture_name, layer in self._texture_layer_map.items():
            image = images[texture_name]
            image_data = image.get_image_data()
            raw = image_data.get_data("RGBA", row_pitch)
            gl.glTexSubImage3D(
                gl.GL_TEXTURE_2D_ARRAY,
                0,
                0,
                0,
                layer,
                self.ATLAS_TILE_SIZE,
                self.ATLAS_TILE_SIZE,
                1,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                raw,
            )

        self._texture_array_id = texture_id.value

    def _init_texture_array_pipeline(self) -> None:
        self._build_texture_array()
        if self._texture_array_id is None:
            self.use_texture_array = False
            return
        self._texture_array_shader = pyglet.gl.current_context.create_program(
            (self._TEXTURE_ARRAY_VERTEX_SOURCE, "vertex"),
            (self._TEXTURE_ARRAY_FRAGMENT_SOURCE, "fragment"),
        )
        self._texture_array_shader["block_texture_array"] = 0
        self._texture_array_shader_group = pyglet.graphics.ShaderGroup(program=self._texture_array_shader)
        self._texture_array_group = TextureArrayGroup(
            texture_id=self._texture_array_id,
            shader_program=self._texture_array_shader,
            parent=self._texture_array_shader_group,
        )

    def _atlas_uv_for_face(self, texture_name: str, face_index: int) -> list[float] | None:
        rect = self._atlas_rects.get(texture_name)
        if rect is None:
            return None
        cache_key = (texture_name, face_index)
        cached = self._atlas_face_uv_cache.get(cache_key)
        if cached is not None:
            return cached

        u0, v0, u1, v1 = rect
        local_uv = self._face_uv(face_index)
        atlas_uv: list[float] = []
        # Pyglet's default textured shader expects 3-component texcoords.
        # Preserve the third component while remapping only UV into atlas space.
        for i in range(0, len(local_uv), 3):
            lu = local_uv[i]
            lv = local_uv[i + 1]
            lw = local_uv[i + 2]
            atlas_uv.extend((u0 + (u1 - u0) * lu, v0 + (v1 - v0) * lv, lw))
        tri_uv = self._quad_to_triangles(atlas_uv)
        self._atlas_face_uv_cache[cache_key] = tri_uv
        return tri_uv

    def _cube_faces(self, x: float, y: float, z: float, n: float) -> list[list[float]]:
        return [
            self._face(x, y, z, n, 0),
            [x - n, y - n, z - n, x - n, y - n, z + n, x - n, y + n, z + n, x - n, y + n, z - n],
            [x - n, y + n, z - n, x - n, y + n, z + n, x + n, y + n, z + n, x + n, y + n, z - n],
            [x - n, y - n, z - n, x + n, y - n, z - n, x + n, y - n, z + n, x - n, y - n, z + n],
            self._face(x, y, z, n, 2),
            [x - n, y - n, z - n, x - n, y + n, z - n, x + n, y + n, z - n, x + n, y - n, z - n],
        ]

    @staticmethod
    def _face_cell(face_index: int, x: int, y: int, z: int) -> tuple[int, int, int]:
        if face_index == 0:  # +X
            return x + 1, z, y
        if face_index == 1:  # -X
            return x, z, y
        if face_index == 2:  # +Y
            return y + 1, x, z
        if face_index == 3:  # -Y
            return y, x, z
        if face_index == 4:  # +Z
            return z + 1, x, y
        return z, x, y  # -Z

    @staticmethod
    def _merged_face_quad(face_index: int, plane: int, u: int, v: int, w: int, h: int) -> list[float]:
        half = 0.5
        if face_index == 0:  # +X
            x = plane - half
            z0, z1 = u - half, u + w - half
            y0, y1 = v - half, v + h - half
            return [x, y0, z0, x, y1, z0, x, y1, z1, x, y0, z1]
        if face_index == 1:  # -X
            x = plane - half
            z0, z1 = u - half, u + w - half
            y0, y1 = v - half, v + h - half
            return [x, y0, z0, x, y0, z1, x, y1, z1, x, y1, z0]
        if face_index == 2:  # +Y
            y = plane - half
            x0, x1 = u - half, u + w - half
            z0, z1 = v - half, v + h - half
            return [x0, y, z0, x0, y, z1, x1, y, z1, x1, y, z0]
        if face_index == 3:  # -Y
            y = plane - half
            x0, x1 = u - half, u + w - half
            z0, z1 = v - half, v + h - half
            return [x0, y, z0, x1, y, z0, x1, y, z1, x0, y, z1]
        if face_index == 4:  # +Z
            z = plane - half
            x0, x1 = u - half, u + w - half
            y0, y1 = v - half, v + h - half
            return [x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z]

        z = plane - half  # -Z
        x0, x1 = u - half, u + w - half
        y0, y1 = v - half, v + h - half
        return [x0, y0, z, x0, y1, z, x1, y1, z, x1, y0, z]

    def _append_face(
        self,
        color_vertices: list[float],
        color_values: list[float],
        textured_vertices: dict[str, list[float]],
        textured_texcoords: dict[str, list[float]],
        textured_colors: dict[str, list[float]],
        position: Vec3,
        block: str,
        face_index: int,
        face: list[float],
        shade: float,
        repeat_u: float = 1.0,
        repeat_v: float = 1.0,
    ) -> None:
        tri_face = self._quad_to_triangles(face)
        texture_name = get_block_texture_for_face(block, face_index)
        if texture_name is None:
            color_vertices.extend(tri_face)
            color_values.extend(self._jittered_color(position, block, shade, vertex_count=6))
            return

        textured_vertices.setdefault(texture_name, []).extend(tri_face)
        textured_texcoords.setdefault(texture_name, []).extend(self._face_uv_tiled(face_index, repeat_u, repeat_v))
        textured_colors.setdefault(texture_name, []).extend((0.0, 0.0, 0.0, 0.0) * 6)

    def make_cube(
        self,
        position: Vec3,
        block: str,
        batch: pyglet.graphics.Batch,
        group: pyglet.graphics.Group,
        visible_faces: list[int] | None = None,
    ) -> RenderedCube:
        x, y, z = position
        n = 0.5
        shades = [0.86, 0.86, 1.0, 0.55, 0.72, 0.72]
        color_vertices: list[float] = []
        color_values: list[float] = []
        textured_vertices: dict[str, list[float]] = {}
        textured_texcoords: dict[str, list[float]] = {}
        textured_colors: dict[str, list[float]] = {}
        faces = self._cube_faces(x, y, z, n)
        face_indices = visible_faces if visible_faces is not None else list(range(len(faces)))

        for face_index in face_indices:
            self._append_face(
                color_vertices,
                color_values,
                textured_vertices,
                textured_texcoords,
                textured_colors,
                position,
                block,
                face_index,
                faces[face_index],
                shades[face_index],
            )

        return self.upload_chunk_mesh(
            ChunkMeshData(
                color_vertices=color_vertices,
                color_values=color_values,
                textured_vertices=textured_vertices,
                textured_texcoords=textured_texcoords,
                textured_colors=textured_colors,
            ),
            batch,
            group,
        )

    def build_chunk_mesh_data(
        self,
        positions: set[Vec3],
        blocks: dict[Vec3, str],
        solid_blocks: set[str],
    ) -> ChunkMeshData:
        color_vertices: list[float] = []
        color_values: list[float] = []
        textured_vertices: dict[str, list[float]] = {}
        textured_texcoords: dict[str, list[float]] = {}
        textured_colors: dict[str, list[float]] = {}
        shades = [0.86, 0.86, 1.0, 0.55, 0.72, 0.72]
        face_offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        face_planes: list[dict[int, dict[tuple[int, int], str]]] = [{}, {}, {}, {}, {}, {}]

        for x, y, z in positions:
            block = blocks.get((x, y, z))
            if block is None:
                continue
            for face_index, (dx, dy, dz) in enumerate(face_offsets):
                if blocks.get((x + dx, y + dy, z + dz)) in solid_blocks:
                    continue
                plane, u, v = self._face_cell(face_index, x, y, z)
                cells = face_planes[face_index].setdefault(plane, {})
                cells[(u, v)] = block

        for face_index, planes in enumerate(face_planes):
            for plane, cells in planes.items():
                visited: set[tuple[int, int]] = set()
                for u0, v0 in sorted(cells.keys(), key=lambda k: (k[1], k[0])):
                    if (u0, v0) in visited:
                        continue
                    block = cells[(u0, v0)]

                    width = 1
                    while cells.get((u0 + width, v0)) == block and (u0 + width, v0) not in visited:
                        width += 1

                    height = 1
                    while True:
                        next_v = v0 + height
                        row_ok = True
                        for du in range(width):
                            key = (u0 + du, next_v)
                            if cells.get(key) != block or key in visited:
                                row_ok = False
                                break
                        if not row_ok:
                            break
                        height += 1

                    for dv in range(height):
                        for du in range(width):
                            visited.add((u0 + du, v0 + dv))

                    merged_face = self._merged_face_quad(face_index, plane, u0, v0, width, height)
                    texture_name = get_block_texture_for_face(block, face_index)
                    if texture_name is not None and (width > 1 or height > 1) and not self.use_texture_array:
                        # Guaranteed no stretching: emit textured merged faces as tiled
                        # 1x1 quads until a shader-backed repeat path is verified.
                        for dv in range(height):
                            for du in range(width):
                                tiled_face = self._merged_face_quad(face_index, plane, u0 + du, v0 + dv, 1, 1)
                                self._append_face(
                                    color_vertices,
                                    color_values,
                                    textured_vertices,
                                    textured_texcoords,
                                    textured_colors,
                                    (u0 + du, v0 + dv, plane),
                                    block,
                                    face_index,
                                    tiled_face,
                                    shades[face_index],
                                )
                    else:
                        self._append_face(
                            color_vertices,
                            color_values,
                            textured_vertices,
                            textured_texcoords,
                            textured_colors,
                            (u0, v0, plane),
                            block,
                            face_index,
                            merged_face,
                            shades[face_index],
                            repeat_u=float(width),
                            repeat_v=float(height),
                        )

        return ChunkMeshData(
            color_vertices=color_vertices,
            color_values=color_values,
            textured_vertices=textured_vertices,
            textured_texcoords=textured_texcoords,
            textured_colors=textured_colors,
        )

    def upload_chunk_mesh(
        self,
        mesh_data: ChunkMeshData,
        batch: pyglet.graphics.Batch,
        group: pyglet.graphics.Group,
    ) -> RenderedCube:
        parts: list[pyglet.graphics.vertexdomain.VertexList] = []
        if mesh_data.color_vertices:
            parts.append(
                self.shader.vertex_list(
                    len(mesh_data.color_vertices) // 3,
                    gl.GL_TRIANGLES,
                    batch=batch,
                    group=group,
                    position=("f/static", mesh_data.color_vertices),
                    colors=("f/static", mesh_data.color_values),
                )
            )

        if self.use_texture_array and self._texture_array_shader is not None and self._texture_array_group is not None:
            array_vertices: list[float] = []
            array_texcoords: list[float] = []
            array_colors: list[float] = []
            array_layers: list[float] = []
            for texture_name, vertices in mesh_data.textured_vertices.items():
                if not vertices:
                    continue
                layer = self._texture_layer_map.get(texture_name)
                if layer is None:
                    continue
                texcoords = mesh_data.textured_texcoords.get(texture_name, [])
                colors = mesh_data.textured_colors.get(texture_name, [])
                vertex_count = len(vertices) // 3
                array_vertices.extend(vertices)
                array_texcoords.extend(texcoords)
                array_colors.extend(colors)
                array_layers.extend([float(layer)] * vertex_count)

            if array_vertices:
                parts.append(
                    self._texture_array_shader.vertex_list(
                        len(array_vertices) // 3,
                        gl.GL_TRIANGLES,
                        batch=batch,
                        group=self._texture_array_group,
                        position=("f/static", array_vertices),
                        colors=("f/static", array_colors),
                        tex_coords=("f/static", array_texcoords),
                        tex_layer=("f/static", array_layers),
                    )
                )
            return RenderedCube(parts)

        for texture_name, vertices in mesh_data.textured_vertices.items():
            if not vertices:
                continue
            texture_group = self._texture_group_for_texture(texture_name, group)
            if texture_group is None:
                continue
            texcoords = mesh_data.textured_texcoords.get(texture_name, [])
            colors = mesh_data.textured_colors.get(texture_name, [])
            parts.append(
                self.shader.vertex_list(
                    len(vertices) // 3,
                    gl.GL_TRIANGLES,
                    batch=batch,
                    group=texture_group,
                    position=("f/static", vertices),
                    colors=("f/static", colors),
                    tex_coords=("f/static", texcoords),
                )
            )

        return RenderedCube(parts)

    def make_chunk_mesh(
        self,
        positions: set[Vec3],
        blocks: dict[Vec3, str],
        batch: pyglet.graphics.Batch,
        group: pyglet.graphics.Group,
        solid_blocks: set[str],
    ) -> RenderedCube:
        mesh_data = self.build_chunk_mesh_data(positions, blocks, solid_blocks)
        return self.upload_chunk_mesh(mesh_data, batch, group)
