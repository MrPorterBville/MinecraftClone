import pyglet
from pyglet import gl

from engine.blocks import get_block_color, get_block_texture_for_face
from engine.blocks.registry import TEXTURES_DIR
from engine.constants import Vec3


class RenderedCube:
    def __init__(self, parts: list[pyglet.graphics.vertexdomain.VertexList]) -> None:
        self.parts = parts

    def delete(self) -> None:
        for part in self.parts:
            part.delete()


class BlockRenderer:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.shader = pyglet.graphics.get_default_shader()
        self._texture_groups: dict[str, pyglet.graphics.TextureGroup] = {}

    @staticmethod
    def _face_uv(face_index: int) -> list[float]:
        # 4 UVs as vec3 (u, v, layer), matched to each face's vertex order.
        # Side faces are oriented so texture V always maps bottom->top in world Y.
        uv_by_face = {
            0: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # +X
            1: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # -X
            2: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # Top
            3: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # Bottom
            4: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # +Z
            5: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # -Z
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

    def _jittered_color(self, pos: Vec3, block: str, shade: float = 1.0, vertex_count: int = 4) -> tuple[float, ...]:
        base = get_block_color(block)
        r = max(0.0, min(1.0, base[0] * shade))
        g = max(0.0, min(1.0, base[1] * shade))
        b = max(0.0, min(1.0, base[2] * shade))
        c = (r, g, b, 1.0)
        return c * vertex_count

    def _texture_group_for_texture(self, texture_name: str, parent_group: pyglet.graphics.Group) -> pyglet.graphics.TextureGroup:
        cached = self._texture_groups.get(texture_name)
        if cached is not None:
            return cached

        texture_path = TEXTURES_DIR / texture_name
        texture = pyglet.image.load(str(texture_path)).get_texture()
        gl.glBindTexture(texture.target, texture.id)
        gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        group = pyglet.graphics.TextureGroup(texture, parent=parent_group)
        self._texture_groups[texture_name] = group
        return group

    def make_cube(
        self,
        position: Vec3,
        block: str,
        batch: pyglet.graphics.Batch,
        group: pyglet.graphics.Group,
    ) -> RenderedCube:
        x, y, z = position
        n = 0.5
        color_vertices: list[float] = []
        color_values: list[float] = []
        textured_vertices: dict[str, list[float]] = {}
        textured_texcoords: dict[str, list[float]] = {}
        textured_colors: dict[str, list[float]] = {}
        faces = [
            self._face(x, y, z, n, 0),
            [x - n, y - n, z - n, x - n, y - n, z + n, x - n, y + n, z + n, x - n, y + n, z - n],
            self._face(x, y, z, n, 1),
            [x - n, y - n, z - n, x + n, y - n, z - n, x + n, y - n, z + n, x - n, y - n, z + n],
            self._face(x, y, z, n, 2),
            [x - n, y - n, z - n, x - n, y + n, z - n, x + n, y + n, z - n, x + n, y - n, z - n],
        ]
        shades = [0.86, 0.86, 1.0, 0.55, 0.72, 0.72]
        for face_index, (face, shade) in enumerate(zip(faces, shades)):
            tri_face = self._quad_to_triangles(face)
            texture_name = get_block_texture_for_face(block, face_index)
            if texture_name is None:
                color_vertices.extend(tri_face)
                color_values.extend(self._jittered_color(position, block, shade, vertex_count=6))
            else:
                textured_vertices.setdefault(texture_name, []).extend(tri_face)
                textured_texcoords.setdefault(texture_name, []).extend(self._quad_to_triangles(self._face_uv(face_index)))
                textured_colors.setdefault(texture_name, []).extend((0.0, 0.0, 0.0, 0.0) * 6)

        parts: list[pyglet.graphics.vertexdomain.VertexList] = []
        if color_vertices:
            parts.append(
                self.shader.vertex_list(
                    len(color_vertices) // 3,
                    gl.GL_TRIANGLES,
                    batch=batch,
                    group=group,
                    position=("f/static", color_vertices),
                    colors=("f/static", color_values),
                )
            )

        for texture_name, vertices in textured_vertices.items():
            texture_group = self._texture_group_for_texture(texture_name, group)
            parts.append(
                self.shader.vertex_list(
                    len(vertices) // 3,
                    gl.GL_TRIANGLES,
                    batch=batch,
                    group=texture_group,
                    position=("f/static", vertices),
                    colors=("f/static", textured_colors[texture_name]),
                    tex_coords=("f/static", textured_texcoords[texture_name]),
                )
            )

        return RenderedCube(parts)
