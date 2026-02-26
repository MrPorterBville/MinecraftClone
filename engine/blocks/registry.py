from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BlockDefinition:
    name: str
    solid: bool
    color: tuple[float, float, float]
    texture: str | None
    texture_top: str | None
    texture_bottom: str | None
    texture_side: str | None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DIR = PROJECT_ROOT / "Blocks"
TEXTURES_DIR = BLOCKS_DIR / "Textures"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_color(value: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("color must have 3 comma-separated components")

    raw = [float(p) for p in parts]
    if any(c > 1.0 for c in raw):
        raw = [c / 255.0 for c in raw]

    return tuple(max(0.0, min(1.0, c)) for c in raw)  # type: ignore[return-value]


def _load_block_file(path: Path) -> BlockDefinition:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip().lower()] = value.strip()

    name = data.get("name", path.stem)
    solid = _parse_bool(data.get("solid", "true"))
    color = _parse_color(data.get("color", "1.0,0.0,1.0"))
    texture = data.get("texture")
    texture_top = data.get("texture_top")
    texture_bottom = data.get("texture_bottom")
    texture_side = data.get("texture_side")

    return BlockDefinition(
        name=name,
        solid=solid,
        color=color,
        texture=texture,
        texture_top=texture_top,
        texture_bottom=texture_bottom,
        texture_side=texture_side,
    )


def load_block_definitions() -> dict[str, BlockDefinition]:
    definitions: dict[str, BlockDefinition] = {}
    if BLOCKS_DIR.is_dir():
        for path in sorted(BLOCKS_DIR.glob("*.txt")):
            block = _load_block_file(path)
            definitions[block.name] = block

    return definitions


BLOCKS = load_block_definitions()
SOLID_BLOCKS = {name for name, block in BLOCKS.items() if block.solid}


def get_block_color(name: str) -> tuple[float, float, float]:
    block = BLOCKS.get(name)
    if block is None:
        return 1.0, 0.0, 1.0
    # Texture rendering can use block.texture when present.
    # If texture is missing, fall back to the solid color from the block file.
    return block.color


def get_block_definition(name: str) -> BlockDefinition | None:
    return BLOCKS.get(name)


def get_block_texture_for_face(name: str, face_index: int) -> str | None:
    block = BLOCKS.get(name)
    if block is None:
        return None

    candidates: list[str] = []
    if face_index == 2 and block.texture_top:
        candidates.append(block.texture_top)
    elif face_index == 3 and block.texture_bottom:
        candidates.append(block.texture_bottom)
    elif face_index in (0, 1, 4, 5) and block.texture_side:
        candidates.append(block.texture_side)

    if block.texture:
        candidates.append(block.texture)

    for texture_name in candidates:
        if (TEXTURES_DIR / texture_name).is_file():
            return texture_name

    return None
