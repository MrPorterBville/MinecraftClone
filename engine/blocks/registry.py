from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class BlockDefinition:
    name: str
    solid: bool
    occludes: bool
    breakable: bool
    color: tuple[float, float, float]
    # index 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
    face_textures: dict[int, str | None]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKS_DIR = PROJECT_ROOT / "Blocks"
TEXTURES_DIR = BLOCKS_DIR / "Textures"

def _load_block_json(path: Path) -> BlockDefinition:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize name and properties
    name = data.get("name", path.stem)
    solid = data.get("solid", True)
    occludes = data.get("occludes", solid)
    breakable = data.get("breakable", True)
    
    raw_color = data.get("color", [1.0, 0.0, 1.0])
    color = tuple(raw_color[:3])

    # --- Robust Texture Handling ---
    tex_input = data.get("textures", {})
    
    if isinstance(tex_input, str):
        # Case 1: "textures": "stone.png" -> apply to all sides
        all_tex = tex_input
        side_tex = tex_input
        tex_dict = {} 
    else:
        # Case 2: "textures": {"all": "stone.png", "top": "grass.png"}
        tex_dict = tex_input
        all_tex = tex_dict.get("all")
        side_tex = tex_dict.get("side", all_tex)
    
    face_map = {
        0: tex_dict.get("right", side_tex),
        1: tex_dict.get("left", side_tex),
        2: tex_dict.get("top", all_tex),
        3: tex_dict.get("bottom", all_tex),
        4: tex_dict.get("front", side_tex),
        5: tex_dict.get("back", side_tex)
    }

    return BlockDefinition(
        name=name,
        solid=solid,
        occludes=occludes,
        breakable=breakable,
        color=color,
        face_textures=face_map
    )

def load_block_definitions() -> dict[str, BlockDefinition]:
    definitions: dict[str, BlockDefinition] = {}
    
    # DIAGNOSTIC 1: Where are we looking?
    #print(f"--- Registry Search ---")
    #print(f"Looking in: {BLOCKS_DIR.absolute()}")
    #print(f"Directory exists: {BLOCKS_DIR.is_dir()}")

    if BLOCKS_DIR.is_dir():
        json_files = list(BLOCKS_DIR.glob("*.json"))
        # DIAGNOSTIC 2: Did we find files?
        #print(f"Found {len(json_files)} .json files")
        
        for path in sorted(json_files):
            block = _load_block_json(path)
            block_id = block.name.lower()
            definitions[block_id] = block
            #print(f"Successfully registered: {block.name}")

    if not definitions:
        print("!!! WARNING: No block definitions were loaded!")
        
    return definitions

# --- Registry Initialization ---
BLOCKS = load_block_definitions()
SOLID_BLOCKS = {name.lower() for name, block in BLOCKS.items() if block.solid}
OCCLUDING_BLOCKS = {name.lower() for name, block in BLOCKS.items() if block.occludes}
#print(f"Registry initialized. Loaded {len(BLOCKS)} blocks. Solid blocks: {SOLID_BLOCKS}")

# --- Helper Functions for the Renderer ---

def get_block_texture_for_face(name: str, face_index: int) -> str | None:
    #print(f'Getting Texture Name: {name}, {face_index}')
    block = BLOCKS.get(name)
    if not block:
        #print(f'{block} not found.')
        return None
    
    texture_name = block.face_textures.get(face_index)
    
    if texture_name and (TEXTURES_DIR / texture_name).is_file():
        #print(f'{block} found: {texture_name}.')
        return texture_name
    #print(f'System Failure.')
    return None

def get_block_color(name: str) -> tuple[float, float, float]:
    block = BLOCKS.get(name)
    return block.color if block else (1.0, 0.0, 1.0)

def get_block_definition(name: str) -> BlockDefinition | None:
    return BLOCKS.get(name)
