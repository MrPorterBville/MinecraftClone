# block.py

class Block:
    def __init__(self, block_data):
        """
        Takes a dictionary (from a JSON) and sets up the block properties.
        """
        # Identification
        self.id = block_data.get("id", "minecraft:air")
        self.name = block_data.get("name", "Unnamed Block")

        
        
        # Physics/Logic
        self.solid = True
        self.transparent = False
        self.hardness = block_data.get("hardness", 1.0)
        
        # Visuals
        self.texture_name = block_data.get("texture", "missing_texture.png")
        tex_data = block_data.get("textures", {})
        if isinstance(tex_data, str):
            self.tex_top = self.tex_bottom = self.tex_sides = tex_data
        else:
            # .get(key, default) is your best friend here!
            self.tex_top = tex_data.get("top", "default.png")
            self.tex_bottom = tex_data.get("bottom", self.tex_top)
            self.tex_sides = tex_data.get("side", self.tex_top)

        tex = block_data.get("textures", {})
        # Create a map for the renderer (0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z)
        # We use .get() to fall back to a 'default' or 'side' texture
        side = tex.get("side", tex.get("all", "missing.png"))
        
        self.faces = {
            0: tex.get("right", side),
            1: tex.get("left", side),
            2: tex.get("top", tex.get("all", "missing.png")),
            3: tex.get("bottom", tex.get("all", "missing.png")),
            4: tex.get("front", side),
            5: tex.get("back", side)
        }

    def __repr__(self):
        return f"<Block: {self.id}>"