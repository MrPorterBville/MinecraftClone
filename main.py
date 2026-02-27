import argparse

import pyglet

from engine.game.window import GameWindow
from engine.graphics.rendering import setup_gl


def run(seed: int = 90125, use_texture_array: bool = False) -> None:
    window = GameWindow(seed=seed, use_texture_array=use_texture_array)
    setup_gl()
    pyglet.app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python Minecraft Clone")
    parser.add_argument("--seed", type=int, default=90125, help="Terrain seed (same seed => same world)")
    parser.add_argument(
        "--use-texture-array",
        action="store_true",
        help="Experimental fast textured meshing path (may show UV artifacts on some systems)",
    )
    args = parser.parse_args()
    run(seed=args.seed, use_texture_array=args.use_texture_array)
