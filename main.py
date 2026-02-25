import pyglet

from engine.game.window import GameWindow
from engine.graphics.rendering import setup_gl


def run() -> None:
    window = GameWindow()
    setup_gl()
    pyglet.app.run()


if __name__ == "__main__":
    run()
