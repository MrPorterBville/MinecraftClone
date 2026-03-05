# Python OpenGL Minecraft-Style Clone

This project is a pure-Python voxel sandbox inspired by classic Minecraft, using **OpenGL via pyglet** for GPU rendering.

## Features implemented

- Seed-based terrain generation with hills, beaches, water layers, and trees.
- Block world with face culling for performance.
- Pixel-like block color variation by jittering per-block face colors.
- First-person controls with gravity, jumping, flying toggle, and collision.
- Breaking and placing blocks with ray-cast targeting.
- Inventory with hotbar selection (`1`-`9`).
- Crafting system (`E`) with sample recipes.
- Basic mobs that wander the world.

## Controls

- `WASD`: Move
- `Mouse`: Look
- `Left Click`: Break block
- `Right Click`: Place selected block
- `1`..`9`: Select hotbar slot
- `Space`: Jump (or ascend while flying)
- `Tab`: Toggle fly mode
- `E`: Craft first available recipe
- `Esc`: Release mouse capture

## Run

```bash
python -m venv .venv
pip install -r requirements.txt
python main.py
```

In VS Code, set the interpreter to `.venv` so Run/F5 launches with the correct environment.

## Notes

This is intentionally implemented in Python for game logic and world simulation, while rendering is handled by OpenGL on the GPU.

## Rust Terrain Kernel

Terrain generation is compiled in Rust and loaded as a Python extension.

- Source: [`engine/world/_terrain_native/`](engine/world/_terrain_native/)
- Build docs: [`engine/world/_terrain_native/README.md`](engine/world/_terrain_native/README.md)
