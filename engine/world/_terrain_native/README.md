# Native Terrain Kernel (Rust)

Terrain generation is compiled in this Rust extension. Python runtime requires it.

## Build (VS Code, Windows)

1. Select project interpreter in VS Code:
- `D:\Codex Minecraft Clone\MinecraftClone\.venv\Scripts\python.exe`

2. From project root, install build tool:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade maturin
```

3. Build/install extension into the same `.venv`:

```powershell
cd engine\world\_terrain_native
.\..\..\..\.venv\Scripts\python.exe -m maturin develop --release
cd ..\..\..
```

4. Run game:

```powershell
.\.venv\Scripts\python.exe main.py
```
