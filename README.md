# # bileScripture

End-to-end toolkit for generating tileable textures and complete PBR (physically based rendering) map packs with a style control pipeline. The project exposes a FastAPI service used by a Blender material workflow. Containerization is optional.

## Scope

- Generate new tileable albedo textures (seeded, size-configurable).
- Predict or provide PBR maps: normal, roughness, ambient occlusion (AO), optional height.
- Export a Blender-ready bundle and a manifest file.
- Provide an HTTP API for automation and a future Blender add-on.
- Future work (non-breaking): seam/palette metrics, U-Net map predictor, style-steered generator, GPU serving, add-on UI, CI.

## Current status (MVP)

- FastAPI application with `/health` and `/pack_pbr` endpoints.
- `/pack_pbr` writes a deterministic demo texture pack to `export/<name>/`.
- Neutral interface that will remain stable as models are added.

## Architecture (high level)

- **Service layer**: FastAPI app (`src/bile_scripture/serving/api.py`).
- **Utilities**: image generation and I/O helpers (to be expanded under `src/bile_scripture/utils/`).
- **Artifacts**: written to `export/` by default; configurable via environment variables.
- **Blender**: material builder script and a planned add-on import path (future).
- **Container** (optional): Docker image for a reproducible runtime.

## Project structure

```text
repo-root/
  src/bile_scripture/
    serving/
      api.py                 # FastAPI endpoints (/health, /pack_pbr)
    utils/                   # seam metric, palette extraction (future)
    models/                  # model code and wrappers (future)
    training/                # training scripts (future)
    blender/                 # material builder helpers (future)
  blender_addon/             # add-on package (future)
  export/                    # generated outputs (gitignored)
  data/                      # datasets (gitignored)
  models/                    # weights and ONNX files (gitignored)
  tests/                     # pytest tests (optional)
  docker/
    Dockerfile               # optional container build (future-ready)
  requirements.txt           # runtime dependencies (API)
  requirements-dev.txt       # dev-only tooling (ruff/black/mypy/pytest)
  requirements-ml.txt        # ML utilities (optuna/mlflow/etc.)
  requirements-dl.txt        # deep learning stack (CPU-first)
  requirements-dl-gpu.txt    # deep learning stack (CUDA wheels)
  .dockerignore              # excludes data/models/exports from images
  .gitignore                 # excludes virtualenv, caches, large artifacts
  .gitattributes             # Git LFS patterns for images/models (optional)
  README.md                  # this document
```

## Requirements

- Python 3.11
- Windows, macOS, or Linux
- Optional: Docker Desktop (WSL2 backend on Windows) for containerized runs
- Recommended Python virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux/macOS (venv setup)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Installation

Install minimal runtime dependencies for the API:

```bash
pip install -r requirements.txt
```

Optional developer tooling:

```bash
pip install -r requirements-dev.txt
```

Optional ML utilities:

```bash
pip install -r requirements-ml.txt
```

Optional deep learning stack (CPU-first):

```bash
pip install -r requirements-dl.txt
```

Optional deep learning stack (GPU, CUDA 12.1 wheels):

```bash
pip install -r requirements-dl-gpu.txt
```

## Running the API (local, no Docker)

Start the server:

```bash
python -m uvicorn bile_scripture.serving.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Generate a demo PBR pack named demo:

```bash
curl -X POST http://localhost:8000/pack_pbr \
  -H "Content-Type: application/json" \
  -d '{"name":"demo","size":512}'
```

Output will be written to:

```text
export/demo/
  albedo.png
  roughness.png
  metallic.png
  ao.png
  normal.png
  height.png
  manifest.json
```

## Environment variables

- `BS_EXPORT_DIR` (default: `./export`): directory for output bundles.
- `BS_MODEL_PATH` (default: `./models/unet.onnx`): model path for inference when predictors are added.

Set via shell, e.g.:

### Linux/macOS

```bash
export BS_EXPORT_DIR="/absolute/path/to/export"
export BS_MODEL_PATH="/absolute/path/to/models/unet.onnx"
```

### Windows PowerShell

```powershell
$Env:BS_EXPORT_DIR = "C:\path\to\export"
$Env:BS_MODEL_PATH = "C:\path\to\models\unet.onnx"
```

## API reference (MVP)

### GET /health

Response 200:

```json
{
  "ok": true,
  "ts": 1710000000.0,
  "export_dir": "/absolute/path",
  "model_path": "/absolute/path/or/default"
}
```

### POST /pack_pbr

Create a PBR texture pack.

Request body:

```json
{
  "name": "demo_material",
  "size": 512
}
```

- `name`: directory name under BS_EXPORT_DIR.
- `size`: output resolution (square, in pixels).

Response 200:

```json
{
  "ok": true,
  "export_dir": "/absolute/path/export/demo_material"
}
```

### Files produced

- `albedo.png` (sRGB)
- `roughness.png` (Non-Color)
- `metallic.png` (Non-Color)
- `ao.png` (Non-Color)
- `normal.png` (Non-Color; tangent-space)
- `height.png` (Non-Color; 16‑bit)
- `manifest.json` (map names, color spaces, size)

The API contract will remain stable as the internal generators and predictors are introduced.

## Output specification

### Color spaces

- **Albedo/Base Color**: sRGB
- **Roughness, Metallic, AO, Normal, Height**: Non‑Color

### Bit depths

- PNG 8‑bit for albedo, roughness, metallic, AO, normal
- PNG 16‑bit (or EXR float in future) for height/displacement

### Tiling

Images are produced tileable when generation modules are enabled. The demo pack is neutral and tiling-compatible.

### Manifest

JSON file containing map filenames, declared color spaces, and tile_size.

## Blender usage (principled PBR)

Load maps as follows:

- **Base Color**: `albedo.png` → Image Texture (Color Space: sRGB) → Principled BSDF Base Color
- **Roughness/Metallic/AO/Normal/Height**: Image Texture (Color Space: Non-Color)
- **AO** is typically multiplied into Base Color in the shader tree
- **Normal**: Image Texture → Normal Map node → Principled Normal
- **Height**: connect through a Displacement node (Cycles: set Material Settings → Displacement = Displacement and Bump)

A Blender add-on and node group will be provided in a later milestone; the file formats and color spaces above are already aligned with Blender expectations.

## Optional: Docker usage (later)

Build image (CPU runtime):

```bash
docker build -f docker/Dockerfile -t bile-scripture:dev .
```

Run and expose the API; bind-mount exports to the host:

### PowerShell

```powershell
mkdir export
docker run --rm -it -p 8000:8000 -v "${PWD}\export:/app/export" bile-scripture:dev
```

### Bash/WSL/Linux/macOS

```bash
mkdir -p export
docker run --rm -it -p 8000:8000 -v "$PWD/export:/app/export" bile-scripture:dev
```

GPU variant will be introduced when model inference is enabled. Model weights should be mounted (e.g., `-v "$PWD/models:/models" -e BS_MODEL_PATH=/models/unet.onnx`).

## Development

### Style and checks

If dev tools are installed:

```bash
ruff check .
black --check .
isort --check-only .
mypy src
pytest -q
```

### Git LFS (recommended for large assets)

`.gitattributes` example:

```gitattributes
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.tif filter=lfs diff=lfs merge=lfs -text
*.exr filter=lfs diff=lfs merge=lfs -text
*.blend filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
```

### Data/versioning (optional)

Use DVC or a cloud bucket for datasets and models. Keep indexes and metadata in Git, large binaries out of Git.

## Roadmap (non-binding)

- Seam metric and palette extraction utilities.
- Supervised map predictor (U‑Net), ONNX export, and API integration.
- Style-steered generator with tileable constraints.
- Blender add-on with import panel and parameter controls.
- GPU inference image; CI build + health check; model registry.

## Security and privacy

- Do not commit secrets or proprietary datasets.
- Prefer environment variables for configuration.
- Keep large binaries and models outside the image; mount at runtime.

## License

MIT License (see LICENSE).
