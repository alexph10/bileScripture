import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from PIL import Image, ImageOps
from pydantic import BaseModel

EXPORT_DIR = Path(os.getenv("BS_EXPORT_DIR", Path.cwd() / "export"))
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = os.getenv("BS_MODEL_PATH", str(Path.cwd() / "models" / "unet.onnx"))

app = FastAPI(title="API", version="0.1.0")


class PackRequest(BaseModel):
    name: str = "demo_material"
    size: int = 512


def _solid(size: int, gray: int) -> Image.Image:
    return Image.new("L", (size, size), color=int(gray))


def _checker(size: int, tile: int = 64) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    px = img.load()
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                for yy in range(y, min(y + tile, size)):
                    for xx in range(x, (min(x + tile, size))):
                        px[xx, yy] = (205, 205, 205)
    return img


@app.get("/health")  # type: ignore[misc]
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "ts": time.time(),
        "export_dir": str(EXPORT_DIR),
        "model_path": MODEL_PATH,
    }


@app.post("/pack_pbr")  # type: ignore[misc]
def pack_pbr(req: PackRequest) -> dict[str, Any]:
    root = EXPORT_DIR / req.name
    root.mkdir(parents=True, exist_ok=True)

    albedo_path = root / "albedo.png"

    _checker(req.size, max(8, req.size // 8)).save(albedo_path, "PNG")

    roughness_path = root / "roughness.png"
    _solid(req.size, 180).save(roughness_path, "PNG")

    metallic_path = root / "metallic.png"
    _solid(req.size, 10).save(metallic_path, "PNG")

    ao_path = root / "ao.png"
    _solid(req.size, 220).save(ao_path, "PNG")

    normal = Image.merge(
        "RGB",
        (
            Image.new("L", (req.size, req.size), 128),
            Image.new("L", (req.size, req.size), 128),
            Image.new("L", (req.size, req.size), 255),
        ),
    )
    normal_path = root / "normal.png"
    normal.save(normal_path, "PNG")

    height16 = ImageOps.autocontrast(
        Image.linear_gradient("L").resize((req.size, req.size))
    ).convert("I;16")
    height_path = root / "height.png"
    height16.save(height_path, "PNG")

    manifest = {
        "albedo": "albedo.png",
        "roughness": "roughness.png",
        "metallic": "metallic.png",
        "ao": "ao.png",
        "normal": "normal.png",
        "height": "height.png",
        "color_spaces": {"albedo": "sRGB", "others": "Non-Color"},
        "tile_size": req.size,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {"ok": True, "export_dir": str(root)}
