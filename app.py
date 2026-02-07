import io
import os
from collections import OrderedDict
from typing import List

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import Response
from PIL import Image
from rembg import remove, new_session

# ----------------------
# Model configuration
# ----------------------

KNOWN_MODELS: List[str] = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "sam",
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-portrait",
    "birefnet-dis",
    "birefnet-hrsod",
    "birefnet-cod",
    "birefnet-massive",
    "bria-rmbg",
]

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "birefnet-general-lite")
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1"))

# simple LRU cache of rembg model sessions
_session_cache: "OrderedDict[str, object]" = OrderedDict()


def get_session(model: str):
    """
    Return a cached rembg session for the given model,
    creating and caching it if needed.
    """
    if model not in KNOWN_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Must be one of: {', '.join(KNOWN_MODELS)}",
        )

    if model in _session_cache:
        _session_cache.move_to_end(model)
        return _session_cache[model]

    # rembg will automatically use GPU if installed with [gpu] and the environment supports it
    sess = new_session(model_name=model)
    _session_cache[model] = sess
    _session_cache.move_to_end(model)

    while len(_session_cache) > MAX_SESSIONS:
        _session_cache.popitem(last=False)

    return sess


# ----------------------
# Image utilities
# ----------------------

def ensure_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode != "RGBA" else img


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def upscale_image(img: Image.Image, scale: int) -> Image.Image:
    """
    Simple Pillow-based upscaling with LANCZOS.
    scale=1 => no change
    """
    if scale <= 1:
        return img

    w, h = img.size
    return img.resize((w * scale, h * scale), resample=Image.Resampling.LANCZOS)


# ----------------------
# FastAPI app
# ----------------------

app = FastAPI(title="rembg-gpu-api", version="1.0.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models")
def models():
    return {"models": KNOWN_MODELS, "default": DEFAULT_MODEL}


@app.post("/remove")
async def remove_background(
    image: UploadFile = File(...),
    model: str = Query(
        DEFAULT_MODEL,
        description="Background removal model. Must be one of /models output.",
    ),
    upscale: int = Query(
        2,
        ge=1,
        le=4,
        description="Upscale factor for the output image (1=original, 2, 3, 4).",
    ),
):
    """
    Remove the background from an image and upscale the result.

    - Input: any common image format (jpg/png/webp/etc.)
    - Output: PNG with alpha channel (transparent background), optionally upscaled.
    """
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    # Get / create model session
    sess = get_session(model)

    # Step 1: remove background (rembg returns PNG bytes)
    try:
        out_bytes = remove(data, session=sess)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {e}")

    # Step 2: decode and upscale
    try:
        img = ensure_rgba(Image.open(io.BytesIO(out_bytes)))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to decode processed image from background remover.",
        )

    img = upscale_image(img, scale=upscale)
    out_png = pil_to_png_bytes(img)

    return Response(
        content=out_png,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{image.filename or "output"}.png"'},
    )
