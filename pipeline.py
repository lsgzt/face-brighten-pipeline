"""Low-memory portrait complexion adjustment service.

The service deliberately produces a web-sized result.  Processing camera originals at
full resolution is the main cause of memory failures on small (512 MB) instances.
"""
import io
import os
import sys
import traceback
from contextlib import suppress

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

# These are deliberately conservative for a 512 MB Render instance.  A 1600x1000
# BGR image is about 4.8 MB; all OpenCV/Numpy working buffers then remain modest.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 12 * 1024 * 1024))
MAX_DECODE_PIXELS = int(os.getenv("MAX_DECODE_PIXELS", 12_000_000))
MAX_PROCESS_PIXELS = int(os.getenv("MAX_PROCESS_PIXELS", 1_600_000))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 88))

# Avoid OpenCV allocating several large thread-local workspaces on a small instance.
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS

app = FastAPI(title="Portrait Complexion Lightening API", version="2.0.0")


def _web_size(image: np.ndarray) -> np.ndarray:
    """Downscale before *any* expensive processing, preserving aspect ratio."""
    height, width = image.shape[:2]
    pixels = height * width
    if pixels <= MAX_PROCESS_PIXELS:
        return image
    scale = (MAX_PROCESS_PIXELS / pixels) ** 0.5
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def get_skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Return a feathered, single-channel skin mask using in-place operations."""
    height, width = image_bgr.shape[:2]
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    del ycrcb

    # Prefer the central portrait region, but fall back for off-centre portraits.
    ellipse = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(ellipse, (width // 2, int(height * 0.45)),
                (int(width * 0.35), int(height * 0.45)), 0, 0, 360, 255, -1)
    cv2.bitwise_and(mask, ellipse, dst=mask)
    if cv2.countNonZero(mask) < 500:
        # Recreate the threshold directly rather than retaining a second colour image.
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        cv2.inRange(ycrcb, lower, upper, dst=mask)
        del ycrcb
    del ellipse

    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cv2.morphologyEx(mask, cv2.MORPH_OPEN, small, dst=mask)
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, large, dst=mask)
    # dst avoids allocating another full-resolution mask.
    cv2.GaussianBlur(mask, (0, 0), 5, dst=mask)
    return mask


def lighten_complexion(image_bgr: np.ndarray, lightness_boost: float = 25.0,
                       whitening_tone: float = 10.0, smooth_skin: bool = True) -> np.ndarray:
    """Apply the effect using bounded-size, mostly single-channel temporary arrays."""
    mask = get_skin_mask(image_bgr)

    base = image_bgr
    if smooth_skin:
        # At the bounded processing size this is inexpensive; do not retain a full-size
        # half-resolution/upscaled pair as the previous implementation did.
        smoothed = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=30, sigmaSpace=30)
        softened = cv2.addWeighted(image_bgr, 0.70, smoothed, 0.30, 0)
        base = image_bgr.copy()
        # A masked copy is much cheaper than a 3-channel float alpha mask.
        cv2.copyTo(softened, mask, base)
        del smoothed, softened

    lab = cv2.cvtColor(base, cv2.COLOR_BGR2Lab)
    if base is not image_bgr:
        del base

    # One float mask (not the former HxWx3 float mask) is sufficient.
    alpha = mask.astype(np.float32)
    alpha *= 1.0 / 255.0
    lightness = lab[:, :, 0].astype(np.float32)
    lightness += lightness_boost * (1.0 - 0.4 * lightness / 255.0) * alpha
    lab[:, :, 0] = np.clip(lightness, 0, 255).astype(np.uint8)
    del lightness

    tone = lab[:, :, 2].astype(np.float32)
    tone -= (whitening_tone * 0.4) * alpha
    lab[:, :, 2] = np.clip(tone, 0, 255).astype(np.uint8)
    del tone

    adjusted = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    del lab

    # Blend channel-by-channel, avoiding several HxWx3 float work arrays.
    output = image_bgr.copy()
    for channel in range(3):
        blended = image_bgr[:, :, channel].astype(np.float32)
        blended += (adjusted[:, :, channel].astype(np.float32) - blended) * alpha
        output[:, :, channel] = np.clip(blended, 0, 255).astype(np.uint8)
    return output


def _validate_image_header(contents: bytes) -> None:
    try:
        with Image.open(io.BytesIO(contents)) as probe:
            width, height = probe.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Upload a valid JPEG, PNG, or WebP image.") from exc
    if width <= 0 or height <= 0 or width * height > MAX_DECODE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum decoded size is {MAX_DECODE_PIXELS:,} pixels.",
        )


@app.get("/", response_class=HTMLResponse)
def root():
    return """<!doctype html><html><head><title>Portrait Complexion Lightener</title>
<style>body{font-family:Arial;margin:40px;background:#f4f4f9;color:#333}.container{max-width:700px;margin:auto;background:#fff;padding:30px;border-radius:8px}label{display:block;font-weight:bold;margin-top:16px}input,select{width:100%;box-sizing:border-box;padding:8px}button{margin-top:20px;background:#0878d1;color:#fff;border:0;padding:11px 18px;border-radius:4px}.results{display:flex;gap:20px;flex-wrap:wrap;margin-top:25px}.result-box{max-width:320px}img{max-width:100%;border:1px solid #ddd}#error{color:#b00020;white-space:pre-wrap;margin-top:15px}</style>
</head><body><div class=container><h1>Portrait Complexion Lightener</h1><p>Large photos are safely resized for fast web output.</p><form id=f><label>Portrait image<input name=file type=file accept="image/jpeg,image/png,image/webp" required></label><label>Lightness boost (0–50)<input name=lightness_boost type=number value=25 min=0 max=50></label><label>Whitening tone shift (0–30)<input name=whitening_tone type=number value=10 min=0 max=30></label><label>Smooth skin<select name=smooth_skin><option value=true selected>True</option><option value=false>False</option></select></label><button>Process portrait</button></form><div id=error></div><div id=r class=results hidden><div class=result-box><h3>Original</h3><img id=o></div><div class=result-box><h3>Processed</h3><img id=p></div></div></div><script>f.onsubmit=async e=>{e.preventDefault();error.textContent='Processing…';r.hidden=true;let d=new FormData(f),x=d.get('file');o.src=URL.createObjectURL(x);try{let q=await fetch('/process',{method:'POST',body:d});if(!q.ok)throw Error(await q.text());p.src=URL.createObjectURL(await q.blob());r.hidden=false;error.textContent=''}catch(z){error.textContent=z.message}}</script></body></html>"""


@app.post("/process")
async def process_image(file: UploadFile = File(...), lightness_boost: float = Form(25.0),
                        whitening_tone: float = Form(10.0), smooth_skin: bool = Form(True)):
    if not 0 <= lightness_boost <= 50 or not 0 <= whitening_tone <= 30:
        raise HTTPException(status_code=422, detail="Boost must be 0–50 and tone shift must be 0–30.")
    try:
        # Read at most one byte beyond the limit; UploadFile itself is spooled to disk.
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload is too large (maximum 12 MB).")
        _validate_image_header(contents)
        encoded = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        del encoded, contents
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode this image.")
        image = _web_size(image)
        processed = lighten_complexion(image, lightness_boost, whitening_tone, smooth_skin)
        ok, response_image = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise HTTPException(status_code=500, detail="Image encoding failed.")
        return Response(response_image.tobytes(), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as exc:
        print("Processing failure:\n" + traceback.format_exc(), file=sys.stderr)
        raise HTTPException(status_code=500, detail="Image processing failed.") from exc
    finally:
        with suppress(Exception):
            await file.close()
