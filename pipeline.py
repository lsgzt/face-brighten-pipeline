import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

app = FastAPI(title="Portrait Complexion Lightening API", version="1.0.0")

def get_skin_mask(image_bgr):
    h, w, _ = image_bgr.shape
    
    # 1. YCrCb Color-based skin segmentation
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    # Standard skin color range in YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # 2. Centered Ellipse Prior (Portraits usually have the face in the center/upper-center region)
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, int(h * 0.45))
    axes = (int(w * 0.35), int(h * 0.45))
    cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1)
    
    # Combine skin color mask with central face region prior
    skin_mask = cv2.bitwise_and(skin_mask, ellipse_mask)
    
    # 3. Morphological cleanup to remove small spots and smooth skin region
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # 4. Feather mask edges for seamless blending and zero halos
    skin_mask = cv2.GaussianBlur(skin_mask, (31, 31), 0)
    
    return skin_mask

def lighten_complexion(image_bgr, lightness_boost=25.0, whitening_tone=10.0, smooth_skin=True):
    h, w, _ = image_bgr.shape
    
    # Get skin mask normalized float (0.0 to 1.0)
    mask = get_skin_mask(image_bgr)
    mask_float = mask.astype(np.float32) / 255.0
    mask_float = np.stack([mask_float, mask_float, mask_float], axis=-1)
    
    # Optional skin smoothing (bilateral filter preserves facial structure and sharp edges while smoothing skin texture)
    if smooth_skin:
        smoothed = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=50, sigmaSpace=50)
        base_img = image_bgr * (1.0 - 0.4 * mask_float) + smoothed * (0.4 * mask_float)
        base_img = base_img.astype(np.uint8)
    else:
        base_img = image_bgr.copy()
        
    # Convert to CIELAB color space for perceptual lightness and color manipulation
    lab = cv2.cvtColor(base_img, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)
    
    L_float = L.astype(np.float32)
    
    # Non-linear lightness adjustment: boost shadows and midtones more than highlights to prevent washed-out skin
    boost_map = lightness_boost * (1.0 - (L_float / 255.0) * 0.4)
    L_adjusted = L_float + boost_map * mask_float[:, :, 0]
    
    # Whitening / porcelain tone adjustment: subtly shift B channel (yellow-blue axis) toward cooler/brighter tone
    B_float = B.astype(np.float32)
    B_adjusted = B_float - (whitening_tone * 0.4) * mask_float[:, :, 0]
    
    L_adjusted = np.clip(L_adjusted, 0, 255).astype(np.uint8)
    B_adjusted = np.clip(B_adjusted, 0, 255).astype(np.uint8)
    
    # Merge and convert back to BGR
    merged_lab = cv2.merge([L_adjusted, A, B_adjusted])
    result_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2BGR)
    
    # Final blending with original image using feathered mask to protect hair, clothes, eyes, background
    final_img = (image_bgr.astype(np.float32) * (1.0 - mask_float) + result_bgr.astype(np.float32) * mask_float)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    return final_img

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Portrait Complexion Lightening API",
        "description": "Optimized CPU-friendly portrait skin lightening pipeline for Render free tier.",
        "endpoints": {
            "POST /process": "Upload image file with optional parameters (lightness_boost, whitening_tone, smooth_skin)."
        }
    }

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    lightness_boost: float = Form(25.0),
    whitening_tone: float = Form(10.0),
    smooth_skin: bool = Form(True)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file or unsupported format.")
            
        processed = lighten_complexion(img, lightness_boost=lightness_boost, whitening_tone=whitening_tone, smooth_skin=smooth_skin)
        
        success, encoded_img = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise HTTPException(status_code=500, detail="Image encoding failed.")
            
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
