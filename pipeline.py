import cv2
import numpy as np
import traceback
import sys
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, HTMLResponse

app = FastAPI(title="Portrait Complexion Lightening API", version="1.1.0")

def get_skin_mask(image_bgr):
    try:
        h, w, _ = image_bgr.shape
        
        # 1. YCrCb Color-based skin segmentation
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # 2. Centered Ellipse Prior
        ellipse_mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, int(h * 0.45))
        axes = (int(w * 0.35), int(h * 0.45))
        cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1)
        
        skin_mask = cv2.bitwise_and(skin_mask, ellipse_mask)
        
        # If mask is empty (e.g. extreme lighting or atypical pose), fallback to pure YCrCb skin mask
        if np.sum(skin_mask) < 500:
            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
            
        # 3. Morphological cleanup
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
        
        # 4. Feather mask edges
        skin_mask = cv2.GaussianBlur(skin_mask, (31, 31), 0)
        
        return skin_mask
    except Exception as e:
        print(f"Error in get_skin_mask: {e}", file=sys.stderr)
        traceback.print_exc()
        # Fallback to full image mask if skin segmentation encounters any issue
        h, w, _ = image_bgr.shape
        return np.full((h, w), 128, dtype=np.uint8)

def lighten_complexion(image_bgr, lightness_boost=25.0, whitening_tone=10.0, smooth_skin=True):
    try:
        h, w, _ = image_bgr.shape
        
        mask = get_skin_mask(image_bgr)
        mask_float = mask.astype(np.float32) / 255.0
        mask_float = np.stack([mask_float, mask_float, mask_float], axis=-1)
        
        if smooth_skin:
            smoothed = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=50, sigmaSpace=50)
            base_img = image_bgr * (1.0 - 0.4 * mask_float) + smoothed * (0.4 * mask_float)
            base_img = base_img.astype(np.uint8)
        else:
            base_img = image_bgr.copy()
            
        lab = cv2.cvtColor(base_img, cv2.COLOR_BGR2Lab)
        L, A, B = cv2.split(lab)
        
        L_float = L.astype(np.float32)
        boost_map = lightness_boost * (1.0 - (L_float / 255.0) * 0.4)
        L_adjusted = L_float + boost_map * mask_float[:, :, 0]
        
        B_float = B.astype(np.float32)
        B_adjusted = B_float - (whitening_tone * 0.4) * mask_float[:, :, 0]
        
        L_adjusted = np.clip(L_adjusted, 0, 255).astype(np.uint8)
        B_adjusted = np.clip(B_adjusted, 0, 255).astype(np.uint8)
        
        merged_lab = cv2.merge([L_adjusted, A, B_adjusted])
        result_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2BGR)
        
        final_img = (image_bgr.astype(np.float32) * (1.0 - mask_float) + result_bgr.astype(np.float32) * mask_float)
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        
        return final_img
    except Exception as e:
        print(f"Error in lighten_complexion: {e}", file=sys.stderr)
        traceback.print_exc()
        raise e

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Portrait Complexion Lightener - Testing UI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f4f4f9; color: #333; }
            .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            h1 { font-size: 24px; margin-bottom: 10px; }
            p { color: #666; }
            .form-group { margin-bottom: 20px; }
            label { display: block; font-weight: bold; margin-bottom: 5px; }
            input[type="file"], input[type="number"], select { width: 100%; padding: 8px; box-sizing: border-box; }
            button { background: #007BFF; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .results { margin-top: 30px; display: flex; gap: 20px; justify-content: space-around; flex-wrap: wrap; }
            .result-box { text-align: center; max-width: 400px; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }
            #errorDiv { margin-top: 20px; padding: 15px; background: #ffe6e6; color: #cc0000; border: 1px solid #ff9999; border-radius: 4px; white-space: pre-wrap; display: none; font-family: monospace; font-size: 13px; }
            #loading { display: none; margin-top: 15px; font-weight: bold; color: #007BFF; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Portrait Complexion Lightener</h1>
            <p>Upload a portrait photo to test the selective complexion-lightening pipeline.</p>
            
            <form id="uploadForm">
                <div class="form-group">
                    <label>Select Portrait Image:</label>
                    <input type="file" id="fileInput" name="file" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label>Lightness Boost (0 - 50):</label>
                    <input type="number" id="boostInput" name="lightness_boost" value="25" step="1">
                </div>
                <div class="form-group">
                    <label>Whitening Tone Shift (0 - 30):</label>
                    <input type="number" id="toneInput" name="whitening_tone" value="10" step="1">
                </div>
                <div class="form-group">
                    <label>Smooth Skin:</label>
                    <select id="smoothInput" name="smooth_skin">
                        <option value="true" selected>True</option>
                        <option value="false">False</option>
                    </select>
                </div>
                <button type="submit" id="submitBtn">Process Portrait</button>
            </form>
            
            <div id="loading">Processing image... Please wait.</div>
            <div id="errorDiv"></div>
            
            <div class="results" id="resultsDiv" style="display:none;">
                <div class="result-box">
                    <h3>Original</h3>
                    <img id="origImg" />
                </div>
                <div class="result-box">
                    <h3>Lightened & Refined</h3>
                    <img id="procImg" />
                </div>
            </div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                if (fileInput.files.length === 0) return;
                
                const errorDiv = document.getElementById('errorDiv');
                const resultsDiv = document.getElementById('resultsDiv');
                const loadingDiv = document.getElementById('loading');
                
                errorDiv.style.display = 'none';
                errorDiv.innerText = '';
                resultsDiv.style.display = 'none';
                loadingDiv.style.display = 'block';
                
                const file = fileInput.files[0];
                const origUrl = URL.createObjectURL(file);
                document.getElementById('origImg').src = origUrl;
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('lightness_boost', document.getElementById('boostInput').value);
                formData.append('whitening_tone', document.getElementById('toneInput').value);
                formData.append('smooth_skin', document.getElementById('smoothInput').value);
                
                try {
                    const response = await fetch('/process', {
                        method: 'POST',
                        body: formData
                    });
                    
                    loadingDiv.style.display = 'none';
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const procUrl = URL.createObjectURL(blob);
                        document.getElementById('procImg').src = procUrl;
                        resultsDiv.style.display = 'flex';
                    } else {
                        const errText = await response.text();
                        let errorMsg = errText;
                        try {
                            const errJson = JSON.parse(errText);
                            if (errJson.detail) {
                                errorMsg = typeof errJson.detail === 'object' ? JSON.stringify(errJson.detail, null, 2) : errJson.detail;
                            }
                        } catch (err) {}
                        errorDiv.innerText = "Server Error (" + response.status + "):\\n" + errorMsg;
                        errorDiv.style.display = 'block';
                    }
                } catch (err) {
                    loadingDiv.style.display = 'none';
                    errorDiv.innerText = "Network or Client Error:\\n" + err.message;
                    errorDiv.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    lightness_boost: float = Form(25.0),
    whitening_tone: float = Form(10.0),
    smooth_skin: bool = Form(True)
):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image file. Please ensure it is a valid JPEG/PNG image.")
            
        processed = lighten_complexion(img, lightness_boost=lightness_boost, whitening_tone=whitening_tone, smooth_skin=smooth_skin)
        
        success, encoded_img = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise HTTPException(status_code=500, detail="Image encoding failed during JPEG compression.")
            
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
    except HTTPException as he:
        raise he
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        tb_str = "".join(traceback.format_exception(exc_type, exc_obj, exc_tb))
        print(f"Unhandled exception in /process: {e}\\n{tb_str}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"{str(e)}\\n\\nTraceback:\\n{tb_str}")