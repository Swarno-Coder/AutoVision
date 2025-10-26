from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from src.preprocess import preprocess_image
from src.gradcam import GradCAM  # Note: GradCAM uses PyTorch, so for ONNX we approximate or skip full CAM
import cv2
from PIL import Image
import io

app = FastAPI(title="AutoVision API")

# Load ONNX model
ort_session = ort.InferenceSession('../models/resnet18_anomaly.onnx')
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_np = np.array(image)
    
    input_data = preprocess_image(image_np)
    
    ort_inputs = {ort_session.get_inputs()[0].name: input_data.astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    pred = np.argmax(ort_outs[0])
    confidence = np.max(ort_outs[0])
    
    # For Grad-CAM, we'd need to load PyTorch model separately for explainability
    # Here, return pred and conf; overlay handled in UI
    
    return JSONResponse({
        "prediction": classes[pred],
        "confidence": float(confidence),
        "class_id": int(pred)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)