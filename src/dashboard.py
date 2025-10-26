import streamlit as st
import cv2
import numpy as np
from src.preprocess import preprocess_image, camera_stream, overlay_heatmap
from src.gradcam import GradCAM
import torch
from PIL import Image
import requests

st.title("AutoVision: Real-Time Defect Detection")

# Load models
@st.cache_resource
def load_models():
    gradcam = GradCAM('../models/resnet18_anomaly.pth')
    return gradcam

gradcam = load_models()
classes = ['normal', 'crazing', 'inclusion', 'pits', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Real-time camera feed
frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# Optional: Use API for prediction (if backend running)
use_api = st.checkbox("Use FastAPI Backend for Inference")

for frame in camera_stream():
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preprocess for inference
    input_data = preprocess_image(rgb_frame)
    input_tensor = torch.from_numpy(input_data).float()
    
    if use_api:
        # Upload to API (simplified; in practice, use multipart)
        # For demo, use local PyTorch
        pass
    
    # Local inference with PyTorch (for Grad-CAM compatibility)
    with torch.no_grad():
        output = gradcam.model(input_tensor)
        pred = output.argmax().item()
        confidence = torch.softmax(output, dim=1).max().item()
    
    # Generate Grad-CAM
    heatmap = gradcam.generate(input_tensor, pred)
    
    # Overlay
    overlaid = overlay_heatmap(rgb_frame, heatmap)
    
    frame_placeholder.image(overlaid, channels="RGB")
    
    prediction_placeholder.markdown(f"**Prediction:** {classes[pred]} ({confidence:.2%})")

st.info("Press Ctrl+C to stop camera.")