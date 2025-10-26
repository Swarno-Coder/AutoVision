import cv2
import numpy as np
from PIL import Image
from src.model import get_transforms

def preprocess_image(image_path_or_np):
    """Preprocess single image for inference."""
    if isinstance(image_path_or_np, str):
        image = cv2.imread(image_path_or_np)
    else:
        image = image_path_or_np
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    transform = get_transforms()
    return transform(image).unsqueeze(0).numpy()  # To numpy for ONNX

def camera_stream():
    """Generator for real-time camera feed."""
    cap = cv2.VideoCapture(0)  # Use default camera
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break
    cap.release()

def overlay_heatmap(frame, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on frame."""
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    return superimposed