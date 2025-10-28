"""
Image preprocessing utilities for AutoVision
Handles image loading, transformation, and visualization
"""
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_transforms

def preprocess_image(image_path_or_np):
    """
    Preprocess single image for inference
    
    Args:
        image_path_or_np: Either file path (str) or numpy array
    
    Returns:
        numpy array: Preprocessed image tensor
    """
    if isinstance(image_path_or_np, str):
        image = cv2.imread(image_path_or_np)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path_or_np
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from cv2
            if isinstance(image_path_or_np, np.ndarray):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    image = Image.fromarray(image)
    
    # Apply transforms
    transform = get_transforms()
    transformed = transform(image).unsqueeze(0).numpy()
    
    return transformed

def load_image(image_path):
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
    
    Returns:
        numpy array: RGB image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_image(image, size=(224, 224)):
    """
    Resize image to specified size
    
    Args:
        image: Input image (numpy array or PIL Image)
        size: Target size (width, height)
    
    Returns:
        Resized image (same type as input)
    """
    if isinstance(image, np.ndarray):
        return cv2.resize(image, size)
    elif isinstance(image, Image.Image):
        return image.resize(size)
    else:
        raise TypeError("Image must be numpy array or PIL Image")

def camera_stream(camera_id=0):
    """
    Generator for real-time camera feed
    
    Args:
        camera_id: Camera device ID (default: 0 for primary camera)
    
    Yields:
        numpy array: Video frame in BGR format
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    finally:
        cap.release()

def overlay_heatmap(frame, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on frame
    
    Args:
        frame: Original image (RGB, numpy array)
        heatmap: Normalized heatmap (H, W) with values in [0, 1]
        alpha: Transparency of heatmap (0=transparent, 1=opaque)
        colormap: OpenCV colormap to use
    
    Returns:
        numpy array: RGB image with heatmap overlay
    """
    # Ensure frame is the right type
    if frame.dtype != np.uint8:
        frame = np.uint8(frame)
    
    # Resize heatmap to match frame size
    if frame.shape[:2] != heatmap.shape:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    
    # Convert heatmap to 0-255 range
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert heatmap from BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend frame and heatmap
    superimposed = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
    
    return superimposed

def normalize_image(image):
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image (numpy array)
    
    Returns:
        numpy array: Normalized image
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image

def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image from ImageNet normalization
    
    Args:
        image: Normalized image tensor or array
        mean: Mean values used for normalization
        std: Std values used for normalization
    
    Returns:
        numpy array: Denormalized image in [0, 255] range
    """
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    
    # Denormalize
    image = image * std + mean
    
    # Clip to [0, 1] and convert to [0, 255]
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    return image

def draw_bounding_box(image, bbox, label, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box on image
    
    Args:
        image: Input image (numpy array)
        bbox: Bounding box coordinates [xmin, ymin, xmax, ymax]
        label: Label text
        color: Box color (B, G, R) or (R, G, B) depending on color space
        thickness: Line thickness
    
    Returns:
        numpy array: Image with bounding box drawn
    """
    image_copy = image.copy()
    
    xmin, ymin, xmax, ymax = bbox
    
    # Draw rectangle
    cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), color, thickness)
    
    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        image_copy,
        (xmin, ymin - label_size[1] - 10),
        (xmin + label_size[0], ymin),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        image_copy,
        label,
        (xmin, ymin - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return image_copy

def augment_image(image, rotation=0, flip=None, brightness=1.0):
    """
    Apply augmentations to image
    
    Args:
        image: Input image (numpy array)
        rotation: Rotation angle in degrees
        flip: 'horizontal', 'vertical', or None
        brightness: Brightness multiplier (1.0 = no change)
    
    Returns:
        numpy array: Augmented image
    """
    aug_image = image.copy()
    
    # Rotation
    if rotation != 0:
        h, w = aug_image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        aug_image = cv2.warpAffine(aug_image, matrix, (w, h))
    
    # Flip
    if flip == 'horizontal':
        aug_image = cv2.flip(aug_image, 1)
    elif flip == 'vertical':
        aug_image = cv2.flip(aug_image, 0)
    
    # Brightness
    if brightness != 1.0:
        aug_image = np.clip(aug_image * brightness, 0, 255).astype(np.uint8)
    
    return aug_image


# Example usage
if __name__ == '__main__':
    """
    Test preprocessing functions
    """
    print("Testing preprocessing utilities...")
    
    # Test image loading
    test_image_path = './data/NEU-DET/validation/images/crazing/crazing_241.jpg'
    
    if os.path.exists(test_image_path):
        # Load image
        image = load_image(test_image_path)
        print(f"✓ Loaded image: {image.shape}")
        
        # Preprocess for model
        preprocessed = preprocess_image(test_image_path)
        print(f"✓ Preprocessed shape: {preprocessed.shape}")
        
        # Create dummy heatmap
        heatmap = np.random.rand(224, 224)
        
        # Overlay heatmap
        image_resized = resize_image(image, (224, 224))
        overlaid = overlay_heatmap(image_resized, heatmap)
        print(f"✓ Overlaid shape: {overlaid.shape}")
        
        # Save result
        cv2.imwrite('./preprocess_test.jpg', cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
        print("✓ Test image saved to preprocess_test.jpg")
    else:
        print(f"Test image not found: {test_image_path}")