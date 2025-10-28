"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Provides visual explanations for CNN predictions
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model

class GradCAM:
    """
    Grad-CAM implementation for visual explanation of model predictions
    """
    def __init__(self, model_path, target_layer='layer4', device=None):
        """
        Initialize Grad-CAM
        
        Args:
            model_path: Path to trained model weights
            target_layer: Layer name to extract activations from (default: 'layer4' for ResNet)
            device: Computing device (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = get_model(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.target_layer_name = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        def backward_hook(module, grad_input, grad_output):
            """Captures gradients during backward pass"""
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            """Captures activations during forward pass"""
            self.activations = output
        
        # Find and hook the target layer
        target_module = None
        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                target_module = module
                break
        
        if target_module is None:
            # Fallback: use last conv layer
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_module = module
        
        if target_module:
            self.hooks.append(target_module.register_forward_hook(forward_hook))
            self.hooks.append(target_module.register_full_backward_hook(backward_hook))
        else:
            raise ValueError(f"Could not find target layer: {self.target_layer_name}")
    
    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            class_idx: Target class index (if None, uses predicted class)
        
        Returns:
            numpy array: Normalized heatmap (H, W) with values in [0, 1]
        """
        # Ensure tensor is on correct device
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Determine target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass for target class
        target_score = output[0, class_idx]
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=False)
        
        # Apply ReLU to focus on positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        return cam
    
    def generate_multiple(self, input_tensor):
        """
        Generate Grad-CAM for all classes
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W)
        
        Returns:
            dict: Dictionary mapping class indices to heatmaps
        """
        num_classes = self.model.fc.out_features
        heatmaps = {}
        
        for class_idx in range(num_classes):
            heatmap = self.generate(input_tensor, class_idx=class_idx)
            heatmaps[class_idx] = heatmap
        
        return heatmaps
    
    def __del__(self):
        """Cleanup hooks when object is destroyed"""
        self.remove_hooks()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def apply_colormap(heatmap, colormap=cv2.COLORMAP_JET):
    """
    Apply colormap to heatmap
    
    Args:
        heatmap: Normalized heatmap (H, W) in [0, 1]
        colormap: OpenCV colormap
    
    Returns:
        RGB heatmap image
    """
    # Convert to 0-255 range
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert BGR to RGB
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    return colored_heatmap

def overlay_heatmap_on_image(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image
    
    Args:
        image: Original RGB image (H, W, 3)
        heatmap: Normalized heatmap (H', W') in [0, 1]
        alpha: Transparency of heatmap (0=transparent, 1=opaque)
    
    Returns:
        RGB image with heatmap overlay
    """
    # Resize heatmap to match image size
    if image.shape[:2] != heatmap.shape:
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    else:
        heatmap_resized = heatmap
    
    # Apply colormap
    heatmap_colored = apply_colormap(heatmap_resized)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    # Blend images
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlaid


# Example usage
if __name__ == '__main__':
    """
    Example usage of Grad-CAM
    """
    from src.model import get_transforms
    from PIL import Image
    
    # Initialize Grad-CAM
    model_path = './models/resnet18_anomaly.pth'
    gradcam = GradCAM(model_path, target_layer='layer4')
    
    # Load and preprocess image
    image_path = './data/NEU-DET/validation/images/crazing/crazing_241.jpg'
    image = Image.open(image_path).convert('RGB')
    
    transform = get_transforms()
    input_tensor = transform(image).unsqueeze(0)
    
    # Generate heatmap
    heatmap = gradcam.generate(input_tensor)
    
    # Create overlay
    image_np = np.array(image.resize((224, 224)))
    overlaid = overlay_heatmap_on_image(image_np, heatmap)
    
    # Save result
    cv2.imwrite('./gradcam_example.jpg', cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
    print("Grad-CAM example saved to gradcam_example.jpg")