import torch
import torch.nn.functional as F
import cv2
import numpy as np
from src.model import get_model

class GradCAM:
    def __init__(self, model_path, target_layer='layer4'):
        self.model = get_model(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        for name, module in self.model.named_modules():
            if target_layer in name:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
    
    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        
        return cam
    
    def __del__(self):
        for hook in self.hooks:
            hook.remove()

# Usage example:
# gradcam = GradCAM('../models/resnet18_anomaly.pth')
# input_tensor = torch.randn(1, 3, 224, 224)  # From preprocess
# heatmap = gradcam.generate(input_tensor)