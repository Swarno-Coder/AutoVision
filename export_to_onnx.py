import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def export_to_onnx():
    """Export the saved PyTorch model to ONNX format."""
    # Load the trained model
    model = get_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load('./models/resnet18_anomaly.pth', map_location=DEVICE))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        './models/resnet18_anomaly.onnx',
        export_params=True, 
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output']
    )
    
    print('Model successfully exported to ONNX format!')
    print('Model saved at: ./models/resnet18_anomaly.onnx')

if __name__ == '__main__':
    export_to_onnx()
