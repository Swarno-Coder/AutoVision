import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model, get_transforms
import numpy as np
from sklearn.metrics import accuracy_score

# NEU-DET dataset with 6 defect classes
# Dataset structure: data/NEU-DET/train and data/NEU-DET/validation
DATA_DIR = './data/NEU-DET'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    transform = get_transforms()
    
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train', 'images'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'validation', 'images'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = get_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}')
        
        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                trues.extend(labels.cpu().numpy())
        
        acc = accuracy_score(trues, preds)
        print(f'Validation Accuracy: {acc:.4f}')
    
    # Save PyTorch model
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), './models/resnet18_anomaly.pth')
    
    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    torch.onnx.export(model, dummy_input, './models/resnet18_anomaly.onnx',
                      export_params=True, opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'], output_names=['output'])
    
    print('Model trained and exported to ONNX!')

if __name__ == '__main__':
    main()