# 🔍 AutoVision - Intelligent Visual Defect Detection System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)

A plug-and-play AI visual inspection system that uses transfer learning (ResNet-18) to detect manufacturing defects in real-time. Perfect for industrial quality control applications.

## 🎯 Features

- **🤖 Transfer Learning**: ResNet-18 pre-trained on ImageNet, fine-tuned for defect detection
- **🔥 Visual Explanations**: Grad-CAM heatmaps show which regions influenced predictions
- **⚡ Real-time Inference**: Process images from camera feed or uploaded files
- **🌐 REST API**: FastAPI backend for easy integration into existing systems
- **📊 Interactive Dashboard**: Streamlit web interface for monitoring and analysis
- **📦 ONNX Export**: Deploy models in production environments
- **🎨 6 Defect Classes**: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 99.72% |
| **Macro Precision** | 99.73% |
| **Macro Recall** | 99.72% |
| **Macro F1-Score** | 99.72% |

### Per-Class Performance

| Defect Class | Accuracy | Samples |
|--------------|----------|---------|
| Crazing | 100.00% | 60/60 |
| Inclusion | 98.33% | 59/60 |
| Patches | 100.00% | 60/60 |
| Pitted Surface | 100.00% | 60/60 |
| Rolled-in Scale | 100.00% | 60/60 |
| Scratches | 100.00% | 60/60 |

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch 2.0+
- **Computer Vision**: OpenCV, torchvision
- **Backend API**: FastAPI, Uvicorn
- **Frontend Dashboard**: Streamlit
- **Explainability**: Grad-CAM
- **Deployment**: ONNX, Docker (optional)
- **Dataset**: NEU-DET Surface Defect Database

## 📁 Project Structure

```
AutoVision/
├── data/
│   └── NEU-DET/                  # Dataset
│       ├── train/
│       │   ├── images/           # Training images (6 classes)
│       │   └── annotations/      # XML annotations
│       └── validation/
│           ├── images/           # Validation images
│           └── annotations/      # XML annotations
├── models/
│   ├── resnet18_anomaly.pth      # PyTorch model weights
│   └── resnet18_anomaly.onnx     # ONNX format (for deployment)
├── src/
│   ├── model.py                  # Model architecture
│   ├── train.py                  # Training script
│   ├── preprocess.py             # Image preprocessing utilities
│   ├── gradcam.py                # Grad-CAM implementation
│   ├── app.py                    # FastAPI backend
│   └── dashboard.py              # Streamlit dashboard
├── results/                      # Inference results & visualizations
├── test_model.py                 # Full model evaluation
├── inference_with_bbox.py        # Batch inference with bounding boxes
├── single_inference.py           # Single image detailed analysis
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Swarno-Coder/AutoVision.git
cd AutoVision
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model (if not already trained)

```bash
python src/train.py
```

**Training Results:**
- Duration: ~10-15 minutes (on GPU)
- Final Accuracy: 99.72%
- Models saved to `models/`

### 4. Test Model

```bash
# Full evaluation on validation set
python test_model.py

# Visualize predictions with bounding boxes
python inference_with_bbox.py

# Detailed single image analysis
python single_inference.py
```

### 5. Start FastAPI Backend

```bash
python src/app.py
```

**Access Points:**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 6. Launch Dashboard

```bash
streamlit run src/dashboard.py
```

Dashboard will open at: http://localhost:8501

## 📡 API Usage

### Basic Prediction

```python
import requests

# Upload image
url = "http://localhost:8000/predict"
files = {"file": open("defect_image.jpg", "rb")}
response = requests.post(url, files=files)

result = response.json()
print(f"Defect: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

**Response:**
```json
{
  "success": true,
  "prediction": "crazing",
  "confidence": 0.9234,
  "class_id": 0,
  "all_probabilities": {
    "crazing": 0.9234,
    "inclusion": 0.0421,
    ...
  },
  "top_3_predictions": [
    {"class": "crazing", "probability": 0.9234},
    {"class": "inclusion", "probability": 0.0421},
    {"class": "patches", "probability": 0.0198}
  ]
}
```

### Prediction with Grad-CAM

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Upload image and get visual explanation
url = "http://localhost:8000/predict/gradcam"
files = {"file": open("defect_image.jpg", "rb")}
response = requests.post(url, files=files)

result = response.json()

# Decode and display heatmap overlay
img_data = base64.b64decode(result["gradcam_image"])
img = Image.open(BytesIO(img_data))
img.show()

print(f"Explanation: {result['explanation']}")
```

### Batch Prediction

```python
import requests

url = "http://localhost:8000/batch/predict"
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
    ("files", open("image3.jpg", "rb"))
]
response = requests.post(url, files=files)

results = response.json()
for item in results["results"]:
    print(f"{item['filename']}: {item['prediction']} ({item['confidence']:.1%})")
```

## 🎨 Dashboard Features

### 1. Upload Image Mode
- Drag & drop or browse to upload images
- Instant defect classification
- Grad-CAM visual explanations
- Probability distribution charts
- Confidence threshold settings

### 2. Real-time Camera Mode
- Live webcam feed processing
- Real-time defect detection
- Heatmap overlay on video
- FPS monitoring
- Continuous inference

### 3. API Integration Mode
- Interactive API documentation
- Live endpoint testing
- Code examples (Python, cURL, JavaScript)
- Request/response visualization

## 🔥 Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) provides visual explanations:

- **Red regions**: High influence on prediction
- **Blue regions**: Low influence on prediction
- Helps understand what the model "sees"
- Useful for debugging and building trust

Example:
```python
from src.gradcam import GradCAM, overlay_heatmap_on_image
from src.model import get_transforms
from PIL import Image

# Initialize
gradcam = GradCAM('./models/resnet18_anomaly.pth')
transform = get_transforms()

# Load image
image = Image.open('defect.jpg')
input_tensor = transform(image).unsqueeze(0)

# Generate heatmap
heatmap = gradcam.generate(input_tensor)

# Overlay on image
import numpy as np
image_np = np.array(image.resize((224, 224)))
result = overlay_heatmap_on_image(image_np, heatmap)
```

## 📊 Dataset

**NEU-DET** (Northeastern University Surface Defect Database)
- **Total Images**: 1,800 (300 per class)
- **Image Size**: 200x200 pixels
- **Format**: Grayscale converted to RGB
- **Split**: 80% train (1,440), 20% validation (360)

### Defect Classes:

1. **Crazing**: Network of fine cracks
2. **Inclusion**: Embedded impurities
3. **Patches**: Irregular surface patches
4. **Pitted Surface**: Corrosion pitting
5. **Rolled-in Scale**: Oxide scale defects
6. **Scratches**: Linear surface scratches

## 🧪 Evaluation Scripts

### 1. Full Model Testing
```bash
python test_model.py
```
Generates:
- Confusion matrix
- Per-class metrics
- Accuracy charts
- Classification report

### 2. Batch Inference with Bounding Boxes
```bash
python inference_with_bbox.py
```
Generates:
- Side-by-side ground truth vs prediction
- Bounding box overlays
- Confidence scores
- 12 sample visualizations

### 3. Single Image Analysis
```bash
python single_inference.py
```
Generates:
- Detailed visualization
- Probability distribution
- Grad-CAM overlay
- Comprehensive metadata

## 🐳 Docker Deployment

```dockerfile
# Build image
docker build -t autovision .

# Run container
docker run -p 8000:8000 -p 8501:8501 autovision
```

## 🔧 Configuration

### Model Settings (`src/model.py`)
- **Architecture**: ResNet-18
- **Input Size**: 224x224
- **Classes**: 6
- **Pretrained**: ImageNet weights

### Training Settings (`src/train.py`)
- **Batch Size**: 32
- **Epochs**: 10
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss

### API Settings (`src/app.py`)
- **Host**: 0.0.0.0
- **Port**: 8000
- **CORS**: Enabled
- **Device**: Auto-detect (CUDA/CPU)

## 📈 Performance Optimization

### For Training:
- Use GPU if available (10x faster)
- Increase batch size for faster convergence
- Data augmentation for better generalization

### For Inference:
- Use ONNX model for production (faster)
- Batch predictions for multiple images
- GPU acceleration for real-time processing

### For API:
- Deploy with Gunicorn/Uvicorn workers
- Use caching for repeated predictions
- Load balancing for high traffic

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: NEU Surface Defect Database
- **Framework**: PyTorch team
- **Inspiration**: MVTec AD Dataset
- **Community**: FastAPI & Streamlit developers

## 📞 Contact

**Developer**: Swarno-Coder
**GitHub**: [@Swarno-Coder](https://github.com/Swarno-Coder)
**Project**: [AutoVision](https://github.com/Swarno-Coder/AutoVision)

## 🎯 Roadmap

- [ ] Add more defect classes
- [ ] Multi-object detection (YOLOv8 integration)
- [ ] Anomaly detection mode
- [ ] Mobile app integration
- [ ] Cloud deployment guide
- [ ] Continuous learning pipeline
- [ ] Production monitoring dashboard
- [ ] Alert system for critical defects

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{autovision2025,
  author = {Swarno-Coder},
  title = {AutoVision: Intelligent Visual Defect Detection System},
  year = {2025},
  url = {https://github.com/Swarno-Coder/AutoVision}
}
```

---

<div align="center">
  <strong>Built with ❤️ for Manufacturing Quality Control</strong>
  <br>
  <sub>Making defect detection intelligent and accessible</sub>
</div>
