# AutoVision Inference Results

## 🎯 Model Performance Summary

### Training Results
- **Dataset**: NEU-DET Steel Surface Defect Detection
- **Training Samples**: 1,440 images (240 per class)
- **Validation Samples**: 360 images (60 per class)
- **Training Epochs**: 10
- **Final Validation Accuracy**: **99.72%**

### Defect Classes (6 types)
1. **crazing** - Surface cracking patterns
2. **inclusion** - Embedded impurities
3. **patches** - Surface patch defects
4. **pitted_surface** - Pitting corrosion
5. **rolled-in_scale** - Scale rolled into surface
6. **scratches** - Linear scratch marks

---

## 📊 Available Inference Scripts

### 1. **test_model.py** - Complete Model Evaluation
**Purpose**: Comprehensive testing on entire validation set

**Features**:
- Overall accuracy metrics
- Per-class precision, recall, F1-score
- Confusion matrix visualization
- Misclassification analysis
- Performance charts

**Usage**:
```bash
python test_model.py
```

**Outputs**:
- `results/confusion_matrix.png` - Heatmap of predictions
- `results/per_class_accuracy.png` - Bar chart of class accuracies

**Results**:
- Overall Accuracy: **99.72%**
- Only 1 misclassification: inclusion → pitted_surface
- Perfect accuracy on 5 out of 6 classes

---

### 2. **inference_with_bbox.py** - Multi-Sample Visualization
**Purpose**: Visualize predictions with bounding boxes on sample images

**Features**:
- Side-by-side ground truth vs prediction
- Bounding box overlay from annotations
- Top-3 prediction probabilities
- Color-coded by defect class
- Batch processing of multiple samples

**Usage**:
```bash
python inference_with_bbox.py
```

**Outputs**:
- `results/inference_samples/` - 12 visualization images (2 per class)
- Each shows ground truth (left) and prediction (right)

**Sample Results** (from last run):
- Processed: 12 images
- Correct: 12/12 (100%)
- Confidence range: 88.6% - 100%

---

### 3. **single_inference.py** - Detailed Single Image Analysis
**Purpose**: In-depth analysis of a single image with comprehensive visualization

**Features**:
- Ground truth vs prediction comparison
- Bounding box visualization
- Probability bar chart for all classes
- Detailed metadata panel
- High-resolution output

**Usage**:
```bash
python single_inference.py
```

You can modify the script to test different images by changing:
```python
sample_class = 'inclusion'  # Choose: crazing, inclusion, patches, etc.
sample_num = '242'           # Image number
```

**Outputs**:
- `results/single_inference_result.png` - Comprehensive visualization

**Example Output**:
- Image: inclusion_242.jpg
- Ground Truth: INCLUSION
- Prediction: INCLUSION ✓
- Confidence: 88.63%

**Probability Breakdown**:
```
→ inclusion          88.63% ████████████████████████████████████████████ (GT)
  pitted_surface      9.67% ████
  scratches           1.60% 
  rolled-in_scale     0.08% 
  crazing             0.01% 
  patches             0.00% 
```

---

## 🎨 Visualization Features

### Bounding Box Colors
Each defect class has a unique color for easy identification:
- 🔴 **crazing**: Red (#FF6B6B)
- 🔵 **inclusion**: Teal (#4ECDC4)
- 🔵 **patches**: Blue (#45B7D1)
- 🟠 **pitted_surface**: Light Salmon (#FFA07A)
- 🟢 **rolled-in_scale**: Mint (#98D8C8)
- 🟡 **scratches**: Yellow (#F7DC6F)

### Visual Elements
- ✅ Green border/text for correct predictions
- ❌ Red border/text for incorrect predictions
- Solid lines for ground truth
- Dashed lines for predictions
- Confidence percentage displayed on bounding boxes
- Top-3 predictions shown at bottom

---

## 📈 Model Architecture

**Base Model**: ResNet-18 (pretrained on ImageNet)
**Modifications**: 
- Final FC layer adapted for 6 classes
- Input size: 224x224 pixels
- Normalization: ImageNet statistics

**Files**:
- `models/resnet18_anomaly.pth` - PyTorch model weights
- `models/resnet18_anomaly.onnx` - ONNX format (for deployment)

---

## 🚀 Quick Start Guide

### Test the model:
```bash
# Full evaluation on validation set
python test_model.py

# Visualize multiple samples with bounding boxes
python inference_with_bbox.py

# Detailed single image analysis
python single_inference.py
```

### View Results:
All outputs are saved in `results/` directory:
- Confusion matrix heatmap
- Per-class accuracy charts
- Individual inference visualizations

---

## 📊 Performance Metrics

### Per-Class Accuracy
| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| crazing | 100.00% | 60/60 |
| inclusion | 98.33% | 59/60 |
| patches | 100.00% | 60/60 |
| pitted_surface | 100.00% | 60/60 |
| rolled-in_scale | 100.00% | 60/60 |
| scratches | 100.00% | 60/60 |

### Overall Performance
- **Accuracy**: 99.72%
- **Macro Precision**: 99.73%
- **Macro Recall**: 99.72%
- **Macro F1-Score**: 99.72%

---

## 🎯 Next Steps

1. **Real-time Inference**: Use `app.py` for FastAPI deployment
2. **Dashboard**: Use `dashboard.py` for Streamlit interface
3. **Grad-CAM**: Use `gradcam.py` for visual explanations
4. **ONNX Deployment**: Use `resnet18_anomaly.onnx` for production

---

*Generated on October 29, 2025*
*Model trained on NEU-DET Steel Surface Defect Database*
