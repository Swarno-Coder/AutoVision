import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model, get_transforms

# Configuration
MODEL_PATH = './models/resnet18_anomaly.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names and colors
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CLASS_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

def parse_xml_annotation(xml_path):
    """Parse XML annotation file to extract bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})
    
    return boxes

def show_single_inference(image_path, annotation_path):
    """Run inference on a single image and display results."""
    
    print("=" * 80)
    print("SINGLE IMAGE INFERENCE WITH BOUNDING BOX")
    print("=" * 80)
    
    # Load model
    print(f"\n📦 Loading model from: {MODEL_PATH}")
    model = get_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✓ Model loaded successfully!")
    
    # Load and preprocess image
    print(f"\n📷 Loading image: {image_path}")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Predict
    print("\n🔍 Running inference...")
    transform = get_transforms()
    image_pil = Image.open(image_path).convert('RGB')
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()
    
    # Get all class probabilities
    all_probs = probabilities[0].cpu().numpy()
    
    # Parse annotations
    annotations = parse_xml_annotation(annotation_path)
    true_class = annotations[0]['name'] if annotations else 'unknown'
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n✅ Ground Truth:  {true_class.upper()}")
    print(f"🎯 Prediction:    {predicted_class.upper()}")
    print(f"📊 Confidence:    {confidence_score*100:.2f}%")
    
    is_correct = predicted_class == true_class
    if is_correct:
        print(f"✓ Status:        CORRECT ✓")
    else:
        print(f"✗ Status:        INCORRECT ✗")
    
    print("\n" + "-" * 80)
    print("ALL CLASS PROBABILITIES:")
    print("-" * 80)
    
    # Sort by probability
    sorted_indices = np.argsort(all_probs)[::-1]
    for idx in sorted_indices:
        class_name = CLASS_NAMES[idx]
        prob = all_probs[idx]
        bar = '█' * int(prob * 50)
        marker = '→' if idx == predicted_idx.item() else ' '
        gt_marker = '(GT)' if class_name == true_class else ''
        print(f"{marker} {class_name:<18} {prob*100:6.2f}% {bar} {gt_marker}")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Main image with ground truth bbox
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title(f'Ground Truth: {true_class}', fontsize=16, fontweight='bold', pad=10)
    ax1.axis('off')
    
    for ann in annotations:
        bbox = ann['bbox']
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=4, edgecolor='lime', facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.text(bbox[0], bbox[1] - 5, f'GT: {true_class}',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lime', alpha=0.9),
                fontsize=12, fontweight='bold', color='black')
    
    # Predicted image with bbox
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_rgb)
    title_color = 'green' if is_correct else 'red'
    status = '✓ CORRECT' if is_correct else '✗ INCORRECT'
    ax2.set_title(f'Prediction: {predicted_class} - {status}', 
                 fontsize=16, fontweight='bold', color=title_color, pad=10)
    ax2.axis('off')
    
    for ann in annotations:
        bbox = ann['bbox']
        pred_color = 'green' if is_correct else 'red'
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=4, edgecolor=pred_color, facecolor='none', linestyle='--'
        )
        ax2.add_patch(rect)
        ax2.text(bbox[0], bbox[1] - 5, f'PRED: {predicted_class}\n{confidence_score*100:.1f}%',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=pred_color, alpha=0.9),
                fontsize=12, fontweight='bold', color='white')
    
    # Probability bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['green' if i == predicted_idx.item() else 'skyblue' for i in range(len(CLASS_NAMES))]
    y_pos = np.arange(len(CLASS_NAMES))
    ax3.barh(y_pos, all_probs * 100, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(CLASS_NAMES)
    ax3.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, 105])
    ax3.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (prob, color) in enumerate(zip(all_probs * 100, colors)):
        ax3.text(prob + 1, i, f'{prob:.1f}%', va='center', fontweight='bold', fontsize=10)
    
    # Info panel (bottom)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    info_text = f"""
    📁 Image: {os.path.basename(image_path)}
    📏 Image Size: {img_rgb.shape[1]}x{img_rgb.shape[0]} pixels
    🎯 Ground Truth: {true_class.upper()}
    🤖 Prediction: {predicted_class.upper()}
    📊 Confidence: {confidence_score*100:.2f}%
    {'✅ CORRECT PREDICTION' if is_correct else '❌ INCORRECT PREDICTION'}
    🖥️  Device: {DEVICE}
    📦 Model: {os.path.basename(MODEL_PATH)}
    """
    
    ax4.text(0.5, 0.5, info_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8),
            family='monospace')
    
    # Save
    output_path = './results/single_inference_result.png'
    os.makedirs('./results', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualization saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    # Example: Run inference on a sample validation image
    # You can change these paths to test different images
    
    sample_class = 'inclusion'  # Change this to test different classes
    sample_num = '242'  # Change this to test different images
    
    image_path = f'./data/NEU-DET/validation/images/{sample_class}/{sample_class}_{sample_num}.jpg'
    annotation_path = f'./data/NEU-DET/validation/annotations/{sample_class}_{sample_num}.xml'
    
    if os.path.exists(image_path) and os.path.exists(annotation_path):
        show_single_inference(image_path, annotation_path)
    else:
        print(f"Error: Could not find image or annotation file!")
        print(f"Image path: {image_path}")
        print(f"Annotation path: {annotation_path}")
