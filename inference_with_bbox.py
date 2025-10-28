import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from PIL import Image
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model, get_transforms

# Configuration
DATA_DIR = './data/NEU-DET'
MODEL_PATH = './models/resnet18_anomaly.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names and colors
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CLASS_COLORS = {
    'crazing': '#FF6B6B',        # Red
    'inclusion': '#4ECDC4',      # Teal
    'patches': '#45B7D1',        # Blue
    'pitted_surface': '#FFA07A',  # Light Salmon
    'rolled-in_scale': '#98D8C8', # Mint
    'scratches': '#F7DC6F'       # Yellow
}

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
        
        boxes.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return boxes

def predict_image(model, image_path, transform):
    """Predict class for a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()
    
    # Get top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
    top3_classes = [(CLASS_NAMES[idx.item()], prob.item()) 
                    for idx, prob in zip(top3_idx[0], top3_prob[0])]
    
    return predicted_class, confidence_score, top3_classes

def visualize_inference(image_path, annotation_path, model, transform, save_path):
    """Visualize inference result with bounding box and predictions."""
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get predictions
    predicted_class, confidence, top3 = predict_image(model, image_path, transform)
    
    # Parse annotations
    annotations = parse_xml_annotation(annotation_path)
    true_class = annotations[0]['name'] if annotations else 'unknown'
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Original image with ground truth
    axes[0].imshow(img_rgb)
    axes[0].set_title(f'Ground Truth: {true_class}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Draw ground truth bounding boxes
    for ann in annotations:
        bbox = ann['bbox']
        class_name = ann['name']
        color = CLASS_COLORS.get(class_name, '#FFFFFF')
        
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1],
            linewidth=3, 
            edgecolor=color, 
            facecolor='none',
            linestyle='-'
        )
        axes[0].add_patch(rect)
        
        # Add label
        axes[0].text(
            bbox[0], bbox[1] - 5,
            f'GT: {class_name}',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8),
            fontsize=10,
            fontweight='bold',
            color='black'
        )
    
    # Right plot: Predicted image with bounding box
    axes[1].imshow(img_rgb)
    is_correct = predicted_class == true_class
    title_color = 'green' if is_correct else 'red'
    status = '✓ CORRECT' if is_correct else '✗ INCORRECT'
    axes[1].set_title(
        f'Prediction: {predicted_class} ({confidence*100:.1f}%) - {status}', 
        fontsize=14, 
        fontweight='bold',
        color=title_color
    )
    axes[1].axis('off')
    
    # Draw predicted bounding boxes (using ground truth bbox with predicted class)
    for ann in annotations:
        bbox = ann['bbox']
        pred_color = CLASS_COLORS.get(predicted_class, '#FFFFFF')
        
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1],
            linewidth=3, 
            edgecolor=pred_color, 
            facecolor='none',
            linestyle='--'
        )
        axes[1].add_patch(rect)
        
        # Add predicted label
        axes[1].text(
            bbox[0], bbox[1] - 5,
            f'PRED: {predicted_class}\n{confidence*100:.1f}%',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=pred_color, alpha=0.8),
            fontsize=10,
            fontweight='bold',
            color='black'
        )
    
    # Add prediction details at the bottom
    details_text = "Top 3 Predictions:\n"
    for i, (cls, prob) in enumerate(top3, 1):
        marker = "→ " if i == 1 else "  "
        details_text += f"{marker}{i}. {cls}: {prob*100:.2f}%\n"
    
    fig.text(0.5, 0.02, details_text, 
             ha='center', fontsize=11, 
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8),
             family='monospace')
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'image': image_path,
        'true_class': true_class,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'is_correct': is_correct,
        'top3': top3
    }

def run_inference_samples(num_samples_per_class=2):
    """Run inference on sample images from each class."""
    print("=" * 80)
    print("INFERENCE WITH BOUNDING BOX VISUALIZATION")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print()
    
    # Load model
    print("Loading model...")
    model = get_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✓ Model loaded successfully!\n")
    
    # Get transform
    transform = get_transforms()
    
    # Create output directory
    output_dir = './results/inference_samples'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process samples from each class
    all_results = []
    
    for class_name in CLASS_NAMES:
        print(f"\nProcessing class: {class_name}")
        print("-" * 60)
        
        # Get images from this class
        class_img_dir = os.path.join(DATA_DIR, 'validation', 'images', class_name)
        class_ann_dir = os.path.join(DATA_DIR, 'validation', 'annotations')
        
        # List all images in this class
        image_files = [f for f in os.listdir(class_img_dir) if f.endswith('.jpg')]
        
        # Randomly sample
        sampled_images = random.sample(image_files, min(num_samples_per_class, len(image_files)))
        
        for img_file in sampled_images:
            # Construct paths
            img_path = os.path.join(class_img_dir, img_file)
            ann_file = img_file.replace('.jpg', '.xml')
            ann_path = os.path.join(class_ann_dir, ann_file)
            
            # Skip if annotation doesn't exist
            if not os.path.exists(ann_path):
                print(f"  ⚠ Skipping {img_file} - annotation not found")
                continue
            
            # Output path
            save_path = os.path.join(output_dir, f'{class_name}_{img_file}')
            
            # Run inference and visualize
            result = visualize_inference(img_path, ann_path, model, transform, save_path)
            all_results.append(result)
            
            # Print result
            status = "✓" if result['is_correct'] else "✗"
            print(f"  {status} {img_file}")
            print(f"     True: {result['true_class']}")
            print(f"     Pred: {result['predicted_class']} ({result['confidence']*100:.1f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    
    correct = sum(1 for r in all_results if r['is_correct'])
    total = len(all_results)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nTotal samples processed: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    print(f"\n✓ All visualizations saved to: {output_dir}/")
    
    # Show incorrect predictions
    incorrect = [r for r in all_results if not r['is_correct']]
    if incorrect:
        print(f"\n❌ Incorrect Predictions ({len(incorrect)}):")
        for r in incorrect:
            print(f"   {os.path.basename(r['image'])}")
            print(f"      True: {r['true_class']} → Predicted: {r['predicted_class']} ({r['confidence']*100:.1f}%)")
    else:
        print("\n🎉 All predictions were correct!")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Run inference on 2 samples per class (12 total images)
    run_inference_samples(num_samples_per_class=2)
