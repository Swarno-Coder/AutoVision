import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model, get_transforms

# Configuration
DATA_DIR = './data/NEU-DET'
MODEL_PATH = './models/resnet18_anomaly.pth'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model():
    """Test the trained model on validation set with detailed metrics."""
    
    print("=" * 80)
    print("TESTING TRAINED MODEL ON VALIDATION SET")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print()
    
    # Load validation dataset
    transform = get_transforms()
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'validation', 'images'), 
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = val_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print()
    
    # Load trained model
    print("Loading model...")
    model = get_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully!")
    print()
    
    # Perform inference
    print("Running inference on validation set...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE}/{len(val_dataset)} samples...")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE METRICS")
    print("=" * 80)
    
    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(class_names))
    )
    
    print("\n" + "-" * 80)
    print("PER-CLASS METRICS")
    print("-" * 80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10.0f}")
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    print("-" * 80)
    print(f"{'Macro Average':<20} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
    print(f"{'Weighted Average':<20} {precision_weighted:<12.4f} {recall_weighted:<12.4f} {f1_weighted:<12.4f}")
    print("-" * 80)
    
    # Detailed classification report
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    
    # Print header
    header = "True\\Pred".ljust(20)
    for class_name in class_names:
        header += f"{class_name[:8]:<10}"
    print(header)
    print("-" * (20 + len(class_names) * 10))
    
    # Print matrix
    for i, class_name in enumerate(class_names):
        row = f"{class_name:<20}"
        for j in range(len(class_names)):
            row += f"{cm[i][j]:<10}"
        print(row)
    
    # Calculate and display per-class accuracy from confusion matrix
    print("\n" + "=" * 80)
    print("PER-CLASS ACCURACY (from confusion matrix)")
    print("=" * 80)
    for i, class_name in enumerate(class_names):
        class_acc = cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"{class_name:<20}: {class_acc:.4f} ({class_acc*100:.2f}%) - {cm[i][i]}/{cm[i].sum()} correct")
    
    # Misclassification analysis
    print("\n" + "=" * 80)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 80)
    
    total_errors = len(all_labels) - np.sum(all_labels == all_preds)
    print(f"\nTotal misclassifications: {total_errors}/{len(all_labels)} ({total_errors/len(all_labels)*100:.2f}%)")
    
    if total_errors > 0:
        print("\nMost common misclassifications:")
        misclass_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i][j] > 0:
                    misclass_pairs.append((class_names[i], class_names[j], cm[i][j]))
        
        misclass_pairs.sort(key=lambda x: x[2], reverse=True)
        for true_class, pred_class, count in misclass_pairs[:10]:
            print(f"  {true_class} → {pred_class}: {count} times")
    else:
        print("\n🎉 Perfect classification! No errors detected!")
    
    # Save confusion matrix plot
    print("\n" + "=" * 80)
    print("SAVING VISUALIZATIONS")
    print("=" * 80)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Validation Set', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    # Save per-class accuracy plot
    plt.figure(figsize=(12, 6))
    class_accuracies = [cm[i][i] / cm[i].sum() * 100 for i in range(len(class_names))]
    colors = ['green' if acc >= 95 else 'orange' if acc >= 90 else 'red' for acc in class_accuracies]
    
    bars = plt.bar(class_names, class_accuracies, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=100, color='g', linestyle='--', linewidth=1, alpha=0.5, label='100%')
    plt.axhline(y=95, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='95%')
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy on Validation Set', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    acc_path = os.path.join(output_dir, 'per_class_accuracy.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class accuracy plot saved to: {acc_path}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Overall Accuracy: {accuracy*100:.2f}%")
    print(f"✓ All results saved to: {output_dir}/")
    print()

if __name__ == '__main__':
    test_model()
