#!/usr/bin/env python3
"""
Download and prepare Kaggle malaria cell dataset for YOLOv8 training.
This script converts the classification dataset to detection format with synthetic bounding boxes.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np
from PIL import Image
import kagglehub

def download_kaggle_dataset() -> str:
    """Download the Kaggle malaria cell dataset."""
    print("Downloading Kaggle malaria cell dataset...")
    path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
    print(f"Dataset downloaded to: {path}")
    return path

def create_yolo_structure(base_path: str) -> dict:
    """Create YOLOv8 dataset directory structure."""
    yolo_path = Path(base_path) / "yolo_malaria"
    
    # Create directory structure
    dirs = {
        'train': yolo_path / "train",
        'val': yolo_path / "val", 
        'test': yolo_path / "test"
    }
    
    for split_name, split_path in dirs.items():
        (split_path / "images").mkdir(parents=True, exist_ok=True)
        (split_path / "labels").mkdir(parents=True, exist_ok=True)
    
    print(f"Created YOLOv8 structure at: {yolo_path}")
    return dirs

def generate_cell_bbox(image_path: str, padding_ratio: float = 0.1) -> Tuple[float, float, float, float]:
    """
    Generate bounding box for cell image using contour detection.
    Returns normalized YOLO format: (center_x, center_y, width, height)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        # Fallback: assume cell occupies most of the image
        return (0.5, 0.5, 0.8, 0.8)
    
    h, w = image.shape[:2]
    
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assumed to be the cell)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
        
        # Add padding
        pad_x = int(bbox_w * padding_ratio)
        pad_y = int(bbox_h * padding_ratio)
        
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        bbox_w = min(w - x, bbox_w + 2 * pad_x)
        bbox_h = min(h - y, bbox_h + 2 * pad_y)
        
        # Convert to YOLO format (normalized)
        center_x = (x + bbox_w / 2) / w
        center_y = (y + bbox_h / 2) / h
        norm_width = bbox_w / w
        norm_height = bbox_h / h
        
        return (center_x, center_y, norm_width, norm_height)
    else:
        # Fallback: assume cell occupies center 80% of image
        return (0.5, 0.5, 0.8, 0.8)

def convert_to_yolo_format(kaggle_path: str, yolo_dirs: dict, train_ratio: float = 0.7, val_ratio: float = 0.2):
    """Convert Kaggle classification dataset to YOLOv8 detection format."""
    
    kaggle_path = Path(kaggle_path)
    
    # Find the cell_images directory
    cell_images_path = None
    for root, dirs, files in os.walk(kaggle_path):
        if 'Parasitized' in dirs and 'Uninfected' in dirs:
            cell_images_path = Path(root)
            break
    
    if not cell_images_path:
        raise ValueError("Could not find Parasitized and Uninfected directories in the dataset")
    
    print(f"Found cell images at: {cell_images_path}")
    
    # Process each class
    all_files = []
    
    for class_name in ['Parasitized', 'Uninfected']:
        class_path = cell_images_path / class_name
        if not class_path.exists():
            print(f"Warning: {class_path} not found, skipping...")
            continue
            
        class_files = list(class_path.glob('*.png'))
        print(f"Found {len(class_files)} images in {class_name}")
        
        # Only parasitized cells get bounding boxes (class 0)
        class_id = 0 if class_name == 'Parasitized' else None
        
        for img_path in class_files:
            all_files.append((img_path, class_id))
    
    # Shuffle and split
    random.shuffle(all_files)
    total_files = len(all_files)
    
    train_end = int(total_files * train_ratio)
    val_end = int(total_files * (train_ratio + val_ratio))
    
    splits = {
        'train': all_files[:train_end],
        'val': all_files[train_end:val_end],
        'test': all_files[val_end:]
    }
    
    print(f"Dataset split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # Process each split
    for split_name, files in splits.items():
        print(f"Processing {split_name} split...")
        
        images_dir = yolo_dirs[split_name] / "images"
        labels_dir = yolo_dirs[split_name] / "labels"
        
        for i, (img_path, class_id) in enumerate(files):
            # Copy image
            new_img_name = f"{split_name}_{i:06d}.png"
            new_img_path = images_dir / new_img_name
            shutil.copy2(img_path, new_img_path)
            
            # Create label file
            label_path = labels_dir / f"{split_name}_{i:06d}.txt"
            
            if class_id is not None:  # Parasitized cell
                # Generate bounding box
                bbox = generate_cell_bbox(str(img_path))
                
                # Write YOLO format label
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            else:  # Uninfected cell - create empty label file
                label_path.touch()
        
        print(f"Completed {split_name}: {len(files)} images processed")

def create_data_yaml(yolo_path: Path):
    """Create data.yaml file for YOLOv8 training."""
    
    yaml_content = f"""# Malaria Cell Detection Dataset
path: {yolo_path.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: 1  # number of classes
names: ['malaria_parasite']  # class names

# Dataset info
download: |
  # Dataset converted from Kaggle malaria cell classification dataset
  # Original: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
  # Converted to YOLOv8 detection format with synthetic bounding boxes
"""
    
    yaml_path = yolo_path / "malaria_data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created data.yaml at: {yaml_path}")
    return yaml_path

def visualize_samples(yolo_dirs: dict, num_samples: int = 5):
    """Visualize some samples with bounding boxes."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    train_images = list((yolo_dirs['train'] / "images").glob("*.png"))
    train_labels = yolo_dirs['train'] / "labels"
    
    # Select random samples
    samples = random.sample(train_images, min(num_samples, len(train_images)))
    
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 3))
    if len(samples) == 1:
        axes = [axes]
    
    for i, img_path in enumerate(samples):
        # Load image
        image = Image.open(img_path)
        w, h = image.size
        
        # Load corresponding label
        label_path = train_labels / f"{img_path.stem}.txt"
        
        axes[i].imshow(image)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
        
        # Draw bounding box if exists
        if label_path.exists() and label_path.stat().st_size > 0:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    class_id = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:5])
                    
                    # Convert to pixel coordinates
                    x = (center_x - width/2) * w
                    y = (center_y - height/2) * h
                    box_w = width * w
                    box_h = height * h
                    
                    # Draw bounding box
                    rect = patches.Rectangle((x, y), box_w, box_h, 
                                           linewidth=2, edgecolor='red', facecolor='none')
                    axes[i].add_patch(rect)
                    axes[i].text(x, y-5, 'Parasite', color='red', fontsize=8, weight='bold')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Sample visualization saved as 'dataset_samples.png'")

def main():
    """Main function to download and convert the dataset."""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("=== Kaggle Malaria Dataset Conversion ===")
    
    # Download dataset
    kaggle_path = download_kaggle_dataset()
    
    # Create YOLOv8 structure
    base_path = "data"
    os.makedirs(base_path, exist_ok=True)
    yolo_dirs = create_yolo_structure(base_path)
    
    # Convert to YOLO format
    convert_to_yolo_format(kaggle_path, yolo_dirs)
    
    # Create data.yaml
    yolo_path = Path(base_path) / "yolo_malaria"
    yaml_path = create_data_yaml(yolo_path)
    
    # Visualize samples
    print("\nGenerating sample visualizations...")
    visualize_samples(yolo_dirs)
    
    print(f"\n=== Conversion Complete ===")
    print(f"YOLOv8 dataset created at: {yolo_path}")
    print(f"Data config file: {yaml_path}")
    print(f"Ready for training with: python train_yolo.py --data {yaml_path}")
    
    # Print dataset statistics
    for split_name, split_dir in yolo_dirs.items():
        images_count = len(list((split_dir / "images").glob("*.png")))
        labels_count = len([f for f in (split_dir / "labels").glob("*.txt") 
                           if f.stat().st_size > 0])
        print(f"{split_name.capitalize()}: {images_count} images, {labels_count} with parasites")

if __name__ == "__main__":
    main()
