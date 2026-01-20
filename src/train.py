#!/usr/bin/env python3
"""
Colab Pro Optimized YOLOv8 Malaria Detection Training Script
Optimized for V100/A100 GPUs with enhanced batch sizes and full training cycles.
"""

# ============================================================================
# COLAB PRO SETUP - Run this first
# ============================================================================

# Check GPU and install dependencies
import subprocess
import sys

def install_packages():
    """Install required packages for Colab Pro."""
    packages = [
        'ultralytics>=8.0.0',
        'kagglehub>=0.2.0', 
        'wandb',
        'opencv-python',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
    
    print("✅ All packages installed successfully")

# Run installation
install_packages()

# Check GPU availability
import torch
print(f"🚀 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected - training will be slow")

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import os
import shutil
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import kagglehub
import yaml
from PIL import Image
import time
import zipfile
from datetime import datetime

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("📚 Libraries imported successfully")

# ============================================================================
# COLAB PRO OPTIMIZED CONFIGURATION
# ============================================================================

# Colab Pro optimized settings
COLAB_PRO_CONFIG = {
    # Training parameters optimized for V100/A100
    'epochs': 100,              # Full training cycle
    'batch_size': 32,           # Large batch for stable gradients
    'image_size': 640,          # High resolution
    'workers': 8,               # Fast data loading
    'cache': 'disk',            # Cache dataset on disk (deterministic)
    
    # Learning parameters
    'lr0': 0.01,               # Initial learning rate
    'weight_decay': 0.0005,    # L2 regularization
    'momentum': 0.937,         # SGD momentum
    'patience': 25,            # Early stopping patience
    
    # Augmentation (clinical-safe)
    'hsv_h': 0.015,           # Hue augmentation
    'hsv_s': 0.7,             # Saturation augmentation  
    'hsv_v': 0.4,             # Value augmentation
    'degrees': 10.0,          # Rotation degrees
    'translate': 0.1,         # Translation
    'scale': 0.5,             # Scale augmentation
    'shear': 2.0,             # Shear augmentation
    'perspective': 0.0,       # No perspective (preserve medical accuracy)
    'flipud': 0.0,            # No vertical flip
    'fliplr': 0.5,            # Horizontal flip OK
    'mosaic': 1.0,            # Mosaic augmentation
    'mixup': 0.1,             # Mixup augmentation
    
    # Optimization
    'optimizer': 'AdamW',      # AdamW optimizer
    'cos_lr': True,           # Cosine learning rate
    'amp': True,              # Mixed precision
    'close_mosaic': 10,       # Disable mosaic in last 10 epochs
    
    # Validation
    'val': True,
    'save_period': 10,        # Save every 10 epochs
    'plots': True,            # Generate plots
}

print("⚙️  Colab Pro configuration loaded")
print(f"   Batch size: {COLAB_PRO_CONFIG['batch_size']}")
print(f"   Epochs: {COLAB_PRO_CONFIG['epochs']}")
print(f"   Workers: {COLAB_PRO_CONFIG['workers']}")

# ============================================================================
# DATASET DOWNLOAD AND PREPARATION
# ============================================================================

def download_and_prepare_dataset():
    """Download Kaggle dataset and convert to YOLOv8 format."""
    
    print("📥 Downloading Kaggle malaria dataset...")
    start_time = time.time()
    
    # Download dataset
    kaggle_path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
    
    download_time = time.time() - start_time
    print(f"✅ Dataset downloaded in {download_time:.1f}s to: {kaggle_path}")
    
    return kaggle_path

def generate_cell_bbox(image_path, padding_ratio=0.15):
    """Generate bounding box using advanced contour detection."""
    
    image = cv2.imread(image_path)
    if image is None:
        return (0.5, 0.5, 0.85, 0.85)  # Larger default bbox
    
    h, w = image.shape[:2]
    
    # Enhanced preprocessing for better contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Adaptive thresholding for better cell detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter contours by area (remove noise)
        min_area = (w * h) * 0.01  # At least 1% of image
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if valid_contours:
            # Get largest valid contour
            largest_contour = max(valid_contours, key=cv2.contourArea)
            x, y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
            
            # Add padding
            pad_x = int(bbox_w * padding_ratio)
            pad_y = int(bbox_h * padding_ratio)
            
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            bbox_w = min(w - x, bbox_w + 2 * pad_x)
            bbox_h = min(h - y, bbox_h + 2 * pad_y)
            
            # Convert to YOLO format
            center_x = (x + bbox_w / 2) / w
            center_y = (y + bbox_h / 2) / h
            norm_width = bbox_w / w
            norm_height = bbox_h / h
            
            # Ensure reasonable bounds
            center_x = max(0.1, min(0.9, center_x))
            center_y = max(0.1, min(0.9, center_y))
            norm_width = max(0.1, min(0.8, norm_width))
            norm_height = max(0.1, min(0.8, norm_height))
            
            return (center_x, center_y, norm_width, norm_height)
    
    # Enhanced fallback
    return (0.5, 0.5, 0.85, 0.85)

def create_yolo_dataset(kaggle_path):
    """Convert Kaggle dataset to YOLOv8 format with enhanced processing."""
    
    print("🔄 Converting dataset to YOLOv8 format...")
    conversion_start = time.time()
    
    # Create directory structure
    yolo_path = Path("yolo_malaria_pro")
    for split in ['train', 'val', 'test']:
        (yolo_path / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_path / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Find cell images directory
    kaggle_path = Path(kaggle_path)
    cell_images_path = None
    
    for root, dirs, files in os.walk(kaggle_path):
        if 'Parasitized' in dirs and 'Uninfected' in dirs:
            cell_images_path = Path(root)
            break
    
    if not cell_images_path:
        raise ValueError("Could not find Parasitized and Uninfected directories")
    
    print(f"📁 Found cell images at: {cell_images_path}")
    
    # Process each class
    all_files = []
    class_stats = {}
    
    for class_name in ['Parasitized', 'Uninfected']:
        class_path = cell_images_path / class_name
        if not class_path.exists():
            continue
            
        class_files = list(class_path.glob('*.png'))
        class_stats[class_name] = len(class_files)
        print(f"📊 {class_name}: {len(class_files)} images")
        
        # Class 0 for parasitized, None for uninfected
        class_id = 0 if class_name == 'Parasitized' else None
        
        for img_path in class_files:
            all_files.append((img_path, class_id))
    
    # Enhanced dataset split for better validation
    random.shuffle(all_files)
    total = len(all_files)
    
    # 70% train, 20% val, 10% test
    train_end = int(total * 0.70)
    val_end = int(total * 0.90)
    
    splits = {
        'train': all_files[:train_end],
        'val': all_files[train_end:val_end],
        'test': all_files[val_end:]
    }
    
    print(f"📈 Dataset split:")
    for split_name, files in splits.items():
        parasitized_count = sum(1 for _, class_id in files if class_id is not None)
        print(f"   {split_name}: {len(files)} total, {parasitized_count} parasitized")
    
    # Process files with progress tracking
    for split_name, files in splits.items():
        print(f"🔄 Processing {split_name} split...")
        
        images_dir = yolo_path / split_name / "images"
        labels_dir = yolo_path / split_name / "labels"
        
        for i, (img_path, class_id) in enumerate(files):
            if i % 2000 == 0 and i > 0:
                print(f"   Progress: {i}/{len(files)} ({i/len(files)*100:.1f}%)")
            
            # Copy image with new naming
            new_img_name = f"{split_name}_{i:06d}.png"
            new_img_path = images_dir / new_img_name
            shutil.copy2(img_path, new_img_path)
            
            # Create label file
            label_path = labels_dir / f"{split_name}_{i:06d}.txt"
            
            if class_id is not None:  # Parasitized cell
                bbox = generate_cell_bbox(str(img_path))
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            else:  # Uninfected cell
                label_path.touch()  # Empty file
        
        print(f"✅ Completed {split_name}: {len(files)} images processed")
    
    # Create enhanced data.yaml
    yaml_content = f"""# Malaria Detection Dataset - Colab Pro Optimized
path: {yolo_path.absolute()}
train: train/images
val: val/images  
test: test/images

# Classes
nc: 1
names: ['malaria_parasite']

# Dataset statistics
total_images: {total}
train_images: {len(splits['train'])}
val_images: {len(splits['val'])}
test_images: {len(splits['test'])}
parasitized_total: {class_stats['Parasitized']}
uninfected_total: {class_stats['Uninfected']}

# Conversion info
converted_on: {datetime.now().isoformat()}
conversion_method: "Enhanced contour detection with CLAHE preprocessing"
bbox_padding: 0.15
"""
    
    yaml_path = yolo_path / "malaria_data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    conversion_time = time.time() - conversion_start
    print(f"✅ Dataset conversion completed in {conversion_time:.1f}s")
    print(f"📄 Data config: {yaml_path}")
    
    return yaml_path, yolo_path

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

def train_yolov8_pro(data_yaml):
    """Train YOLOv8 with Colab Pro optimized settings."""
    
    print("🚀 Starting YOLOv8 training with Colab Pro optimization...")
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    print(f"📦 Model loaded: {sum(p.numel() for p in model.model.parameters()):,} parameters")
    
    # Training configuration
    train_args = {
        'data': str(data_yaml),
        'epochs': COLAB_PRO_CONFIG['epochs'],
        'batch': COLAB_PRO_CONFIG['batch_size'],
        'imgsz': COLAB_PRO_CONFIG['image_size'],
        'workers': COLAB_PRO_CONFIG['workers'],
        'cache': COLAB_PRO_CONFIG['cache'],
        'device': 0 if torch.cuda.is_available() else 'cpu',
        
        # Learning parameters
        'lr0': COLAB_PRO_CONFIG['lr0'],
        'weight_decay': COLAB_PRO_CONFIG['weight_decay'],
        'momentum': COLAB_PRO_CONFIG['momentum'],
        'patience': COLAB_PRO_CONFIG['patience'],
        
        # Augmentation
        'hsv_h': COLAB_PRO_CONFIG['hsv_h'],
        'hsv_s': COLAB_PRO_CONFIG['hsv_s'],
        'hsv_v': COLAB_PRO_CONFIG['hsv_v'],
        'degrees': COLAB_PRO_CONFIG['degrees'],
        'translate': COLAB_PRO_CONFIG['translate'],
        'scale': COLAB_PRO_CONFIG['scale'],
        'shear': COLAB_PRO_CONFIG['shear'],
        'perspective': COLAB_PRO_CONFIG['perspective'],
        'flipud': COLAB_PRO_CONFIG['flipud'],
        'fliplr': COLAB_PRO_CONFIG['fliplr'],
        'mosaic': COLAB_PRO_CONFIG['mosaic'],
        'mixup': COLAB_PRO_CONFIG['mixup'],
        
        # Optimization
        'optimizer': COLAB_PRO_CONFIG['optimizer'],
        'cos_lr': COLAB_PRO_CONFIG['cos_lr'],
        'amp': COLAB_PRO_CONFIG['amp'],
        'close_mosaic': COLAB_PRO_CONFIG['close_mosaic'],
        
        # Validation and saving
        'val': COLAB_PRO_CONFIG['val'],
        'save_period': COLAB_PRO_CONFIG['save_period'],
        'plots': COLAB_PRO_CONFIG['plots'],
        
        # Project settings
        'project': 'malaria_detection_pro',
        'name': 'yolov8n_colab_pro',
        'exist_ok': True,
        'verbose': True,
        'seed': 42,
    }
    
    print("⚙️  Training configuration:")
    key_params = ['epochs', 'batch', 'imgsz', 'workers', 'lr0', 'optimizer']
    for param in key_params:
        print(f"   {param}: {train_args[param]}")
    
    # Start training
    training_start = time.time()
    print(f"🏁 Training started at {datetime.now().strftime('%H:%M:%S')}")
    
    results = model.train(**train_args)
    
    training_time = time.time() - training_start
    print(f"🏆 Training completed in {training_time/3600:.1f} hours")
    print(f"💾 Best model: {model.trainer.best}")
    
    return model, results

# ============================================================================
# EVALUATION AND EXPORT
# ============================================================================

def evaluate_and_export(model, data_yaml):
    """Comprehensive model evaluation and export."""
    
    print("📊 Evaluating model performance...")
    
    # Load best model
    best_model = YOLO(model.trainer.best)
    
    # Validate on test set
    test_results = best_model.val(data=str(data_yaml), split='test')
    
    # Print results
    print("🎯 Test Results:")
    print(f"   mAP50: {test_results.box.map50:.4f}")
    print(f"   mAP50-95: {test_results.box.map:.4f}")
    print(f"   Precision: {test_results.box.mp:.4f}")
    print(f"   Recall: {test_results.box.mr:.4f}")
    
    # Export models
    print("📦 Exporting models...")
    
    exports = {}
    
    # ONNX export (production ready)
    try:
        onnx_path = best_model.export(format='onnx', optimize=True)
        exports['onnx'] = onnx_path
        print(f"✅ ONNX: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
    
    # TorchScript export
    try:
        torchscript_path = best_model.export(format='torchscript')
        exports['torchscript'] = torchscript_path
        print(f"✅ TorchScript: {torchscript_path}")
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
    
    return best_model, test_results, exports

def create_results_package(model, test_results, exports, yolo_path):
    """Create downloadable results package."""
    
    print("📦 Creating results package...")
    
    # Create zip file
    zip_name = f"malaria_detection_pro_results_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # Add models
        zipf.write(model.trainer.best, 'models/best_model.pt')
        
        for format_name, path in exports.items():
            if path and os.path.exists(path):
                zipf.write(path, f'models/best_model.{format_name}')
        
        # Add data config
        zipf.write(yolo_path / "malaria_data.yaml", 'config/malaria_data.yaml')
        
        # Add training results
        results_dir = Path(model.trainer.save_dir)
        for file in results_dir.glob('*.png'):
            zipf.write(file, f'results/{file.name}')
        
        if (results_dir / 'results.csv').exists():
            zipf.write(results_dir / 'results.csv', 'results/training_metrics.csv')
        
        # Add summary
        summary = f"""# Malaria Detection Training Results - Colab Pro

## Model Performance
- mAP50: {test_results.box.map50:.4f}
- mAP50-95: {test_results.box.map:.4f}  
- Precision: {test_results.box.mp:.4f}
- Recall: {test_results.box.mr:.4f}

## Training Configuration
- Model: YOLOv8n
- Epochs: {COLAB_PRO_CONFIG['epochs']}
- Batch Size: {COLAB_PRO_CONFIG['batch_size']}
- Image Size: {COLAB_PRO_CONFIG['image_size']}
- Optimizer: {COLAB_PRO_CONFIG['optimizer']}

## Files Included
- models/best_model.pt - PyTorch model
- models/best_model.onnx - ONNX model (production)
- models/best_model.torchscript - TorchScript model
- config/malaria_data.yaml - Dataset configuration
- results/ - Training plots and metrics

## Usage
Load the model: model = YOLO('models/best_model.pt')
Run inference: results = model('image.jpg')

Generated on: {datetime.now().isoformat()}
"""
        
        zipf.writestr('README.md', summary)
    
    print(f"✅ Results package created: {zip_name}")
    return zip_name

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline for Colab Pro."""
    
    print("🚀 YOLOv8 Malaria Detection - Colab Pro Training")
    print("=" * 60)
    
    total_start = time.time()
    
    try:
        # Step 1: Download dataset
        kaggle_path = download_and_prepare_dataset()
        
        # Step 2: Convert to YOLOv8 format
        data_yaml, yolo_path = create_yolo_dataset(kaggle_path)
        
        # Step 3: Train model
        model, training_results = train_yolov8_pro(data_yaml)
        
        # Step 4: Evaluate and export
        best_model, test_results, exports = evaluate_and_export(model, data_yaml)
        
        # Step 5: Create results package
        results_zip = create_results_package(model, test_results, exports, yolo_path)
        
        # Final summary
        total_time = time.time() - total_start
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {total_time/3600:.1f} hours")
        print(f"🎯 Final mAP50: {test_results.box.map50:.4f}")
        print(f"📦 Results package: {results_zip}")
        print("=" * 60)
        
        return {
            'model': best_model,
            'results': test_results,
            'exports': exports,
            'package': results_zip
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    # Run the complete training pipeline
    results = main()
