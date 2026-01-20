import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import yaml


class MalariaYOLOv8:
    """YOLOv8-based malaria parasite detection and counting system."""
    
    def __init__(self, model_size='n', num_classes=1, pretrained=True):
        """
        Initialize YOLOv8 model for malaria detection.
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
            num_classes: Number of classes (1 for malaria parasites)
            pretrained: Use COCO pretrained weights
        """
        self.model_size = model_size
        self.num_classes = num_classes
        
        # Initialize YOLOv8 model
        if pretrained:
            self.model = YOLO(f'yolov8{model_size}.pt')
        else:
            self.model = YOLO(f'yolov8{model_size}.yaml')
        
        # Configure for malaria detection
        self._configure_model()
    
    def _configure_model(self):
        """Configure model for malaria-specific detection."""
        # Update model configuration for single class
        if hasattr(self.model.model, 'yaml'):
            self.model.model.yaml['nc'] = self.num_classes
        
        # Set class names
        self.class_names = ['malaria_parasite']
    
    def train(self, data_config, epochs=100, imgsz=640, batch_size=16, 
              lr0=0.01, weight_decay=0.0005, **kwargs):
        """
        Train YOLOv8 model on malaria dataset.
        
        Args:
            data_config: Path to data configuration YAML
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Training batch size
            lr0: Initial learning rate
            weight_decay: Weight decay for optimizer
        """
        
        # Training parameters optimized for medical imaging
        train_args = {
            'data': data_config,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'lr0': lr0,
            'weight_decay': weight_decay,
            'patience': 20,  # Early stopping patience
            'save_period': 10,  # Save checkpoint every N epochs
            'val': True,
            'plots': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            
            # Medical imaging specific augmentations
            'hsv_h': 0.015,  # Hue augmentation (reduced for medical images)
            'hsv_s': 0.7,    # Saturation augmentation
            'hsv_v': 0.4,    # Value augmentation
            'degrees': 10.0,  # Rotation (reduced for medical accuracy)
            'translate': 0.1, # Translation
            'scale': 0.2,     # Scale augmentation
            'shear': 2.0,     # Shear augmentation
            'perspective': 0.0, # Disable perspective (preserve medical accuracy)
            'flipud': 0.0,    # No vertical flip (preserve orientation)
            'fliplr': 0.5,    # Horizontal flip OK
            'mosaic': 0.5,    # Reduced mosaic for medical images
            'mixup': 0.1,     # Light mixup
            'copy_paste': 0.0, # Disable copy-paste for medical accuracy
        }
        
        # Add any additional arguments
        train_args.update(kwargs)
        
        # Start training
        results = self.model.train(**train_args)
        return results
    
    def predict(self, source, conf=0.25, iou=0.7, save=False, **kwargs):
        """
        Run inference on images.
        
        Args:
            source: Image path, directory, or array
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Save prediction results
        """
        
        predict_args = {
            'source': source,
            'conf': conf,
            'iou': iou,
            'save': save,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        predict_args.update(kwargs)
        
        results = self.model.predict(**predict_args)
        return results
    
    def validate(self, data_config, **kwargs):
        """Validate model performance."""
        val_args = {
            'data': data_config,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        val_args.update(kwargs)
        
        results = self.model.val(**val_args)
        return results
    
    def export(self, format='onnx', **kwargs):
        """Export model to various formats."""
        export_args = {
            'format': format,
            'imgsz': 640,
            'optimize': True,
        }
        export_args.update(kwargs)
        
        path = self.model.export(**export_args)
        return path
    
    def count_parasites(self, results, confidence_threshold=0.5):
        """
        Count parasites from detection results.
        
        Args:
            results: YOLOv8 prediction results
            confidence_threshold: Minimum confidence for counting
            
        Returns:
            dict: Parasite count and statistics
        """
        
        if not results:
            return {'count': 0, 'detections': [], 'avg_confidence': 0.0}
        
        total_count = 0
        all_detections = []
        confidences = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                confs = boxes.conf.cpu().numpy()
                
                # Filter by confidence
                valid_detections = confs >= confidence_threshold
                count = valid_detections.sum()
                total_count += count
                
                if count > 0:
                    valid_confs = confs[valid_detections]
                    confidences.extend(valid_confs)
                    
                    # Get bounding box coordinates
                    xyxy = boxes.xyxy.cpu().numpy()[valid_detections]
                    
                    for i, (conf, box) in enumerate(zip(valid_confs, xyxy)):
                        all_detections.append({
                            'confidence': float(conf),
                            'bbox': box.tolist(),  # [x1, y1, x2, y2]
                            'area': float((box[2] - box[0]) * (box[3] - box[1]))
                        })
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'count': total_count,
            'detections': all_detections,
            'avg_confidence': float(avg_confidence),
            'confidence_std': float(np.std(confidences)) if confidences else 0.0
        }
    
    def clinical_analysis(self, results, image_area=None):
        """
        Perform clinical analysis of detection results.
        
        Args:
            results: YOLOv8 prediction results
            image_area: Total image area for density calculation
            
        Returns:
            dict: Clinical analysis results
        """
        
        parasite_stats = self.count_parasites(results)
        
        # Calculate parasite density if image area provided
        density = None
        if image_area and parasite_stats['count'] > 0:
            density = parasite_stats['count'] / image_area
        
        # Determine infection level based on count
        count = parasite_stats['count']
        if count == 0:
            infection_level = 'negative'
            clinical_significance = 'No parasites detected'
        elif count <= 5:
            infection_level = 'low'
            clinical_significance = 'Low parasitemia - monitor patient'
        elif count <= 20:
            infection_level = 'moderate'
            clinical_significance = 'Moderate parasitemia - treat immediately'
        else:
            infection_level = 'high'
            clinical_significance = 'High parasitemia - urgent treatment required'
        
        # Confidence assessment
        avg_conf = parasite_stats['avg_confidence']
        if avg_conf >= 0.8:
            confidence_level = 'high'
        elif avg_conf >= 0.5:
            confidence_level = 'moderate'
        else:
            confidence_level = 'low'
        
        return {
            'parasite_count': count,
            'infection_level': infection_level,
            'clinical_significance': clinical_significance,
            'confidence_level': confidence_level,
            'avg_confidence': avg_conf,
            'parasite_density': density,
            'detections': parasite_stats['detections']
        }


def create_data_config(train_dir, val_dir, test_dir=None, save_path='malaria_data.yaml'):
    """
    Create YOLOv8 data configuration file.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory  
        test_dir: Test data directory (optional)
        save_path: Path to save configuration file
    """
    
    config = {
        'path': str(Path(train_dir).parent),  # Root directory
        'train': str(Path(train_dir).name),   # Relative to path
        'val': str(Path(val_dir).name),       # Relative to path
        'nc': 1,  # Number of classes
        'names': ['malaria_parasite']  # Class names
    }
    
    if test_dir:
        config['test'] = str(Path(test_dir).name)
    
    # Save configuration
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Data configuration saved to: {save_path}")
    return save_path


def convert_coco_to_yolo(coco_annotations, image_width, image_height):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_annotations: List of COCO format bounding boxes
        image_width: Image width
        image_height: Image height
        
    Returns:
        list: YOLO format annotations
    """
    
    yolo_annotations = []
    
    for ann in coco_annotations:
        # COCO format: [x, y, width, height] (top-left corner)
        x, y, w, h = ann['bbox']
        
        # Convert to YOLO format: [class, x_center, y_center, width, height] (normalized)
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        norm_width = w / image_width
        norm_height = h / image_height
        
        # Class ID (0 for malaria parasite)
        class_id = 0
        
        yolo_annotations.append([class_id, x_center, y_center, norm_width, norm_height])
    
    return yolo_annotations


if __name__ == "__main__":
    # Test YOLOv8 malaria detection
    print("Testing YOLOv8 Malaria Detection...")
    
    # Initialize model
    model = MalariaYOLOv8(model_size='n', num_classes=1)
    
    print(f"Model initialized: YOLOv8{model.model_size}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Class names: {model.class_names}")
    
    # Test data config creation
    config_path = create_data_config(
        train_dir='data/train',
        val_dir='data/val',
        test_dir='data/test'
    )
    
    print(f"Data configuration created: {config_path}")
    
    # Test COCO to YOLO conversion
    sample_coco = [
        {'bbox': [100, 150, 50, 60]},  # x, y, w, h
        {'bbox': [200, 250, 40, 45]}
    ]
    
    yolo_format = convert_coco_to_yolo(sample_coco, 640, 480)
    print(f"COCO to YOLO conversion test: {yolo_format}")
