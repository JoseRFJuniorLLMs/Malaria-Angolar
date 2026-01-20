import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


class ClinicalVisualization:
    """Clinical visualization tools for YOLOv8 malaria detection results."""
    
    def __init__(self):
        self.colors = {
            'parasite': (0, 255, 0),      # Green for parasites
            'high_conf': (0, 255, 0),     # Green for high confidence
            'medium_conf': (255, 165, 0),  # Orange for medium confidence
            'low_conf': (255, 0, 0),      # Red for low confidence
            'text': (255, 255, 255),      # White text
            'background': (0, 0, 0)       # Black background
        }
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           save_path: Optional[str] = None, 
                           show_confidence: bool = True,
                           show_count: bool = True) -> np.ndarray:
        """
        Visualize parasite detections on microscope image.
        
        Args:
            image: Input microscope image (numpy array)
            detections: List of detection dictionaries with bbox, confidence
            save_path: Optional path to save visualization
            show_confidence: Whether to show confidence scores
            show_count: Whether to show parasite count
        
        Returns:
            Annotated image
        """
        
        # Create copy of image
        vis_image = image.copy()
        
        # Draw bounding boxes and labels
        for i, detection in enumerate(detections):
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            confidence = detection['confidence']
            
            # Determine color based on confidence
            if confidence >= 0.8:
                color = self.colors['high_conf']
            elif confidence >= 0.5:
                color = self.colors['medium_conf']
            else:
                color = self.colors['low_conf']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add confidence label if requested
            if show_confidence:
                label = f"P{i+1}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Background for text
                cv2.rectangle(vis_image, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Text
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          self.colors['text'], 1)
        
        # Add parasite count if requested
        if show_count:
            count_text = f"Parasites: {len(detections)}"
            cv2.putText(vis_image, count_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                       self.colors['parasite'], 2)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def create_clinical_report_image(self, image: np.ndarray, 
                                   detection_results: Dict,
                                   patient_info: Optional[Dict] = None,
                                   save_path: Optional[str] = None) -> np.ndarray:
        """
        Create comprehensive clinical report visualization.
        
        Args:
            image: Original microscope image
            detection_results: Detection results from inference service
            patient_info: Optional patient information
            save_path: Optional path to save report
        
        Returns:
            Clinical report image
        """
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Malaria Detection Clinical Report', fontsize=16, fontweight='bold')
        
        # Original image with detections
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Microscope Field with Detections')
        axes[0, 0].axis('off')
        
        # Draw bounding boxes on matplotlib plot
        for detection in detection_results.get('detections', []):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Color based on confidence
            if confidence >= 0.8:
                color = 'green'
            elif confidence >= 0.5:
                color = 'orange'
            else:
                color = 'red'
            
            rect = Rectangle((bbox[0], bbox[1]), 
                           bbox[2] - bbox[0], bbox[3] - bbox[1],
                           linewidth=2, edgecolor=color, facecolor='none')
            axes[0, 0].add_patch(rect)
        
        # Detection statistics
        axes[0, 1].axis('off')
        stats_text = self._create_stats_text(detection_results, patient_info)
        axes[0, 1].text(0.1, 0.9, stats_text, transform=axes[0, 1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[0, 1].set_title('Detection Summary')
        
        # Confidence distribution
        if detection_results.get('detections'):
            confidences = [d['confidence'] for d in detection_results['detections']]
            axes[1, 0].hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Number of Detections')
            axes[1, 0].set_title('Confidence Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No detections found', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].set_title('Confidence Distribution')
        
        # Clinical interpretation
        axes[1, 1].axis('off')
        clinical_text = self._create_clinical_interpretation(detection_results)
        axes[1, 1].text(0.1, 0.9, clinical_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1, 1].set_title('Clinical Interpretation')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        report_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        report_image = report_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return report_image
    
    def _create_stats_text(self, results: Dict, patient_info: Optional[Dict] = None) -> str:
        """Create statistics text for clinical report."""
        
        lines = []
        
        # Patient information
        if patient_info:
            lines.append("PATIENT INFORMATION")
            lines.append("-" * 20)
            lines.append(f"Patient ID: {patient_info.get('patient_id', 'N/A')}")
            lines.append(f"Clinic: {patient_info.get('clinic_id', 'N/A')}")
            lines.append(f"Date: {patient_info.get('date', 'N/A')}")
            lines.append("")
        
        # Detection results
        lines.append("DETECTION RESULTS")
        lines.append("-" * 20)
        lines.append(f"Parasite Count: {results.get('parasite_count', 0)}")
        lines.append(f"Infection Level: {results.get('infection_level', 'Unknown').title()}")
        lines.append(f"Avg Confidence: {results.get('avg_confidence', 0):.3f}")
        lines.append(f"Confidence Level: {results.get('confidence_level', 'Unknown').title()}")
        
        if results.get('parasite_density'):
            lines.append(f"Density: {results['parasite_density']:.2f}/mm²")
        
        lines.append("")
        lines.append("PROCESSING INFO")
        lines.append("-" * 20)
        lines.append(f"Processing Time: {results.get('processing_time_ms', 0):.1f}ms")
        
        if results.get('image_dimensions'):
            w, h = results['image_dimensions']
            lines.append(f"Image Size: {w}×{h}")
        
        return "\n".join(lines)
    
    def _create_clinical_interpretation(self, results: Dict) -> str:
        """Create clinical interpretation text."""
        
        lines = []
        
        lines.append("CLINICAL SIGNIFICANCE")
        lines.append("-" * 25)
        lines.append(results.get('clinical_significance', 'No interpretation available'))
        lines.append("")
        
        # Recommendations based on results
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 15)
        
        infection_level = results.get('infection_level', '').lower()
        confidence_level = results.get('confidence_level', '').lower()
        
        if infection_level == 'negative':
            lines.append("• No parasites detected")
            if confidence_level == 'low':
                lines.append("• Consider manual review")
                lines.append("• Repeat test if clinical suspicion high")
        
        elif infection_level == 'low':
            lines.append("• Low parasitemia detected")
            lines.append("• Monitor patient closely")
            lines.append("• Consider treatment based on clinical symptoms")
        
        elif infection_level == 'moderate':
            lines.append("• Moderate parasitemia detected")
            lines.append("• Initiate antimalarial treatment")
            lines.append("• Monitor treatment response")
        
        elif infection_level == 'high':
            lines.append("• High parasitemia detected")
            lines.append("• URGENT: Immediate treatment required")
            lines.append("• Consider hospitalization")
            lines.append("• Monitor for complications")
        
        # Confidence-based recommendations
        if confidence_level == 'low':
            lines.append("")
            lines.append("⚠ LOW CONFIDENCE DETECTION")
            lines.append("• Manual microscopy review recommended")
            lines.append("• Consider repeat sample")
        
        return "\n".join(lines)
    
    def create_detection_heatmap(self, image_shape: Tuple[int, int], 
                               detections: List[Dict],
                               save_path: Optional[str] = None) -> np.ndarray:
        """
        Create heatmap showing parasite detection density.
        
        Args:
            image_shape: (height, width) of original image
            detections: List of detection dictionaries
            save_path: Optional path to save heatmap
        
        Returns:
            Heatmap image
        """
        
        # Create heatmap grid
        grid_size = 50
        h, w = image_shape[:2]
        heatmap = np.zeros((h // grid_size + 1, w // grid_size + 1))
        
        # Count detections in each grid cell
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            grid_x = int(center_x // grid_size)
            grid_y = int(center_y // grid_size)
            
            if 0 <= grid_y < heatmap.shape[0] and 0 <= grid_x < heatmap.shape[1]:
                heatmap[grid_y, grid_x] += 1
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap, cmap='Reds', annot=True, fmt='g', 
                   cbar_kws={'label': 'Parasite Count'})
        plt.title('Parasite Detection Density Heatmap')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to numpy array
        plt.tight_layout()
        fig = plt.gcf()
        fig.canvas.draw()
        heatmap_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        heatmap_image = heatmap_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return heatmap_image
    
    def create_comparison_view(self, images: List[np.ndarray], 
                             results_list: List[Dict],
                             titles: List[str],
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Create side-by-side comparison of multiple detection results.
        
        Args:
            images: List of microscope images
            results_list: List of detection results for each image
            titles: List of titles for each image
            save_path: Optional path to save comparison
        
        Returns:
            Comparison image
        """
        
        n_images = len(images)
        fig, axes = plt.subplots(2, n_images, figsize=(5*n_images, 10))
        
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        for i, (image, results, title) in enumerate(zip(images, results_list, titles)):
            # Original image with detections
            vis_image = self.visualize_detections(image, results.get('detections', []))
            axes[0, i].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f"{title}\nCount: {results.get('parasite_count', 0)}")
            axes[0, i].axis('off')
            
            # Statistics
            axes[1, i].axis('off')
            stats_text = f"""
            Parasites: {results.get('parasite_count', 0)}
            Level: {results.get('infection_level', 'Unknown').title()}
            Confidence: {results.get('avg_confidence', 0):.3f}
            """
            axes[1, i].text(0.1, 0.9, stats_text, transform=axes[1, i].transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Detection Results Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        comparison_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        comparison_image = comparison_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return comparison_image
    
    def save_detection_annotations(self, detections: List[Dict], 
                                 image_path: str, 
                                 output_path: str):
        """
        Save detection annotations in COCO format for future reference.
        
        Args:
            detections: List of detection dictionaries
            image_path: Path to original image
            output_path: Path to save annotations JSON
        """
        
        # Load image to get dimensions
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Create COCO-style annotation
        annotation = {
            "image": {
                "file_name": Path(image_path).name,
                "width": width,
                "height": height,
                "path": image_path
            },
            "annotations": []
        }
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            
            # Convert to COCO format [x, y, width, height]
            coco_bbox = [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1]
            ]
            
            annotation["annotations"].append({
                "id": i,
                "category_id": 1,  # Malaria parasite
                "bbox": coco_bbox,
                "area": coco_bbox[2] * coco_bbox[3],
                "confidence": detection['confidence']
            })
        
        # Save annotation
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)


def create_batch_visualization_report(image_dir: str, results_dir: str, 
                                    output_dir: str):
    """
    Create visualization reports for a batch of images.
    
    Args:
        image_dir: Directory containing microscope images
        results_dir: Directory containing detection results JSON files
        output_dir: Directory to save visualization reports
    """
    
    visualizer = ClinicalVisualization()
    
    image_dir = Path(image_dir)
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for image_path in image_dir.glob("*.jpg"):
        result_path = results_dir / f"{image_path.stem}_results.json"
        
        if not result_path.exists():
            continue
        
        # Load image and results
        image = cv2.imread(str(image_path))
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        # Create clinical report
        report_image = visualizer.create_clinical_report_image(
            image, results, 
            save_path=output_dir / f"{image_path.stem}_clinical_report.png"
        )
        
        # Create detection visualization
        vis_image = visualizer.visualize_detections(
            image, results.get('detections', []),
            save_path=output_dir / f"{image_path.stem}_detections.png"
        )
        
        print(f"Created visualizations for {image_path.name}")


if __name__ == "__main__":
    # Test clinical visualization
    print("Testing Clinical Visualization...")
    
    # Create dummy data
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    dummy_detections = [
        {
            'bbox': [100, 150, 150, 200],
            'confidence': 0.95,
            'center': [125, 175],
            'area': 2500
        },
        {
            'bbox': [300, 250, 340, 290],
            'confidence': 0.78,
            'center': [320, 270],
            'area': 1600
        }
    ]
    
    dummy_results = {
        'parasite_count': 2,
        'infection_level': 'low',
        'clinical_significance': 'Low parasitemia detected - monitor patient',
        'confidence_level': 'high',
        'avg_confidence': 0.865,
        'detections': dummy_detections,
        'processing_time_ms': 125.5,
        'image_dimensions': [640, 480]
    }
    
    # Test visualization
    visualizer = ClinicalVisualization()
    
    # Test detection visualization
    vis_image = visualizer.visualize_detections(dummy_image, dummy_detections)
    print(f"Detection visualization created: {vis_image.shape}")
    
    # Test clinical report
    report_image = visualizer.create_clinical_report_image(dummy_image, dummy_results)
    print(f"Clinical report created: {report_image.shape}")
    
    print("Clinical visualization testing completed!")
