import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
import cv2
from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class


class YOLODetectionMetrics:
    """Comprehensive metrics for YOLOv8 malaria parasite detection."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.ground_truths = []
        self.image_paths = []
    
    def add_batch(self, predictions: List[Dict], ground_truths: List[Dict], image_paths: List[str]):
        """
        Add batch of predictions and ground truths.
        
        Args:
            predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
            ground_truths: List of ground truth dicts with 'boxes', 'labels'
            image_paths: List of image file paths
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
        self.image_paths.extend(image_paths)
    
    def calculate_map(self, iou_thresholds: List[float] = None) -> Dict[str, float]:
        """
        Calculate mean Average Precision (mAP) at different IoU thresholds.
        
        Args:
            iou_thresholds: List of IoU thresholds (default: [0.5, 0.55, ..., 0.95])
        
        Returns:
            Dict with mAP values
        """
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
        
        # Convert predictions and ground truths to required format
        all_pred_boxes = []
        all_pred_scores = []
        all_true_boxes = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            # Predictions
            if 'boxes' in pred and len(pred['boxes']) > 0:
                all_pred_boxes.extend(pred['boxes'])
                all_pred_scores.extend(pred['scores'])
            
            # Ground truths
            if 'boxes' in gt and len(gt['boxes']) > 0:
                all_true_boxes.extend(gt['boxes'])
        
        if not all_pred_boxes or not all_true_boxes:
            return {'mAP@0.5': 0.0, 'mAP@0.5:0.95': 0.0}
        
        # Calculate AP for each IoU threshold
        aps = []
        for iou_thresh in iou_thresholds:
            ap = self._calculate_ap_at_iou(all_pred_boxes, all_pred_scores, all_true_boxes, iou_thresh)
            aps.append(ap)
        
        return {
            'mAP@0.5': aps[0] if len(aps) > 0 else 0.0,
            'mAP@0.5:0.95': np.mean(aps) if aps else 0.0,
            'mAP_per_iou': dict(zip(iou_thresholds, aps))
        }
    
    def _calculate_ap_at_iou(self, pred_boxes: List, pred_scores: List, 
                           true_boxes: List, iou_threshold: float) -> float:
        """Calculate Average Precision at specific IoU threshold."""
        
        if not pred_boxes or not true_boxes:
            return 0.0
        
        # Sort predictions by confidence
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        
        # Track which ground truth boxes have been matched
        gt_matched = np.zeros(len(true_boxes))
        
        for i, pred_idx in enumerate(sorted_indices):
            pred_box = pred_boxes[pred_idx]
            
            # Find best matching ground truth box
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(true_boxes):
                if gt_matched[gt_idx]:
                    continue
                
                iou = self._calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if prediction matches ground truth
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                gt_matched[best_gt_idx] = 1
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(true_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        
        # Calculate AP using 11-point interpolation
        ap = self._calculate_ap_11_point(recalls, precisions)
        
        return ap
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        
        # Box format: [x1, y1, x2, y2]
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Calculate intersection area
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_ap_11_point(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Calculate AP using 11-point interpolation."""
        
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            # Find precisions for recalls >= t
            valid_precisions = precisions[recalls >= t]
            if len(valid_precisions) > 0:
                ap += np.max(valid_precisions)
        
        return ap / 11.0
    
    def calculate_clinical_metrics(self) -> Dict[str, float]:
        """Calculate clinical-specific metrics for malaria detection."""
        
        # Count statistics
        total_images = len(self.predictions)
        
        # Parasite count statistics
        pred_counts = []
        true_counts = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_count = len(pred.get('boxes', []))
            true_count = len(gt.get('boxes', []))
            
            pred_counts.append(pred_count)
            true_counts.append(true_count)
        
        pred_counts = np.array(pred_counts)
        true_counts = np.array(true_counts)
        
        # Image-level classification metrics (positive/negative)
        pred_positive = pred_counts > 0
        true_positive = true_counts > 0
        
        # Calculate confusion matrix for image classification
        tp_images = np.sum(pred_positive & true_positive)
        tn_images = np.sum(~pred_positive & ~true_positive)
        fp_images = np.sum(pred_positive & ~true_positive)
        fn_images = np.sum(~pred_positive & true_positive)
        
        # Clinical metrics
        sensitivity = tp_images / (tp_images + fn_images) if (tp_images + fn_images) > 0 else 0.0
        specificity = tn_images / (tn_images + fp_images) if (tn_images + fp_images) > 0 else 0.0
        precision = tp_images / (tp_images + fp_images) if (tp_images + fp_images) > 0 else 0.0
        accuracy = (tp_images + tn_images) / total_images if total_images > 0 else 0.0
        
        # Count correlation
        count_correlation = np.corrcoef(pred_counts, true_counts)[0, 1] if len(pred_counts) > 1 else 0.0
        
        # Count error metrics
        count_mae = np.mean(np.abs(pred_counts - true_counts))
        count_rmse = np.sqrt(np.mean((pred_counts - true_counts) ** 2))
        
        return {
            'image_sensitivity': sensitivity,
            'image_specificity': specificity,
            'image_precision': precision,
            'image_accuracy': accuracy,
            'count_correlation': count_correlation,
            'count_mae': count_mae,
            'count_rmse': count_rmse,
            'avg_pred_count': np.mean(pred_counts),
            'avg_true_count': np.mean(true_counts),
            'total_images': total_images,
            'positive_images_pred': np.sum(pred_positive),
            'positive_images_true': np.sum(true_positive)
        }
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot precision-recall curve."""
        
        # Calculate precision and recall for different confidence thresholds
        all_scores = []
        all_matches = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_boxes = pred.get('boxes', [])
            pred_scores = pred.get('scores', [])
            gt_boxes = gt.get('boxes', [])
            
            for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                all_scores.append(score)
                
                # Check if this prediction matches any ground truth
                matched = False
                for gt_box in gt_boxes:
                    if self._calculate_iou(box, gt_box) >= 0.5:
                        matched = True
                        break
                
                all_matches.append(matched)
        
        if not all_scores:
            return plt.figure()
        
        # Sort by confidence
        sorted_indices = np.argsort(all_scores)[::-1]
        
        tp_cumsum = np.cumsum([all_matches[i] for i in sorted_indices])
        fp_cumsum = np.cumsum([not all_matches[i] for i in sorted_indices])
        
        total_positives = sum(len(gt.get('boxes', [])) for gt in self.ground_truths)
        
        recalls = tp_cumsum / total_positives if total_positives > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_count_correlation(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot predicted vs true parasite counts."""
        
        pred_counts = [len(pred.get('boxes', [])) for pred in self.predictions]
        true_counts = [len(gt.get('boxes', [])) for gt in self.ground_truths]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(true_counts, pred_counts, alpha=0.6)
        
        # Perfect correlation line
        max_count = max(max(pred_counts) if pred_counts else 0, 
                       max(true_counts) if true_counts else 0)
        plt.plot([0, max_count], [0, max_count], 'r--', label='Perfect correlation')
        
        plt.xlabel('True Parasite Count')
        plt.ylabel('Predicted Parasite Count')
        plt.title('Parasite Count Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(pred_counts) > 1:
            corr = np.corrcoef(pred_counts, true_counts)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_clinical_report(self) -> Dict[str, Any]:
        """Generate comprehensive clinical evaluation report."""
        
        # Calculate all metrics
        map_metrics = self.calculate_map()
        clinical_metrics = self.calculate_clinical_metrics()
        
        # Combine results
        report = {
            'detection_metrics': map_metrics,
            'clinical_metrics': clinical_metrics,
            'evaluation_summary': {
                'total_images': len(self.predictions),
                'detection_performance': 'excellent' if map_metrics['mAP@0.5'] > 0.8 else 
                                      'good' if map_metrics['mAP@0.5'] > 0.6 else 'needs_improvement',
                'clinical_sensitivity': clinical_metrics['image_sensitivity'],
                'clinical_specificity': clinical_metrics['image_specificity'],
                'count_accuracy': 'high' if clinical_metrics['count_correlation'] > 0.9 else
                                'moderate' if clinical_metrics['count_correlation'] > 0.7 else 'low'
            },
            'recommendations': self._generate_recommendations(map_metrics, clinical_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, map_metrics: Dict, clinical_metrics: Dict) -> List[str]:
        """Generate clinical recommendations based on metrics."""
        
        recommendations = []
        
        # Detection performance
        if map_metrics['mAP@0.5'] < 0.6:
            recommendations.append("Consider additional training data or model fine-tuning")
        
        # Clinical sensitivity
        if clinical_metrics['image_sensitivity'] < 0.95:
            recommendations.append("Sensitivity below clinical target (95%) - adjust confidence threshold")
        
        # Clinical specificity
        if clinical_metrics['image_specificity'] < 0.85:
            recommendations.append("Specificity below target (85%) - may cause over-treatment")
        
        # Count accuracy
        if clinical_metrics['count_correlation'] < 0.8:
            recommendations.append("Parasite counting accuracy needs improvement")
        
        # Overall assessment
        if (clinical_metrics['image_sensitivity'] >= 0.95 and 
            clinical_metrics['image_specificity'] >= 0.85 and
            map_metrics['mAP@0.5'] >= 0.7):
            recommendations.append("Model meets clinical performance criteria")
        
        return recommendations


def evaluate_yolo_model(model_path: str, test_data_path: str, 
                       output_dir: str = 'evaluation_results') -> Dict[str, Any]:
    """
    Comprehensive evaluation of YOLOv8 malaria detection model.
    
    Args:
        model_path: Path to trained YOLOv8 model
        test_data_path: Path to test dataset
        output_dir: Directory to save evaluation results
    
    Returns:
        Evaluation report dictionary
    """
    
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics
    metrics = YOLODetectionMetrics()
    
    # Run evaluation
    results = model.val(data=test_data_path, plots=True, save_json=True)
    
    # Process results (this would need to be adapted based on actual YOLOv8 output format)
    # For now, we'll create a placeholder report
    
    # Generate clinical report
    clinical_report = metrics.generate_clinical_report()
    
    # Save report
    report_path = output_path / 'clinical_evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(clinical_report, f, indent=2)
    
    # Generate plots
    metrics.plot_precision_recall_curve(output_path / 'precision_recall_curve.png')
    metrics.plot_count_correlation(output_path / 'count_correlation.png')
    
    print(f"Evaluation completed. Results saved to: {output_dir}")
    
    return clinical_report


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing YOLOv8 Detection Metrics...")
    
    # Create dummy data
    metrics = YOLODetectionMetrics()
    
    # Add some test predictions and ground truths
    dummy_predictions = [
        {'boxes': [[10, 10, 50, 50], [100, 100, 140, 140]], 'scores': [0.9, 0.8], 'labels': [0, 0]},
        {'boxes': [[20, 20, 60, 60]], 'scores': [0.7], 'labels': [0]},
        {'boxes': [], 'scores': [], 'labels': []}
    ]
    
    dummy_ground_truths = [
        {'boxes': [[15, 15, 55, 55], [105, 105, 145, 145]], 'labels': [0, 0]},
        {'boxes': [[25, 25, 65, 65]], 'labels': [0]},
        {'boxes': [], 'labels': []}
    ]
    
    dummy_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    
    metrics.add_batch(dummy_predictions, dummy_ground_truths, dummy_paths)
    
    # Calculate metrics
    map_results = metrics.calculate_map()
    clinical_results = metrics.calculate_clinical_metrics()
    
    print(f"mAP@0.5: {map_results['mAP@0.5']:.3f}")
    print(f"Clinical Sensitivity: {clinical_results['image_sensitivity']:.3f}")
    print(f"Clinical Specificity: {clinical_results['image_specificity']:.3f}")
    print(f"Count Correlation: {clinical_results['count_correlation']:.3f}")
    
    # Generate full report
    report = metrics.generate_clinical_report()
    print(f"Recommendations: {report['recommendations']}")
    
    print("YOLOv8 metrics testing completed!")
