#!/usr/bin/env python3
"""
YOLOv8 Model Export Utilities
Export trained YOLOv8 models for clinical deployment.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import torch


class YOLOModelExporter:
    """Export YOLOv8 models to various formats for deployment."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)
        
        print(f"Loaded YOLOv8 model: {model_path}")
        
    def export_onnx(self, output_path: str = None, imgsz: int = 640, 
                   optimize: bool = True, simplify: bool = True) -> str:
        """Export model to ONNX format."""
        
        print("Exporting to ONNX format...")
        
        export_path = self.model.export(
            format='onnx',
            imgsz=imgsz,
            optimize=optimize,
            simplify=simplify
        )
        
        if output_path:
            # Move to specified location
            import shutil
            shutil.move(export_path, output_path)
            export_path = output_path
        
        # Verify ONNX model
        self._verify_onnx_export(export_path, imgsz)
        
        print(f"ONNX export completed: {export_path}")
        return export_path
    
    def export_torchscript(self, output_path: str = None, imgsz: int = 640) -> str:
        """Export model to TorchScript format."""
        
        print("Exporting to TorchScript format...")
        
        export_path = self.model.export(
            format='torchscript',
            imgsz=imgsz
        )
        
        if output_path:
            import shutil
            shutil.move(export_path, output_path)
            export_path = output_path
        
        # Verify TorchScript model
        self._verify_torchscript_export(export_path, imgsz)
        
        print(f"TorchScript export completed: {export_path}")
        return export_path
    
    def export_tflite(self, output_path: str = None, imgsz: int = 640, 
                     int8: bool = False) -> str:
        """Export model to TensorFlow Lite format."""
        
        print("Exporting to TensorFlow Lite format...")
        
        format_type = 'tflite'
        if int8:
            format_type = 'int8'
        
        export_path = self.model.export(
            format=format_type,
            imgsz=imgsz
        )
        
        if output_path:
            import shutil
            shutil.move(export_path, output_path)
            export_path = output_path
        
        print(f"TensorFlow Lite export completed: {export_path}")
        return export_path
    
    def export_openvino(self, output_path: str = None, imgsz: int = 640) -> str:
        """Export model to OpenVINO format."""
        
        print("Exporting to OpenVINO format...")
        
        export_path = self.model.export(
            format='openvino',
            imgsz=imgsz
        )
        
        if output_path:
            import shutil
            shutil.move(export_path, output_path)
            export_path = output_path
        
        print(f"OpenVINO export completed: {export_path}")
        return export_path
    
    def _verify_onnx_export(self, onnx_path: str, imgsz: int):
        """Verify ONNX model works correctly."""
        
        try:
            import onnxruntime as ort
            
            # Load ONNX model
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            
            # Test with dummy input
            dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
            input_name = session.get_inputs()[0].name
            
            # Run inference
            start_time = time.time()
            outputs = session.run(None, {input_name: dummy_input})
            inference_time = (time.time() - start_time) * 1000
            
            print(f"✓ ONNX verification passed (inference: {inference_time:.1f}ms)")
            
        except Exception as e:
            print(f"⚠ ONNX verification failed: {e}")
    
    def _verify_torchscript_export(self, script_path: str, imgsz: int):
        """Verify TorchScript model works correctly."""
        
        try:
            # Load TorchScript model
            model = torch.jit.load(script_path, map_location='cpu')
            
            # Test with dummy input
            dummy_input = torch.randn(1, 3, imgsz, imgsz)
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                outputs = model(dummy_input)
            inference_time = (time.time() - start_time) * 1000
            
            print(f"✓ TorchScript verification passed (inference: {inference_time:.1f}ms)")
            
        except Exception as e:
            print(f"⚠ TorchScript verification failed: {e}")
    
    def benchmark_model(self, format_path: str, format_type: str, 
                       num_runs: int = 100, imgsz: int = 640) -> dict:
        """Benchmark model inference performance."""
        
        print(f"Benchmarking {format_type} model...")
        
        if format_type == 'onnx':
            return self._benchmark_onnx(format_path, num_runs, imgsz)
        elif format_type == 'torchscript':
            return self._benchmark_torchscript(format_path, num_runs, imgsz)
        elif format_type == 'pytorch':
            return self._benchmark_pytorch(num_runs, imgsz)
        else:
            print(f"Benchmarking not supported for {format_type}")
            return {}
    
    def _benchmark_onnx(self, onnx_path: str, num_runs: int, imgsz: int) -> dict:
        """Benchmark ONNX model."""
        
        import onnxruntime as ort
        
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            session.run(None, {input_name: dummy_input})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000
        throughput = 1000 / avg_time
        
        return {
            'format': 'ONNX',
            'avg_inference_time_ms': avg_time,
            'throughput_fps': throughput,
            'num_runs': num_runs
        }
    
    def _benchmark_torchscript(self, script_path: str, num_runs: int, imgsz: int) -> dict:
        """Benchmark TorchScript model."""
        
        model = torch.jit.load(script_path, map_location='cpu')
        dummy_input = torch.randn(1, 3, imgsz, imgsz)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                model(dummy_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000
        throughput = 1000 / avg_time
        
        return {
            'format': 'TorchScript',
            'avg_inference_time_ms': avg_time,
            'throughput_fps': throughput,
            'num_runs': num_runs
        }
    
    def _benchmark_pytorch(self, num_runs: int, imgsz: int) -> dict:
        """Benchmark original PyTorch model."""
        
        dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.model.predict(dummy_input, verbose=False)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            self.model.predict(dummy_input, verbose=False)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000
        throughput = 1000 / avg_time
        
        return {
            'format': 'PyTorch',
            'avg_inference_time_ms': avg_time,
            'throughput_fps': throughput,
            'num_runs': num_runs
        }
    
    def create_deployment_config(self, exported_models: dict, save_path: str):
        """Create deployment configuration file."""
        
        # Get model info
        model_info = {
            'model_type': 'YOLOv8',
            'task': 'detection',
            'classes': ['malaria_parasite'],
            'num_classes': 1,
            'input_size': [640, 640],
            'exported_models': exported_models
        }
        
        # Clinical configuration
        clinical_config = {
            'infection_thresholds': {
                'low': 5,
                'moderate': 20
            },
            'confidence_threshold': 0.25,
            'iou_threshold': 0.7,
            'min_confidence_clinical': 0.5
        }
        
        # Preprocessing configuration
        preprocessing_config = {
            'input_format': 'RGB',
            'normalization': 'yolo_default',  # YOLOv8 handles normalization internally
            'resize_method': 'letterbox'
        }
        
        # Postprocessing configuration
        postprocessing_config = {
            'nms_enabled': True,
            'max_detections': 100,
            'output_format': 'xyxy'  # [x1, y1, x2, y2]
        }
        
        config = {
            'model_info': model_info,
            'clinical': clinical_config,
            'preprocessing': preprocessing_config,
            'postprocessing': postprocessing_config,
            'deployment': {
                'recommended_format': 'onnx',
                'cpu_optimized': True,
                'gpu_support': True
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Deployment configuration saved: {save_path}")
        return config


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 malaria detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLOv8 model (.pt file)')
    parser.add_argument('--output-dir', type=str, default='exports',
                       help='Output directory for exported models')
    parser.add_argument('--formats', type=str, nargs='+', 
                       choices=['onnx', 'torchscript', 'tflite', 'openvino'],
                       default=['onnx'], help='Export formats')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark exported models')
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Optimize exported models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize exporter
    exporter = YOLOModelExporter(args.model)
    
    # Get model name for file naming
    model_name = Path(args.model).stem
    
    exported_models = {}
    benchmark_results = {}
    
    # Export to requested formats
    for format_type in args.formats:
        output_path = output_dir / f"{model_name}.{format_type}"
        
        try:
            if format_type == 'onnx':
                export_path = exporter.export_onnx(
                    str(output_path), args.imgsz, args.optimize
                )
            elif format_type == 'torchscript':
                export_path = exporter.export_torchscript(
                    str(output_path), args.imgsz
                )
            elif format_type == 'tflite':
                export_path = exporter.export_tflite(
                    str(output_path), args.imgsz
                )
            elif format_type == 'openvino':
                export_path = exporter.export_openvino(
                    str(output_path), args.imgsz
                )
            
            exported_models[format_type] = export_path
            
            # Benchmark if requested
            if args.benchmark:
                benchmark_results[format_type] = exporter.benchmark_model(
                    export_path, format_type, imgsz=args.imgsz
                )
                
        except Exception as e:
            print(f"Failed to export to {format_type}: {e}")
    
    # Benchmark original PyTorch model for comparison
    if args.benchmark:
        benchmark_results['pytorch'] = exporter.benchmark_model(
            args.model, 'pytorch', imgsz=args.imgsz
        )
    
    # Print benchmark results
    if benchmark_results:
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        for format_name, results in benchmark_results.items():
            if results:
                print(f"\n{results['format']}:")
                print(f"  Average inference time: {results['avg_inference_time_ms']:.2f} ms")
                print(f"  Throughput: {results['throughput_fps']:.1f} FPS")
    
    # Create deployment configuration
    config_path = output_dir / "yolo_deployment_config.json"
    deployment_config = exporter.create_deployment_config(exported_models, str(config_path))
    
    print(f"\n✓ Export completed successfully!")
    print(f"Exported models saved in: {output_dir}")
    print(f"Deployment config: {config_path}")
    
    # Print usage instructions
    print(f"\nUsage Instructions:")
    print(f"1. Copy exported models to your deployment environment")
    print(f"2. Use the deployment config for inference service setup")
    print(f"3. Recommended format for production: ONNX")
    
    if 'onnx' in exported_models:
        print(f"\nONNX model ready for inference service:")
        print(f"python inference_service_yolo.py --model-path {exported_models['onnx']}")


if __name__ == "__main__":
    main()
