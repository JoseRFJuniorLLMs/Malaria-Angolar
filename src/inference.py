#!/usr/bin/env python3
"""
YOLOv8 Inference Service for Malaria Parasite Detection
Clinical microscope integration with parasite counting and localization.
"""

import os
import sys
import io
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid

import numpy as np
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Add src to path
sys.path.append('src')

from models.yolov8_malaria import MalariaYOLOv8


# Metrics for monitoring
DETECTION_COUNTER = Counter('malaria_detections_total', 'Total detections made', ['infection_level'])
DETECTION_LATENCY = Histogram('malaria_detection_duration_seconds', 'Detection latency')
ERROR_COUNTER = Counter('malaria_errors_total', 'Total errors', ['error_type'])
PARASITE_COUNT_HISTOGRAM = Histogram('malaria_parasite_count', 'Distribution of parasite counts')


class DetectionRequest(BaseModel):
    """Request model for parasite detection."""
    confidence_threshold: Optional[float] = 0.25
    iou_threshold: Optional[float] = 0.7
    patient_id: Optional[str] = None
    clinic_id: Optional[str] = None
    microscope_id: Optional[str] = None
    magnification: Optional[str] = None


class ParasiteDetection(BaseModel):
    """Individual parasite detection."""
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    center: List[float]  # [x_center, y_center]
    area: float


class DetectionResponse(BaseModel):
    """Response model for parasite detection."""
    detection_id: str
    parasite_count: int
    infection_level: str
    clinical_significance: str
    confidence_level: str
    avg_confidence: float
    parasite_density: Optional[float] = None
    detections: List[ParasiteDetection]
    processing_time_ms: float
    image_dimensions: List[int]  # [width, height]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: str
    version: str
    uptime_seconds: float


class MalariaYOLOInferenceService:
    """Production YOLOv8 inference service for clinical malaria detection."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.config_path = config_path
        self.start_time = time.time()
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        # Load model
        self.model = None
        self._load_model()
        
        # Clinical thresholds
        clinical_config = self.config.get('clinical', {})
        self.low_threshold = clinical_config.get('low_threshold', 5)
        self.moderate_threshold = clinical_config.get('moderate_threshold', 20)
        self.min_confidence = clinical_config.get('min_confidence', 0.5)
        
        self.logger.info("YOLOv8 malaria inference service initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        if self.config_path and os.path.exists(self.config_path):
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'clinical': {
                'low_threshold': 5,
                'moderate_threshold': 20,
                'min_confidence': 0.5
            },
            'detection': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.7,
                'image_size': 640
            }
        }
    
    def _setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('yolo_inference_service.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            
            # Warm up model
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_image, verbose=False)
            
            self.logger.info(f"YOLOv8 model loaded: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image, target_size: int = 640) -> np.ndarray:
        """Preprocess image for YOLOv8 inference."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        return img_array
    
    def _determine_infection_level(self, parasite_count: int) -> tuple:
        """Determine infection level and clinical significance."""
        
        if parasite_count == 0:
            level = 'negative'
            significance = 'No parasites detected - negative for malaria'
        elif parasite_count <= self.low_threshold:
            level = 'low'
            significance = f'Low parasitemia ({parasite_count} parasites) - monitor patient, consider treatment'
        elif parasite_count <= self.moderate_threshold:
            level = 'moderate'
            significance = f'Moderate parasitemia ({parasite_count} parasites) - treat immediately'
        else:
            level = 'high'
            significance = f'High parasitemia ({parasite_count} parasites) - urgent treatment required'
        
        return level, significance
    
    def _determine_confidence_level(self, avg_confidence: float) -> str:
        """Determine confidence level for clinical reporting."""
        if avg_confidence >= 0.8:
            return 'high'
        elif avg_confidence >= 0.5:
            return 'moderate'
        else:
            return 'low'
    
    def _calculate_parasite_density(self, parasite_count: int, image_area: int) -> float:
        """Calculate parasite density per unit area."""
        if image_area > 0:
            return parasite_count / (image_area / 1000000)  # parasites per mm²
        return 0.0
    
    def detect_parasites(self, image: Image.Image, request: DetectionRequest) -> DetectionResponse:
        """Main parasite detection function."""
        start_time = time.time()
        detection_id = str(uuid.uuid4())
        
        try:
            # Preprocess image
            img_array = self._preprocess_image(image)
            img_height, img_width = img_array.shape[:2]
            image_area = img_width * img_height
            
            # Run YOLOv8 inference
            results = self.model.predict(
                img_array,
                conf=request.confidence_threshold or 0.25,
                iou=request.iou_threshold or 0.7,
                verbose=False
            )
            
            # Process detection results
            detections = []
            total_confidence = 0.0
            parasite_count = 0
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        conf = float(boxes.conf[i])
                        
                        # Filter by minimum confidence
                        if conf >= self.min_confidence:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            
                            # Calculate center and area
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            bbox_area = (x2 - x1) * (y2 - y1)
                            
                            detection = ParasiteDetection(
                                confidence=conf,
                                bbox=[float(x1), float(y1), float(x2), float(y2)],
                                center=[float(center_x), float(center_y)],
                                area=float(bbox_area)
                            )
                            
                            detections.append(detection)
                            total_confidence += conf
                            parasite_count += 1
            
            # Calculate metrics
            avg_confidence = total_confidence / parasite_count if parasite_count > 0 else 0.0
            infection_level, clinical_significance = self._determine_infection_level(parasite_count)
            confidence_level = self._determine_confidence_level(avg_confidence)
            parasite_density = self._calculate_parasite_density(parasite_count, image_area)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            DETECTION_COUNTER.labels(infection_level=infection_level).inc()
            DETECTION_LATENCY.observe(processing_time / 1000)
            PARASITE_COUNT_HISTOGRAM.observe(parasite_count)
            
            # Log detection
            self.logger.info(
                f"Detection {detection_id}: {parasite_count} parasites, "
                f"level={infection_level}, confidence={avg_confidence:.3f}, "
                f"time={processing_time:.1f}ms"
            )
            
            response = DetectionResponse(
                detection_id=detection_id,
                parasite_count=parasite_count,
                infection_level=infection_level,
                clinical_significance=clinical_significance,
                confidence_level=confidence_level,
                avg_confidence=avg_confidence,
                parasite_density=parasite_density,
                detections=detections,
                processing_time_ms=processing_time,
                image_dimensions=[img_width, img_height],
                timestamp=datetime.now().isoformat()
            )
            
            return response
            
        except Exception as e:
            ERROR_COUNTER.labels(error_type='detection_error').inc()
            self.logger.error(f"Detection error {detection_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    def get_health_status(self) -> HealthResponse:
        """Get service health status."""
        uptime = time.time() - self.start_time
        
        return HealthResponse(
            status='healthy',
            model_loaded=self.model is not None,
            model_type='YOLOv8',
            version='1.0.0',
            uptime_seconds=uptime
        )


# Initialize service
service = None

def get_service():
    """Dependency to get service instance."""
    global service
    if service is None:
        model_path = os.getenv('MODEL_PATH', 'runs/detect/train/weights/best.pt')
        config_path = os.getenv('CONFIG_PATH', 'configs/yolov8_malaria.yaml')
        service = MalariaYOLOInferenceService(model_path, config_path)
    return service


# FastAPI app
app = FastAPI(
    title="Malaria YOLOv8 Detection API",
    description="Clinical malaria parasite detection and counting service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detect", response_model=DetectionResponse)
async def detect_malaria_parasites(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = 0.25,
    iou_threshold: Optional[float] = 0.7,
    patient_id: Optional[str] = None,
    clinic_id: Optional[str] = None,
    microscope_id: Optional[str] = None,
    magnification: Optional[str] = None,
    service: MalariaYOLOInferenceService = Depends(get_service)
):
    """Detect and count malaria parasites in microscope field image."""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Create request
        request = DetectionRequest(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            patient_id=patient_id,
            clinic_id=clinic_id,
            microscope_id=microscope_id,
            magnification=magnification
        )
        
        # Get detection results
        result = service.detect_parasites(image, request)
        
        return result
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type='api_error').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check(service: MalariaYOLOInferenceService = Depends(get_service)):
    """Health check endpoint."""
    return service.get_health_status()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Malaria YOLOv8 Detection API",
        "version": "1.0.0",
        "model_type": "YOLOv8",
        "capabilities": [
            "Parasite detection and counting",
            "Clinical infection level assessment",
            "Confidence scoring",
            "Bounding box localization"
        ],
        "endpoints": {
            "detect": "/detect",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run YOLOv8 malaria detection inference service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--model-path', type=str, default='runs/detect/train/weights/best.pt',
                       help='Path to YOLOv8 model file')
    parser.add_argument('--config-path', type=str, default='configs/yolov8_malaria.yaml',
                       help='Path to configuration file')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['MODEL_PATH'] = args.model_path
    os.environ['CONFIG_PATH'] = args.config_path
    
    # Run server
    uvicorn.run(
        "inference:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
