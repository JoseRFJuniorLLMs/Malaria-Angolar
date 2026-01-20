# Deployment Guide

## Quick Start

### 1. Local Development Setup

```bash
# Clone and setup
cd malaria-detection
pip install -r requirements.txt

# Create data directory structure
mkdir -p data/{train/{parasitized,uninfected},val/{parasitized,uninfected},test/{parasitized,uninfected}}

# Download malaria datasets (you'll need to source these):
# - NIH Malaria Dataset: https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html
# - Place images in appropriate directories
```

### 2. Training

```bash
# Train model
python train.py --config configs/efficientnet_b0.yaml

# Export for deployment
python export_model.py --checkpoint checkpoints/best_checkpoint.pth --format onnx --optimize --benchmark
```

### 3. Local Inference Service

```bash
# Run API server
python inference_service.py --model-path exports/efficientnet-b0.onnx

# Test API
python test_api.py --create-test-image
python test_api.py
```

### 4. Docker Deployment

```bash
# Build and run with Docker Compose (includes monitoring)
docker-compose up --build

# Or build standalone container
docker build -t malaria-api .
docker run -p 8080:8080 -v $(pwd)/exports:/app/models malaria-api
```

## Cloud Deployment Options

### Google Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/malaria-api

# Deploy to Cloud Run
gcloud run deploy malaria-api \
  --image gcr.io/PROJECT_ID/malaria-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

### AWS ECS/Fargate

```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag malaria-api:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/malaria-api:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/malaria-api:latest

# Deploy with ECS (use provided task definition)
aws ecs create-service --cluster malaria-cluster --service-name malaria-api --task-definition malaria-api:1 --desired-count 2
```

### Azure Container Instances

```bash
# Push to ACR
az acr build --registry malariaregistry --image malaria-api .

# Deploy to ACI
az container create \
  --resource-group malaria-rg \
  --name malaria-api \
  --image malariaregistry.azurecr.io/malaria-api:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8080
```

## Production Configuration

### Environment Variables

```bash
# Required
MODEL_PATH=/app/models/model.onnx
CONFIG_PATH=/app/models/deployment_config.json

# Optional
LOG_LEVEL=INFO
MAX_WORKERS=4
ENABLE_CORS=true
```

### Model Files Required

- `model.onnx` or `model.pt` - Trained model
- `deployment_config.json` - Configuration file

### Security Considerations

1. **Authentication**: Add API key authentication for production
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **Input Validation**: Validate image size and format
4. **Logging**: Ensure no PHI is logged
5. **HTTPS**: Always use HTTPS in production

### Monitoring Setup

The Docker Compose setup includes:
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Visualization dashboard (port 3000, admin/admin)
- **API Metrics**: Prediction counts, latency, confidence distribution

Key metrics to monitor:
- Prediction latency (target: <500ms)
- Error rates (target: <1%)
- Confidence score distribution
- Memory and CPU usage

### Scaling Guidelines

**CPU-based scaling:**
- 1 vCPU can handle ~10-20 requests/second
- Scale horizontally for higher throughput

**Memory requirements:**
- EfficientNet-B0: ~1GB RAM
- Add 512MB for API overhead
- Recommended: 2GB per instance

**Auto-scaling triggers:**
- CPU > 70% for 2 minutes
- Memory > 80% for 2 minutes
- Request queue length > 10

## Clinical Validation

### Required Testing

1. **Sensitivity Validation**: Test with known positive samples
2. **Specificity Validation**: Test with known negative samples
3. **Cross-device Testing**: Validate across different microscope/camera setups
4. **Field Testing**: Deploy in controlled clinical environment

### Performance Targets

- **Sensitivity**: ≥95% (critical for patient safety)
- **Specificity**: ≥85% (minimize over-treatment)
- **Latency**: <2 seconds end-to-end
- **Availability**: 99.9% uptime

### Regulatory Considerations

- Check local medical device regulations
- May require clinical validation studies
- Consider liability and insurance requirements
- Implement audit trails for predictions

## Troubleshooting

### Common Issues

**Model loading errors:**
```bash
# Check model file exists and is readable
ls -la exports/
# Verify ONNX model
python -c "import onnxruntime; ort.InferenceSession('exports/model.onnx')"
```

**Memory issues:**
```bash
# Monitor memory usage
docker stats
# Reduce batch size or use quantized model
```

**Slow predictions:**
```bash
# Check if using GPU
nvidia-smi
# Benchmark model performance
python export_model.py --benchmark
```

### Health Checks

```bash
# API health
curl http://localhost:8080/health

# Detailed metrics
curl http://localhost:8080/metrics

# Test prediction
curl -X POST -F "file=@test_image.jpg" http://localhost:8080/predict
```

## Data Sources

For training data, you'll need to source malaria cell images:

1. **NIH Malaria Dataset**: Public dataset with segmented cells
2. **Partner Clinics**: Field data from actual deployments
3. **Synthetic Data**: Augmented versions of existing datasets

Ensure proper data licensing and patient consent for all datasets.

## Support

For issues and questions:
1. Check logs: `docker logs malaria-api`
2. Review metrics in Grafana dashboard
3. Test with known good images
4. Verify model performance on validation set
