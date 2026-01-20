#!/usr/bin/env python3
"""
Test script for malaria detection API
"""

import requests
import json
from pathlib import Path
import time


def test_api_endpoints():
    """Test all API endpoints."""
    base_url = "http://localhost:8080"
    
    print("Testing Malaria Detection API")
    print("=" * 40)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✓ Health check passed")
            print(f"   Status: {health_data['status']}")
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   Uptime: {health_data['uptime_seconds']:.1f}s")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Health check error: {e}")
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("   ✓ Root endpoint accessible")
        else:
            print(f"   ✗ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Root endpoint error: {e}")
    
    # Test metrics endpoint
    print("\n3. Testing metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics")
        if response.status_code == 200:
            print("   ✓ Metrics endpoint accessible")
            print(f"   Metrics size: {len(response.text)} characters")
        else:
            print(f"   ✗ Metrics endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Metrics endpoint error: {e}")
    
    # Test prediction endpoint (requires a test image)
    print("\n4. Testing prediction endpoint...")
    test_image_path = "test_image.jpg"
    
    if Path(test_image_path).exists():
        try:
            with open(test_image_path, 'rb') as f:
                files = {'file': ('test_image.jpg', f, 'image/jpeg')}
                data = {
                    'include_gradcam': False,
                    'patient_id': 'test_patient_001',
                    'clinic_id': 'test_clinic'
                }
                
                start_time = time.time()
                response = requests.post(f"{base_url}/predict", files=files, data=data)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    print("   ✓ Prediction successful")
                    print(f"   Label: {result['label']}")
                    print(f"   Probability: {result['probability']:.4f}")
                    print(f"   Confidence: {result['confidence_level']}")
                    print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
                    print(f"   Total request time: {(end_time - start_time) * 1000:.1f}ms")
                else:
                    print(f"   ✗ Prediction failed: {response.status_code}")
                    print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   ✗ Prediction error: {e}")
    else:
        print("   ⚠ No test image found. Create a test_image.jpg to test predictions.")
    
    print("\n" + "=" * 40)
    print("API testing completed")


def create_test_image():
    """Create a dummy test image for API testing."""
    try:
        from PIL import Image
        import numpy as np
        
        # Create a random RGB image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save('test_image.jpg')
        print("Created test_image.jpg for API testing")
        
    except ImportError:
        print("PIL not available. Cannot create test image.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test malaria detection API')
    parser.add_argument('--create-test-image', action='store_true',
                       help='Create a dummy test image')
    
    args = parser.parse_args()
    
    if args.create_test_image:
        create_test_image()
    
    test_api_endpoints()
