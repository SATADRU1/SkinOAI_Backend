
"""
Test script for SkinOAI backend API
"""

import requests
import base64
import json
from PIL import Image
import io

def create_test_image():
    """Create a simple test image"""
    # Create a simple RGB image
    img = Image.new('RGB', (224, 224), color='red')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_data = buffer.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    
    return img_base64

def test_api():
    """Test the API endpoints"""
    base_url = "http://192.168.0.140:5000"
    
    print("Testing SkinOAI Backend API...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/")
        print(f"✓ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return
    
    # Test 2: Ping endpoint
    try:
        response = requests.get(f"{base_url}/ping")
        print(f"✓ Ping: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"✗ Ping failed: {e}")
    
    # Test 3: Prediction endpoint
    try:
        test_image = create_test_image()
        payload = {
            "image": test_image,
            "text": "Red patches on skin with itching"
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{base_url}/predict",
            data=json.dumps(payload),
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction successful:")
            print(f"  - Predicted class: {result.get('predicted_class')}")
            print(f"  - Confidence: {result.get('confidence')}%")
            print(f"  - Recommendation: {result.get('recommendation')[:100]}...")
        else:
            print(f"✗ Prediction failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")

if __name__ == "__main__":
    test_api()
