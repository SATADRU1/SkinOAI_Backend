import requests
import base64

def test_prediction():
    """Test the prediction endpoint with a sample image"""
    
    # Load the test image
    try:
        with open("img1.jpg", "rb") as image_file:
            # Convert to base64
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print("Test image img1.jpg not found")
        return
    
    # Prepare the request
    url = "http://localhost:5000/predict"
    data = {
        "image": image_data
    }
    
    try:
        # Make the request
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"Class: {result.get('class')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"Message: {result.get('message')}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the Flask app is running.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_prediction() 