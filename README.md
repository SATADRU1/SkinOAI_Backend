# SkinOAI Backend

A simplified Flask API for skin condition prediction using Roboflow.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

- `GET /` - Health check
- `GET /ping` - Simple ping endpoint
- `POST /predict` - Predict skin condition from image

### Predict Endpoint

Send a POST request to `/predict` with JSON data:

```json
{
  "image": "base64_encoded_image_data"
}
```

Response:
```json
{
  "success": true,
  "class": "predicted_class",
  "confidence": 0.95,
  "message": "Prediction successful"
}
```

## Testing

Run the test script to verify the API:
```bash
python test_simple.py
```

## Files

- `app.py` - Main Flask application
- `model1.py` - Original Roboflow model reference
- `requirements.txt` - Python dependencies
- `test_simple.py` - Test script
- `img1.jpg` - Sample test image