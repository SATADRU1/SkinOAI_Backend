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

# SkinOAI: AI-Powered Skin Disease Detection

SkinOAI is a mobile application that uses machine learning to detect various skin conditions from images. The application provides an intuitive interface for users to upload skin images and receive potential condition predictions.

## 🚀 Features

- **Image-based Skin Disease Detection**: Upload an image of a skin condition to get predictions
- **Condition Classification**: Detects multiple skin conditions
- **Confidence Scoring**: Get confidence levels for each prediction
- **Responsive Web Interface**: Modern, user-friendly interface built with React Native and Expo
- **RESTful API**: Backend service for processing image predictions

## 🛠️ Tech Stack

### Backend
- **Python 3.10+**
- **Flask**: Web framework for building the REST API
- **Roboflow**: For model inference and predictions
- **Flask-CORS**: For handling Cross-Origin Resource Sharing
- **Pillow**: For image processing

### Frontend
- **React Native**: For building cross-platform mobile applications
- **Expo**: Development platform for React Native
- **TypeScript**: For type-safe JavaScript development

## 🔑 Important Note

This project uses a private API key for the Roboflow service. For security reasons, the API key has been intentionally excluded from the repository. To run this project locally, you'll need to:

1. Sign up for a Roboflow account
2. Obtain your API key from the Roboflow dashboard
3. Set up the API key in your environment variables

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Node.js 16+ and npm
- Expo CLI (for frontend development)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SkinOAI.git
   cd SkinOAI
   ```

2. **Backend Setup**
   ```bash
   cd Backend
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd ../Fronetnd
   npm install
   ```

### Running the Application

1. **Start the Backend Server**
   ```bash
   cd Backend
   python app.py
   ```
   The API will be available at `http://localhost:5000`

2. **Start the Frontend**
   ```bash
   cd Fronetnd
   npm start
   ```
   Use the Expo Go app on your mobile device to scan the QR code, or use an emulator.

## 📚 API Endpoints

- `GET /`: Health check endpoint
- `GET /ping`: Simple ping endpoint
- `POST /predict`: Submit an image for skin disease prediction
  - Request body: `{ "image": "base64_encoded_image" }`
  - Response: `{ "prediction": "disease_name", "confidence": 0.95, "description": "..." }`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Thanks to all the open-source contributors who made this project possible
- Medical professionals who provided guidance on skin condition identification
- The PyTorch and React Native communities

## 📧 Contact

For any inquiries, please contact [riddhimondal50@gmail.com]
