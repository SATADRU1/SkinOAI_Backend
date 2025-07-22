# SkinOAI: AI-Powered Skin Disease Detection

SkinOAI is a cutting-edge application that leverages machine learning to detect various skin conditions from images. The application combines computer vision and natural language processing to provide accurate skin disease predictions and detailed information.

## 🚀 Features

- **Image-based Skin Disease Detection**: Upload an image of a skin condition to get instant predictions
- **Multi-class Classification**: Detects 15+ different skin conditions including:
  - Actinic Keratosis
  - Basal Cell Carcinoma
  - Melanoma
  - Psoriasis
  - Eczema
  - And more...
- **Confidence Scoring**: Get confidence levels for each prediction
- **Detailed Information**: Receive comprehensive information about detected conditions
- **Responsive Web Interface**: Modern, user-friendly interface built with React Native
- **RESTful API**: Easy integration with other applications

## 🛠️ Tech Stack

### Backend
- **Python 3.10+**
- **Flask**: Web framework for building the REST API
- **PyTorch**: Deep learning framework for the image classification model
- **TinyLlama**: For natural language understanding and response generation
- **Flask-CORS**: For handling Cross-Origin Resource Sharing

### Frontend
- **React Native**: For building cross-platform mobile applications
- **Expo**: Development platform for React Native
- **TypeScript**: For type-safe JavaScript development
- **React Navigation**: For handling navigation between screens

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Node.js 16+ and npm
- PyTorch (CPU or GPU version)
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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all the open-source contributors who made this project possible
- Medical professionals who provided guidance on skin condition identification
- The PyTorch and React Native communities

## 📧 Contact

For any inquiries, please contact [riddhimondal50@gmail.com]
