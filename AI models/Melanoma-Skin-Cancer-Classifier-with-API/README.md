# 🔬 Melanoma Skin Cancer Classifier API

A FastAPI-based melanoma skin cancer classification system using a VGG19 deep learning model. This API provides an intuitive web interface and RESTful endpoints for early-stage melanoma detection screening.

## ⚠️ Medical Disclaimer

**This tool is for educational and screening purposes only. It is NOT a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and proper diagnosis.**

## Before Setting Up this repo

Go to this [Google Drive folder](https://drive.google.com/drive/folders/1YJotdT9KDw4qULasBzvAw-piBJ9OufW4?usp=sharing) `AI Models` and download:
- `melanoma-skin-cancer_vgg19.pkl`
- `melanoma.efficientnetb7.pkl`

## 🎯 Features

- 🖼️ **Image Upload Interface**: Drag-and-drop web interface for easy image uploads
- 🧠 **AI-Powered Analysis**: VGG19-based deep learning model for classification
- 📊 **Confidence Scoring**: Provides prediction confidence levels
- 🏥 **Medical Recommendations**: Contextual recommendations based on prediction results
- 🔗 **RESTful API**: Complete API endpoints for integration
- 📚 **Interactive Documentation**: Built-in Swagger UI and ReDoc documentation
- 🐳 **Docker Support**: Containerized deployment ready

## 📋 Prerequisites

- Python 3.9+
- pip or conda
- Docker (optional, for containerized deployment)

## 🚀 Quick Start

### Option 1: Local Development

1. **Clone and navigate to the project directory**:
   ```bash
   cd "Skin Cancer"
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv melanoma-env
   
   # On Windows:
   melanoma-env\Scripts\activate
   
   # On macOS/Linux:
   source melanoma-env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure your model file is present**:
   - Place your `melanoma.vgg19.pkl` file in the project directory
   - The model should be saved using joblib or pickle

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Access the application**:
   - **Web Interface**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **Alternative Docs**: http://localhost:8000/redoc

### Option 2: Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t melanoma-classifier .
   ```

2. **Run the container**:
   ```bash
   docker run -d -p 8000:8000 melanoma-classifier
   ```

3. **Access the application at**: http://localhost:8000

## 🔧 API Endpoints

### Health Check
```http
GET /health
```
Returns the API status and model loading state.

### Image Prediction
```http
POST /predict
```
Upload an image for melanoma classification.

**Request**: Multipart form data with image file
**Response**:
```json
{
  "filename": "lesion.jpg",
  "predicted_class": "Benign",
  "confidence": 0.8234,
  "risk_level": "Low Risk",
  "recommendation": "LOW RISK: Continue regular skin monitoring..."
}
```

### Model Information
```http
GET /model-info
```
Returns information about the loaded model.

## 🧪 Testing the API

### Using cURL:
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Image prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/skin_lesion_image.jpg"
```

### Using Python:
```python
import requests

# Test prediction
url = "http://localhost:8000/predict"
files = {"file": open("skin_lesion.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 🏗️ Project Structure

```
Skin Cancer/
├── app.py                    # Main FastAPI application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── README.md               # This file
├── melanoma.vgg19.pkl      # Model file (you need to add this)
└── melanoma-skin-cancer-classifier-sic-final-project.ipynb
```

## 🎨 Model Information

- **Architecture**: VGG19-based classifier
- **Input Size**: 300×300 pixels
- **Classes**: Benign, Malignant
- **Preprocessing**: RGB conversion, resizing, normalization
- **Framework**: Trained with TensorFlow/Keras, saved with joblib

## 🔍 Usage Guidelines

1. **Image Requirements**:
   - Format: JPG, PNG, or other common image formats
   - Quality: Clear, well-lit images work best
   - Content: Should show the skin lesion clearly

2. **Interpretation**:
   - **High Confidence (>80%)**: More reliable prediction
   - **Low Confidence (<50%)**: Uncertain, requires professional evaluation
   - **Risk Levels**: Guide for understanding urgency

3. **Recommendations**:
   - **High Risk**: Immediate dermatologist consultation
   - **Medium Risk**: Consider professional evaluation
   - **Low Risk**: Continue monitoring, routine check-ups

## 🛠️ Troubleshooting

### Model Not Loading
- Ensure `melanoma.vgg19.pkl` is in the project directory
- Check file permissions
- Verify the model was saved correctly from your notebook

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.9+ recommended)

### Port Already in Use
- Change the port in app.py: `uvicorn.run(..., port=8001)`
- Or kill the process using port 8000

### Docker Issues
- Ensure Docker is running
- Check if port 8000 is available: `docker ps`
- View logs: `docker logs <container_id>`

## 📈 Performance Considerations

- **Image Size**: Larger images take longer to process
- **Batch Processing**: For multiple images, consider using the API programmatically
- **Scaling**: For production use, consider using multiple workers or container orchestration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with medical software regulations in your jurisdiction before using in clinical settings.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Ensure your model file is correctly placed and formatted

---

**Remember**: This tool is designed to assist in screening but should never replace professional medical diagnosis. Always consult healthcare professionals for medical advice. 
