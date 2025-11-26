import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from io import BytesIO
from PIL import Image
from typing import Dict, Any
import logging
import os
import sys
from pathlib import Path
import cv2
import pydicom
from PIL import ImageFile
import pillow_heif
from contextlib import asynccontextmanager

# Enable HEIF support for iOS photos
pillow_heif.register_heif_opener()

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
MODEL = None
MODEL_NAME = None
CLASS_NAMES = ['Benign', 'Malignant']  # Binary classification for melanoma

def resize_image(image, x, y):
    """Resize image using OpenCV"""
    # Convert image to OpenCV format (if necessary)
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume RGB, convert to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize the image
    resized_image = cv2.resize(image, (x, y))

    return resized_image

def ben_Graham(image):
    """Apply Ben Graham preprocessing (grayscale + Gaussian blur)"""
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Adjust kernel size and sigmaX as needed

    return blurred_image

def hair_remove(image):
    """Remove hair artifacts using morphological operations"""
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1, (17, 17))

    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # apply thresholding to blackhat
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)

    return final_image

def combined_preprocessing(image, x, y):
    """Combined preprocessing pipeline"""
    image = hair_remove(image)
    image = ben_Graham(image)
    image = resize_image(image, x, y)

    return image

async def load_model():
    """Load the melanoma model - try multiple available formats"""
    global MODEL, MODEL_NAME
    try:
        # Use the simple loader
        from simple_model_loader import find_and_load_model
        MODEL, MODEL_NAME = find_and_load_model()
        
        if MODEL is not None:
            logger.info(f"Model loaded successfully: {MODEL_NAME}")
        else:
            logger.warning("No compatible model file found!")
            logger.info("Available files in directory:")
            for f in Path(".").glob("*.pkl"):
                logger.info(f"  - {f.name}")
            for f in Path(".").glob("*.h5"):
                logger.info(f"  - {f.name}")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        MODEL = None
        MODEL_NAME = "Error loading model"

def load_image_from_bytes(image_data: bytes, filename: str = "") -> Image.Image:
    """
    Load image from bytes, handling multiple formats including DICOM and HEIC
    """
    try:
        # Check if it's a DICOM file based on filename or content
        if filename.lower().endswith('.dcm') or filename.lower().endswith('.dicom'):
            # Handle DICOM files
            logger.info("Processing DICOM file")
            try:
                # Load DICOM from bytes
                dicom_file = pydicom.dcmread(BytesIO(image_data))
                
                # Extract pixel array
                if hasattr(dicom_file, 'pixel_array'):
                    pixel_array = dicom_file.pixel_array
                    
                    # Handle different DICOM image formats
                    if len(pixel_array.shape) == 2:
                        # Grayscale image - convert to RGB
                        pixel_array = np.stack([pixel_array] * 3, axis=-1)
                    elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 1:
                        # Single channel - expand to RGB
                        pixel_array = np.repeat(pixel_array, 3, axis=2)
                    
                    # Normalize to 0-255 range
                    if pixel_array.max() > 255:
                        pixel_array = ((pixel_array - pixel_array.min()) / 
                                     (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
                    else:
                        pixel_array = pixel_array.astype(np.uint8)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(pixel_array)
                    logger.info(f"DICOM image loaded successfully: {pil_image.size}")
                    return pil_image
                else:
                    raise ValueError("DICOM file does not contain pixel data")
                    
            except Exception as e:
                logger.error(f"Error processing DICOM file: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid DICOM file: {str(e)}")
        
        else:
            # Handle standard image formats (JPEG, PNG, HEIC, etc.)
            try:
                pil_image = Image.open(BytesIO(image_data))
                logger.info(f"Standard image loaded successfully: {pil_image.format}, {pil_image.size}")
                return pil_image
            except Exception as e:
                logger.error(f"Error loading standard image: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def preprocess_image(image_data: bytes, filename: str = "") -> np.ndarray:
    """
    Preprocess the uploaded image for model prediction using notebook preprocessing pipeline
    Supports DICOM (.dcm), HEIC (iOS photos), JPEG, PNG and other standard formats
    Based on notebook analysis: resize to 128x128 pixels with hair removal and Ben Graham preprocessing
    
    Returns:
        np.ndarray: Preprocessed image with shape (1, 128, 128, 3) for VGG19 model
    """
    try:
        # Load image using the new multi-format loader
        pil_image = load_image_from_bytes(image_data, filename)
        
        # Convert to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            logger.info(f"Converted image to RGB from {pil_image.mode}")
        
        # Convert PIL image to numpy array (RGB format)
        image_array = np.array(pil_image)
        logger.info(f"Original image array shape: {image_array.shape}")
        
        # Apply the combined preprocessing pipeline from the notebook
        # This includes: hair removal -> Ben Graham -> resize
        processed_image = combined_preprocessing(image_array, 128, 128)
        
        # Handle the case where Ben Graham returns grayscale image
        if len(processed_image.shape) == 2:
            # Convert grayscale back to 3-channel for consistency
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
            logger.info("Converted grayscale to RGB")
        elif len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            # Convert from BGR to RGB if needed (OpenCV uses BGR)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            logger.info("Converted BGR to RGB")
        
        # Ensure we have the correct shape (128, 128, 3)
        if processed_image.shape != (128, 128, 3):
            logger.warning(f"Unexpected shape after preprocessing: {processed_image.shape}")
            # Resize to correct dimensions if needed
            processed_image = cv2.resize(processed_image, (128, 128))
            if len(processed_image.shape) == 2:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        
        # Normalize pixel values (0-255 to 0-1)
        processed_image = processed_image.astype(np.float32) / 255.0
        
        # Add batch dimension for VGG19 model: (1, 128, 128, 3)
        # This is the correct format for Keras/TensorFlow CNN models
        image_batch = np.expand_dims(processed_image, axis=0)
        
        logger.info(f"Final preprocessed shape: {image_batch.shape}")
        return image_batch
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model()
    yield
    # Shutdown
    pass

# Initialize FastAPI app
app = FastAPI(
    title="Melanoma Skin Cancer Classifier API",
    description="FastAPI-based melanoma skin cancer classification using deep learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information and upload interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Melanoma Classifier API</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header { 
                text-align: center; 
                color: white; 
                background: linear-gradient(45deg, #2c3e50, #3498db);
                padding: 30px;
                margin: 0;
            }
            .header h1 { margin: 0; font-size: 2.5em; }
            .header p { margin: 10px 0 0 0; opacity: 0.9; }
            .content { padding: 40px; }
            .upload-section {
                border: 3px dashed #3498db;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background: #f8f9fa;
                transition: all 0.3s ease;
            }
            .upload-section:hover {
                border-color: #2980b9;
                background: #e3f2fd;
            }
            .file-input {
                display: none;
            }
            .upload-btn {
                background: #3498db;
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                transition: background 0.3s ease;
            }
            .upload-btn:hover {
                background: #2980b9;
            }
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .info-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #3498db;
            }
            .info-card h3 {
                color: #2c3e50;
                margin-top: 0;
            }
            .endpoints {
                background: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .endpoints h3 { margin-top: 0; }
            .endpoints a { 
                color: #3498db; 
                text-decoration: none;
                font-weight: bold;
            }
            .endpoints a:hover { color: #2980b9; }
            .result-section {
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                display: none;
            }
            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 Melanoma Classifier</h1>
                <p>AI-Powered Skin Lesion Analysis</p>
            </div>
            
            <div class="content">
                <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                    <h3>📸 Upload Skin Lesion Image</h3>
                    <p>Click here or drag and drop an image file</p>
                    <p><small>Supports: JPEG, PNG, HEIC (iOS photos), DICOM (.dcm)</small></p>
                    <input type="file" id="fileInput" class="file-input" accept="image/*,.dcm,.dicom" onchange="uploadImage()">
                    <button class="upload-btn">Choose Image</button>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing image...</p>
                </div>
                
                <div class="result-section" id="results">
                    <h3>📊 Analysis Results</h3>
                    <div id="resultContent"></div>
                </div>
                
                <div class="info-grid">
                    <div class="info-card">
                        <h3>🎯 Model Information</h3>
                        <ul>
                            <li>Architecture: VGG19</li>
                            <li>Input Size: 128×128 pixels</li>
                            <li>Preprocessing: Hair removal + Ben Graham</li>
                            <li>Classes: Benign, Malignant</li>
                            <li>Purpose: Early detection screening</li>
                        </ul>
                    </div>
                    
                    <div class="info-card">
                        <h3>⚠️ Important Notice</h3>
                        <ul>
                            <li>This tool is for screening purposes only</li>
                            <li>Not a substitute for professional medical diagnosis</li>
                            <li>Always consult healthcare professionals</li>
                            <li>Seek immediate medical attention if concerned</li>
                        </ul>
                    </div>
                </div>
                
                <div class="endpoints">
                    <h3>🔗 API Endpoints</h3>
                    <ul>
                        <li><strong>GET /health</strong> - Check API status</li>
                        <li><strong>POST /predict</strong> - Upload image for analysis</li>
                        <li><strong>GET <a href="/docs">/docs</a></strong> - Interactive API documentation</li>
                        <li><strong>GET <a href="/redoc">/redoc</a></strong> - Alternative documentation</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) return;
                
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                const resultContent = document.getElementById('resultContent');
                
                loading.style.display = 'block';
                results.style.display = 'none';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        resultContent.innerHTML = `
                            <div style="background: ${data.predicted_class === 'Malignant' ? '#ffe6e6' : '#e6ffe6'}; padding: 15px; border-radius: 8px; margin: 10px 0;">
                                <h4>🎯 Prediction: ${data.predicted_class}</h4>
                                <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                                <p><strong>Risk Level:</strong> ${data.risk_level}</p>
                                <p><strong>Recommendation:</strong> ${data.recommendation}</p>
                            </div>
                        `;
                    } else {
                        resultContent.innerHTML = `
                            <div style="background: #ffe6e6; padding: 15px; border-radius: 8px; color: #d32f2f;">
                                <h4>❌ Error</h4>
                                <p>${data.detail}</p>
                            </div>
                        `;
                    }
                    
                    results.style.display = 'block';
                    
                } catch (error) {
                    resultContent.innerHTML = `
                        <div style="background: #ffe6e6; padding: 15px; border-radius: 8px; color: #d32f2f;">
                            <h4>❌ Error</h4>
                            <p>Failed to connect to the API. Please check if the server is running.</p>
                        </div>
                    `;
                    results.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if MODEL is not None else "model_not_loaded",
        "model_loaded": MODEL is not None,
        "message": "Melanoma Classifier API is running",
        "model_name": MODEL_NAME if MODEL_NAME else "No model loaded"
    }

@app.post("/predict")
async def predict_melanoma(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict melanoma from uploaded skin lesion image
    Supports multiple formats: JPEG, PNG, HEIC (iOS photos), DICOM (.dcm)
    
    Args:
        file: Uploaded image file (JPG, PNG, HEIC, DCM, etc.)
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Get filename for format detection
        filename = file.filename or ""
        logger.info(f"Processing file: {filename}, Content-Type: {file.content_type}")
        
        # Validate file type - allow more formats including DICOM
        if file.content_type:
            allowed_types = ['image/', 'application/dicom', 'application/octet-stream']
            if not any(file.content_type.startswith(t) for t in allowed_types):
                # Also check by filename extension
                allowed_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.dcm', '.dicom']
                if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
                    raise HTTPException(
                        status_code=400, 
                        detail="File must be an image (JPG, PNG, HEIC) or DICOM file (.dcm)"
                    )
        
        # Check if model is loaded
        if MODEL is None:
            raise HTTPException(
                status_code=503, 
                detail=f"Model not loaded. Current status: {MODEL_NAME}. Please check model files and restart server."
            )
        
        # Read and preprocess image
        image_data = await file.read()
        processed_image = preprocess_image(image_data, filename)
        
        # Make prediction with improved error handling
        try:
            # Check if it's a deep learning model (Keras/TensorFlow)
            if hasattr(MODEL, 'predict') and callable(getattr(MODEL, 'predict', None)):
                # For deep learning models (VGG19, etc.)
                if hasattr(MODEL, 'predict_proba'):
                    # Model supports probability prediction
                    probabilities = MODEL.predict_proba(processed_image)
                    if len(probabilities.shape) > 1:
                        probabilities = probabilities[0]  # Get first batch
                    predicted_class_idx = np.argmax(probabilities)
                    confidence = float(probabilities[predicted_class_idx])
                else:
                    # Model only supports class prediction
                    prediction = MODEL.predict(processed_image)
                    if len(prediction.shape) > 1:
                        prediction = prediction[0]  # Get first batch
                    
                    if len(prediction) == 1:
                        # Binary classification
                        predicted_class_idx = int(prediction[0] > 0.5)
                        confidence = float(prediction[0]) if predicted_class_idx == 1 else float(1 - prediction[0])
                    else:
                        predicted_class_idx = int(np.argmax(prediction))
                        confidence = float(np.max(prediction))
            else:
                # For traditional ML models (scikit-learn)
                if hasattr(MODEL, 'predict_proba'):
                    probabilities = MODEL.predict_proba(processed_image)[0]
                    predicted_class_idx = np.argmax(probabilities)
                    confidence = float(probabilities[predicted_class_idx])
                else:
                    prediction = MODEL.predict(processed_image)
                    if len(prediction) == 1:
                        predicted_class_idx = int(prediction[0] > 0.5)
                        confidence = float(prediction[0]) if predicted_class_idx == 1 else float(1 - prediction[0])
                    else:
                        predicted_class_idx = int(np.argmax(prediction))
                        confidence = float(np.max(prediction))
                        
        except Exception as model_error:
            logger.error(f"Model prediction error: {str(model_error)}")
            logger.error(f"Processed image shape: {processed_image.shape}")
            logger.error(f"Model type: {type(MODEL)}")
            raise HTTPException(
                status_code=500,
                detail=f"Model prediction failed. Error: {str(model_error)}"
            )
        
        # Ensure predicted_class_idx is within bounds
        if predicted_class_idx >= len(CLASS_NAMES):
            logger.warning(f"Predicted class index {predicted_class_idx} out of bounds, using 0")
            predicted_class_idx = 0
        
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Prepare response
        result = {
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "risk_level": "High Risk" if predicted_class == "Malignant" else "Low Risk",
            "recommendation": get_recommendation(predicted_class, confidence),
            "model_used": MODEL_NAME,
            "image_shape": list(processed_image.shape)
        }
        
        logger.info(f"Prediction made for {filename}: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def get_recommendation(predicted_class: str, confidence: float) -> str:
    """Generate medical recommendation based on prediction"""
    if predicted_class == "Malignant":
        if confidence > 0.8:
            return "HIGH PRIORITY: Consult a dermatologist immediately for professional evaluation."
        else:
            return "MEDIUM PRIORITY: Consider consulting a dermatologist for professional evaluation."
    else:
        if confidence > 0.8:
            return "LOW RISK: Continue regular skin monitoring. Consult a doctor if changes occur."
        else:
            return "UNCERTAIN: Recommend professional medical evaluation for accurate diagnosis."

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_name": MODEL_NAME if MODEL_NAME else "No model loaded",
        "architecture": MODEL_NAME if MODEL_NAME else "Unknown",
        "input_shape": [128, 128, 3],
        "preprocessing_pipeline": ["Hair removal", "Ben Graham (grayscale + blur)", "Resize to 128x128"],
        "classes": CLASS_NAMES,
        "model_loaded": MODEL is not None,
        "model_file_exists": any(Path(f).exists() for f in ["melanoma-skin-cancer_vgg19.pkl", "melanoma.efficientnetb7.pkl", "melanoma.vgg19.pkl"]),
        "supported_formats": ["JPEG", "PNG", "HEIC (iOS photos)", "DICOM (.dcm)"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8002, 
        reload=True,
        log_level="info"
    ) 