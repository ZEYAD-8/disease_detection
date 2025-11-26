from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List, Dict, Any
import uvicorn
import pandas as pd

app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes based on health indicators using machine learning",
    version="1.0.0"
)

# Load the pre-trained model
try:
    model = joblib.load("./diabetes_best_model.pkl")
except FileNotFoundError:
    raise RuntimeError("Model file 'diabetes_best_model.pkl' not found. Please ensure the model file is in the correct directory.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Pydantic model for input validation
class DiabetesFeatures(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20, description="Number of times pregnant")
    glucose: float = Field(..., ge=0, le=250, description="Plasma glucose concentration (mg/dL)")
    blood_pressure: float = Field(..., ge=0, le=150, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(..., ge=0, le=1000, description="2-Hour serum insulin (mu U/ml)")
    bmi: float = Field(..., ge=0, le=70, description="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree_function: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    age: int = Field(..., ge=1, le=120, description="Age in years")

    class Config:
        json_schema_extra = {
            "example": {
                "pregnancies": 6,
                "glucose": 148.0,
                "blood_pressure": 72.0,
                "skin_thickness": 35.0,
                "insulin": 0.0,
                "bmi": 33.6,
                "diabetes_pedigree_function": 0.627,
                "age": 50
            }
        }

# Pydantic model for batch predictions
class BatchDiabetesFeatures(BaseModel):
    patients: List[DiabetesFeatures] = Field(..., description="List of patient data for batch prediction")

# Pydantic model for prediction response
class DiabetesPrediction(BaseModel):
    prediction: int = Field(..., description="Diabetes prediction (0: No diabetes, 1: Diabetes)")
    probability: float = Field(..., description="Prediction probability")
    risk_level: str = Field(..., description="Risk level classification")

# Pydantic model for batch prediction response
class BatchDiabetesPrediction(BaseModel):
    predictions: List[DiabetesPrediction] = Field(..., description="List of predictions for each patient")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")

def get_risk_level(probability: float) -> str:
    """Classify risk level based on prediction probability."""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Diabetes Prediction API",
        "description": "Use /predict for single predictions or /predict-batch for multiple predictions",
        "endpoints": {
            "/predict": "POST - Single diabetes prediction",
            "/predict-batch": "POST - Batch diabetes predictions",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=DiabetesPrediction)
async def predict_diabetes(data: DiabetesFeatures):
    """
    Predict diabetes for a single patient.
    
    Returns:
    - prediction: 0 (No diabetes) or 1 (Diabetes)
    - probability: Prediction probability (0-1)
    - risk_level: Risk classification (Low/Moderate/High Risk)
    """
    try:
        # Create engineered features
        features = create_engineered_features(data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probability
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0][1]  # Probability of diabetes (class 1)
        else:
            # For models without predict_proba, use decision function or return 0.5
            if hasattr(model, 'decision_function'):
                # Convert decision function to probability-like score
                decision_score = model.decision_function(features)[0]
                probability = 1 / (1 + np.exp(-decision_score))  # Sigmoid function
            else:
                probability = 0.5  # Fallback for models without probability methods
        
        # Classify risk level
        risk_level = get_risk_level(probability)
        
        return DiabetesPrediction(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def create_engineered_features(data: DiabetesFeatures) -> pd.DataFrame:
    """Create engineered features from the input data as a DataFrame."""
    # Create DataFrame with correct column names from the start
    features = {
        'Pregnancies': [data.pregnancies],
        'Glucose': [data.glucose],
        'BloodPressure': [data.blood_pressure],
        'SkinThickness': [data.skin_thickness],
        'Insulin': [data.insulin],
        'BMI': [data.bmi],
        'DiabetesPedigreeFunction': [data.diabetes_pedigree_function],
        'Age': [data.age]
    }
    
    df = pd.DataFrame(features)
    
    # Add engineered features
    df['BMI_Glucose'] = df['BMI'] * df['Glucose']
    df['Age_Glucose'] = df['Age'] * df['Glucose']
    df['BP_Glucose'] = df['BloodPressure'] * df['Glucose']
    df['Insulin_Glucose'] = df['Insulin'] * df['Glucose']
    df['Glucose_sq'] = df['Glucose'] ** 2
    df['BMI_sq'] = df['BMI'] ** 2
    df['Age_sq'] = df['Age'] ** 2
    
    return df

@app.post("/predict-batch", response_model=BatchDiabetesPrediction)
async def predict_diabetes_batch(data: BatchDiabetesFeatures):
    """
    Predict diabetes for multiple patients.
    
    Returns predictions for all patients plus summary statistics.
    """
    try:
        predictions = []
        probabilities = []
        
        for patient in data.patients:
            # Create engineered features
            features = create_engineered_features(patient)
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get prediction probability
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features)[0][1]
            else:
                if hasattr(model, 'decision_function'):
                    decision_score = model.decision_function(features)[0]
                    probability = 1 / (1 + np.exp(-decision_score))
                else:
                    probability = 0.5
            
            probabilities.append(probability)
            risk_level = get_risk_level(probability)
            
            predictions.append(DiabetesPrediction(
                prediction=int(prediction),
                probability=float(probability),
                risk_level=risk_level
            ))
        
        # Calculate summary statistics
        total_patients = len(predictions)
        diabetes_count = sum(1 for p in predictions if p.prediction == 1)
        no_diabetes_count = total_patients - diabetes_count
        avg_probability = np.mean(probabilities)
        high_risk_count = sum(1 for p in predictions if p.risk_level == "High Risk")
        
        summary = {
            "total_patients": total_patients,
            "diabetes_predictions": diabetes_count,
            "no_diabetes_predictions": no_diabetes_count,
            "diabetes_percentage": round((diabetes_count / total_patients) * 100, 2),
            "average_probability": round(float(avg_probability), 3),
            "high_risk_patients": high_risk_count
        }
        
        return BatchDiabetesPrediction(predictions=predictions, summary=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)