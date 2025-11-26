# 📞 Complete Step-by-Step Guide: How to Call the Diabetes Prediction API

This guide shows you exactly how to call your Diabetes Prediction API using different methods, from beginner-friendly to advanced.

## 🚀 Prerequisites

1. **Start the API Server** (choose one method):
   ```bash
   # Method 1: Easy start script
   python start_server.py
   
   # Method 2: Direct uvicorn
   uvicorn app:app --host 127.0.0.1 --port 8001 --reload
   
   # Method 3: Python directly
   python app.py
   ```

2. **Verify the server is running**:
   - You should see: `INFO: Uvicorn running on http://127.0.0.1:8001`
   - The API will be available at: `http://127.0.0.1:8001`

---

## Method 1: 🌐 Web Interface (Recommended for Beginners)

### Step 1: Open the Test Client
1. Navigate to your project folder: `C:/Users/Waleed/Code/Diabetes`
2. Find the file `test_client.html`
3. **Double-click** on `test_client.html` to open it in your web browser
4. You should see a beautiful interface with the title "🩺 Diabetes Prediction API Test Client"

### Step 2: Check API Status
- At the top of the page, you should see: **"✅ API is online and healthy (Model loaded: true)"**
- If you see a red error message, make sure your API server is running

### Step 3: Use Preset Data (Easy Testing)
Click one of these buttons to load sample data:
- **"Low Risk Patient"** - loads data for a healthy patient
- **"High Risk Patient"** - loads data for a patient with diabetes risk factors
- **"Example Patient"** - loads the sample data from documentation

### Step 4: Make a Prediction
1. After clicking a preset (or entering your own data), click **"🔮 Predict Diabetes"**
2. Wait for the "🔄 Making prediction..." message
3. View the results:
   - **Prediction**: "DIABETES DETECTED" or "NO DIABETES"
   - **Probability**: Percentage chance (e.g., 31.8%)
   - **Risk Level**: Low Risk 🟢, Moderate Risk 🟡, or High Risk 🔴

### Step 5: Try Different Scenarios
- Click **"🗑️ Clear Form"** to reset
- Try different preset buttons to see various risk levels
- Modify individual values to see how they affect the prediction

---

## Method 2: 📚 Swagger UI (Interactive Documentation)

### Step 1: Open Swagger UI
1. Open your web browser
2. Go to: **`http://127.0.0.1:8001/docs`**
3. You'll see the interactive API documentation

### Step 2: Test the Health Endpoint
1. Find the **"GET /health"** section
2. Click on it to expand
3. Click the **"Try it out"** button
4. Click **"Execute"**
5. You should see a response like:
   ```json
   {
     "status": "healthy",
     "model_loaded": true
   }
   ```

### Step 3: Test Single Prediction
1. Find the **"POST /predict"** section
2. Click on it to expand
3. Click **"Try it out"**
4. You'll see a JSON example in the "Request body" field
5. **Replace** the example with your data:
   ```json
   {
     "pregnancies": 6,
     "glucose": 148.0,
     "blood_pressure": 72.0,
     "skin_thickness": 35.0,
     "insulin": 0.0,
     "bmi": 33.6,
     "diabetes_pedigree_function": 0.627,
     "age": 50
   }
   ```
6. Click **"Execute"**
7. View the response in the "Response body" section

### Step 4: Test Batch Prediction
1. Find the **"POST /predict-batch"** section
2. Click **"Try it out"**
3. Replace the example with multiple patients:
   ```json
   {
     "patients": [
       {
         "pregnancies": 1,
         "glucose": 85.0,
         "blood_pressure": 66.0,
         "skin_thickness": 29.0,
         "insulin": 0.0,
         "bmi": 26.6,
         "diabetes_pedigree_function": 0.351,
         "age": 31
       },
       {
         "pregnancies": 8,
         "glucose": 180.0,
         "blood_pressure": 90.0,
         "skin_thickness": 40.0,
         "insulin": 200.0,
         "bmi": 40.0,
         "diabetes_pedigree_function": 1.5,
         "age": 65
       }
     ]
   }
   ```
4. Click **"Execute"**
5. View the batch results with summary statistics

---

## Method 3: 💻 Command Line (curl)

### Step 1: Test Health Check
```bash
curl http://127.0.0.1:8001/health
```
**Expected Response:**
```json
{"status":"healthy","model_loaded":true}
```

### Step 2: Get API Information
```bash
curl http://127.0.0.1:8001/
```
**Expected Response:**
```json
{
  "message": "Diabetes Prediction API",
  "description": "Use /predict for single predictions or /predict-batch for multiple predictions",
  "endpoints": {
    "/predict": "POST - Single diabetes prediction",
    "/predict-batch": "POST - Batch diabetes predictions",
    "/health": "GET - Health check",
    "/docs": "GET - API documentation"
  }
}
```

### Step 3: Single Prediction
```bash
curl -X POST "http://127.0.0.1:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6,
    "glucose": 148.0,
    "blood_pressure": 72.0,
    "skin_thickness": 35.0,
    "insulin": 0.0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50
  }'
```
**Expected Response:**
```json
{
  "prediction": 0,
  "probability": 0.318,
  "risk_level": "Moderate Risk"
}
```

### Step 4: Batch Prediction
```bash
curl -X POST "http://127.0.0.1:8001/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {
        "pregnancies": 1,
        "glucose": 85.0,
        "blood_pressure": 66.0,
        "skin_thickness": 29.0,
        "insulin": 0.0,
        "bmi": 26.6,
        "diabetes_pedigree_function": 0.351,
        "age": 31
      },
      {
        "pregnancies": 8,
        "glucose": 180.0,
        "blood_pressure": 90.0,
        "skin_thickness": 40.0,
        "insulin": 200.0,
        "bmi": 40.0,
        "diabetes_pedigree_function": 1.5,
        "age": 65
      }
    ]
  }'
```

---

## Method 4: 🐍 Python Code

### Step 1: Create a Python Script
Create a file called `call_api.py`:

```python
import requests
import json

# API base URL
BASE_URL = "http://127.0.0.1:8001"

def test_health():
    """Test if API is healthy"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def predict_single_patient():
    """Predict diabetes for a single patient"""
    patient_data = {
        "pregnancies": 6,
        "glucose": 148.0,
        "blood_pressure": 72.0,
        "skin_thickness": 35.0,
        "insulin": 0.0,
        "bmi": 33.6,
        "diabetes_pedigree_function": 0.627,
        "age": 50
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    result = response.json()
    
    print(f"Prediction: {result['prediction']} ({'Diabetes' if result['prediction'] == 1 else 'No Diabetes'})")
    print(f"Probability: {result['probability']:.3f}")
    print(f"Risk Level: {result['risk_level']}")

def predict_batch():
    """Predict diabetes for multiple patients"""
    batch_data = {
        "patients": [
            {
                "pregnancies": 1,
                "glucose": 85.0,
                "blood_pressure": 66.0,
                "skin_thickness": 29.0,
                "insulin": 0.0,
                "bmi": 26.6,
                "diabetes_pedigree_function": 0.351,
                "age": 31
            },
            {
                "pregnancies": 8,
                "glucose": 180.0,
                "blood_pressure": 90.0,
                "skin_thickness": 40.0,
                "insulin": 200.0,
                "bmi": 40.0,
                "diabetes_pedigree_function": 1.5,
                "age": 65
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict-batch", json=batch_data)
    result = response.json()
    
    print("Batch Prediction Results:")
    print(f"Total Patients: {result['summary']['total_patients']}")
    print(f"Diabetes Cases: {result['summary']['diabetes_predictions']}")
    print(f"Average Probability: {result['summary']['average_probability']:.3f}")

if __name__ == "__main__":
    print("🩺 Testing Diabetes Prediction API")
    print("=" * 40)
    
    # Test health
    test_health()
    print()
    
    # Single prediction
    print("Single Patient Prediction:")
    predict_single_patient()
    print()
    
    # Batch prediction
    print("Batch Prediction:")
    predict_batch()
```

### Step 2: Run the Python Script
```bash
python call_api.py
```

---

## Method 5: 🧪 Automated Testing

### Step 1: Run the Test Suite
```bash
python test_api.py
```

This will run comprehensive tests covering:
- ✅ Health check endpoint
- ✅ Root endpoint information
- ✅ Single prediction with multiple scenarios
- ✅ Batch prediction processing
- ✅ Input validation and error handling

---

## 📊 Understanding the Response

### Single Prediction Response
```json
{
  "prediction": 0,           // 0 = No Diabetes, 1 = Diabetes
  "probability": 0.318,      // Confidence score (0.0 to 1.0)
  "risk_level": "Moderate Risk"  // Low/Moderate/High Risk
}
```

### Risk Level Classification
- 🟢 **Low Risk**: probability < 0.3
- 🟡 **Moderate Risk**: 0.3 ≤ probability < 0.7
- 🔴 **High Risk**: probability ≥ 0.7

### Batch Prediction Response
```json
{
  "predictions": [
    {
      "prediction": 0,
      "probability": 0.089,
      "risk_level": "Low Risk"
    },
    {
      "prediction": 1,
      "probability": 0.816,
      "risk_level": "High Risk"
    }
  ],
  "summary": {
    "total_patients": 2,
    "diabetes_predictions": 1,
    "no_diabetes_predictions": 1,
    "diabetes_percentage": 50.0,
    "average_probability": 0.453,
    "high_risk_patients": 1
  }
}
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

**1. "Connection refused" or "Could not connect"**
- ✅ Make sure the API server is running
- ✅ Check the correct port (8001, not 8000)
- ✅ Verify the URL: `http://127.0.0.1:8001`

**2. "Port already in use" error**
```bash
# Use a different port
uvicorn app:app --host 127.0.0.1 --port 8002 --reload
# Then update your URLs to use port 8002
```

**3. "Model file not found" error**
- ✅ Ensure `diabetes_best_model.pkl` is in the same directory as `app.py`

**4. Validation errors (422 status)**
- ✅ Check that all required fields are included
- ✅ Verify data types (integers vs floats)
- ✅ Ensure values are within the specified ranges

**5. Web interface shows "API is offline"**
- ✅ Start the API server first
- ✅ Wait a few seconds for the server to fully start
- ✅ Refresh the web page

---

## 🎯 Quick Reference

| Method | Difficulty | Best For |
|--------|------------|----------|
| Web Interface | ⭐ Easy | Beginners, quick testing |
| Swagger UI | ⭐⭐ Medium | Interactive exploration |
| Command Line | ⭐⭐⭐ Advanced | Automation, scripting |
| Python Code | ⭐⭐⭐ Advanced | Integration, custom apps |
| Test Suite | ⭐⭐ Medium | Comprehensive validation |

---

**🎉 Congratulations!** You now know how to call your Diabetes Prediction API using multiple methods. Start with the web interface for easy testing, then move to more advanced methods as needed. 