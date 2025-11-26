import requests
import json

# API base URL
BASE_URL = "http://127.0.0.1:8001"

def test_health():
    """Test if API is healthy"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("Health Check:", response.json())
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure the server is running.")
        return False

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
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=patient_data)
        result = response.json()
        
        print(f"Prediction: {result['prediction']} ({'Diabetes' if result['prediction'] == 1 else 'No Diabetes'})")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
        return True
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return False

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
    
    try:
        response = requests.post(f"{BASE_URL}/predict-batch", json=batch_data)
        result = response.json()
        
        print("Batch Prediction Results:")
        print(f"Total Patients: {result['summary']['total_patients']}")
        print(f"Diabetes Cases: {result['summary']['diabetes_predictions']}")
        print(f"No Diabetes Cases: {result['summary']['no_diabetes_predictions']}")
        print(f"Diabetes Percentage: {result['summary']['diabetes_percentage']}%")
        print(f"Average Probability: {result['summary']['average_probability']:.3f}")
        print(f"High Risk Patients: {result['summary']['high_risk_patients']}")
        
        print("\nIndividual Results:")
        for i, prediction in enumerate(result['predictions'], 1):
            status = "Diabetes" if prediction['prediction'] == 1 else "No Diabetes"
            print(f"  Patient {i}: {status} (Probability: {prediction['probability']:.3f}, Risk: {prediction['risk_level']})")
        
        return True
    except Exception as e:
        print(f"❌ Error making batch prediction: {e}")
        return False

def test_custom_patient():
    """Test with custom patient data"""
    print("Testing with a high-risk patient profile:")
    
    high_risk_patient = {
        "pregnancies": 10,
        "glucose": 200.0,
        "blood_pressure": 100.0,
        "skin_thickness": 50.0,
        "insulin": 300.0,
        "bmi": 45.0,
        "diabetes_pedigree_function": 2.0,
        "age": 70
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=high_risk_patient)
        result = response.json()
        
        print(f"High-Risk Patient Prediction:")
        print(f"  Prediction: {result['prediction']} ({'Diabetes' if result['prediction'] == 1 else 'No Diabetes'})")
        print(f"  Probability: {result['probability']:.3f}")
        print(f"  Risk Level: {result['risk_level']}")
        return True
    except Exception as e:
        print(f"❌ Error making custom prediction: {e}")
        return False

def main():
    """Main function to run all API tests"""
    print("🩺 Testing Diabetes Prediction API")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("\n❌ API is not available. Please start the server first:")
        print("   python start_server.py")
        return
    
    print("\n" + "=" * 50)
    
    # Single prediction
    print("📊 Single Patient Prediction:")
    print("-" * 30)
    predict_single_patient()
    
    print("\n" + "=" * 50)
    
    # Batch prediction
    print("📈 Batch Prediction (2 patients):")
    print("-" * 30)
    predict_batch()
    
    print("\n" + "=" * 50)
    
    # Custom high-risk patient
    print("🔴 Custom High-Risk Patient:")
    print("-" * 30)
    test_custom_patient()
    
    print("\n" + "=" * 50)
    print("✅ All API tests completed successfully!")
    print("\n💡 Try these next steps:")
    print("   • Open test_client.html in your browser for a web interface")
    print("   • Visit http://127.0.0.1:8001/docs for interactive documentation")
    print("   • Run 'python test_api.py' for comprehensive testing")

if __name__ == "__main__":
    main() 