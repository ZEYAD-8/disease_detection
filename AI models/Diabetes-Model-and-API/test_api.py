#!/usr/bin/env python3
"""
Test script for the Diabetes Prediction API
Run this after starting the FastAPI server to test all endpoints
"""

import requests
import json

# API base URL
BASE_URL = "http://127.0.0.1:8001"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🔍 Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint with different scenarios"""
    print("\n🔍 Testing Single Prediction Endpoint...")
    
    # Test case 1: Low risk patient
    print("\n📊 Test Case 1: Low Risk Patient")
    low_risk_data = {
        "pregnancies": 1,
        "glucose": 85.0,
        "blood_pressure": 66.0,
        "skin_thickness": 29.0,
        "insulin": 0.0,
        "bmi": 26.6,
        "diabetes_pedigree_function": 0.351,
        "age": 31
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=low_risk_data)
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Prediction: {result['prediction']} ({'Diabetes' if result['prediction'] == 1 else 'No Diabetes'})")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test case 2: High risk patient
    print("\n📊 Test Case 2: High Risk Patient")
    high_risk_data = {
        "pregnancies": 8,
        "glucose": 180.0,
        "blood_pressure": 90.0,
        "skin_thickness": 40.0,
        "insulin": 200.0,
        "bmi": 40.0,
        "diabetes_pedigree_function": 1.5,
        "age": 65
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=high_risk_data)
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Prediction: {result['prediction']} ({'Diabetes' if result['prediction'] == 1 else 'No Diabetes'})")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test case 3: Example from the model
    print("\n📊 Test Case 3: Example Patient (from model documentation)")
    example_data = {
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
        response = requests.post(f"{BASE_URL}/predict", json=example_data)
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Prediction: {result['prediction']} ({'Diabetes' if result['prediction'] == 1 else 'No Diabetes'})")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n🔍 Testing Batch Prediction Endpoint...")
    
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
            },
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
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict-batch", json=batch_data)
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        print(f"\n📈 Batch Results Summary:")
        print(f"Total Patients: {result['summary']['total_patients']}")
        print(f"Diabetes Predictions: {result['summary']['diabetes_predictions']}")
        print(f"No Diabetes Predictions: {result['summary']['no_diabetes_predictions']}")
        print(f"Diabetes Percentage: {result['summary']['diabetes_percentage']}%")
        print(f"Average Probability: {result['summary']['average_probability']}")
        print(f"High Risk Patients: {result['summary']['high_risk_patients']}")
        
        print(f"\n📊 Individual Results:")
        for i, prediction in enumerate(result['predictions'], 1):
            print(f"Patient {i}: {prediction['prediction']} ({'Diabetes' if prediction['prediction'] == 1 else 'No Diabetes'}) - "
                  f"Probability: {prediction['probability']:.3f} - Risk: {prediction['risk_level']}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_error_handling():
    """Test API error handling with invalid data"""
    print("\n🔍 Testing Error Handling...")
    
    # Test with missing required field
    print("\n📊 Test Case: Missing Required Field")
    invalid_data = {
        "pregnancies": 6,
        "glucose": 148.0,
        # Missing other required fields
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error Response: {response.json()}")
            print("✅ API correctly handled invalid input")
        else:
            print("⚠️ API should have returned an error for invalid input")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test with out-of-range values
    print("\n📊 Test Case: Out-of-Range Values")
    invalid_range_data = {
        "pregnancies": -1,  # Should be >= 0
        "glucose": 300.0,   # Should be <= 250
        "blood_pressure": 72.0,
        "skin_thickness": 35.0,
        "insulin": 0.0,
        "bmi": 33.6,
        "diabetes_pedigree_function": 0.627,
        "age": 50
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_range_data)
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error Response: {response.json()}")
            print("✅ API correctly handled out-of-range values")
        else:
            print("⚠️ API should have returned an error for out-of-range values")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def main():
    """Run all API tests"""
    print("🚀 Starting Diabetes Prediction API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Root Endpoint", test_root_endpoint),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the API implementation.")

if __name__ == "__main__":
    main() 