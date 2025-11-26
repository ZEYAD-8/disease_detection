import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "features": {
        "age": 55,
        "sex": 1,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "thalach": 160,
        "exang": 0,
        "oldpeak": 1.5,
        "ca": 0,
        "cp_1": 0,
        "cp_2": 1,
        "cp_3": 0,
        "restecg_1": 0,
        "restecg_2": 1,
        "slope_1": 1,
        "slope_2": 0,
        "thal_1": 0,
        "thal_2": 1,
        "thal_3": 0,
        "age_group_Middle": 1,
        "age_group_Senior": 0,
        "age_group_Elderly": 0
    }
}

response = requests.post(url, json=data)
print(response.json())
