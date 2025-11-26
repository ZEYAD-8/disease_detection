import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import HeartPredictionSerializer
from .models import HeartPredictionAttempt

class HeartPredictionView(APIView):
    permission_classes = []
    authentication_classes = []
    def post(self, request):
        user = request.user
        input_data = request.data

        try:
            REQUIRED_FIELDS = [
                "age",
                "sex", 
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal"
            ]

            if not isinstance(input_data, dict):
                return Response({"Error": "Input must be a JSON object."}, status=status.HTTP_400_BAD_REQUEST)

            for key, value in input_data.items():
                print(key, value)
                if not isinstance(value, (int, float)):
                    return Response({"Error": "All Values must be floating point values or Integers."}, status=status.HTTP_400_BAD_REQUEST)

                if key not in REQUIRED_FIELDS:
                    return Response({"Error": f"Unknown Key: {key}", "Accepted": REQUIRED_FIELDS}, status=status.HTTP_400_BAD_REQUEST)

            missing = set(REQUIRED_FIELDS) - input_data.keys()
            if missing:
                return Response({"Error": f"Missing required fields: {list(missing)}"}, status=status.HTTP_400_BAD_REQUEST)

            data = {
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
                    "restecg_2": 0,

                    "slope_1": 0,
                    "slope_2": 0,

                    "thal_1": 0,
                    "thal_2": 0,
                    "thal_3": 0,
                    "age_group_Middle": 0,
                    "age_group_Senior": 0,
                    "age_group_Elderly": 0
            }

            if input_data['age'] <= 40:
                data["age_group_Middle"] = 1
            elif input_data["age"] <= 55:
                data["age_group_Senior"] = 1
            else:
                data["age_group_Elderly"] = 1
            
            if input_data['thal'] <= 1:
                data["thal_1"] = 1
            elif input_data["thal"] <= 2:
                data['thal_2'] = 1
            else:
                data['thal_3'] = 1
            
            if input_data['slope'] <= 1:
                data['slope_1'] = 1
            else:
                data['slope_2'] = 1

            if input_data['restecg'] == 1:
                data["restecg_1"] = 1
            else:
                data['restecg_2'] = 1

            if input_data["cp"] == 1:
                data['cp_1'] = 1
            elif input_data["cp"] == 2:
                data['cp_2'] = 1
            else:
                data['cp_3'] = 1

            data["age"] = input_data["age"]
            data["sex"] = input_data["sex"]
            data["trestbps"] = input_data["trestbps"]
            data["chol"] = input_data["chol"]
            data["fbs"] = input_data["fbs"]
            data["thalach"] = input_data["thalach"]
            data["exang"] = input_data["exang"]
            data["oldpeak"] = input_data["oldpeak"]
            data["ca"] = input_data["ca"]

            input_data = {'features': data}
            response = requests.post("http://localhost:5005/predict", json=input_data)
            if response.status_code != 200:
                return Response({"Error": "AI service error"}, status=status.HTTP_424_FAILED_DEPENDENCY)

            result = response.json()
            combined_data = {**input_data, **result}

            serializer = HeartPredictionSerializer(data=combined_data)
            if serializer.is_valid():
                serializer.save(user=user)
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except requests.RequestException as e:
            return Response({"error": "Failed to connect to AI model", "details": str(e)}, status=500)
