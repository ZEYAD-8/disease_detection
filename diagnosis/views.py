from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
import requests
from .serializers import DiabetesAttemptsSerializer
class DiabetesAIPredictAPIView(APIView):
    """
    API endpoint to handle AI predictions.
    """
    def post(self, request, *args, **kwargs):
        try:
            # Extract input data from the request
            input_data = request.data  # Input JSON as a dictionary
            # Validate input data
            REQUIRED_FIELDS = [ 
                "pregnancies",
                "glucose",
                "blood_pressure",
                "skin_thickness", 
                "insulin",
                "bmi",
                "diabetes_pedigree_function",
                "age"
            ]

            if not isinstance(input_data, dict):
                return Response({"Error": "Input must be a JSON object."}, status=status.HTTP_400_BAD_REQUEST)

            for key, value in input_data.items():
                if not isinstance(value, (int, float)):
                    return Response({"Error": "All Values must be floating point values or Integers."}, status=status.HTTP_400_BAD_REQUEST)

                if key not in REQUIRED_FIELDS:
                    return Response({"Error": f"Unknown Key: {key}", "Accepted": REQUIRED_FIELDS}, status=status.HTTP_400_BAD_REQUEST)

            missing = set(REQUIRED_FIELDS) - input_data.keys()
            if missing:
                return Response({"Error": f"Missing required fields: {list(missing)}"}, status=status.HTTP_400_BAD_REQUEST)

            # Make a prediction
            try:
                response = requests.post("http://localhost:8001/predict", json=input_data) # Connecting to the model on my docker container
                api_result = response.json()
            except Exception as e:
                print(f"Error reaching the diabetes model:\n{str(e)}")
                return Response({"Error": "Error with the AI Model."}, status=status.HTTP_424_FAILED_DEPENDENCY)

            # Saving the Attempt
            if "prediction" in api_result:
                user_data = {f'user_{key}': value for key, value in input_data.items()}
                ai_data = {f'ai_{key}': value for key, value in api_result.items()}
                data = {**user_data, **ai_data}
                serializer = DiabetesAttemptsSerializer(data=data, context={'request': request})
                if serializer.is_valid():
                    serializer.save()
                else:
                    print(serializer.errors)
            else:
                return Response({"Error": api_result['detail']}, status=status.HTTP_400_BAD_REQUEST)

            return Response(api_result, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"Error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def options(self, request, *args, **kwargs):
        print("Method:", request.method)
        print("GET params:", request.GET)
        print("POST data:", request.POST)
        print("Headers:", dict(request.headers))
        print("Body:", request.body.decode('utf-8'))
        print("Path:", request.path)
        print("User:", request.user)
        
        return super().options(request, *args, **kwargs)

class DiabetesAIHealthView(APIView):
    permission_classes = []
    authentication_classes = []

    def get(self, request):
        try:
            try:
                response = requests.get("http://localhost:8001/health")
                api_result = response.json()
                return Response(api_result, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"Error": "Error with the AI Model."}, status=status.HTTP_424_FAILED_DEPENDENCY)
        except Exception as e:
            return Response({"Error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)