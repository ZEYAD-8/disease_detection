from django.shortcuts import render

# Create your views here.
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class SkinCancerPredictionView(APIView):
    def post(self, request):
        try:
            # Replace with your actual Docker/AI endpoint
            external_api_url = "http://localhost:6001/predict"

            # Forward the incoming request's data to the external API
            response = requests.post(external_api_url, files=request.FILES, data=request.data)

            if response.status_code == 200:
                return Response(response.json(), status=status.HTTP_200_OK)
            else:
                return Response(
                    {"error": "AI model returned an error", "details": response.text},
                    status=response.status_code
                )

        except requests.RequestException as e:
            return Response({"error": "Failed to contact AI model", "details": str(e)}, status=500)