from django.urls import path
from .views import DiabetesAIPredictAPIView, DiabetesAIHealthView

urlpatterns = [
    path('predict/', DiabetesAIPredictAPIView.as_view(), name='ai_predict'),
    path('health/', DiabetesAIHealthView.as_view(), name='ai_health')
]
