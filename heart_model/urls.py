from django.urls import path
from .views import HeartPredictionView

urlpatterns = [
    path('predict/', HeartPredictionView.as_view(), name='heart-predict'),
]