from django.urls import path
from .views import SkinCancerPredictionView

urlpatterns = [
    path('predict/', SkinCancerPredictionView.as_view(), name='skin-cancer-predict'),
]