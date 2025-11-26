from rest_framework import serializers
from .models import HeartPredictionAttempt

class HeartPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = HeartPredictionAttempt
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'user']