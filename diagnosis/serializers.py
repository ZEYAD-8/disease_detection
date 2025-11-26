from rest_framework import serializers
from .models import DiabetesAttempts

class DiabetesAttemptsSerializer(serializers.ModelSerializer):
    class Meta:
        model = DiabetesAttempts
        fields = [
            'id',
            'user',
            'created_at',
            'user_pregnancies',
            'user_glucose',
            'user_blood_pressure',
            'user_skin_thickness',
            'user_insulin',
            'user_bmi',
            'user_diabetes_pedigree_function',
            'user_age',
            'ai_prediction',
            'ai_probability',
            'ai_risk_level'
        ]
        read_only_fields = ['id', 'created_at', 'user']

    def create(self, validated_data):
        request = self.context.get('request')
        user = request.user if request else None
        if not user:
            raise serializers.ValidationError("User must be authenticated to create a diabetes attempt.")
        return DiabetesAttempts.objects.create(user=user, **validated_data)