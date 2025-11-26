from django.db import models
from django.conf import settings
from users.models import UserCustom
# Create your models here.

class HeartPredictionAttempt(models.Model):
    user = models.ForeignKey(UserCustom, on_delete=models.CASCADE, related_name='heart_attempts')
    created_at = models.DateTimeField(auto_now_add=True)

    user_age = models.IntegerField()
    user_sex = models.IntegerField()
    user_trestbps = models.IntegerField()
    user_chol = models.IntegerField()
    user_fbs = models.IntegerField()
    user_thalach = models.IntegerField()
    user_exang = models.IntegerField()
    user_oldpeak = models.FloatField()
    user_ca = models.IntegerField()
    
    user_cp_1 = models.IntegerField()
    user_cp_2 = models.IntegerField()
    user_cp_3 = models.IntegerField()
    
    user_restecg_1 = models.IntegerField()
    user_restecg_2 = models.IntegerField()
    
    user_slope_1 = models.IntegerField()
    user_slope_2 = models.IntegerField()
    
    user_thal_1 = models.IntegerField()
    user_thal_2 = models.IntegerField()
    user_thal_3 = models.IntegerField()
    
    user_age_group_Middle = models.IntegerField()
    user_age_group_Senior = models.IntegerField()
    user_age_group_Elderly = models.IntegerField()

    ai_probability = models.FloatField()
    ai_prediction = models.IntegerField()

    def __str__(self):
        return f"Heart Prediction by {self.user.email} at {self.created_at}"