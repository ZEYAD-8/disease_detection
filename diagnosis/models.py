from django.db import models
from users.models import UserCustom
# Create your models here.


class DiabetesAttempts(models.Model):
    user = models.ForeignKey(UserCustom, on_delete=models.CASCADE, related_name="diabetes_attempts")
    created_at = models.DateTimeField(auto_now_add=True)

    user_pregnancies = models.FloatField()
    user_glucose = models.FloatField()
    user_blood_pressure = models.FloatField()
    user_skin_thickness = models.FloatField()
    user_insulin = models.FloatField()
    user_bmi = models.FloatField()
    user_diabetes_pedigree_function = models.FloatField()
    user_age = models.FloatField()

    ai_prediction = models.IntegerField(blank=True, null=True)
    ai_probability = models.FloatField(blank=True, null=True)
    ai_risk_level = models.CharField(max_length=50, blank=True, null=True)

    def __str__(self):
        return f"An attempt for user: [{self.user}]"

