from django.urls import path
from .views import RegisterUserView, LoginUserView, UserProfileView, PasswordResetView, VerifyPinCodeView, AttemptsView, ContactView

urlpatterns = [
    path('register/', RegisterUserView.as_view(), name='register_user'),
    path('login/', LoginUserView.as_view(), name='login_user'),
    path('reset_password/', PasswordResetView.as_view(), name='reset_password'),

    path('verify-code/', VerifyPinCodeView.as_view(), name='verify-code-registration'),
    path('verify-reset/', VerifyPinCodeView.as_view(), name='verify-code-reset'),

    path('profile/', UserProfileView.as_view(), name='user_profile'),
    path('attempts/', AttemptsView.as_view(), name='user_attempts'),

    path('contact/', ContactView.as_view(), name='contact-form'),
]
