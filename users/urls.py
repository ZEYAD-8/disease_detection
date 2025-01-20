from django.urls import path
from .views import RegisterUserView, LoginUserView, UserProfileView, ChangePasswordView, ForgotPasswordView

urlpatterns = [
    path('register/', RegisterUserView.as_view(), name='register_user'),
    path('login/', LoginUserView.as_view(), name='login_user'),
    # path('change_password/', ChangePasswordView.as_view(), name='change_password'),
    path('forgot_password/', ForgotPasswordView.as_view(), name='forgot_password'),

    path('profile/', UserProfileView.as_view(), name='user_profile'),
]
