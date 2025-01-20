from django.contrib.auth.backends import BaseBackend
from rest_framework.authentication import TokenAuthentication

from .models import UserCustom


class EmailBackend(BaseBackend):
    def authenticate(self, request, email=None, username=None, password=None, **kwargs):
        try:
            email = email or username
            user = UserCustom.objects.get(email=email)
            if user.check_password(password):
                return user
        except UserCustom.DoesNotExist:
            return None

    def get_user(self, user_id):
        try:
            return UserCustom.objects.get(pk=user_id)
        except UserCustom.DoesNotExist:
            return None


# overriding TokenAuthentication class to use custom keyword
# and then let django continue with the rest of the process
class CustomTokenAuthentication(TokenAuthentication):
    keyword = 'Token'

    def authenticate(self, request):
        token = request.headers.get(self.keyword)
        if not token:
            return None

        return self.authenticate_credentials(token)
