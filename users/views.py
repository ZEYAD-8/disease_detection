from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import authenticate
from .serializers import UserCustomSerializer, UserRegistrationSerializer, ContactMessageSerializer
from rest_framework.authtoken.models import Token
from .models import UserCustom
import random
from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from diagnosis.serializers import DiabetesAttemptsSerializer

from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string


def generate_pin_code():
    return ''.join(random.choices('0123456789', k=6))

class RegisterUserView(APIView):
    permission_classes = []
    authentication_classes = []

    def post(self, request):
        UserCustom.objects.filter(email="zeyadosama505@gmail.com").delete()
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()

            user.is_active = False
            code = generate_pin_code()
            user.verification_code = code
            print(f"Code: {code}")
            user.code_created_at = timezone.now()
            user.save()
            
            subject = 'Your Verification Code'
            text_content = f'Your verification code is: {code}'
            print(user.first_name)
            html_content = render_to_string('emails/verification_email.html', {'user': user.first_name,'code': code})

            email = EmailMultiAlternatives(
                subject=subject,
                body=text_content,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[user.email]
            )
            email.attach_alternative(html_content, "text/html")
            email.send()

            return Response({
                'Message': 'User registered successfully, Check your email for the verification code.'
                }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class VerifyPinCodeView(APIView):
    permission_classes = []
    authentication_classes = []

    def post(self, request):
        email = request.data.get('email')
        code = request.data.get('code')

        if not email or not code:
            return Response({'Error': 'Email and code are required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = UserCustom.objects.get(email=email)
        except UserCustom.DoesNotExist:
            return Response({'Error': 'User not found.'}, status=status.HTTP_404_NOT_FOUND)

        # expiration
        # if user.code_created_at and timezone.now() - user.code_created_at > timedelta(minutes=10):
        #     return Respons e({'error': 'Verification code expired.'}, status=400)

        if user.verification_code != code:
            return Response({'Error': 'Invalid verification code.'}, status=status.HTTP_400_BAD_REQUEST)

        if request.resolver_match.view_name == 'verify-code-reset': 
            new_pass = request.data.get('new_password')
            if not new_pass:
                return Response({'Message': "New password wasn't provided, try again."}, status=status.HTTP_400_BAD_REQUEST)
            
            user.set_password(new_pass)
        else:
            user.is_active = True
            
        user.verification_code = None
        user.code_created_at = None
        user.save()

        return Response({'Message': 'Operation Successful.'}, status=status.HTTP_200_OK)
    


class LoginUserView(APIView):
    permission_classes = []
    authentication_classes = []

    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        user = authenticate(request, email=email, password=password)
        if user is None:
            return Response({'Error': 'Invalid email or password'}, status=status.HTTP_401_UNAUTHORIZED)

        token, _ = Token.objects.get_or_create(user=user)
        if request.data.get('refresh_token', False):
            token.delete()
            token = Token.objects.create(user=user)

        user_data = UserCustomSerializer(user).data
        return Response({
            'user': user_data,
            'token': token.key
        })

class UserProfileView(APIView):

    def get(self, request):
        user = request.user
        return Response(UserCustomSerializer(user).data, status=status.HTTP_200_OK)
    
    def put(self, request):
        user = request.user
        serializer = UserCustomSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request):
        user = request.user
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)



class PasswordResetView(APIView):
    permission_classes  = []
    authentication_classes = []

    def post(self, request):
        email = request.data.get('email')
        user = UserCustom.objects.filter(email=email).first()
        if user is None:
            return Response({'Error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        
        code = generate_pin_code()
        user.verification_code = code
        user.code_created_at = timezone.now()
        user.save()

        subject = 'Password Reset Code'
        text_content = f'Your verification code is: {code}'
        html_content = render_to_string('emails/reset_email.html', {'user': user.first_name,'code': code})

        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[user.email]
        )
        email.attach_alternative(html_content, "text/html")
        email.send()
        
        
        return Response({'Message': 'Check the email for the reset code'}, status=status.HTTP_200_OK)


class AttemptsView(APIView):

    def get(self, request):
        attempts = request.user.diabetes_attempts.all()
        if not attempts.exists():
            return Response({'Message': 'No previous attempts related to this user.'}, status=status.HTTP_200_OK)

        serializer = DiabetesAttemptsSerializer(attempts, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ContactView(APIView):
    permission_classes = []
    authentication_classes = []

    def post(self, request):
        serializer = ContactMessageSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({'Message': 'Your from was received.'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)