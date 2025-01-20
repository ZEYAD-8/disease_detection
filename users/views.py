from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import authenticate
from .serializers import UserCustomSerializer, UserRegistrationSerializer
from rest_framework.authtoken.models import Token
from .models import UserCustom

class RegisterUserView(APIView):
    permission_classes = []
    authentication_classes = []

    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            token, _ = Token.objects.get_or_create(user=user)
            user_data = UserCustomSerializer(user).data
            return Response({
                'message': 'User registered successfully',
                'token': token.key,
                'user': user_data
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginUserView(APIView):
    permission_classes = []
    authentication_classes = []

    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        user = authenticate(request, email=email, password=password)
        if user is None:
            return Response({'error': 'Invalid email or password'}, status=status.HTTP_401_UNAUTHORIZED)

        token, _ = Token.objects.get_or_create(user=user)
        if request.data.get('refresh_token', False):
            token.delete()
            token = Token.objects.create(user=user)

        user_data = UserCustomSerializer(user).data
        return Response({
            'user': user_data,
            'token': token.key
        })

class ChangePasswordView(APIView):

    def post(self, request):
        user = request.user
        old_password = request.data.get('old_password')
        new_password = request.data.get('new_password')

        if not user.check_password(old_password):
            return Response({'error': 'Invalid old password'}, status=status.HTTP_400_BAD_REQUEST)

        user.set_password(new_password)
        user.save()
        return Response({'message': 'Password changed successfully'}, status=status.HTTP_200_OK)

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



class ForgotPasswordView(APIView):
    permission_classes  = []
    authentication_classes = []

    def post(self, request):
        email = request.data.get('email')
        user = UserCustom.objects.filter(email=email).first()
        if user is None:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

        new_pass = request.data.get('new_password')
        user.set_password(new_pass)
        user.save()
        return Response({'message': 'Password reset successfully'}, status=status.HTTP_200_OK)
