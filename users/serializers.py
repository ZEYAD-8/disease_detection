from rest_framework import serializers
from .models import UserCustom, ContactMessage
from django.contrib.auth.password_validation import validate_password

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)
    is_creator = serializers.BooleanField(default=False)

    class Meta:
        model = UserCustom
        fields = ['email', 'password', 'password_confirm', 'is_creator', 'first_name', 'last_name']

    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError("Passwords do not match.")
        return data

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        validated_data['username'] = "Username is not used"
        validated_data['is_admin'] = False
        user = UserCustom.objects.create_user(**validated_data)
        return user


class UserCustomSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])
    id = serializers.IntegerField(read_only=True)
    email = serializers.EmailField(read_only=True)

    class Meta:
        model = UserCustom
        fields = ['id', 'email', 'is_creator', 'password', 'first_name', 'last_name']


    def create(self, validated_data):
        password = validated_data.pop('password')
        user = UserCustom.objects.create_user(password=password, **validated_data)
        return user


class ContactMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ContactMessage
        fields = '__all__'