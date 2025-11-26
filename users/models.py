from django.db import models
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.contrib.auth.models import UserManager


class UserCustomManager(UserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, **extra_fields)


class UserCustom(AbstractUser):
    email = models.EmailField(unique=True)
    is_creator = models.BooleanField(default=False)
    is_admin = models.BooleanField(default=False)
    username = models.CharField(max_length=30, unique=False, blank=True, null=True)
    first_name = models.CharField(max_length=30, unique=False, blank=True, null=True)
    last_name = models.CharField(max_length=30, unique=False, blank=True, null=True)

    verification_code = models.CharField(max_length=6, blank=True, null=True)
    code_created_at = models.DateTimeField(blank=True, null=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['is_admin']

    groups = models.ManyToManyField(
        Group,
        related_name='usercustom_groups',
        blank=True
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name='usercustom_premissions',
        blank=True
    )

    objects = UserCustomManager()

    def __str__(self):
        return f"[{self.id}] {self.email}"


class ContactMessage(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone_number = models.CharField(max_length=20, blank=True)
    subject = models.CharField(max_length=512)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message from {self.name} - {self.subject}"