# from rest_framework.permissions import BasePermission

# class IsCreator(BasePermission):

#     def has_permission(self, request, view):
#         if request.method == 'POST':
#             return request.user.is_authenticated and request.user.is_creator
#         return True

#     def has_object_permission(self, request, view, obj):
#         return request.user.is_authenticated and obj.user == request.user
