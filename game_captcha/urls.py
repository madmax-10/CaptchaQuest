from django.urls import path
from . import views

urlpatterns = [
    path('', views.game_view, name='game'),
    path('verify/', views.verify_captcha, name='verify_captcha'),
    path('success/', views.success_view, name='success'),

]
