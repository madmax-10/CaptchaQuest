from django.contrib import admin
from game_captcha.models import CaptchaSession, MouseMovement

# Register your models here.
admin.site.register(CaptchaSession)
admin.site.register(MouseMovement)