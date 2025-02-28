from django.db import models
import uuid
import json

class CaptchaSession(models.Model):
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    completed = models.BooleanField(default=False)
    passed = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Session {self.session_id} - {'Passed' if self.passed else 'Failed' if self.completed else 'Incomplete'}"

class MouseMovement(models.Model):
    session = models.ForeignKey(CaptchaSession, on_delete=models.CASCADE, related_name='movements')
    timestamp = models.FloatField()  # Client-side timestamp
    x = models.FloatField()
    y = models.FloatField()
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"Movement at ({self.x}, {self.y}) - {self.timestamp}"
