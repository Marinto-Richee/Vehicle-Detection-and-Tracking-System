from django.db import models
from django.utils import timezone
# Create your models here.

class CCTV(models.Model):
    name = models.CharField(max_length=100)
    category = models.CharField(max_length=3)
    rtsp = models.TextField() 
    running_status = models.BooleanField(default=False)
    start_detection = models.BooleanField(default=False)
    def __str__(self):
        return self.name

class PolygonZone(models.Model):
    name = models.CharField(max_length=100)
    camera = models.ForeignKey(CCTV, on_delete=models.CASCADE)
    # Polygon coordinates
    coordinates = models.JSONField()
    def __str__(self):
        return self.name
    
class LineZone(models.Model):
    name = models.CharField(max_length=100)
    camera = models.ForeignKey(CCTV, on_delete=models.CASCADE)
    # Line coordinates
    coordinates = models.JSONField()
    def __str__(self):
        return self.name
    
class tracker_status(models.Model):
    tracker_id = models.CharField(max_length=100,primary_key=True)
    status = models.CharField(max_length=100,default='Unknown')
    category = models.CharField(max_length=100,default='Unknown')
    completed = models.BooleanField(default=False)
    last_detection_time = models.DateTimeField(default=timezone.now) 
    def __str__(self):
        return f'{self.tracker_id}'

class Detection(models.Model):
    unique_id = models.CharField(max_length=100, default='Unknown',primary_key=True)
    tracker = models.ForeignKey(tracker_status, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)  # Date and Time of detection
    camera = models.CharField(max_length=100, default='Unknown')  # Camera name
    image = models.ImageField(upload_to='detections/', null=True, blank=True)  # Save vehicle image


class Vehicle(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)  # Date and Time of detection
    camera = models.CharField(max_length=100)  # Camera name
    image = models.ImageField(upload_to='vehicles/', null=True, blank=True)  # Save vehicle image
    license_plate = models.CharField(max_length=100, default='UNREGISTERED')  # License plate number
    license_plate_image = models.ImageField(upload_to='license_plates/', null=True, blank=True)  # Save license plate image
    def __str__(self):
        return self.license_plate
    


class ScriptStatus(models.Model):
    id = models.AutoField(primary_key=True)
    script_name = models.CharField(max_length=255)
    status = models.CharField(max_length=50)  # e.g., "Running", "Stopped", "Error"
    last_updated = models.DateTimeField(auto_now=True)  # Automatically updates with each status change

    def __str__(self):
        return f"{self.script_name}: {self.status}"