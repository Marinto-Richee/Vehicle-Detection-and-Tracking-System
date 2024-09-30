from django.contrib import admin
from django.utils.html import format_html
from .models import *

@admin.register(CCTV)
class CCTVAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'rtsp',"running_status")  # Ensure fields are listed here

@admin.register(PolygonZone)
class PolygonZoneAdmin(admin.ModelAdmin):
    list_display = ('name', 'camera')  # List relevant fields

@admin.register(LineZone)
class LineZoneAdmin(admin.ModelAdmin):
    list_display = ('name', 'camera')  # List relevant fields


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('tracker','unique_id', 'timestamp', 'camera', 'vehicle_image')

    def vehicle_image(self, obj):
        if obj.image:
            return format_html('<img src="{}" style="width: 100px; height: auto;" />', obj.image.url)
        return "No Image"
    vehicle_image.short_description = 'Vehicle Image'

@admin.register(tracker_status)
class tracker_statusAdmin(admin.ModelAdmin):
    list_display = ('tracker_id', 'status', 'category', 'completed', 'last_detection_time')

@admin.register(Vehicle)
class VehicleAdmin(admin.ModelAdmin):
    list_display = ('license_plate','timestamp', 'camera', 'vehicle_image',  'license_plate_image')

    def vehicle_image(self, obj):
        if obj.image and hasattr(obj.image, 'url'):
            return format_html('<img src="{}" style="width: 100px; height: auto;" />', obj.image.url)
        return "No Image"
    vehicle_image.short_description = 'Vehicle Image'

    def license_plate_image(self, obj):
        if obj.license_plate_image and hasattr(obj.license_plate_image, 'url'):
            return format_html('<img src="{}" style="width: 100px; height: auto;" />', obj.license_plate_image.url)
        return "No Image"
    license_plate_image.short_description = 'License Plate Image'

@admin.register(ScriptStatus)
class ScriptStatusAdmin(admin.ModelAdmin):
    list_display = ('script_name', 'status')