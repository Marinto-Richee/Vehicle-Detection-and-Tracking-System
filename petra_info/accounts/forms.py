from django import forms
from .models import CCTV, PolygonZone, LineZone

class CCTVForm(forms.ModelForm):
    class Meta:
        model = CCTV
        fields = ['name', 'category', 'rtsp']
        widgets = {
            'name': forms.Select(attrs={'class': 'form-control'}, choices=[('Camera In', 'Camera In'), ('Camera Out', 'Camera Out')]),
            'category': forms.Select(attrs={'class': 'form-control'}, choices=[('In', 'In'), ('Out', 'Out')]),
            'rtsp': forms.URLInput(attrs={'class': 'form-control'}),
        }

