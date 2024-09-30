"""
URL configuration for petra_info project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from accounts.views import *
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('accounts/login/', login_view, name='login'),
    path('admin/', admin.site.urls),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('home/', home_view, name='home'),
    path('config-cameras/', config_cameras, name='config_cameras'),
    path('add-camera/', add_camera, name='add_camera'),
    path('update-camera/<int:camera_id>/', update_camera, name='update_camera'),
    path('delete-camera/<int:camera_id>/', delete_camera, name='delete_camera'),
    path('configure-zones/<int:camera_id>/', configure_zones, name='configure_zones'),
    path('toggle-detection/<int:camera_id>/', toggle_detection, name='toggle_detection'),
    path('save-zones/<int:camera_id>/', save_zones, name='save_zones'),
    path('export_excel/', export_to_excel, name='export_excel'),
    path('', RedirectView.as_view(url='/login/')), 
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

