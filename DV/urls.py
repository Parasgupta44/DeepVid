from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path,include
from . import views
urlpatterns = [

    path('UploadVideo/',views.UploadVideo,name='Upload'),
    path('UploadVideo/contact/',views.contact,name='contact'),
    path('UploadVideo/about/',views.about,name='about')
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
