from django.urls import path

from . import views

app_name = 'file_upload'
urlpatterns = [
    path('', views.index, name='index'),
    path('results', views.file_get, name='results'),

]