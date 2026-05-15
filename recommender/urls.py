from django.urls import path

from . import views

app_name = 'recommender'

urlpatterns = [
    path('', views.home, name='home'),
    path('ai/', views.ai, name='ai'),
    path('explore/', views.explore, name='explore'),
    path('persona/', views.persona, name='persona'),
    path('api/chat/', views.chat_api, name='chat_api'),
]