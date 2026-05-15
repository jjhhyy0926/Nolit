from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('login/',          views.login,          name='login'),
    path('logout/',         views.logout,          name='logout'),
    path('signup/',         views.signup,          name='signup'),
    path('check-username/', views.check_username,  name='check_username'),
]