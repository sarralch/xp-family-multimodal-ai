from django.urls import path
from . import views

urlpatterns = [
    path('assistant1/', views.assistant_1_view, name='assistant1'),
    path('dashboard_assistant/', views.dashboard_assistant, name='dashboard_assistant'),

    #path('assistant2/', views.assistant_2, name='assistant2'),
   #path('assistant3/', views.assistant_3, name='assistant3'),
]