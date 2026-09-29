from django.urls import path
from . import views

urlpatterns = [
    path('assistant2/', views.assistant_2_view, name='assistant2'),
    #path('assistant2/', views.assistant_2, name='assistant2'),
   #path('assistant3/', views.assistant_3, name='assistant3'),
]