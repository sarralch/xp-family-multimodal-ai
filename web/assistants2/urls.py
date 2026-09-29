from django.urls import path
from . import views

urlpatterns = [
    path('assistant3/', views.assistant_3_view, name='assistant3'),
    #path('assistant2/', views.assistant_2, name='assistant2'),
   #path('assistant3/', views.assistant_3, name='assistant3'),
]