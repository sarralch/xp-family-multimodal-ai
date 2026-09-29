from django.urls import path

from . import views

urlpatterns = [
    path("assistant2/", views.assistant_2_view, name="assistant2"),
]
