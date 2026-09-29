from django.urls import path

from . import views

urlpatterns = [
    path("assistant3/", views.assistant_3_view, name="assistant3"),
]
