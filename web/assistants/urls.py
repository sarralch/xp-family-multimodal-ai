from django.urls import path

from . import views

urlpatterns = [
    path("assistant1/", views.assistant_1_view, name="assistant1"),
    path("dashboard_assistant/", views.dashboard_assistant, name="dashboard_assistant"),
]
