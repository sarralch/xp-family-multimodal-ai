from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="dashboard_assistant", permanent=False)),
    path("admin/", admin.site.urls),
    path("assistants/", include("assistants.urls")),
    path("assistants1/", include("assistants1.urls")),
    path("assistants2/", include("assistants2.urls")),
]
