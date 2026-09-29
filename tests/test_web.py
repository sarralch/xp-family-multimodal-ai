"""Smoke tests for the Django UI, run against the same offline services as the API tests."""

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("django")

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


@pytest.fixture(scope="module")
def django_client():
    sys.path.insert(0, str(WEB_DIR))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_project.settings")
    os.environ.setdefault("DJANGO_DEBUG", "true")
    import django
    from django.test import Client
    from django.test.utils import setup_test_environment

    django.setup()
    setup_test_environment()
    return Client()


@pytest.fixture
def offline(monkeypatch, services):
    import assistants.views
    import assistants1.views
    import assistants2.views

    for module in (assistants.views, assistants1.views, assistants2.views):
        monkeypatch.setattr(module, "get_services", lambda: services)


def test_dashboard_renders(django_client):
    assert django_client.get("/assistants/dashboard_assistant/").status_code == 200


def test_root_redirects_to_dashboard(django_client):
    response = django_client.get("/")
    assert response.status_code == 302
    assert response.url.endswith("/assistants/dashboard_assistant/")


def test_question_page_shows_answer_and_sources(django_client, offline):
    body = django_client.post("/assistants1/assistant2/", {"question": "Hydroquinone ?"})
    html = body.content.decode()
    assert body.status_code == 200
    assert 'aria-live="polite"' in html
    assert "Sources" in html


def test_recipe_page_refuses_unsuitable_ingredients(django_client, offline):
    html = django_client.post(
        "/assistants2/assistant3/", {"ingredients": "aloe vera, retinol"}
    ).content.decode()
    assert "non adaptés" in html
    assert "Déconseillé" in html
