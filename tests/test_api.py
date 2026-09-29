import pytest
from fastapi.testclient import TestClient

from xp_family.api.main import app
from xp_family.services import Services, get_services


@pytest.fixture
def client(services: Services):
    app.dependency_overrides[get_services] = lambda: services
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_and_trace_header(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Trace-Id"]


def test_ask_returns_answer_with_sources(client):
    body = client.post("/v1/ask", json={"question": "L'hydroquinone est-elle autorisée ?"}).json()
    assert body["answer"]
    assert body["sources"]
    assert any("Hydroquinone" in s["excerpt"] for s in body["sources"])
    assert "dermatologue" in body["disclaimer"]


def test_safety_check_is_deterministic(client):
    body = client.post("/v1/safety-check", json={"ingredients": "Glycerin, Oxybenzone, Foo"}).json()
    assert [v["status"] for v in body["verdicts"]] == ["safe", "banned", "unknown"]
    assert body["summary"]["banned"] == 1


def test_recipe_refused_when_an_ingredient_is_unsuitable(client):
    body = client.post("/v1/recipe", json={"ingredients": "aloe vera, retinol"}).json()
    assert body["recipe"] is None
    assert "retinol" in body["refused_reason"]


def test_recipe_generated_from_product_context(client):
    body = client.post(
        "/v1/recipe", json={"ingredients": "aloe vera, glycerin", "product_type": "crème"}
    ).json()
    assert body["recipe"]
    assert body["sources"]


def test_missing_api_key_is_a_503_not_a_crash(settings):
    app.dependency_overrides[get_services] = lambda: Services(settings, use_default_models=False)
    try:
        response = TestClient(app).post("/v1/ask", json={"question": "zinc oxide ?"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
