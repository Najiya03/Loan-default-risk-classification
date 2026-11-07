import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID = {
    "status_checking": "A11", "duration_months": 24, "credit_history": "A32", "purpose": "A43",
    "credit_amount": 3000, "savings": "A61", "employment_since": "A73", "installment_rate": 3,
    "other_debtors": "A101", "residence_since": 2, "property": "A121", "age": 35,
    "other_installment_plans": "A143", "housing": "A152", "existing_credits": 1,
    "job": "A173", "num_dependents": 1, "telephone": "A191",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["environment"]["sklearn_match"] is True


def test_valid_prediction(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_class"] in {"high", "lower"}
    assert "notice" in body


def test_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "age"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert any("age" in e for e in r.json()["errors"])


def test_invalid_category(client):
    r = client.post("/predict", json={**VALID, "purpose": "ZZZ"})
    assert r.status_code == 422


def test_out_of_bounds_age(client):
    r = client.post("/predict", json={**VALID, "age": 5})
    assert r.status_code == 422


def test_unknown_field_rejected(client):
    r = client.post("/predict", json={**VALID, "favourite_colour": "blue"})
    assert r.status_code == 422


def test_sensitive_field_ignored(client):
    r = client.post("/predict", json={**VALID, "foreign_worker": "A201"})
    assert r.status_code == 200
    assert "foreign_worker" in r.json()["ignored_fields"]


def test_malformed_body(client):
    r = client.post("/predict", content="not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_metrics_count_predictions(client):
    before = client.get("/metrics").json()["counts"].get("predictions_total", 0)
    client.post("/predict", json=VALID)
    after = client.get("/metrics").json()["counts"]["predictions_total"]
    assert after == before + 1