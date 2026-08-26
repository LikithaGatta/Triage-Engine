"""
tests/test_week4.py
====================

RUN:
    pytest tests/test_week4.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.dependencies import get_models, get_store, InMemoryStore, ModelContainer
from src.models.explainer import ExplanationResult, TokenContribution


def make_mock_models():
    """Create a mock ModelContainer that behaves like the real one."""
    mock = MagicMock()  # No spec= so all attributes are allowed
    mock.is_ready = True
    mock.version  = "1.0.0"
    mock.uptime_seconds = 42.0

    # Mock preprocessor
    mock.preprocessor.clean_text.return_value = "charged twice refund billing"
    mock.preprocessor.tokenize_and_filter.return_value = ["charged", "twice", "refund"]

    # Mock pipeline prediction
    import numpy as np
    mock.pipeline.predict_proba.return_value = np.array([[
        0.05,   # account_access
        0.85,   # billing  ← predicted
        0.04,   # bug_report
        0.02,   # feature_request
        0.02,   # general
        0.02,   # performance
    ]])
    mock.pipeline.classes_ = [
        "account_access", "billing", "bug_report",
        "feature_request", "general", "performance"
    ]

    # Mock SHAP explainer
    mock.explainer.explain.return_value = ExplanationResult(
        ticket_id="pending",
        predicted_category="billing",
        confidence=0.85,
        base_value=0.17,
        top_positive=[
            TokenContribution("charged", 0.42, "positive"),
            TokenContribution("refund",  0.31, "positive"),
        ],
        top_negative=[
            TokenContribution("login", -0.05, "negative"),
        ],
        all_contributions=[],
        explanation_text='Classified as billing (85% confidence). Key signals: "charged", "refund".',
    )

    return mock


@pytest.fixture
def client():
    """
    TestClient with mocked models injected.
    FastAPI's dependency override replaces get_models() with
    our mock — no real model files needed.
    """
    mock_models = make_mock_models()

    # Override the dependency for this test session
    app.dependency_overrides[get_models] = lambda: mock_models

    with TestClient(app) as c:
        yield c

    # Clean up overrides after tests
    app.dependency_overrides.clear()


@pytest.fixture
def fresh_store():
    """A clean InMemoryStore for each test."""
    store = InMemoryStore()
    app.dependency_overrides[get_store] = lambda: store
    yield store
    app.dependency_overrides.pop(get_store, None)



class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have required fields."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status"           in data
        assert "model_loaded"     in data
        assert "explainer_loaded" in data
        assert "uptime_seconds"   in data
        assert "version"          in data

    def test_health_status_is_healthy(self, client):
        """Status should be 'healthy' when models are loaded."""
        response = client.get("/api/v1/health")
        assert response.json()["status"] == "healthy"



class TestSubmitTicket:

    def test_submit_returns_201(self, client, fresh_store):
        """Successful ticket submission should return 201 Created."""
        response = client.post("/api/v1/tickets", json={
            "subject": "Charged twice",
            "body":    "I was charged twice this month for my Pro subscription.",
            "source":  "test",
        })
        assert response.status_code == 201

    def test_submit_response_has_required_fields(self, client, fresh_store):
        """Response must include all required fields."""
        response = client.post("/api/v1/tickets", json={
            "body": "I was charged twice this month."
        })
        data = response.json()
        assert "ticket_id"          in data
        assert "predicted_category" in data
        assert "predicted_urgency"  in data
        assert "confidence"         in data
        assert "auto_routed"        in data
        assert "routed_to"          in data
        assert "explanation"        in data
        assert "processing_time_ms" in data

    def test_submit_requires_body_field(self, client, fresh_store):
        """Request without body field should return 422."""
        response = client.post("/api/v1/tickets", json={
            "subject": "No body field provided"
        })
        assert response.status_code == 422

    def test_submit_rejects_too_short_body(self, client, fresh_store):
        """Body shorter than 5 characters should return 422."""
        response = client.post("/api/v1/tickets", json={"body": "Hi"})
        assert response.status_code == 422

    def test_predicted_category_is_billing(self, client, fresh_store):
        """Mock returns billing — response should show billing."""
        response = client.post("/api/v1/tickets", json={
            "body": "I was charged twice this month."
        })
        assert response.json()["predicted_category"] == "billing"

    def test_high_confidence_ticket_is_auto_routed(self, client, fresh_store):
        """85% confidence is above 75% threshold — should auto-route."""
        response = client.post("/api/v1/tickets", json={
            "body": "I was charged twice this month."
        })
        data = response.json()
        assert data["auto_routed"] == True
        assert data["routed_to"] == "Billing Team"

    def test_explanation_in_response(self, client, fresh_store):
        """Every response must include a SHAP explanation."""
        response = client.post("/api/v1/tickets", json={
            "body": "I was charged twice this month."
        })
        explanation = response.json()["explanation"]
        assert "top_positive"     in explanation
        assert "top_negative"     in explanation
        assert "base_value"       in explanation
        assert "explanation_text" in explanation

    def test_ticket_id_is_assigned(self, client, fresh_store):
        """Submitted ticket should receive a unique ID."""
        response = client.post("/api/v1/tickets", json={
            "body": "I was charged twice this month."
        })
        ticket_id = response.json()["ticket_id"]
        assert ticket_id is not None
        assert len(ticket_id) > 0

    def test_two_tickets_get_different_ids(self, client, fresh_store):
        """Each submission should get a unique ticket ID."""
        r1 = client.post("/api/v1/tickets", json={"body": "Ticket one text here."})
        r2 = client.post("/api/v1/tickets", json={"body": "Ticket two text here."})
        assert r1.json()["ticket_id"] != r2.json()["ticket_id"]


class TestListTickets:

    def test_list_empty_returns_200(self, client, fresh_store):
        """Empty store should return 200 with empty list."""
        response = client.get("/api/v1/tickets")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_after_submit(self, client, fresh_store):
        """After submitting a ticket, list should contain it."""
        client.post("/api/v1/tickets", json={"body": "I need help with billing."})
        response = client.get("/api/v1/tickets")
        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_list_pagination_limit(self, client, fresh_store):
        """Limit parameter should cap results."""
        for i in range(5):
            client.post("/api/v1/tickets", json={"body": f"Ticket number {i} content here."})
        response = client.get("/api/v1/tickets?limit=3")
        assert len(response.json()) == 3


class TestGetTicket:

    def test_get_existing_ticket(self, client, fresh_store):
        """Should return 200 for an existing ticket."""
        submit = client.post("/api/v1/tickets", json={"body": "Billing issue here."})
        ticket_id = submit.json()["ticket_id"]
        response = client.get(f"/api/v1/tickets/{ticket_id}")
        assert response.status_code == 200

    def test_get_nonexistent_ticket_returns_404(self, client, fresh_store):
        """Should return 404 for a ticket that does not exist."""
        response = client.get("/api/v1/tickets/NONEXISTENT-999")
        assert response.status_code == 404

    def test_get_ticket_includes_body(self, client, fresh_store):
        """Detail response should include the original ticket body."""
        body_text = "I was charged twice for my subscription this month."
        submit = client.post("/api/v1/tickets", json={"body": body_text})
        ticket_id = submit.json()["ticket_id"]
        response = client.get(f"/api/v1/tickets/{ticket_id}")
        assert response.json()["body"] == body_text


class TestOverride:

    def test_override_existing_ticket(self, client, fresh_store):
        """Should record an override successfully."""
        submit = client.post("/api/v1/tickets", json={"body": "Some ticket content here."})
        ticket_id = submit.json()["ticket_id"]
        response = client.post(f"/api/v1/tickets/{ticket_id}/override", json={
            "corrected_category": "account_access",
            "corrected_urgency":  "high",
            "agent_id":           "agent_test_001",
            "correction_note":    "Actually an account issue",
        })
        assert response.status_code == 200

    def test_override_invalid_category_returns_422(self, client, fresh_store):
        """Invalid category in override should return 422."""
        submit = client.post("/api/v1/tickets", json={"body": "Some ticket content here."})
        ticket_id = submit.json()["ticket_id"]
        response = client.post(f"/api/v1/tickets/{ticket_id}/override", json={
            "corrected_category": "invalid_category_xyz",
            "agent_id":           "agent_001",
        })
        assert response.status_code == 422

    def test_override_nonexistent_ticket_returns_404(self, client, fresh_store):
        """Overriding a ticket that does not exist should return 404."""
        response = client.post("/api/v1/tickets/FAKE-000001/override", json={
            "corrected_category": "billing",
            "agent_id":           "agent_001",
        })
        assert response.status_code == 404


class TestMetrics:

    def test_metrics_returns_200(self, client, fresh_store):
        """Metrics endpoint should return 200."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

    def test_metrics_has_required_fields(self, client, fresh_store):
        """Metrics response should have all required fields."""
        response = client.get("/api/v1/metrics")
        data = response.json()
        assert "total_tickets"       in data
        assert "auto_routed_count"   in data
        assert "auto_route_rate"     in data
        assert "avg_confidence"      in data
        assert "tickets_by_category" in data
        assert "model_version"       in data

    def test_metrics_counts_tickets(self, client, fresh_store):
        """total_tickets should reflect submitted tickets."""
        client.post("/api/v1/tickets", json={"body": "First ticket body text."})
        client.post("/api/v1/tickets", json={"body": "Second ticket body text."})
        response = client.get("/api/v1/metrics")
        assert response.json()["total_tickets"] == 2