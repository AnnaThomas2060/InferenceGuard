"""Integration tests for production FastAPI endpoints in InferenceGuard.
These tests verify health checks, readiness probes, and request payload validation.
TestClient executes requests against the application without starting a live web server."""

import unittest
from fastapi.testclient import TestClient

from src.api.app import app


class TestApiProduction(unittest.TestCase):
    """Test suite for validating API endpoints and error response schemas.
    It checks that liveness and readiness probes return HTTP 200 with expected keys.
    Input validation guards are tested to ensure malformed requests are rejected cleanly."""

    def setUp(self) -> None:
        """Initialize the FastAPI test client before each test execution.
        The client simulates HTTP interactions directly against the ASGI app instance.
        Startup and shutdown lifecycle events are triggered during client initialization."""
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        """Verify the health check endpoint returns status healthy and current timestamp.
        The endpoint must respond with HTTP 200 to satisfy container liveness probes.
        Returned JSON must contain service identity and status fields."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "inference-guard")
        self.assertIn("version", data)
        self.assertIn("timestamp", data)

    def test_readiness_endpoint(self) -> None:
        """Verify the readiness probe reports model initialization and hardware availability.
        The endpoint returns HTTP 200 indicating the probe itself succeeded.
        Response fields report ready state, configured device, and model identifier."""
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("ready", data)
        self.assertIn("model_id", data)
        self.assertIn("device", data)
        self.assertIn("cuda_available", data)
        self.assertIn("rewriter_model_id", data)

    def test_analyze_empty_payload(self) -> None:
        """Verify that requests missing required text fields return validation errors.
        FastAPI must reject empty JSON objects with HTTP 422 Unprocessable Entity.
        This confirms schema validation protects downstream model inference."""
        response = self.client.post("/analyze", json={})
        self.assertEqual(response.status_code, 422)

    def test_analyze_valid_text(self) -> None:
        """Verify that valid input text produces a structured risk report.
        The response must return HTTP 200 and include report attributes and latency.
        Overall risk scores and categorical risk bands must be populated."""
        response = self.client.post(
            "/analyze",
            json={"text": "I am a doctor living in Chicago and graduated in 2015."},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("report", data)
        self.assertIn("latency_ms", data)
        report = data["report"]
        self.assertIn("overall_risk_score", report)
        self.assertIn("overall_risk_band", report)
        self.assertIn("attributes", report)


if __name__ == "__main__":
    unittest.main()
