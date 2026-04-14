import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


class ApiContractsTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_analyze_cv_returns_expected_shape(self):
        fake_result = {
            "report_id": "CV-20260415-1001",
            "source": "cv_rule_v1",
            "incomplete": False,
            "risk_score": 28,
            "risk_level": "low",
            "action_scores": [],
            "action_metrics": {
                "close_eye_force": {},
                "pout": {},
                "puff_cheek": {},
            },
            "missing_actions": [],
            "top_abnormal_features": [],
            "visual_regions": [],
            "short_advice": "Current CV risk is low.",
            "quality_flags": [],
            "disclaimer": "For early screening support only. Not a clinical diagnosis.",
        }
        files = [
            ("close_eye_force", ("close_eye_force.jpg", b"img-1", "image/jpeg")),
            ("pout", ("pout.jpg", b"img-2", "image/jpeg")),
            ("puff_cheek", ("puff_cheek.jpg", b"img-3", "image/jpeg")),
        ]
        with patch("app.api.inference.run_cv_scoring", return_value=fake_result):
            response = self.client.post("/api/analyze/cv", files=files)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "cv_rule_v1")
        self.assertIn("risk_score", body)
        self.assertIn("action_metrics", body)

    def test_analyze_returns_llm_contract(self):
        fake_result = {
            "report_id": "LLM-20260415-1002",
            "source": "llm_rule_v1",
            "risk_level": "medium",
            "confidence": 68,
            "completeness": "complete",
            "missing_required_actions": [],
            "actions": [],
            "key_findings": [],
            "recommendations": [],
            "disclaimer": "For early screening support only. Not a clinical diagnosis.",
            "analysis_valid": True,
            "invalid_reason": None,
        }
        files = [("files", ("sample.jpg", b"img", "image/jpeg"))]
        with patch("app.api.inference.run_screening_pipeline", new=AsyncMock(return_value=fake_result)):
            response = self.client.post("/api/analyze", files=files)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "llm_rule_v1")
        self.assertIn("risk_level", body)
        self.assertIn("confidence", body)


if __name__ == "__main__":
    unittest.main()
