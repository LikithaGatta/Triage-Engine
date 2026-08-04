"""
tests/test_week3.py
====================
Unit tests for the SHAP explainability layer.

RUN TESTS:
    pytest tests/test_week3.py -v

PREREQUISITE:
    Run scripts/train_baseline.py and scripts/train_explainer.py first.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import joblib
import numpy as np
import pandas as pd

from src.models.explainer import TicketExplainer, TokenContribution, ExplanationResult


@pytest.fixture(scope="module")
def trained_pipeline():
    """Load the trained TF-IDF pipeline. Skip if not available."""
    path = Path("models/baseline_v1.0.0.joblib")
    if not path.exists():
        pytest.skip("No trained model. Run train_baseline.py first.")
    return joblib.load(path)


@pytest.fixture(scope="module")
def fitted_explainer(trained_pipeline):
    """Create and fit a TicketExplainer on sample data."""
    category_names = list(trained_pipeline.classes_)

    explainer = TicketExplainer(
        pipeline=trained_pipeline,
        category_names=category_names,
        n_top_tokens=5,
    )

    # Use minimal background for fast tests
    background_texts = [
        "charged twice refund subscription",
        "cannot login password reset",
        "app crashes error broken",
        "feature request dark mode",
        "slow loading performance timeout",
        "general question how pricing",
    ]
    explainer.fit(background_texts, n_background=6)
    return explainer

class TestTokenContribution:

    def test_token_contribution_creation(self):
        """TokenContribution should be creatable with all fields."""
        tc = TokenContribution(token="refund", shap_value=0.35, direction="positive")
        assert tc.token == "refund"
        assert tc.shap_value == 0.35
        assert tc.direction == "positive"

    def test_negative_direction(self):
        """Negative shap_value should have direction 'negative'."""
        tc = TokenContribution(token="subscription", shap_value=-0.12, direction="negative")
        assert tc.direction == "negative"
        assert tc.shap_value < 0


class TestExplanationResult:

    def test_to_dict_has_required_keys(self):
        """to_dict() must return all required keys for API serialization."""
        result = ExplanationResult(
            ticket_id="T001",
            predicted_category="billing",
            confidence=0.91,
            base_value=0.17,
            top_positive=[TokenContribution("refund", 0.35, "positive")],
            top_negative=[TokenContribution("login", -0.08, "negative")],
            all_contributions=[],
            explanation_text="Classified as billing (91% confidence).",
        )
        d = result.to_dict()
        assert "ticket_id"          in d
        assert "predicted_category" in d
        assert "confidence"         in d
        assert "base_value"         in d
        assert "top_positive"       in d
        assert "top_negative"       in d
        assert "explanation_text"   in d

    def test_to_dict_rounds_floats(self):
        """Floats should be rounded to 4 decimal places for clean JSON."""
        result = ExplanationResult(
            ticket_id="T001",
            predicted_category="billing",
            confidence=0.912345678,
            base_value=0.166666666,
            top_positive=[],
            top_negative=[],
            all_contributions=[],
            explanation_text="Test.",
        )
        d = result.to_dict()
        assert d["confidence"] == 0.9123
        assert d["base_value"] == 0.1667


class TestTicketExplainer:

    def test_explainer_requires_fit_before_explain(self, trained_pipeline):
        """Calling explain() before fit() should raise RuntimeError."""
        explainer = TicketExplainer(
            pipeline=trained_pipeline,
            category_names=list(trained_pipeline.classes_),
        )
        with pytest.raises(RuntimeError, match="Call .fit()"):
            explainer.explain("some ticket text")

    def test_explain_returns_explanation_result(self, fitted_explainer):
        """explain() should return an ExplanationResult object."""
        result = fitted_explainer.explain("charged twice refund billing")
        assert isinstance(result, ExplanationResult)

    def test_explanation_has_predicted_category(self, fitted_explainer):
        """ExplanationResult must have a valid predicted_category."""
        result = fitted_explainer.explain("charged twice refund billing")
        valid_categories = ["billing", "bug_report", "feature_request",
                           "account_access", "performance", "general"]
        assert result.predicted_category in valid_categories

    def test_confidence_is_valid_probability(self, fitted_explainer):
        """Confidence must be between 0 and 1."""
        result = fitted_explainer.explain("app crashes error server broken")
        assert 0.0 <= result.confidence <= 1.0

    def test_top_positive_tokens_have_positive_shap(self, fitted_explainer):
        """All tokens in top_positive must have positive SHAP values."""
        result = fitted_explainer.explain("charged twice refund subscription billing")
        for token in result.top_positive:
            assert token.shap_value > 0, (
                f"Token '{token.token}' in top_positive has negative value {token.shap_value}"
            )

    def test_top_negative_tokens_have_negative_shap(self, fitted_explainer):
        """All tokens in top_negative must have negative SHAP values."""
        result = fitted_explainer.explain("charged twice refund subscription billing")
        for token in result.top_negative:
            assert token.shap_value < 0, (
                f"Token '{token.token}' in top_negative has positive value {token.shap_value}"
            )

    def test_n_top_tokens_respected(self, trained_pipeline):
        """Should not return more than n_top_tokens in top_positive or top_negative."""
        explainer = TicketExplainer(
            pipeline=trained_pipeline,
            category_names=list(trained_pipeline.classes_),
            n_top_tokens=3,  # Explicitly set to 3
        )
        explainer.fit(["charged refund billing subscription payment"], n_background=1)
        result = explainer.explain("charged twice refund billing subscription payment invoice")
        assert len(result.top_positive) <= 3
        assert len(result.top_negative) <= 3

    def test_explanation_text_is_nonempty_string(self, fitted_explainer):
        """explanation_text should always be a non-empty string."""
        result = fitted_explainer.explain("cannot login password account locked")
        assert isinstance(result.explanation_text, str)
        assert len(result.explanation_text) > 10

    def test_base_value_is_valid(self, fitted_explainer):
        """Base value should be a probability (between 0 and 1)."""
        result = fitted_explainer.explain("feature request dark mode export")
        assert 0.0 <= result.base_value <= 1.0

    def test_explain_batch_returns_correct_count(self, fitted_explainer):
        """explain_batch should return one result per input text."""
        texts = [
            "charged twice refund billing",
            "app crashes error broken",
            "dark mode feature request",
        ]
        results = fitted_explainer.explain_batch(texts)
        assert len(results) == 3

    def test_explain_batch_uses_provided_ids(self, fitted_explainer):
        """Ticket IDs should appear in the ExplanationResult objects."""
        texts = ["charged refund billing", "cannot login account"]
        ids   = ["T-001", "T-002"]
        results = fitted_explainer.explain_batch(texts, ticket_ids=ids)
        assert results[0].ticket_id == "T-001"
        assert results[1].ticket_id == "T-002"

    def test_get_category_top_features_returns_all_categories(self, fitted_explainer):
        """get_category_top_features should return a key for every category."""
        features = fitted_explainer.get_category_top_features(n=5)
        for cat in fitted_explainer.category_names:
            assert cat in features, f"Category '{cat}' missing from global features"

    def test_global_features_count(self, fitted_explainer):
        """Each category should have exactly n features returned."""
        n = 5
        features = fitted_explainer.get_category_top_features(n=n)
        for cat, feat_list in features.items():
            assert len(feat_list) <= n, (
                f"Category '{cat}' returned {len(feat_list)} features, expected <= {n}"
            )

    def test_save_and_load(self, fitted_explainer, tmp_path):
        """Saved explainer should produce identical results after loading."""
        save_path = str(tmp_path / "test_explainer.joblib")
        fitted_explainer.save(save_path)

        # Create new explainer and load saved state
        import joblib
        new_explainer = TicketExplainer(
            pipeline=fitted_explainer.pipeline,
            category_names=fitted_explainer.category_names,
        )
        new_explainer.load(save_path)

        assert new_explainer.is_fitted
        assert new_explainer.category_names == fitted_explainer.category_names

        # Both should produce the same prediction on the same input
        text    = "charged twice refund billing subscription"
        result1 = fitted_explainer.explain(text)
        result2 = new_explainer.explain(text)
        assert result1.predicted_category == result2.predicted_category