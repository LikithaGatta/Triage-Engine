"""
tests/test_week2.py
====================
Unit tests for the Week 2 DistilBERT classifier.

RUN TESTS:
    pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd

from src.models.distilbert_classifier import (
    DistilBertClassifier,
    CATEGORY_TO_ID,
    ID_TO_CATEGORY,
    NUM_LABELS,
)
from src.utils.schemas import Category



class TestCategoryMappings:
    """Verify the integer ↔ category string mappings are correct."""

    def test_all_categories_have_an_id(self):
        """Every category in our schema must have a corresponding integer ID."""
        for cat in Category:
            assert cat.value in CATEGORY_TO_ID, (
                f"Category '{cat.value}' missing from CATEGORY_TO_ID mapping"
            )

    def test_mapping_is_reversible(self):
        """Converting category → ID → category should give the original back."""
        for category_name, category_id in CATEGORY_TO_ID.items():
            assert ID_TO_CATEGORY[category_id] == category_name, (
                f"Round-trip failed: {category_name} → {category_id} → {ID_TO_CATEGORY[category_id]}"
            )

    def test_correct_number_of_labels(self):
        """We should have exactly 6 categories."""
        assert NUM_LABELS == 6
        assert len(CATEGORY_TO_ID) == 6
        assert len(ID_TO_CATEGORY) == 6

    def test_ids_are_sequential_from_zero(self):
        """IDs should be 0, 1, 2, 3, 4, 5 — no gaps."""
        ids = sorted(CATEGORY_TO_ID.values())
        assert ids == list(range(NUM_LABELS)), (
            f"IDs should be 0 to {NUM_LABELS-1}, got {ids}"
        )



class TestTokenization:
    """Test that text gets correctly converted to token IDs."""

    @pytest.fixture(scope="class")
    def clf(self):
        """Create classifier and load ONLY the tokenizer (fast)."""
        classifier = DistilBertClassifier()
        # Load just the tokenizer — much faster than loading the full model
        from transformers import AutoTokenizer
        from src.models.distilbert_classifier import MODEL_NAME
        classifier.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return classifier

    def test_tokenize_dataset_adds_required_columns(self, clf):
        """Tokenized dataset must have input_ids, attention_mask, label columns."""
        df = pd.DataFrame({
            "body":     ["I cannot login to my account"],
            "category": ["account_access"],
            "subject":  ["Login broken"],
        })
        dataset = clf._tokenize_dataset(df)
        column_names = dataset.column_names
        assert "input_ids"      in column_names
        assert "attention_mask" in column_names
        assert "label"          in column_names

    def test_label_mapping_is_correct(self, clf):
        """'billing' should map to integer 0."""
        df = pd.DataFrame({
            "body":     ["I was charged twice"],
            "category": ["billing"],
            "subject":  ["Billing issue"],
        })
        dataset = clf._tokenize_dataset(df)
        assert dataset[0]["label"] == CATEGORY_TO_ID["billing"]

    def test_input_ids_length_equals_max_length(self, clf):
        """All sequences should be padded/truncated to exactly max_length."""
        df = pd.DataFrame({
            "body":     ["Short ticket", "A much longer ticket with lots more words and detail"],
            "category": ["general", "billing"],
            "subject":  ["", ""],
        })
        dataset = clf._tokenize_dataset(df, max_length=64)
        for i in range(len(dataset)):
            assert len(dataset[i]["input_ids"]) == 64, (
                f"Expected 64 tokens, got {len(dataset[i]['input_ids'])}"
            )

    def test_long_text_gets_truncated(self, clf):
        """Text longer than max_length should be truncated, not raise an error."""
        very_long_text = "crash error broken " * 200   # Way over max_length
        df = pd.DataFrame({
            "body":     [very_long_text],
            "category": ["bug_report"],
            "subject":  [""],
        })
        # Should not raise any exception
        dataset = clf._tokenize_dataset(df, max_length=32)
        assert len(dataset[0]["input_ids"]) == 32



class TestPredictions:
    """
    Tests for the prediction output format.
    These are skipped if no trained model exists yet.
    After you run train_distilbert.py, these will all pass.
    """

    @pytest.fixture(scope="class")
    def trained_clf(self):
        """Load the trained model if it exists, skip tests if not."""
        model_path = Path("models/distilbert_finetuned")
        if not model_path.exists():
            pytest.skip("No trained DistilBERT model found. Run train_distilbert.py first.")
        clf = DistilBertClassifier()
        clf.load(str(model_path))
        return clf

    def test_predict_single_returns_required_keys(self, trained_clf):
        """Prediction dict must have all 4 required keys."""
        result = trained_clf.predict_single("I was charged twice this month")
        assert "predicted_category"  in result
        assert "confidence"          in result
        assert "all_probabilities"   in result
        assert "auto_routed"         in result

    def test_confidence_is_valid_probability(self, trained_clf):
        """Confidence must be between 0 and 1."""
        result = trained_clf.predict_single("The app keeps crashing when I upload files")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_probabilities_sum_to_one(self, trained_clf):
        """Softmax probabilities must sum to 1.0."""
        result = trained_clf.predict_single("Please add dark mode to the dashboard")
        total = sum(result["all_probabilities"].values())
        assert abs(total - 1.0) < 0.001, f"Probabilities sum to {total}, expected 1.0"

    def test_predicted_category_is_valid(self, trained_clf):
        """Predicted category must be one of our 6 defined categories."""
        valid_categories = {c.value for c in Category}
        result = trained_clf.predict_single("I cannot log into my account at all")
        assert result["predicted_category"] in valid_categories

    def test_auto_route_matches_confidence_threshold(self, trained_clf):
        """auto_routed should be True if and only if confidence >= 0.75."""
        result = trained_clf.predict_single("The search feature is loading very slowly")
        expected = result["confidence"] >= 0.75
        assert result["auto_routed"] == expected

    def test_billing_ticket_prediction(self, trained_clf):
        """A clear billing ticket should be classified as billing."""
        result = trained_clf.predict_single(
            "I was charged twice for my Pro subscription this month. "
            "Please refund the duplicate $49 charge immediately."
        )
        # We check the top prediction is billing with reasonable confidence
        assert result["predicted_category"] == "billing"
        assert result["confidence"] > 0.70

    def test_predict_batch_returns_correct_count(self, trained_clf):
        """predict_batch should return one result per input ticket."""
        tickets = [
            "I cannot login",
            "App is crashing",
            "Please add dark mode",
        ]
        results = trained_clf.predict_batch(tickets)
        assert len(results) == len(tickets)
        for result in results:
            assert "predicted_category" in result
            assert "confidence"         in result