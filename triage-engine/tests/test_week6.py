"""
tests/test_week6.py
====================
Tests for the feedback store and retraining pipeline.

RUN:
    pytest tests/test_week6.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import tempfile

from src.data.feedback_store import FeedbackStore, CORRECTION_COLUMNS


@pytest.fixture
def temp_store(tmp_path):
    """A FeedbackStore backed by a temporary file — deleted after each test."""
    return FeedbackStore(path=tmp_path / "test_corrections.csv")


class TestFeedbackStore:

    def test_store_creates_file_on_init(self, tmp_path):
        """FeedbackStore should create the CSV file with headers on init."""
        path = tmp_path / "corrections.csv"
        assert not path.exists()
        FeedbackStore(path=path)
        assert path.exists()

    def test_created_file_has_headers(self, tmp_path):
        """The created CSV should have the correct column headers."""
        path = tmp_path / "corrections.csv"
        FeedbackStore(path=path)
        df = pd.read_csv(path)
        for col in CORRECTION_COLUMNS:
            assert col in df.columns, f"Column '{col}' missing from corrections CSV"

    def test_save_correction_increments_count(self, temp_store):
        """Saving a correction should increase the count by 1."""
        assert temp_store.count() == 0
        temp_store.save_correction(
            ticket_id="T001",
            body="I was charged twice this month",
            subject="Billing issue",
            original_category="account_access",
            original_confidence=0.61,
            corrected_category="billing",
            agent_id="agent_test",
        )
        assert temp_store.count() == 1

    def test_save_multiple_corrections(self, temp_store):
        """Should correctly count multiple saved corrections."""
        for i in range(5):
            temp_store.save_correction(
                ticket_id=f"T{i:03d}",
                body=f"Ticket body number {i} with enough content",
                subject=f"Subject {i}",
                original_category="general",
                original_confidence=0.55,
                corrected_category="billing",
                agent_id="agent_test",
            )
        assert temp_store.count() == 5

    def test_invalid_category_raises_error(self, temp_store):
        """Saving a correction with an invalid category should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid category"):
            temp_store.save_correction(
                ticket_id="T001",
                body="Some ticket body text here",
                subject="Subject",
                original_category="billing",
                original_confidence=0.7,
                corrected_category="invalid_xyz",  # Not a valid category
                agent_id="agent_test",
            )

    def test_load_corrections_returns_dataframe(self, temp_store):
        """load_corrections() should return a pandas DataFrame."""
        temp_store.save_correction(
            ticket_id="T001",
            body="Cannot login to my account at all",
            subject="Login issue",
            original_category="bug_report",
            original_confidence=0.60,
            corrected_category="account_access",
            agent_id="agent_test",
        )
        df = temp_store.load_corrections()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_load_empty_store_returns_empty_dataframe(self, temp_store):
        """Loading from empty store should return empty DataFrame, not error."""
        df = temp_store.load_corrections()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_saved_data_is_correct(self, temp_store):
        """Saved correction should have correct field values when loaded back."""
        temp_store.save_correction(
            ticket_id="T001",
            body="I was charged twice this month",
            subject="Billing problem",
            original_category="account_access",
            original_confidence=0.61,
            corrected_category="billing",
            agent_id="agent_sarah",
            correction_note="Clear billing issue",
        )
        df = temp_store.load_corrections()
        row = df.iloc[0]

        assert row["ticket_id"]          == "T001"
        assert row["corrected_category"] == "billing"
        assert row["original_category"]  == "account_access"
        assert row["agent_id"]           == "agent_sarah"
        assert row["correction_note"]    == "Clear billing issue"

    def test_get_as_training_data_format(self, temp_store):
        """get_as_training_data() should return DataFrame in training format."""
        temp_store.save_correction(
            ticket_id="T001",
            body="I was charged twice this month for my subscription",
            subject="Double charge",
            original_category="account_access",
            original_confidence=0.58,
            corrected_category="billing",
            agent_id="agent_test",
        )
        training_df = temp_store.get_as_training_data()
        assert "body"     in training_df.columns
        assert "category" in training_df.columns
        assert "source"   in training_df.columns
        assert training_df.iloc[0]["category"] == "billing"
        assert training_df.iloc[0]["source"]   == "agent_correction"

    def test_clear_resets_count_to_zero(self, temp_store):
        """clear() should remove all corrections."""
        temp_store.save_correction(
            ticket_id="T001",
            body="Some ticket body content here",
            subject="Subject",
            original_category="general",
            original_confidence=0.5,
            corrected_category="billing",
            agent_id="agent_test",
        )
        assert temp_store.count() == 1
        temp_store.clear()
        assert temp_store.count() == 0

    def test_corrections_persist_across_instances(self, tmp_path):
        """Corrections saved in one instance should be readable by another."""
        path = tmp_path / "shared_corrections.csv"

        # Save in first instance
        store1 = FeedbackStore(path=path)
        store1.save_correction(
            ticket_id="T001",
            body="Persistent ticket body content for testing",
            subject="Persistent test",
            original_category="general",
            original_confidence=0.55,
            corrected_category="billing",
            agent_id="agent_test",
        )

        # Read in second instance (simulates API restart)
        store2 = FeedbackStore(path=path)
        assert store2.count() == 1
        df = store2.load_corrections()
        assert df.iloc[0]["ticket_id"] == "T001"