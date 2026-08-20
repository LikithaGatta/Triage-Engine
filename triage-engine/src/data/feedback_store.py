"""
src/data/feedback_store.py
===========================
Manages the collection and retrieval of agent corrections.

In Week 4 we stored corrections in memory (lost on restart).
Week 6 persists them to a CSV file so they survive restarts
and accumulate over time into a proper training signal.

WHY CSV AND NOT POSTGRESQL?
  PostgreSQL is the production choice (Week 4's setup_database.py
  creates the schema). For the portfolio, CSV is fine — it is
  readable, version-controllable, and requires zero infrastructure.
  In an interview you explain: "In production this would be
  PostgreSQL with the agent_corrections table from our schema."

WHAT GETS STORED:
  Every correction has:
    - The original ticket text (the training input)
    - The corrected category (the training label)
    - The model's original prediction (to measure improvement)
    - Metadata: agent_id, timestamp, confidence of original prediction
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.utils.logger import logger
from src.utils.schemas import Category, UrgencyLevel


# Where corrections are persisted
FEEDBACK_PATH = Path("data/processed/agent_corrections.csv")

# Columns in the corrections CSV
CORRECTION_COLUMNS = [
    "ticket_id",
    "body",
    "subject",
    "original_category",
    "original_confidence",
    "corrected_category",
    "corrected_urgency",
    "agent_id",
    "correction_note",
    "corrected_at",
    "source",
]


class FeedbackStore:
    """
    Persists agent corrections to disk and loads them for retraining.

    Usage:
        store = FeedbackStore()
        store.save_correction(...)
        df = store.load_corrections()
        print(f"Have {len(df)} corrections ready for retraining")
    """

    def __init__(self, path: Path = FEEDBACK_PATH):
        self.path = path
        # Create the file with headers if it does not exist
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CORRECTION_COLUMNS)
                writer.writeheader()
            logger.info(f"Created feedback store at {self.path}")

    def save_correction(
        self,
        ticket_id: str,
        body: str,
        subject: str,
        original_category: str,
        original_confidence: float,
        corrected_category: str,
        corrected_urgency: str = "normal",
        agent_id: str = "unknown",
        correction_note: Optional[str] = None,
    ) -> None:
        """
        Append one agent correction to the CSV store.

        Each correction becomes a training example — the body is the
        input text and corrected_category is the label.

        Args:
            ticket_id:           Which ticket was corrected
            body:                The ticket's full text (training input)
            subject:             The ticket's subject line
            original_category:   What the model predicted (wrong)
            original_confidence: How confident the model was when it was wrong
            corrected_category:  What the agent says the correct category is
            corrected_urgency:   Agent's urgency assessment
            agent_id:            Which agent made the correction
            correction_note:     Optional explanation from the agent
        """
        # Validate the corrected category is one of our 6
        valid_categories = {c.value for c in Category}
        if corrected_category not in valid_categories:
            raise ValueError(
                f"Invalid category '{corrected_category}'. "
                f"Must be one of: {valid_categories}"
            )

        row = {
            "ticket_id":           ticket_id,
            "body":                body,
            "subject":             subject,
            "original_category":   original_category,
            "original_confidence": round(original_confidence, 4),
            "corrected_category":  corrected_category,
            "corrected_urgency":   corrected_urgency,
            "agent_id":            agent_id,
            "correction_note":     correction_note or "",
            "corrected_at":        datetime.now().isoformat(),
            "source":              "agent_correction",
        }

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CORRECTION_COLUMNS)
            writer.writerow(row)

        logger.info(
            f"Correction saved: ticket {ticket_id} "
            f"{original_category} → {corrected_category} "
            f"by {agent_id}"
        )

    def load_corrections(self) -> pd.DataFrame:
        """
        Load all saved corrections as a DataFrame.

        Returns:
            DataFrame with all corrections, ready to merge with
            training data. Returns empty DataFrame if no corrections yet.
        """
        df = pd.read_csv(self.path)

        if len(df) == 0:
            logger.info("No corrections found in feedback store")
            return df

        logger.info(f"Loaded {len(df)} corrections from {self.path}")
        logger.info(
            f"Category distribution of corrections:\n"
            f"{df['corrected_category'].value_counts().to_string()}"
        )

        # Show which predictions were most often wrong
        wrong_predictions = df.groupby(
            ["original_category", "corrected_category"]
        ).size().reset_index(name="count")
        wrong_predictions = wrong_predictions.sort_values("count", ascending=False)
        logger.info(
            f"Most common corrections:\n"
            f"{wrong_predictions.head(5).to_string(index=False)}"
        )

        return df

    def count(self) -> int:
        """Return the total number of corrections stored."""
        df = pd.read_csv(self.path)
        return len(df)

    def clear(self) -> None:
        """
        Clear all corrections. Used after a successful retraining
        so the next cycle starts fresh.

        WARNING: Only call this after the new model has been saved
        and verified. Clearing before that loses the training signal.
        """
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CORRECTION_COLUMNS)
            writer.writeheader()
        logger.info("Feedback store cleared")

    def get_as_training_data(self) -> pd.DataFrame:
        """
        Convert corrections into training data format.

        Combines subject + body into a single text field.
        Renames corrected_category to category (what the model trains on).
        Adds source column so we can track correction-derived examples.

        Returns:
            DataFrame in the same format as train.csv — ready to concat
            with the original training data for retraining.
        """
        df = self.load_corrections()

        if len(df) == 0:
            return pd.DataFrame(columns=["ticket_id", "subject", "body", "category", "urgency", "source"])

        training_df = pd.DataFrame({
            "ticket_id": df["ticket_id"],
            "subject":   df["subject"].fillna(""),
            "body":      df["body"].fillna(""),
            "category":  df["corrected_category"],   # The corrected label
            "urgency":   df["corrected_urgency"].fillna("normal"),
            "source":    "agent_correction",
        })

        return training_df