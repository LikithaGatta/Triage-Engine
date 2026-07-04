
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator



class Category(str, Enum):
    """The 6 categories our classifier predicts."""
    BILLING          = "billing"
    BUG_REPORT       = "bug_report"
    FEATURE_REQUEST  = "feature_request"
    ACCOUNT_ACCESS   = "account_access"
    PERFORMANCE      = "performance"
    GENERAL          = "general"


class UrgencyLevel(str, Enum):
    """3-tier urgency system."""
    CRITICAL = "critical"   # Service down — respond immediately
    HIGH     = "high"       # Significant problem — respond same day
    NORMAL   = "normal"     # Standard request — respond within SLA


# ----------------------------------------------------------------
# DATA MODELS
# ----------------------------------------------------------------

class RawTicket(BaseModel):
    """A ticket exactly as it arrives """
    ticket_id: str
    subject:   str = ""
    body:      str
    source:    str = "unknown"
    created_at: datetime = Field(default_factory=datetime.now)

    # Validator: converts ticket_id to string even if an int is passed
    @field_validator("ticket_id", mode="before")
    @classmethod
    def coerce_id_to_str(cls, v):
        return str(v)


class ProcessedTicket(BaseModel):
    """A ticket after preprocessing"""
    ticket_id:      str
    raw_subject:    str
    raw_body:       str
    text:           str      # The cleaned combined text fed to TF-IDF or DistilBERT
    char_count:     int
    word_count:     int
    source:         str = "unknown"
    created_at:     datetime = Field(default_factory=datetime.now)


class PredictionResult(BaseModel):
    """Output of the classifier for one ticket."""
    ticket_id:            str
    predicted_category:   Category
    predicted_urgency:    UrgencyLevel
    confidence:           float        # 0.0 to 1.0
    all_probabilities:    dict         # {"billing": 0.91, "bug_report": 0.05, ...}
    auto_routed:          bool         # True if confidence >= threshold
    routed_to:            str          # Team name or "human_review"
    model_version:        str = "1.0.0"
    predicted_at:         datetime = Field(default_factory=datetime.now)