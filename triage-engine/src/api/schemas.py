from datetime import datetime 
from typing import Dict, List, Optional
from pydantic import BaseModel, Field 

class TicketSubmitRequest(BaseModel):
    """
    Body of POST /tickets request.
    A client submits a ticket for triage.
    """
    subject: str = Field(
        default="",
        description="Ticket subject line",
        example="App crashes on file upload",
    )
    body: str = Field(
        ...,                        # ... means required — no default
        min_length=5,               # Reject obviously empty tickets
        description="Full ticket body text",
        example="Every time I try to upload a PDF the app crashes immediately.",
    )
    source: str = Field(
        default="api",
        description="Where the ticket came from",
        example="web_form",
    )

class OverrideRequest(BaseModel):
    """
    Body of POST /tickets/{id}/override request.
    An agent corrects the model's routing decision.
    """
    corrected_category: str = Field(
        ...,
        description="The correct category the agent assigns",
        example="billing",
    )
    corrected_urgency: str = Field(
        default="normal",
        description="The correct urgency level",
        example="high",
    )
    agent_id: str = Field(
        ...,
        description="ID of the agent making the correction",
        example="agent_sarah_k",
    )
    correction_note: Optional[str] = Field(
        default=None,
        description="Optional explanation of why the model was wrong",
        example="Ticket mentions billing issue despite login keywords",
    )

# Response Models

class TokenContributionResponse(BaseModel):
    """One token's SHAP contribution in an API response."""
    token:      str
    shap_value: float


class ExplanationResponse(BaseModel):
    """SHAP explanation included in every prediction response."""
    top_positive:     List[TokenContributionResponse]
    top_negative:     List[TokenContributionResponse]
    base_value:       float
    explanation_text: str


class TicketPredictionResponse(BaseModel):
    """
    Response from POST /tickets.
    Contains the routing decision plus SHAP explanation.
    """
    ticket_id:          str
    predicted_category: str
    predicted_urgency:  str
    confidence:         float
    auto_routed:        bool
    routed_to:          str
    explanation:        ExplanationResponse
    processing_time_ms: float
    model_version:      str
    created_at:         datetime


class TicketSummary(BaseModel):
    """
    Brief ticket info for list responses (GET /tickets).
    Does not include full explanation to keep response small.
    """
    ticket_id:          str
    subject:            str
    predicted_category: str
    predicted_urgency:  str
    confidence:         float
    auto_routed:        bool
    routed_to:          str
    source:             str
    created_at:         datetime


class TicketDetail(TicketSummary):
    """
    Full ticket info for GET /tickets/{id}.
    Includes the body text and full explanation.
    """
    body:        str
    explanation: ExplanationResponse


class MetricsResponse(BaseModel):
    """
    Response from GET /metrics.
    Shows system-wide statistics for the dashboard.
    """
    total_tickets:        int
    auto_routed_count:    int
    auto_route_rate:      float
    avg_confidence:       float
    override_count:       int
    override_rate:        float
    tickets_by_category:  Dict[str, int]
    tickets_by_urgency:   Dict[str, int]
    model_version:        str


class HealthResponse(BaseModel):
    """Response from GET /health."""
    status:        str    # "healthy" or "degraded"
    model_loaded:  bool
    explainer_loaded: bool
    uptime_seconds: float
    version:       str


class OverrideResponse(BaseModel):
    """Response from POST /tickets/{id}/override."""
    ticket_id:          str
    original_category:  str
    corrected_category: str
    agent_id:           str
    message:            str