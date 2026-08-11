"""
src/api/routes.py
==================
All API endpoint definitions.

ENDPOINT SUMMARY:
  POST /tickets              Submit ticket → prediction + SHAP explanation
  GET  /tickets              List tickets (with optional filters)
  GET  /tickets/{ticket_id}  Get one ticket's full details
  POST /tickets/{id}/override  Agent corrects a routing decision
  GET  /metrics              System-wide statistics
  GET  /health               Is the server alive?

ROUTING MAP (what team handles each category):
  billing         → Billing Team
  bug_report      → Engineering Team
  feature_request → Product Team
  account_access  → Account Support Team
  performance     → Engineering Team
  general         → General Support Queue
"""

import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import ModelContainer, InMemoryStore, get_models, get_store
from src.api.schemas import (
    ExplanationResponse,
    HealthResponse,
    MetricsResponse,
    OverrideRequest,
    OverrideResponse,
    TicketDetail,
    TicketPredictionResponse,
    TicketSubmitRequest,
    TicketSummary,
    TokenContributionResponse,
)
from src.utils.logger import logger
from src.utils.schemas import Category, UrgencyLevel

# ---- API version prefix ----
# All routes will be /api/v1/tickets, /api/v1/metrics, etc.
# Versioning lets you ship breaking changes as /api/v2/ without
# breaking existing clients still using /api/v1/
router = APIRouter(prefix="/api/v1")

# ---- Routing rules ----
# Maps category → team name shown in the UI
CATEGORY_TO_TEAM = {
    "billing":          "Billing Team",
    "bug_report":       "Engineering Team",
    "feature_request":  "Product Team",
    "account_access":   "Account Support Team",
    "performance":      "Engineering Team",
    "general":          "General Support Queue",
}

# ---- Urgency detection ----
# Keywords that upgrade a ticket's urgency level
CRITICAL_KEYWORDS = {
    "down", "outage", "data loss", "security breach",
    "cannot access", "locked out", "production", "urgent",
}
HIGH_KEYWORDS = {
    "crash", "broken", "not working", "refund", "charged",
    "cannot login", "blocked", "failed",
}


def detect_urgency(text: str, predicted_category: str) -> str:
    """
    Determine urgency level from ticket text and category.

    Combines rule-based keyword detection with category-based defaults.
    Critical and high keywords override the default.

    Args:
        text:               Raw ticket body text
        predicted_category: The category predicted by the ML model
    Returns:
        Urgency string: "critical", "high", or "normal"
    """
    text_lower = text.lower()

    # Check for critical keywords first (highest priority)
    if any(kw in text_lower for kw in CRITICAL_KEYWORDS):
        return UrgencyLevel.CRITICAL.value

    # Check for high keywords
    if any(kw in text_lower for kw in HIGH_KEYWORDS):
        return UrgencyLevel.HIGH.value

    # Category-based defaults for remaining tickets
    category_urgency_map = {
        "bug_report":     UrgencyLevel.HIGH.value,
        "account_access": UrgencyLevel.HIGH.value,
        "billing":        UrgencyLevel.NORMAL.value,
        "feature_request":UrgencyLevel.NORMAL.value,
        "performance":    UrgencyLevel.NORMAL.value,
        "general":        UrgencyLevel.NORMAL.value,
    }
    return category_urgency_map.get(predicted_category, UrgencyLevel.NORMAL.value)


# ================================================================
# POST /api/v1/tickets
# Submit a ticket for triage
# ================================================================

@router.post(
    "/tickets",
    response_model=TicketPredictionResponse,
    status_code=201,                    # 201 = Created (not 200 = OK)
    summary="Submit a ticket for triage",
    description="Submit a support ticket. Returns prediction, confidence, routing decision, and SHAP explanation.",
)
def submit_ticket(
    request: TicketSubmitRequest,
    models: ModelContainer = Depends(get_models),
    store:  InMemoryStore  = Depends(get_store),
) -> TicketPredictionResponse:
    """
    Main endpoint — the core of the entire system.

    Flow:
      1. Preprocess the raw ticket text
      2. Get prediction + confidence from TF-IDF classifier
      3. Get SHAP explanation for the prediction
      4. Detect urgency from keywords + category
      5. Route to team based on category + confidence threshold
      6. Store ticket in memory
      7. Return full response with prediction + explanation

    Args:
        request: Validated TicketSubmitRequest (FastAPI handles validation)
        models:  Injected ModelContainer with loaded models
        store:   Injected InMemoryStore for persistence
    Returns:
        TicketPredictionResponse with all routing information
    """
    start_time = time.time()

    # ---- Step 1: Preprocess ----
    # Clean the text the same way as training data
    combined_raw = f"{request.subject} {request.body}".strip()
    cleaned      = models.preprocessor.clean_text(combined_raw)
    tokens       = models.preprocessor.tokenize_and_filter(cleaned)
    processed    = " ".join(tokens)

    # ---- Step 2: Predict ----
    # Get probabilities for all 6 categories
    proba             = models.pipeline.predict_proba([processed])[0]
    classes           = list(models.pipeline.classes_)
    predicted_idx     = proba.argmax()
    predicted_category = classes[predicted_idx]
    confidence        = float(proba[predicted_idx])

    # ---- Step 3: SHAP Explanation ----
    explanation = models.explainer.explain(
        text=processed,
        ticket_id="pending",          # ID not assigned yet
        predicted_category=predicted_category,
        confidence=confidence,
    )

    # ---- Step 4: Urgency detection ----
    urgency = detect_urgency(combined_raw, predicted_category)

    # ---- Step 5: Routing decision ----
    # Tickets above confidence threshold route automatically
    # Below threshold go to human review queue
    confidence_threshold = 0.75
    auto_routed          = confidence >= confidence_threshold
    routed_to            = (
        CATEGORY_TO_TEAM.get(predicted_category, "General Support Queue")
        if auto_routed
        else "Human Review Queue"
    )

    # ---- Step 6: Store ----
    ticket_data = {
        "subject":            request.subject,
        "body":               request.body,
        "source":             request.source,
        "predicted_category": predicted_category,
        "predicted_urgency":  urgency,
        "confidence":         confidence,
        "auto_routed":        auto_routed,
        "routed_to":          routed_to,
        "explanation":        explanation.to_dict(),
        "model_version":      models.version,
        "created_at":         datetime.now().isoformat(),
    }
    ticket_id = store.save_ticket(ticket_data)

    # Calculate processing time in milliseconds
    processing_ms = (time.time() - start_time) * 1000

    logger.info(
        f"Ticket {ticket_id}: {predicted_category} "
        f"({confidence:.0%} conf) → {routed_to} "
        f"[{processing_ms:.1f}ms]"
    )

    # ---- Step 7: Build and return response ----
    return TicketPredictionResponse(
        ticket_id=ticket_id,
        predicted_category=predicted_category,
        predicted_urgency=urgency,
        confidence=confidence,
        auto_routed=auto_routed,
        routed_to=routed_to,
        explanation=ExplanationResponse(
            top_positive=[
                TokenContributionResponse(token=t.token, shap_value=t.shap_value)
                for t in explanation.top_positive
            ],
            top_negative=[
                TokenContributionResponse(token=t.token, shap_value=t.shap_value)
                for t in explanation.top_negative
            ],
            base_value=explanation.base_value,
            explanation_text=explanation.explanation_text,
        ),
        processing_time_ms=round(processing_ms, 2),
        model_version=models.version,
        created_at=datetime.now(),
    )


# ================================================================
# GET /api/v1/tickets
# List all tickets
# ================================================================

@router.get(
    "/tickets",
    response_model=list[TicketSummary],
    summary="List all tickets",
)
def list_tickets(
    category: Optional[str] = Query(
        default=None,
        description="Filter by category (e.g. billing, bug_report)",
    ),
    limit:  int = Query(default=50,  ge=1,  le=200),
    offset: int = Query(default=0,   ge=0),
    store:  InMemoryStore = Depends(get_store),
) -> list:
    """
    List tickets with optional filtering and pagination.

    Query parameters:
      ?category=billing  → only billing tickets
      ?limit=20          → return 20 at a time
      ?offset=20         → skip first 20 (for pagination)

    Pagination works by combining limit and offset:
      Page 1: limit=20, offset=0
      Page 2: limit=20, offset=20
      Page 3: limit=20, offset=40
    """
    tickets = store.list_tickets(category=category, limit=limit, offset=offset)

    return [
        TicketSummary(
            ticket_id=t["ticket_id"],
            subject=t.get("subject", ""),
            predicted_category=t["predicted_category"],
            predicted_urgency=t["predicted_urgency"],
            confidence=t["confidence"],
            auto_routed=t["auto_routed"],
            routed_to=t["routed_to"],
            source=t.get("source", "api"),
            created_at=datetime.fromisoformat(t["created_at"]),
        )
        for t in tickets
    ]


# ================================================================
# GET /api/v1/tickets/{ticket_id}
# Get one ticket's full details
# ================================================================

@router.get(
    "/tickets/{ticket_id}",
    response_model=TicketDetail,
    summary="Get ticket details",
)
def get_ticket(
    ticket_id: str,
    store: InMemoryStore = Depends(get_store),
) -> TicketDetail:
    """
    Get the full details of one ticket including body text
    and SHAP explanation.

    Returns 404 if ticket_id does not exist.
    """
    ticket = store.get_ticket(ticket_id)

    if ticket is None:
        # 404 = Not Found — the standard HTTP status for missing resources
        raise HTTPException(
            status_code=404,
            detail=f"Ticket {ticket_id} not found",
        )

    exp = ticket["explanation"]

    return TicketDetail(
        ticket_id=ticket["ticket_id"],
        subject=ticket.get("subject", ""),
        body=ticket.get("body", ""),
        predicted_category=ticket["predicted_category"],
        predicted_urgency=ticket["predicted_urgency"],
        confidence=ticket["confidence"],
        auto_routed=ticket["auto_routed"],
        routed_to=ticket["routed_to"],
        source=ticket.get("source", "api"),
        created_at=datetime.fromisoformat(ticket["created_at"]),
        explanation=ExplanationResponse(
            top_positive=[
                TokenContributionResponse(**t) for t in exp.get("top_positive", [])
            ],
            top_negative=[
                TokenContributionResponse(**t) for t in exp.get("top_negative", [])
            ],
            base_value=exp.get("base_value", 0.0),
            explanation_text=exp.get("explanation_text", ""),
        ),
    )


# ================================================================
# POST /api/v1/tickets/{ticket_id}/override
# Agent corrects a wrong routing decision
# ================================================================

@router.post(
    "/tickets/{ticket_id}/override",
    response_model=OverrideResponse,
    summary="Override a routing decision",
)
def override_ticket(
    ticket_id: str,
    request:   OverrideRequest,
    store:     InMemoryStore = Depends(get_store),
) -> OverrideResponse:
    """
    Record an agent's correction to the model's prediction.

    This is the feedback loop — every correction is stored and
    will feed into the Week 6 retraining pipeline.

    The correction does NOT change the stored ticket's category.
    It records the disagreement separately so we can:
      1. Show the agent's correction in the UI
      2. Use it as training data for model improvement
      3. Track which categories the model gets wrong most often
    """
    ticket = store.get_ticket(ticket_id)

    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    # Validate the corrected category is one of our 6
    valid_categories = {c.value for c in Category}
    if request.corrected_category not in valid_categories:
        raise HTTPException(
            status_code=422,    # 422 = Unprocessable Entity
            detail=f"Invalid category '{request.corrected_category}'. "
                   f"Must be one of: {sorted(valid_categories)}",
        )

    # Store the correction
    override_data = {
        "ticket_id":           ticket_id,
        "original_category":   ticket["predicted_category"],
        "original_confidence": ticket["confidence"],
        "corrected_category":  request.corrected_category,
        "corrected_urgency":   request.corrected_urgency,
        "agent_id":            request.agent_id,
        "correction_note":     request.correction_note,
        "corrected_at":        datetime.now().isoformat(),
    }
    store.save_override(override_data)

    logger.info(
        f"Override: ticket {ticket_id} "
        f"{ticket['predicted_category']} → {request.corrected_category} "
        f"by {request.agent_id}"
    )

    return OverrideResponse(
        ticket_id=ticket_id,
        original_category=ticket["predicted_category"],
        corrected_category=request.corrected_category,
        agent_id=request.agent_id,
        message=(
            f"Correction recorded. Ticket {ticket_id} was predicted as "
            f"{ticket['predicted_category']} and corrected to "
            f"{request.corrected_category}. "
            f"This will feed into the next model retraining."
        ),
    )


# ================================================================
# GET /api/v1/metrics
# System-wide statistics for the dashboard
# ================================================================

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get system metrics",
)
def get_metrics(
    models: ModelContainer = Depends(get_models),
    store:  InMemoryStore  = Depends(get_store),
) -> MetricsResponse:
    """
    Returns aggregate statistics across all tickets.

    Powers the analytics dashboard — shows queue volumes,
    auto-route rate, average confidence, and override rate.
    """
    data = store.get_metrics()

    return MetricsResponse(
        total_tickets=data["total_tickets"],
        auto_routed_count=data["auto_routed_count"],
        auto_route_rate=round(data["auto_route_rate"], 4),
        avg_confidence=round(data["avg_confidence"], 4),
        override_count=data["override_count"],
        override_rate=round(data["override_rate"], 4),
        tickets_by_category=data["tickets_by_category"],
        tickets_by_urgency=data["tickets_by_urgency"],
        model_version=models.version,
    )


# ================================================================
# GET /api/v1/health
# Health check endpoint
# ================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health_check(
    models: ModelContainer = Depends(get_models),
) -> HealthResponse:
    """
    Returns the health status of the API.

    Used by:
      - Load balancers (is this instance alive?)
      - Monitoring systems (is the model loaded?)
      - CI/CD pipelines (did the deployment succeed?)

    Returns 200 if healthy, 503 if models not loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=models.pipeline is not None,
        explainer_loaded=models.explainer is not None,
        uptime_seconds=round(models.uptime_seconds, 1),
        version=models.version,
    )