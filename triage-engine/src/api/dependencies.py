"""
src/api/dependencies.py
========================
Dependency injection functions for FastAPI.

WHAT IS DEPENDENCY INJECTION?
  Instead of loading the ML model inside every endpoint function
  (slow — would reload 250MB on every request), we load models
  ONCE at startup and inject them wherever needed.

  FastAPI's Depends() system handles this automatically.
  Any endpoint that declares models=Depends(get_models) gets
  the already-loaded model container injected at call time.

WHY NOT GLOBAL VARIABLES?
  Global variables work but are hard to test. With dependency
  injection, tests can inject mock models instead of loading
  real ones — making tests fast and independent.

  This pattern is used in every production FastAPI codebase.
"""

import time
from pathlib import Path
from typing import Optional

import joblib
from fastapi import HTTPException

from src.data.preprocessor import TicketPreprocessor
from src.models.explainer import TicketExplainer
from src.utils.logger import logger
from src.utils.schemas import Category, UrgencyLevel


# ----------------------------------------------------------------
# MODEL CONTAINER
# A simple class that holds all loaded models.
# Loaded once at startup, reused for every request.
# ----------------------------------------------------------------

class ModelContainer:
    """
    Holds all loaded ML models and related objects.

    Attributes:
        pipeline:     The trained sklearn Pipeline (TF-IDF + LogReg)
        explainer:    The fitted SHAP TicketExplainer
        preprocessor: The TicketPreprocessor for text cleaning
        is_ready:     True once all models loaded successfully
        load_time:    When models were loaded (for uptime calculation)
        version:      Model version string
    """

    def __init__(self):
        self.pipeline     = None
        self.explainer    = None
        self.preprocessor = None
        self.is_ready     = False
        self.load_time    = time.time()
        self.version      = "1.0.0"

    def load(self) -> None:
        """
        Load all models from disk.
        Called once at application startup.
        Raises an exception if any model file is missing.
        """
        model_path    = Path("models/baseline_v1.0.0.joblib")
        explainer_path = Path("models/shap_explainer.joblib")

        # ---- Load TF-IDF + Logistic Regression pipeline ----
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Run scripts/train_baseline.py first."
            )
        logger.info(f"Loading classifier from {model_path}...")
        self.pipeline = joblib.load(model_path)
        logger.info("Classifier loaded")

        # ---- Load SHAP explainer ----
        if not explainer_path.exists():
            raise FileNotFoundError(
                f"SHAP explainer not found at {explainer_path}. "
                "Run scripts/train_explainer.py first."
            )
        logger.info(f"Loading SHAP explainer from {explainer_path}...")

        # Load raw joblib data and reconstruct the explainer
        explainer_data = joblib.load(explainer_path)
        self.explainer = TicketExplainer(
            pipeline=self.pipeline,
            category_names=explainer_data["category_names"],
            n_top_tokens=5,
        )
        self.explainer.shap_explainer  = explainer_data["shap_explainer"]
        self.explainer.background_mean = explainer_data["background_mean"]
        self.explainer.feature_names   = explainer_data["feature_names"]
        self.explainer.is_fitted       = True
        logger.info("SHAP explainer loaded")

        # ---- Initialize preprocessor ----
        self.preprocessor = TicketPreprocessor()

        self.is_ready = True
        logger.info("All models loaded and ready")

    @property
    def uptime_seconds(self) -> float:
        """How long since models were loaded."""
        return time.time() - self.load_time


# ---- Singleton instance ----
# Created once when the module loads.
# FastAPI startup event calls .load() to populate it.
_model_container = ModelContainer()


def get_models() -> ModelContainer:
    """
    FastAPI dependency — returns the loaded model container.

    Usage in endpoints:
        @app.post("/tickets")
        def predict(models: ModelContainer = Depends(get_models)):
            result = models.pipeline.predict(...)

    Raises HTTPException if models are not loaded yet.
    """
    if not _model_container.is_ready:
        raise HTTPException(
            status_code=503,    # 503 = Service Unavailable
            detail="Models are still loading. Please retry in a moment.",
        )
    return _model_container


def load_models_on_startup() -> None:
    """
    Called by FastAPI's startup event handler.
    Loads all models into the singleton container.
    """
    logger.info("Loading models on startup...")
    _model_container.load()


# ----------------------------------------------------------------
# IN-MEMORY STORAGE
# For Week 4 we store tickets in memory (a Python list).
# Week 4 ends with PostgreSQL integration — for now this lets
# us build and test the full API without needing a database.
# ----------------------------------------------------------------

class InMemoryStore:
    """
    Simple in-memory ticket store.
    Stores tickets as dicts in a list.
    Replaced by PostgreSQL in a production system.
    """

    def __init__(self):
        self.tickets    = []   # List of ticket dicts
        self.overrides  = []   # List of correction dicts
        self._id_counter = 1

    def next_id(self) -> str:
        """Generate a unique ticket ID."""
        tid = f"TKT-{self._id_counter:06d}"
        self._id_counter += 1
        return tid

    def save_ticket(self, ticket_data: dict) -> str:
        """Save a ticket and return its ID."""
        tid = self.next_id()
        ticket_data["ticket_id"] = tid
        self.tickets.append(ticket_data)
        return tid

    def get_ticket(self, ticket_id: str) -> Optional[dict]:
        """Find a ticket by ID. Returns None if not found."""
        for t in self.tickets:
            if t["ticket_id"] == ticket_id:
                return t
        return None

    def list_tickets(
        self,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        """List tickets with optional category filter and pagination."""
        results = self.tickets
        if category:
            results = [t for t in results if t.get("predicted_category") == category]
        return results[offset : offset + limit]

    def save_override(self, override_data: dict) -> None:
        """Save an agent correction."""
        self.overrides.append(override_data)

    def get_metrics(self) -> dict:
        """Calculate aggregate metrics across all stored tickets."""
        if not self.tickets:
            return {
                "total_tickets": 0,
                "auto_routed_count": 0,
                "auto_route_rate": 0.0,
                "avg_confidence": 0.0,
                "override_count": 0,
                "override_rate": 0.0,
                "tickets_by_category": {},
                "tickets_by_urgency": {},
            }

        total         = len(self.tickets)
        auto_routed   = sum(1 for t in self.tickets if t.get("auto_routed"))
        avg_confidence = sum(t.get("confidence", 0) for t in self.tickets) / total
        overrides     = len(self.overrides)

        # Count tickets per category
        by_category = {}
        for t in self.tickets:
            cat = t.get("predicted_category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1

        # Count tickets per urgency
        by_urgency = {}
        for t in self.tickets:
            urg = t.get("predicted_urgency", "normal")
            by_urgency[urg] = by_urgency.get(urg, 0) + 1

        return {
            "total_tickets":       total,
            "auto_routed_count":   auto_routed,
            "auto_route_rate":     auto_routed / total,
            "avg_confidence":      avg_confidence,
            "override_count":      overrides,
            "override_rate":       overrides / total if total > 0 else 0,
            "tickets_by_category": by_category,
            "tickets_by_urgency":  by_urgency,
        }


# ---- Singleton store ----
_store = InMemoryStore()


def get_store() -> InMemoryStore:
    """FastAPI dependency — returns the in-memory store."""
    return _store