"""
src/api/main.py
================
FastAPI application factory and startup configuration.

THIS IS THE ENTRY POINT for the web server.
Running: uvicorn src.api.main:app --reload
  - src.api.main  = this file's module path
  - app           = the FastAPI instance created below
  - --reload      = restart server when code changes (development only)

STARTUP SEQUENCE:
  1. Python imports this file
  2. FastAPI app is created with metadata
  3. CORS middleware is added
  4. Routes are registered
  5. When first request arrives, lifespan loads models
  6. Server is ready to handle requests
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import load_models_on_startup
from src.api.routes import router
from src.utils.logger import logger


# ----------------------------------------------------------------
# LIFESPAN CONTEXT MANAGER
# Controls what happens at startup and shutdown.
# Modern FastAPI pattern (replaces @app.on_event("startup"))
# ----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs at application startup and shutdown.

    The code BEFORE 'yield' runs at startup.
    The code AFTER 'yield' runs at shutdown.

    We load models at startup so they are ready for the first request.
    Loading here (not on first request) means no cold-start delay.
    """
    # ---- STARTUP ----
    logger.info("Starting Triage Engine API...")
    load_models_on_startup()
    logger.info("API ready to accept requests")

    yield   # Server runs while yielded — all requests handled here

    # ---- SHUTDOWN ----
    logger.info("Shutting down Triage Engine API...")
    # In production: close database connections, flush caches, etc.


# ----------------------------------------------------------------
# FASTAPI APP INSTANCE
# The metadata here populates the auto-generated /docs page
# ----------------------------------------------------------------

app = FastAPI(
    title="Support Ticket Triage Engine",
    description=(
        "NLP-powered API that automatically classifies and routes support tickets. "
        "Uses TF-IDF + Logistic Regression with SHAP explainability. "
        "Every prediction includes token-level reasoning."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",     # ReDoc UI at http://localhost:8000/redoc
)


# ----------------------------------------------------------------
# CORS MIDDLEWARE
# CORS = Cross-Origin Resource Sharing
#
# Browsers block requests from one origin (localhost:3000 React app)
# to a different origin (localhost:8000 API) by default — this is a
# security feature. CORS middleware tells the browser "these origins
# are allowed to make requests to this API."
#
# allow_origins=["*"] means ANY origin — fine for development.
# In production you would list specific allowed domains:
#   allow_origins=["https://yourdomain.com"]
# ----------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],          # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],          # Allow all request headers
)


# ----------------------------------------------------------------
# REGISTER ROUTES
# include_router attaches all endpoints defined in routes.py
# The prefix here (/api/v1) is already in the router, so we
# do not add it again here
# ----------------------------------------------------------------

app.include_router(router)


# ----------------------------------------------------------------
# ROOT REDIRECT
# Anyone hitting http://localhost:8000/ sees a helpful message
# pointing them to the docs
# ----------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    """Root endpoint — redirects to API docs."""
    return {
        "message":     "Support Ticket Triage Engine API",
        "version":     "1.0.0",
        "docs":        "/docs",
        "health":      "/api/v1/health",
        "submit":      "POST /api/v1/tickets",
        "list":        "GET /api/v1/tickets",
        "metrics":     "GET /api/v1/metrics",
    }