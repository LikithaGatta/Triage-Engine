

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Support Ticket Triage Engine API")
    print("="*55)
    print("  Docs:    http://localhost:8000/docs")
    print("  Health:  http://localhost:8000/api/v1/health")
    print("  Submit:  POST http://localhost:8000/api/v1/tickets")
    print("="*55 + "\n")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",     # Accept connections from any network interface
        port=8000,
        reload=True,         # Auto-restart when code changes
        log_level="info",
    )