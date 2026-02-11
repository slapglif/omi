#!/usr/bin/env python3
"""
Start a test server for manual testing of the dashboard API.
"""

from fastapi import FastAPI
import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omi.dashboard_api import router

# Create FastAPI app
app = FastAPI(title="OMI Dashboard API Test Server")
app.include_router(router)

if __name__ == "__main__":
    print("Starting test server on http://localhost:8420")
    print("API available at http://localhost:8420/api/v1/dashboard/graph")
    print("Press Ctrl+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8420)
