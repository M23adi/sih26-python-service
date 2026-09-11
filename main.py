"""
Oil Spill Detection & Hindcast — FastAPI Microservice (Prototype)
=================================================================

Person 1 service for SIH26143.
Runs on port 8000. Exposes two endpoints:
  - POST /detect    → oil spill detection + characterization
  - POST /hindcast  → Monte Carlo backward/forward drift simulation

Person 2's Node.js backend calls these over HTTP.
This service does NOT talk to the frontend directly.

Usage:
    cd python-service
    venv\\Scripts\\python.exe -m uvicorn main:app --port 8000 --reload
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add the project root to path so our modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from schemas import (
    DetectRequest, DetectResponse,
    HindcastRequest, HindcastResponse,
    ErrorResponse,
)
from config import SERVICE_PORT, USE_FALLBACK, FALLBACK_DIR

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="SIH26143 — Oil Spill Detection & Hindcast Service",
    description=(
        "Person 1 microservice. Performs SAR-based oil spill detection, "
        "look-alike filtering, spill characterization, and Monte Carlo "
        "Lagrangian hindcast/forecast."
    ),
    version="1.0.0-prototype",
)

# CORS — allow Person 2's Node.js backend (port 5000) and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "SIH26143 Python Detection & Hindcast",
        "version": "1.0.0-prototype",
        "status": "running",
        "endpoints": ["POST /detect", "POST /hindcast"],
        "port": SERVICE_PORT,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


# ──────────────────────────────────────────────
# POST /detect
# ──────────────────────────────────────────────

@app.post("/detect", response_model=DetectResponse, responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect(request: DetectRequest):
    """
    Detect oil spills in the given region and date.
    
    Pipeline stages:
    1. SAR image ingestion & preprocessing (simulated)
    2. DeepLabV3+ model inference (simulated with realistic output)
    3. Look-alike filter (rule-based FAI/NIR threshold check)
    4. Spill characterization (real geodesic geometry calculations)
    
    Returns detection with polygon, confidence, and geometric properties.
    """
    try:
        # Validate input
        if request.region.min_lon >= request.region.max_lon:
            raise HTTPException(
                status_code=422,
                detail={"status": "error", "message": "min_lon must be less than max_lon"}
            )
        if request.region.min_lat >= request.region.max_lat:
            raise HTTPException(
                status_code=422,
                detail={"status": "error", "message": "min_lat must be less than max_lat"}
            )

        # Check for fallback mode
        if USE_FALLBACK:
            fallback_path = FALLBACK_DIR / "detect_response.json"
            if fallback_path.exists():
                with open(fallback_path) as f:
                    return json.load(f)

        # Run the detection pipeline
        from detection.detect_pipeline import run_detection
        result = run_detection(request)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Detection pipeline failed: {str(e)}"}
        )


# ──────────────────────────────────────────────
# POST /hindcast
# ──────────────────────────────────────────────

@app.post("/hindcast", response_model=HindcastResponse, responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def hindcast(request: HindcastRequest):
    """
    Run Monte Carlo Lagrangian hindcast (backward) and forecast (forward).
    
    Pipeline stages:
    1. Parse detection position and time
    2. Run backward ensemble (50-100 perturbed simulations) for origin estimation
    3. Compute origin probability area (convex hull of backward endpoints)
    4. Run forward ensemble for drift forecast
    5. Compute forecast path polygon
    
    Returns origin probability area, time window, and forecast path.
    """
    try:
        # Validate timestamp format
        try:
            datetime.fromisoformat(request.detection_timestamp.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={"status": "error", "message": "Invalid detection_timestamp format. Use ISO 8601 UTC, e.g. 2024-03-15T14:30:00Z"}
            )

        # Check for fallback mode
        if USE_FALLBACK:
            fallback_path = FALLBACK_DIR / "hindcast_response.json"
            if fallback_path.exists():
                with open(fallback_path) as f:
                    return json.load(f)

        # Run the hindcast pipeline
        from hindcast.hindcast_pipeline import run_hindcast
        result = run_hindcast(request)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Hindcast pipeline failed: {str(e)}"}
        )


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"\n{'='*60}")
    print(f"  SIH26143 — Oil Spill Detection & Hindcast Service")
    print(f"  Starting on http://localhost:{SERVICE_PORT}")
    print(f"  Fallback mode: {'ON' if USE_FALLBACK else 'OFF'}")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
