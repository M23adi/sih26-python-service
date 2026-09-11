"""
Pydantic schemas matching the EXACT API contract from 01-PYTHON-SPEC.md Section 4.
Every field name, type, and nesting level is exactly as specified.
Coordinate order: [longitude, latitude] throughout (GeoJSON standard).
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


# ──────────────────────────────────────────────
# Shared / nested models
# ──────────────────────────────────────────────

class Region(BaseModel):
    """Bounding box for the area of interest."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class GeoJSONPolygon(BaseModel):
    """
    Standard GeoJSON Polygon.
    coordinates is a list of rings; each ring is a list of [lon, lat] pairs.
    The first and last point in each ring MUST be identical (closed ring).
    """
    type: str = "Polygon"
    coordinates: List[List[List[float]]]


class Centroid(BaseModel):
    lon: float
    lat: float


class SpillGeometry(BaseModel):
    """Geometric characterization of the detected spill."""
    area_km2: float
    perimeter_km: float
    centroid: Centroid
    aspect_ratio: float
    compactness: float


# ──────────────────────────────────────────────
# POST /detect
# ──────────────────────────────────────────────

class DetectRequest(BaseModel):
    region: Region
    date: str  # ISO date string, e.g. "2024-03-15"


class DetectResponse(BaseModel):
    detection_id: str
    timestamp: str                     # ISO 8601 UTC, e.g. "2024-03-15T14:30:00Z"
    confidence: float                  # 0.0 – 1.0
    is_lookalike_filtered: bool
    polygon: GeoJSONPolygon
    geometry: SpillGeometry


# ──────────────────────────────────────────────
# POST /hindcast
# ──────────────────────────────────────────────

class HindcastRequest(BaseModel):
    centroid: Centroid
    polygon: GeoJSONPolygon
    detection_timestamp: str           # ISO 8601 UTC
    region: Region


class TimeWindow(BaseModel):
    start: str                         # ISO 8601 UTC
    end: str                           # ISO 8601 UTC


class HindcastResponse(BaseModel):
    origin_probability_area: GeoJSONPolygon
    estimated_origin_time_window: TimeWindow
    forward_forecast_path: GeoJSONPolygon
    monte_carlo_runs: int


# ──────────────────────────────────────────────
# Error response (shared)
# ──────────────────────────────────────────────

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
