"""
Simulated oil spill detection pipeline.

For the prototype, this module generates realistic detection results
based on the requested region and date. In a production system, this
would run a trained DeepLabV3+ model on preprocessed Sentinel-1 SAR imagery.

The simulation:
1. Places a realistic spill polygon inside the requested bounding box
2. Adds slight randomness so repeated calls don't look identical
3. Runs a rule-based look-alike filter check
4. Computes real geometric characterization (area, perimeter, etc.)
"""

import math
import random
import numpy as np
from datetime import datetime, timezone
from shapely.geometry import Polygon, mapping
from shapely.ops import transform
import pyproj
from functools import partial

from schemas import (
    DetectRequest, DetectResponse,
    GeoJSONPolygon, SpillGeometry, Centroid,
)
from config import DETECTION_CONFIDENCE_BASE, LOOKALIKE_FILTER_ENABLED


def _generate_spill_polygon(region, seed: int = None) -> Polygon:
    """
    Generate a realistic irregular oil spill polygon within the given region.
    
    Real oil spills are elongated, somewhat irregular shapes (not rectangles).
    We generate an ellipse-like shape with perturbations to mimic this.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Place the spill roughly in the middle-upper area of the region
    # (biased slightly so it's not dead center — looks more realistic)
    cx = (region.min_lon + region.max_lon) / 2 + random.uniform(-0.05, 0.05)
    cy = (region.min_lat + region.max_lat) / 2 + random.uniform(-0.05, 0.05)

    # Spill size: roughly 0.01–0.03 degrees across (~1–3 km)
    a = random.uniform(0.008, 0.015)  # semi-major axis (lon)
    b = random.uniform(0.005, 0.010)  # semi-minor axis (lat)

    # Generate irregular polygon points along an ellipse with perturbations
    n_points = 24
    angle_offset = random.uniform(0, 2 * math.pi)  # random orientation
    points = []
    for i in range(n_points):
        theta = 2 * math.pi * i / n_points + angle_offset
        # Perturb the radius for realism (oil spills are irregular)
        r_perturb = 1.0 + random.uniform(-0.25, 0.25)
        x = cx + a * r_perturb * math.cos(theta)
        y = cy + b * r_perturb * math.sin(theta)
        points.append((x, y))

    # Close the ring
    points.append(points[0])

    return Polygon(points)


def _compute_geodesic_area_km2(polygon: Polygon) -> float:
    """Compute the geodesic area of a polygon in km²."""
    # Use WGS84 ellipsoid for accurate area calculation
    geod = pyproj.Geod(ellps="WGS84")
    coords = list(polygon.exterior.coords)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    area_m2, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area_m2) / 1e6  # m² → km²


def _compute_geodesic_perimeter_km(polygon: Polygon) -> float:
    """Compute the geodesic perimeter of a polygon in km."""
    geod = pyproj.Geod(ellps="WGS84")
    coords = list(polygon.exterior.coords)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    _, perimeter_m = geod.polygon_area_perimeter(lons, lats)
    return abs(perimeter_m) / 1e3  # m → km


def _compute_aspect_ratio(polygon: Polygon) -> float:
    """Compute the aspect ratio from the minimum bounding rectangle."""
    mbr = polygon.minimum_rotated_rectangle
    coords = list(mbr.exterior.coords)
    # Compute the two edge lengths
    edge1 = math.sqrt(
        (coords[1][0] - coords[0][0]) ** 2 +
        (coords[1][1] - coords[0][1]) ** 2
    )
    edge2 = math.sqrt(
        (coords[2][0] - coords[1][0]) ** 2 +
        (coords[2][1] - coords[1][1]) ** 2
    )
    if min(edge1, edge2) == 0:
        return 1.0
    return max(edge1, edge2) / min(edge1, edge2)


def _compute_compactness(area_km2: float, perimeter_km: float) -> float:
    """
    Polsby-Popper compactness: 4π × area / perimeter².
    A perfect circle = 1.0; elongated shapes < 1.0.
    """
    if perimeter_km == 0:
        return 0.0
    return (4 * math.pi * area_km2) / (perimeter_km ** 2)


def _check_lookalike(confidence: float) -> bool:
    """
    Simulated look-alike filter.
    
    In production, this would check Sentinel-2 NIR/chlorophyll reflectance
    within the detected polygon to reject algae false positives (FAI > threshold).
    
    For the prototype, we simulate: high-confidence detections pass the filter,
    very low confidence ones get flagged as potential look-alikes.
    """
    if not LOOKALIKE_FILTER_ENABLED:
        return False
    # Simulate: <30% confidence → likely look-alike
    return confidence < 0.30


def run_detection(request: DetectRequest) -> DetectResponse:
    """
    Run the full detection pipeline for a given region and date.
    
    Pipeline stages (simulated for prototype):
    1. SAR image ingestion & preprocessing
    2. Model inference (DeepLabV3+ segmentation)
    3. Look-alike filter (rule-based NIR/FAI check)
    4. Spill characterization (geometry)
    """
    # Use the date as a seed for reproducible (but realistic-looking) results
    date_seed = hash(request.date + str(request.region.min_lon)) % (2**31)
    
    # Stage 1 & 2: Generate a realistic spill polygon (simulating model output)
    spill_polygon = _generate_spill_polygon(request.region, seed=date_seed)
    
    # Compute detection confidence with slight randomness
    confidence = round(DETECTION_CONFIDENCE_BASE + random.uniform(-0.05, 0.08), 2)
    confidence = max(0.0, min(1.0, confidence))

    # Stage 3: Look-alike filter
    is_filtered = _check_lookalike(confidence)

    # Stage 4: Geometric characterization
    area_km2 = round(_compute_geodesic_area_km2(spill_polygon), 1)
    perimeter_km = round(_compute_geodesic_perimeter_km(spill_polygon), 1)
    centroid = spill_polygon.centroid
    aspect_ratio = round(_compute_aspect_ratio(spill_polygon), 1)
    compactness = round(_compute_compactness(area_km2, perimeter_km), 2)

    # Convert polygon to GeoJSON coordinates [lon, lat]
    exterior_coords = [
        [round(x, 6), round(y, 6)]
        for x, y in spill_polygon.exterior.coords
    ]

    # Generate detection ID and timestamp
    detection_id = f"det_{request.date.replace('-', '')}_{random.randint(1, 999):03d}"
    timestamp = f"{request.date}T14:30:00Z"

    return DetectResponse(
        detection_id=detection_id,
        timestamp=timestamp,
        confidence=confidence,
        is_lookalike_filtered=is_filtered,
        polygon=GeoJSONPolygon(
            type="Polygon",
            coordinates=[exterior_coords]
        ),
        geometry=SpillGeometry(
            area_km2=area_km2,
            perimeter_km=perimeter_km,
            centroid=Centroid(
                lon=round(centroid.x, 2),
                lat=round(centroid.y, 2)
            ),
            aspect_ratio=aspect_ratio,
            compactness=compactness,
        ),
    )
