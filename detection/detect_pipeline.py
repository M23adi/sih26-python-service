"""
Demo oil-spill detection pipeline for SIH26143.

TEMPORARY PROTOTYPE IMPLEMENTATION
----------------------------------
This version uses deterministic demo spill locations for the three
supported demo regions. It preserves the same API contract that the
future Sentinel-1 + ML detector will use.

Future replacement:
    Sentinel-1 SAR
        -> preprocessing
        -> DeepLabV3+ / segmentation model
        -> look-alike filtering
        -> polygon extraction
        -> geometry
        -> DetectResponse
"""

import math
import random
from datetime import datetime

import pyproj
from shapely.geometry import Polygon

from schemas import (
    DetectRequest,
    DetectResponse,
    GeoJSONPolygon,
    SpillGeometry,
    Centroid,
)
from config import (
    DETECTION_CONFIDENCE_BASE,
    LOOKALIKE_FILTER_ENABLED,
)


# ============================================================
# DEMO SPILL SCENARIOS
# ============================================================

DEMO_SCENARIOS = {
    "mumbai": {
        "center": (72.62, 18.43),
        "confidence": 0.92,
        "size": (0.025, 0.012),
    },

    "kochi": {
        "center": (76.55, 9.72),
        "confidence": 0.94,
        "size": (0.022, 0.011),
    },

    "chennai": {
        "center": (80.55, 13.10),
        "confidence": 0.91,
        "size": (0.024, 0.012),
    },
}


# ============================================================
# REGION DETECTION
# ============================================================

def _identify_demo_region(region):
    """
    Identify which supported demo region the request represents.

    Returns:
        "mumbai", "kochi", "chennai", or None
    """

    center_lon = (region.min_lon + region.max_lon) / 2
    center_lat = (region.min_lat + region.max_lat) / 2

    # Mumbai / Maharashtra
    if (
        17.0 <= center_lat <= 19.5
        and 71.0 <= center_lon <= 74.5
    ):
        return "mumbai"

    # Kochi
    if (
        8.5 <= center_lat <= 11.5
        and 74.5 <= center_lon <= 77.5
    ):
        return "kochi"

    # Chennai
    if (
        12.0 <= center_lat <= 14.5
        and 79.0 <= center_lon <= 81.5
    ):
        return "chennai"

    return None


# ============================================================
# POLYGON GENERATION
# ============================================================

def _generate_demo_spill_polygon(
    center_lon,
    center_lat,
    size_lon,
    size_lat,
    seed,
):
    """
    Generate a deterministic irregular polygon around a predefined
    offshore demo location.

    This is ONLY a temporary replacement for ML segmentation output.
    """

    random.seed(seed)

    n_points = 24
    points = []

    rotation = random.uniform(0, 2 * math.pi)

    for i in range(n_points):
        theta = (
            2 * math.pi * i / n_points
            + rotation
        )

        perturbation = random.uniform(0.85, 1.15)

        x = (
            center_lon
            + size_lon
            * perturbation
            * math.cos(theta)
        )

        y = (
            center_lat
            + size_lat
            * perturbation
            * math.sin(theta)
        )

        points.append((x, y))

    points.append(points[0])

    return Polygon(points)


# ============================================================
# GEOMETRY
# ============================================================

def _compute_geodesic_area_km2(polygon):
    """Compute polygon area in km² using WGS84."""

    geod = pyproj.Geod(ellps="WGS84")

    coords = list(polygon.exterior.coords)

    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]

    area_m2, _ = geod.polygon_area_perimeter(
        lons,
        lats,
    )

    return abs(area_m2) / 1_000_000


def _compute_geodesic_perimeter_km(polygon):
    """Compute polygon perimeter in km using WGS84."""

    geod = pyproj.Geod(ellps="WGS84")

    coords = list(polygon.exterior.coords)

    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]

    _, perimeter_m = geod.polygon_area_perimeter(
        lons,
        lats,
    )

    return abs(perimeter_m) / 1_000


def _compute_aspect_ratio(polygon):
    """Compute aspect ratio from minimum rotated rectangle."""

    mbr = polygon.minimum_rotated_rectangle

    coords = list(mbr.exterior.coords)

    edge1 = math.sqrt(
        (coords[1][0] - coords[0][0]) ** 2
        + (coords[1][1] - coords[0][1]) ** 2
    )

    edge2 = math.sqrt(
        (coords[2][0] - coords[1][0]) ** 2
        + (coords[2][1] - coords[1][1]) ** 2
    )

    if min(edge1, edge2) == 0:
        return 1.0

    return max(edge1, edge2) / min(edge1, edge2)


def _compute_compactness(area_km2, perimeter_km):
    """Polsby-Popper compactness."""

    if perimeter_km == 0:
        return 0.0

    return (
        4
        * math.pi
        * area_km2
        / (perimeter_km ** 2)
    )


# ============================================================
# LOOK-ALIKE FILTER
# ============================================================

def _check_lookalike(confidence):
    """
    Temporary look-alike filter.

    Future implementation:
        Sentinel-2 NIR / FAI analysis.
    """

    if not LOOKALIKE_FILTER_ENABLED:
        return False

    return confidence < 0.30


# ============================================================
# MAIN DETECTION
# ============================================================

def run_detection(request: DetectRequest) -> DetectResponse:
    """
    Run the temporary deterministic demo detection.

    The output format intentionally matches the future ML detector.
    """

    demo_region = _identify_demo_region(request.region)

    if demo_region is None:
        raise ValueError(
            "Unsupported demo region. "
            "Supported regions: Mumbai, Kochi, Chennai."
        )

    scenario = DEMO_SCENARIOS[demo_region]

    center_lon, center_lat = scenario["center"]

    size_lon, size_lat = scenario["size"]

    # Deterministic seed based on region + date
    seed = hash(
        f"{demo_region}-{request.date}"
    ) % (2 ** 31)

    # Generate demo spill polygon
    spill_polygon = _generate_demo_spill_polygon(
        center_lon=center_lon,
        center_lat=center_lat,
        size_lon=size_lon,
        size_lat=size_lat,
        seed=seed,
    )

    # Confidence
    confidence = round(
        scenario["confidence"],
        2,
    )

    # Look-alike filter
    is_filtered = _check_lookalike(
        confidence
    )

    # Geometry
    area_km2 = round(
        _compute_geodesic_area_km2(
            spill_polygon
        ),
        2,
    )

    perimeter_km = round(
        _compute_geodesic_perimeter_km(
            spill_polygon
        ),
        2,
    )

    centroid = spill_polygon.centroid

    aspect_ratio = round(
        _compute_aspect_ratio(
            spill_polygon
        ),
        2,
    )

    compactness = round(
        _compute_compactness(
            area_km2,
            perimeter_km,
        ),
        2,
    )

    # GeoJSON coordinates
    exterior_coords = [
        [
            round(x, 6),
            round(y, 6),
        ]
        for x, y in spill_polygon.exterior.coords
    ]

    # Detection ID
    detection_id = (
        f"det_"
        f"{request.date.replace('-', '')}_"
        f"{demo_region}"
    )

    timestamp = (
        f"{request.date}T14:30:00Z"
    )

    return DetectResponse(
        detection_id=detection_id,
        timestamp=timestamp,
        confidence=confidence,
        is_lookalike_filtered=is_filtered,

        polygon=GeoJSONPolygon(
            type="Polygon",
            coordinates=[
                exterior_coords
            ],
        ),

        geometry=SpillGeometry(
            area_km2=area_km2,
            perimeter_km=perimeter_km,

            centroid=Centroid(
                lon=round(
                    centroid.x,
                    4,
                ),
                lat=round(
                    centroid.y,
                    4,
                ),
            ),

            aspect_ratio=aspect_ratio,
            compactness=compactness,
        ),
    )