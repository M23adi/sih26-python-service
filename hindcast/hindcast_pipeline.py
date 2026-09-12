"""
Simulated Monte Carlo Lagrangian hindcast & forward forecast.

For the prototype, this module simulates what OpenDrift would produce:
- Backward particle trajectories → origin probability area
- Forward particle trajectories → forecast drift path

The simulation uses simplified physics:
  drift = surface_current + wind_drift_factor × wind + random_diffusion

This produces realistic-looking origin and forecast polygons that are
physically plausible (particles drift with currents/wind and spread
due to diffusion uncertainty).
"""

import math
import random
import numpy as np
from datetime import datetime, timedelta, timezone
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union

from schemas import (
    HindcastRequest, HindcastResponse,
    GeoJSONPolygon, TimeWindow,
)
from config import (
    MONTE_CARLO_RUNS,
    BACKWARD_HOURS,
    FORWARD_HOURS,
    PARTICLE_COUNT,
    WIND_DRIFT_FACTOR,
    CURRENT_SPEED_MS,
    CURRENT_DIRECTION_DEG,
    WIND_SPEED_MS,
    WIND_DIRECTION_DEG,
    DIFFUSION_COEFFICIENT,
)


def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _meters_to_deg_lon(meters: float, lat: float) -> float:
    """Convert meters to degrees longitude at a given latitude."""
    return meters / (111320.0 * math.cos(_deg_to_rad(lat)))


def _meters_to_deg_lat(meters: float) -> float:
    """Convert meters to degrees latitude."""
    return meters / 110540.0


def _simulate_particle_drift(
    start_lon: float,
    start_lat: float,
    hours: float,
    direction: int,  # 1 = forward, -1 = backward
    current_speed: float = None,
    current_dir: float = None,
    wind_speed: float = None,
    wind_dir: float = None,
) -> tuple[float, float]:
    """
    Simulate a single particle's drift trajectory over the given hours.
    
    Uses simplified Lagrangian advection:
      displacement = (current + wind_drift) × time + diffusion_noise
    
    Args:
        direction: 1 for forward forecast, -1 for backward hindcast
    
    Returns:
        (final_lon, final_lat) of the particle
    """
    if current_speed is None:
        current_speed = CURRENT_SPEED_MS
    if current_dir is None:
        current_dir = CURRENT_DIRECTION_DEG
    if wind_speed is None:
        wind_speed = WIND_SPEED_MS
    if wind_dir is None:
        wind_dir = WIND_DIRECTION_DEG

    # Add Monte Carlo perturbation to physical parameters
    perturbed_current_speed = current_speed * (1 + random.gauss(0, 0.2))
    perturbed_current_dir = current_dir + random.gauss(0, 15)  # ±15° uncertainty
    perturbed_wind_speed = wind_speed * (1 + random.gauss(0, 0.15))
    perturbed_wind_dir = wind_dir + random.gauss(0, 10)

    # Total drift velocity components (m/s)
    # Current contribution
    vx_current = perturbed_current_speed * math.sin(_deg_to_rad(perturbed_current_dir))
    vy_current = perturbed_current_speed * math.cos(_deg_to_rad(perturbed_current_dir))

    # Wind drift contribution (oil drifts ~3% of wind speed)
    vx_wind = WIND_DRIFT_FACTOR * perturbed_wind_speed * math.sin(_deg_to_rad(perturbed_wind_dir))
    vy_wind = WIND_DRIFT_FACTOR * perturbed_wind_speed * math.cos(_deg_to_rad(perturbed_wind_dir))

    # Total advection
    vx = vx_current + vx_wind
    vy = vy_current + vy_wind

    # Time in seconds
    dt = hours * 3600

    # Advection displacement (meters)
    dx_advection = direction * vx * dt
    dy_advection = direction * vy * dt

    # Diffusion (random walk) displacement (meters)
    # Standard deviation grows as sqrt(2 * K * t)
    diffusion_std = math.sqrt(2 * DIFFUSION_COEFFICIENT * abs(dt))
    dx_diffusion = random.gauss(0, diffusion_std)
    dy_diffusion = random.gauss(0, diffusion_std)

    # Total displacement in meters
    dx_total = dx_advection + dx_diffusion
    dy_total = dy_advection + dy_diffusion

    # Convert to degrees
    final_lon = start_lon + _meters_to_deg_lon(dx_total, start_lat)
    final_lat = start_lat + _meters_to_deg_lat(dy_total)

    return final_lon, final_lat


def _run_ensemble(
    center_lon: float,
    center_lat: float,
    hours: float,
    direction: int,
    n_runs: int,
) -> list[tuple[float, float]]:
    """
    Run a Monte Carlo ensemble of particle drift simulations.
    
    Returns a list of (lon, lat) endpoint positions.
    """
    endpoints = []
    for _ in range(n_runs):
        # Add slight scatter to starting position (simulates spill extent)
        start_lon = center_lon + random.gauss(0, 0.005)
        start_lat = center_lat + random.gauss(0, 0.003)

        lon, lat = _simulate_particle_drift(
            start_lon, start_lat, hours, direction
        )
        endpoints.append((lon, lat))
    return endpoints


def _endpoints_to_polygon(endpoints: list[tuple[float, float]], buffer_deg: float = 0.01) -> list[list[float]]:
    """
    Convert a set of particle endpoints into a GeoJSON polygon.
    
    Uses the convex hull of all endpoints with a small buffer
    to create the probability area polygon.
    """
    if len(endpoints) < 3:
        # Not enough points for a polygon, create a simple box
        lons = [p[0] for p in endpoints]
        lats = [p[1] for p in endpoints]
        min_lon, max_lon = min(lons) - buffer_deg, max(lons) + buffer_deg
        min_lat, max_lat = min(lats) - buffer_deg, max(lats) + buffer_deg
        coords = [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]
        return [[round(c[0], 6), round(c[1], 6)] for c in coords]

    points = MultiPoint(endpoints)
    hull = points.convex_hull

    # Buffer slightly for visual clarity
    hull_buffered = hull.buffer(buffer_deg)

    # Simplify to reduce point count while keeping shape
    hull_simplified = hull_buffered.simplify(0.002)

    coords = list(hull_simplified.exterior.coords)
    return [[round(c[0], 6), round(c[1], 6)] for c in coords]


def run_hindcast(request: HindcastRequest) -> HindcastResponse:
    """
    Run the full hindcast/forecast pipeline.
    
    1. Monte Carlo backward simulations → origin probability area
    2. Compute origin time window
    3. Monte Carlo forward simulations → forecast path
    """
    # Parse detection timestamp
    det_time = datetime.fromisoformat(request.detection_timestamp.replace("Z", "+00:00"))

    center_lon = request.centroid.lon
    center_lat = request.centroid.lat

    # Seed the random generators to ensure deterministic output per region
    seed_str = request.detection_timestamp + str(center_lon) + str(center_lat)
    seed_val = hash(seed_str) % (2**31)
    random.seed(seed_val)
    np.random.seed(seed_val)

    # ── Backward hindcast ──
    backward_endpoints = _run_ensemble(
        center_lon, center_lat,
        hours=BACKWARD_HOURS,
        direction=-1,
        n_runs=MONTE_CARLO_RUNS,
    )

    origin_coords = _endpoints_to_polygon(backward_endpoints)

    # Origin time window: detection_time - BACKWARD_HOURS ± some spread
    origin_start = det_time - timedelta(hours=BACKWARD_HOURS + 1)
    origin_end = det_time - timedelta(hours=BACKWARD_HOURS - 2)

    # ── Forward forecast ──
    forward_endpoints = _run_ensemble(
        center_lon, center_lat,
        hours=FORWARD_HOURS,
        direction=1,
        n_runs=MONTE_CARLO_RUNS,
    )

    forecast_coords = _endpoints_to_polygon(forward_endpoints)

    return HindcastResponse(
        origin_probability_area=GeoJSONPolygon(
            type="Polygon",
            coordinates=[origin_coords],
        ),
        estimated_origin_time_window=TimeWindow(
            start=origin_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=origin_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        forward_forecast_path=GeoJSONPolygon(
            type="Polygon",
            coordinates=[forecast_coords],
        ),
        monte_carlo_runs=MONTE_CARLO_RUNS,
    )
