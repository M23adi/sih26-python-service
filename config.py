"""
Configuration for the Oil Spill Detection & Hindcast prototype service.
All constants, paths, and tunable parameters live here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FALLBACK_DIR = DATA_DIR / "fallback"
DEMO_AIS_CSV = BASE_DIR.parent / "demoData.txt"  # The team's shared demo AIS data

# ──────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────
SERVICE_PORT = int(os.getenv("PORT", "8000"))
USE_FALLBACK = os.getenv("USE_FALLBACK", "false").lower() == "true"

# ──────────────────────────────────────────────
# Detection simulation parameters
# ──────────────────────────────────────────────
DETECTION_CONFIDENCE_BASE = 0.87       # Base confidence for simulated detection
LOOKALIKE_FILTER_ENABLED = True        # Whether to run the look-alike filter step
FAI_THRESHOLD = 0.3                    # Floating Algae Index threshold for rejection

# ──────────────────────────────────────────────
# Hindcast simulation parameters
# ──────────────────────────────────────────────
MONTE_CARLO_RUNS = 100                 # Number of Monte Carlo simulations
BACKWARD_HOURS = 6                     # Hours to trace backward from detection
FORWARD_HOURS = 6                      # Hours to forecast forward
PARTICLE_COUNT = 50                    # Particles per simulation run

# ──────────────────────────────────────────────
# Physical constants for drift simulation
# ──────────────────────────────────────────────
WIND_DRIFT_FACTOR = 0.03              # Oil drift = ~3% of wind speed
CURRENT_SPEED_MS = 0.3                # Default surface current (m/s)
CURRENT_DIRECTION_DEG = 45            # Default current direction (degrees from N)
WIND_SPEED_MS = 8.0                   # Default wind speed (m/s)
WIND_DIRECTION_DEG = 225              # Default wind from direction (SW)
DIFFUSION_COEFFICIENT = 100           # Horizontal diffusion (m²/s)
