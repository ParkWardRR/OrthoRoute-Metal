"""
Named Constants for PathFinder Routing Algorithm

Extracted from unified_pathfinder.py to eliminate magic numbers.
These constants control the PathFinder negotiation loop behavior,
layer balancing, via costing, and routing heuristics.

Import from here instead of using raw numbers in code.
"""

from __future__ import annotations

# =============================================================================
# ROUTING MARGINS & GEOMETRY
# =============================================================================

ROUTING_MARGIN_MM: float = 3.0
"""Safety margin (mm) added around pad bounding box for ROI extraction."""

DEFAULT_GRID_PITCH_MM: float = 0.4
"""Default grid pitch in millimeters for the Manhattan routing lattice."""

# =============================================================================
# PATHFINDER NEGOTIATION PARAMETERS
# =============================================================================

EWMA_ALPHA: float = 0.85
"""Exponential weighted moving average alpha for layer bias smoothing.
Higher values weight history more heavily (range: 0.0-1.0)."""

EWMA_COMPLEMENT: float = 0.15
"""1 - EWMA_ALPHA, the weight for new observations in EWMA updates."""

PRESSURE_MULTIPLIER_DEFAULT: float = 1.10
"""Default pressure factor multiplier per iteration.
Controls how fast congestion pressure escalates (was 1.15, reduced for stability)."""

HIST_COST_WEIGHT_MULT_MANY_LAYERS: float = 0.8
"""History cost weight multiplier for boards with many signal layers.
Lower values reduce history influence on high-layer-count boards."""

PRES_FAC_MAX_DEFAULT: float = 8.0
"""Default maximum pressure factor cap (scaled by layer count in practice)."""

# =============================================================================
# VIA COSTING
# =============================================================================

DEFAULT_VIA_COST: float = 0.7
"""Base via transition cost. Lower values encourage layer hopping."""

VIA_COST_MULTIPLIER_CONGESTED: float = 1.5
"""Via cost multiplier applied when routing through congested via columns."""

# =============================================================================
# LAYER BALANCING
# =============================================================================

LAYER_BIAS_MIN: float = 0.85
"""Minimum layer bias value (prevents over-penalizing any layer)."""

LAYER_BIAS_MAX: float = 1.20
"""Maximum layer bias value (prevents over-favoring any layer)."""

# =============================================================================
# ROI & SEARCH LIMITS
# =============================================================================

GPU_ROI_THRESHOLD: int = 5000
"""Minimum ROI node count to trigger GPU acceleration over CPU fallback."""

DEFAULT_MAX_SEARCH_NODES: int = 2_000_000
"""Maximum nodes explored per single-net pathfinding call."""

HOT_SET_BASE_SIZE: int = 100
"""Base target size for hot-set selection (subset of nets to reroute per iteration)."""

LOW_OVERUSE_THRESHOLD: int = 100
"""If total overuse edges <= this, switch to fine-grained convergence mode."""

# =============================================================================
# CONVERGENCE DETECTION
# =============================================================================

CONVERGENCE_STALL_THRESHOLD: int = 5
"""Number of iterations without improvement before declaring stall."""

MAX_ITERATIONS_DEFAULT: int = 40
"""Default maximum PathFinder negotiation iterations."""
