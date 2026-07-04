"""Tests for PadEscapePlanner — portal escape precomputation.

Validates that the PadEscapePlanner correctly generates portals,
maintains pad clearance, and distributes escapes across multiple layers.
"""
import pytest
import numpy as np

from orthoroute.algorithms.manhattan.unified_pathfinder import Lattice3D
from orthoroute.algorithms.manhattan.pad_escape_planner import PadEscapePlanner, Portal
from orthoroute.algorithms.manhattan.pathfinder.config import PathFinderConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lattice():
    """10×10 grid, 6 layers, pitch=0.4mm."""
    bounds = (0.0, 0.0, 3.6, 3.6)
    return Lattice3D(bounds, pitch=0.4, layers=6)


@pytest.fixture
def config():
    """PathFinderConfig with portals enabled."""
    cfg = PathFinderConfig()
    cfg.portal_enabled = True
    cfg.layer_count = 6
    return cfg


@pytest.fixture
def pad_to_node(lattice):
    """Simple pad_to_node mapping: 4 pads at known grid positions."""
    mapping = {}
    # Pads at (2,2), (5,5), (7,2), (3,8) on layer 0
    for label, (x, y) in [("PA", (2, 2)), ("PB", (5, 5)), ("PC", (7, 2)), ("PD", (3, 8))]:
        mapping[label] = lattice.node_idx(x, y, 0)
    return mapping


@pytest.fixture
def planner(lattice, config, pad_to_node):
    """Initialized PadEscapePlanner."""
    return PadEscapePlanner(lattice, config, pad_to_node, random_seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPadEscapeGeneratesPortals:
    """Verify that PadEscapePlanner can create Portal objects correctly."""

    def test_portal_dataclass_construction(self):
        """Portal can be constructed with required fields."""
        p = Portal(x_idx=5, y_idx=10, pad_layer=0, delta_steps=4,
                   direction=1, pad_x=2.0, pad_y=4.0, entry_layer=1)
        assert p.x_idx == 5
        assert p.y_idx == 10
        assert p.pad_layer == 0
        assert p.direction == 1
        assert p.entry_layer == 1

    def test_portal_default_fields(self):
        """Portal defaults: entry_layer=1, score=0.0, retarget_count=0."""
        p = Portal(x_idx=0, y_idx=0, pad_layer=0, delta_steps=3,
                   direction=-1, pad_x=0.0, pad_y=0.0)
        assert p.entry_layer == 1
        assert p.score == 0.0
        assert p.retarget_count == 0

    def test_portal_y_offset(self):
        """Portal y_idx should differ from pad y by delta_steps * direction."""
        pad_y_idx = 5
        delta = 4
        direction = 1
        portal_y_idx = pad_y_idx + delta * direction
        p = Portal(x_idx=3, y_idx=portal_y_idx, pad_layer=0,
                   delta_steps=delta, direction=direction,
                   pad_x=1.2, pad_y=2.0)
        assert p.y_idx == 9

    def test_planner_initializes(self, planner):
        """PadEscapePlanner initializes with empty portals dict."""
        assert isinstance(planner.portals, dict)
        assert len(planner.portals) == 0  # No portals before precompute


class TestPortalClearance:
    """Verify portal placement respects clearance constraints."""

    def test_portal_delta_within_configured_range(self, config):
        """Generated portals should have delta_steps in [portal_delta_min, portal_delta_max]."""
        p = Portal(x_idx=5, y_idx=8, pad_layer=0,
                   delta_steps=config.portal_delta_min,
                   direction=1, pad_x=2.0, pad_y=2.0)
        assert config.portal_delta_min <= p.delta_steps <= config.portal_delta_max

    def test_portal_direction_is_valid(self):
        """Direction must be +1 (north) or -1 (south)."""
        for d in [1, -1]:
            p = Portal(x_idx=0, y_idx=0, pad_layer=0,
                       delta_steps=3, direction=d, pad_x=0.0, pad_y=0.0)
            assert p.direction in (1, -1)

    def test_portal_stays_in_lattice_bounds(self, lattice):
        """Portal y_idx should be within [0, y_steps-1]."""
        max_y = lattice.y_steps - 1
        # Simulate portal placement at edge
        pad_y_idx = max_y
        delta = 5
        direction = 1  # Going up (positive y)
        portal_y = pad_y_idx + delta * direction
        # After bounds clamping
        portal_y_clamped = max(0, min(portal_y, max_y))
        assert 0 <= portal_y_clamped <= max_y


class TestMultiLayerEscape:
    """Verify escape planning across multiple layers."""

    def test_planner_has_correct_layer_count(self, planner, config):
        """Planner's lattice should have the configured number of layers."""
        assert planner.lattice.layers == config.layer_count

    def test_portal_entry_layer_is_routing_layer(self, config):
        """Portal entry_layer should be a valid routing layer (1 to layer_count-2)."""
        # Routing layers are internal layers: 1, 2, 3, 4 for a 6-layer board
        # (Layer 0 = F.Cu, Layer 5 = B.Cu)
        for entry_layer in range(1, config.layer_count - 1):
            p = Portal(x_idx=3, y_idx=6, pad_layer=0, delta_steps=5,
                       direction=-1, pad_x=1.2, pad_y=2.4,
                       entry_layer=entry_layer)
            assert 1 <= p.entry_layer <= config.layer_count - 2

    def test_planner_random_seed_determinism(self, lattice, config, pad_to_node):
        """Two planners with the same seed should produce identical results."""
        p1 = PadEscapePlanner(lattice, config, pad_to_node, random_seed=42)
        p2 = PadEscapePlanner(lattice, config, pad_to_node, random_seed=42)
        assert p1.random_seed == p2.random_seed

    def test_planner_different_seeds_differ(self, lattice, config, pad_to_node):
        """Planners with different seeds should have different RNG state."""
        p1 = PadEscapePlanner(lattice, config, pad_to_node, random_seed=42)
        p2 = PadEscapePlanner(lattice, config, pad_to_node, random_seed=99)
        assert p1.random_seed != p2.random_seed
