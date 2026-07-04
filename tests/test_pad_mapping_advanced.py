"""
Advanced Pad Mapping Tests

Tests for pad-to-lattice-node mapping covering:
- Nearest lattice node snapping (world_to_lattice + node_idx)
- Back layer (B.Cu) maps to z = num_layers - 1
- Internal layer mapping (In2.Cu → z=2)
- Out-of-bounds pad handling (clamp to edge, no crash)
- Multiple pads on same net both mapped
- Clearance between mapped nodes of different nets
"""

import pytest
import numpy as np

from orthoroute.algorithms.manhattan.unified_pathfinder import Lattice3D
from orthoroute.algorithms.manhattan.pathfinder.config import PathFinderConfig, GRID_PITCH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lattice():
    """Standard 6-layer lattice over 40×40 mm at 0.4 mm pitch."""
    return Lattice3D(bounds=(0, 0, 40, 40), pitch=0.4, layers=6)


@pytest.fixture
def small_lattice():
    """Small 2-layer lattice for edge-case tests."""
    return Lattice3D(bounds=(0, 0, 4, 4), pitch=0.4, layers=2)


# ---------------------------------------------------------------------------
# Helper: simulate _map_pads snap logic
# ---------------------------------------------------------------------------

def snap_to_node(lattice, x_mm, y_mm, layer=0):
    """Replicate the _snap_to_node logic from _map_pads."""
    x_idx, y_idx = lattice.world_to_lattice(x_mm, y_mm)
    # Clamp to valid range
    x_idx = max(0, min(x_idx, lattice.x_steps - 1))
    y_idx = max(0, min(y_idx, lattice.y_steps - 1))
    return lattice.node_idx(x_idx, y_idx, layer), (x_idx, y_idx, layer)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPadMapsToNearestLatticeNode:
    """Verify pads snap to the closest lattice node."""

    def test_pad_maps_to_nearest_lattice_node(self, lattice):
        """Pad at (1.2, 1.6) on a 0.4 mm pitch grid should snap to (3, 4, 0).

        Calculation: x_idx = round(1.2 / 0.4) = 3, y_idx = round(1.6 / 0.4) = 4
        """
        node, (x, y, z) = snap_to_node(lattice, 1.2, 1.6, layer=0)
        assert (x, y, z) == (3, 4, 0), f"Expected (3, 4, 0), got ({x}, {y}, {z})"
        assert node == lattice.node_idx(3, 4, 0)

    def test_pad_at_origin(self, lattice):
        """Pad at (0, 0) should map to lattice node (0, 0, 0)."""
        node, (x, y, z) = snap_to_node(lattice, 0.0, 0.0, layer=0)
        assert (x, y, z) == (0, 0, 0)

    def test_pad_at_grid_boundary(self, lattice):
        """Pad exactly at grid max should map to last valid node."""
        node, (x, y, z) = snap_to_node(lattice, 40.0, 40.0, layer=0)
        assert x == lattice.x_steps - 1
        assert y == lattice.y_steps - 1

    def test_pad_midpoint_rounding(self, lattice):
        """Pad at exactly between two grid points rounds to nearest."""
        # 1.0 mm / 0.4 = 2.5 → round to 2 or 3 depending on Python round()
        node, (x, y, z) = snap_to_node(lattice, 1.0, 1.0, layer=0)
        expected_x = round(1.0 / 0.4)  # Python banker's rounding: 2
        expected_y = round(1.0 / 0.4)
        assert x == expected_x
        assert y == expected_y


class TestPadOnBackLayer:
    """Verify back copper layer mapping."""

    def test_pad_on_back_layer_maps_to_z_max(self, lattice):
        """Pad on B.Cu should map to z = num_layers - 1 = 5."""
        back_layer = lattice.layers - 1  # 5 for 6-layer
        node, (x, y, z) = snap_to_node(lattice, 5.0, 5.0, layer=back_layer)
        assert z == back_layer, f"Expected z={back_layer}, got z={z}"

    def test_back_layer_node_idx_distinct(self, lattice):
        """Same XY on F.Cu vs B.Cu should produce different node indices."""
        node_front, _ = snap_to_node(lattice, 5.0, 5.0, layer=0)
        node_back, _ = snap_to_node(lattice, 5.0, 5.0, layer=lattice.layers - 1)
        assert node_front != node_back


class TestPadOnInternalLayer:
    """Verify internal layer mapping."""

    def test_pad_on_internal_layer(self, lattice):
        """Pad on In2.Cu should map to z=2."""
        node, (x, y, z) = snap_to_node(lattice, 10.0, 10.0, layer=2)
        assert z == 2, f"Expected z=2, got z={z}"

    def test_all_layers_produce_valid_nodes(self, lattice):
        """Every layer index produces a valid node within total node count."""
        for layer_z in range(lattice.layers):
            node, _ = snap_to_node(lattice, 5.0, 5.0, layer=layer_z)
            assert 0 <= node < lattice.num_nodes, (
                f"Node {node} out of range for layer {layer_z}"
            )


class TestPadOutsideBounds:
    """Verify graceful handling of out-of-bounds pads."""

    def test_pad_outside_bounds_handled(self, lattice):
        """Pad at (999, 999) shouldn't crash; it clamps to nearest edge node."""
        node, (x, y, z) = snap_to_node(lattice, 999.0, 999.0, layer=0)
        # Should clamp to max valid indices
        assert x == lattice.x_steps - 1, f"Expected clamped x={lattice.x_steps - 1}, got {x}"
        assert y == lattice.y_steps - 1, f"Expected clamped y={lattice.y_steps - 1}, got {y}"
        assert 0 <= node < lattice.num_nodes

    def test_pad_negative_coords_handled(self, lattice):
        """Pad at negative coords clamps to (0, 0)."""
        node, (x, y, z) = snap_to_node(lattice, -10.0, -10.0, layer=0)
        assert x == 0
        assert y == 0

    def test_large_negative_no_crash(self, lattice):
        """Very large negative coordinates don't crash."""
        node, (x, y, z) = snap_to_node(lattice, -1e6, -1e6, layer=0)
        assert x == 0
        assert y == 0
        assert 0 <= node < lattice.num_nodes


class TestMultiplePadsSameNet:
    """Verify multiple pads on the same net all get mapped."""

    def test_multiple_pads_same_net(self, lattice):
        """Two pads on the same net should both get valid, distinct node mappings."""
        pad1_pos = (4.0, 4.0)
        pad2_pos = (20.0, 20.0)

        node1, coords1 = snap_to_node(lattice, *pad1_pos, layer=0)
        node2, coords2 = snap_to_node(lattice, *pad2_pos, layer=0)

        assert node1 != node2, "Same-net pads should map to different nodes"
        assert 0 <= node1 < lattice.num_nodes
        assert 0 <= node2 < lattice.num_nodes

    def test_three_pad_net(self, lattice):
        """Three pads on one net all map to valid, distinct nodes."""
        positions = [(2.0, 2.0), (10.0, 10.0), (30.0, 30.0)]
        nodes = set()
        for x, y in positions:
            node, _ = snap_to_node(lattice, x, y, layer=0)
            nodes.add(node)
        assert len(nodes) == 3, "All three pads should map to distinct nodes"


class TestPadClearanceMaintained:
    """Verify mapped nodes for different nets don't collide."""

    def test_pad_clearance_maintained(self, lattice):
        """Mapped nodes for pads on different nets must not be the same node."""
        # Net A pads
        node_a1, _ = snap_to_node(lattice, 4.0, 4.0, layer=0)
        node_a2, _ = snap_to_node(lattice, 8.0, 8.0, layer=0)

        # Net B pads — different positions
        node_b1, _ = snap_to_node(lattice, 12.0, 12.0, layer=0)
        node_b2, _ = snap_to_node(lattice, 16.0, 16.0, layer=0)

        net_a_nodes = {node_a1, node_a2}
        net_b_nodes = {node_b1, node_b2}

        collision = net_a_nodes & net_b_nodes
        assert len(collision) == 0, (
            f"Cross-net node collision at nodes: {collision}"
        )

    def test_adjacent_pads_distinct_mapping(self, lattice):
        """Even pads separated by 1 pitch should map to distinct nodes."""
        pitch = lattice.pitch
        node1, _ = snap_to_node(lattice, 10.0, 10.0, layer=0)
        node2, _ = snap_to_node(lattice, 10.0 + pitch, 10.0, layer=0)

        assert node1 != node2, "Adjacent grid positions should be distinct nodes"
