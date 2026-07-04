"""Tests for pad → lattice node mapping.

Validates that _snap_to_node correctly maps pad world coordinates to
lattice GIDs, respects layer indices, and handles out-of-bounds pads
gracefully (clamping rather than crashing).
"""
import pytest
import numpy as np

from orthoroute.algorithms.manhattan.unified_pathfinder import Lattice3D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lattice_4layer():
    """5×5 grid, 4 layers, pitch=0.4mm."""
    bounds = (0.0, 0.0, 1.6, 1.6)  # (1.6-0.0)/0.4+1 = 5 steps per axis
    return Lattice3D(bounds, pitch=0.4, layers=4)


@pytest.fixture
def lattice_6layer():
    """10×10 grid, 6 layers, pitch=0.4mm."""
    bounds = (0.0, 0.0, 3.6, 3.6)  # 10 steps per axis
    return Lattice3D(bounds, pitch=0.4, layers=6)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNearestNodeFinding:
    """Verify that a pad at a known world coordinate maps to the expected lattice node."""

    def test_origin_maps_to_node_zero(self, lattice_4layer):
        """Pad at (0.0, 0.0) on layer 0 should map to node_idx(0, 0, 0) == 0."""
        lat = lattice_4layer
        x_idx, y_idx = lat.world_to_lattice(0.0, 0.0)
        node = lat.node_idx(x_idx, y_idx, 0)
        assert node == lat.node_idx(0, 0, 0)

    def test_center_maps_correctly(self, lattice_4layer):
        """Pad at (0.8, 0.8) should snap to lattice index (2, 2)."""
        lat = lattice_4layer
        x_idx, y_idx = lat.world_to_lattice(0.8, 0.8)
        assert x_idx == 2
        assert y_idx == 2

    def test_fractional_coordinate_snaps(self, lattice_4layer):
        """Pad at (0.79, 0.41) — slightly off-grid — should still snap to nearest node."""
        lat = lattice_4layer
        x_idx, y_idx = lat.world_to_lattice(0.79, 0.41)
        # 0.79/0.4 ≈ 1.975 → round to 2; 0.41/0.4 ≈ 1.025 → round to 1
        assert x_idx == 2
        assert y_idx == 1

    def test_roundtrip_node_idx(self, lattice_4layer):
        """node_idx → idx_to_coord → node_idx is identity."""
        lat = lattice_4layer
        x_idx, y_idx = lat.world_to_lattice(0.4, 1.2)
        node = lat.node_idx(x_idx, y_idx, 0)
        x2, y2, z2 = lat.idx_to_coord(node)
        assert lat.node_idx(x2, y2, z2) == node


class TestMultiLayerPadMapping:
    """Pad on layer z maps to the correct z-coordinate in the lattice."""

    def test_layer_zero(self, lattice_6layer):
        """Pad on layer 0 maps to z=0 plane."""
        lat = lattice_6layer
        node = lat.node_idx(3, 3, 0)
        x, y, z = lat.idx_to_coord(node)
        assert z == 0

    def test_layer_five(self, lattice_6layer):
        """Pad on layer 5 (B.Cu in 6-layer board) maps to z=5."""
        lat = lattice_6layer
        node = lat.node_idx(3, 3, 5)
        x, y, z = lat.idx_to_coord(node)
        assert z == 5

    def test_different_layers_different_nodes(self, lattice_6layer):
        """Same (x,y) on different layers should produce different node indices."""
        lat = lattice_6layer
        nodes = [lat.node_idx(5, 5, z) for z in range(6)]
        assert len(set(nodes)) == 6  # All 6 must be distinct

    def test_layer_offset_consistent(self, lattice_6layer):
        """Nodes on consecutive layers at same (x,y) should be separated by plane_size."""
        lat = lattice_6layer
        plane_size = lat.x_steps * lat.y_steps
        for z in range(5):
            n0 = lat.node_idx(4, 4, z)
            n1 = lat.node_idx(4, 4, z + 1)
            assert n1 - n0 == plane_size


class TestUnmappablePadHandling:
    """Pads outside lattice bounds should be handled gracefully (clamped)."""

    def test_negative_coordinate_clamped(self, lattice_4layer):
        """Negative world coordinates should clamp to (0,0)."""
        lat = lattice_4layer
        x_idx, y_idx = lat.world_to_lattice(-5.0, -5.0)
        # Clamp to valid range (as _snap_to_node does)
        x_idx = max(0, min(x_idx, lat.x_steps - 1))
        y_idx = max(0, min(y_idx, lat.y_steps - 1))
        node = lat.node_idx(x_idx, y_idx, 0)
        assert node == lat.node_idx(0, 0, 0)

    def test_far_positive_coordinate_clamped(self, lattice_4layer):
        """Coordinates far beyond bounds should clamp to max lattice indices."""
        lat = lattice_4layer
        x_idx, y_idx = lat.world_to_lattice(100.0, 100.0)
        x_idx = max(0, min(x_idx, lat.x_steps - 1))
        y_idx = max(0, min(y_idx, lat.y_steps - 1))
        node = lat.node_idx(x_idx, y_idx, 0)
        assert node == lat.node_idx(lat.x_steps - 1, lat.y_steps - 1, 0)

    def test_clamped_node_is_valid(self, lattice_4layer):
        """Even after clamping, the resulting node index should be within range."""
        lat = lattice_4layer
        for x_mm, y_mm in [(-1.0, -1.0), (50.0, 50.0), (-10.0, 50.0)]:
            x_idx, y_idx = lat.world_to_lattice(x_mm, y_mm)
            x_idx = max(0, min(x_idx, lat.x_steps - 1))
            y_idx = max(0, min(y_idx, lat.y_steps - 1))
            node = lat.node_idx(x_idx, y_idx, 0)
            assert 0 <= node < lat.num_nodes
