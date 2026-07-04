"""Tests for KiCadGeometry – coordinate conversion and layer direction rules.

Covers: creation with various parameters, lattice/world coordinate round-trips,
node index uniqueness and range, layer direction alternation, edge validity
checks (horizontal, vertical, via, diagonal), and pitch alignment.
"""
import pytest

from orthoroute.algorithms.manhattan.pathfinder.kicad_geometry import KiCadGeometry


# ---------------------------------------------------------------------------
# Creation & grid dimensions
# ---------------------------------------------------------------------------


class TestKiCadGeometryCreation:
    """Tests for KiCadGeometry instantiation and grid sizing."""

    def test_geometry_creation_defaults(self):
        """KiCadGeometry with pitch 0.4 and 6 layers creates a valid object."""
        geo = KiCadGeometry((0, 0, 40, 40), 0.4, 6)
        assert geo.pitch == pytest.approx(0.4)
        assert geo.layer_count == 6
        assert geo.x_steps > 0
        assert geo.y_steps > 0

    def test_geometry_creation_2layer(self):
        """Default 2-layer geometry creates correctly."""
        geo = KiCadGeometry((0, 0, 10, 10), 0.4, 2)
        assert geo.layer_count == 2
        assert len(geo.layer_directions) == 2

    def test_grid_dimensions(self):
        """x_steps and y_steps match (grid_max - grid_min)/pitch + 1."""
        geo = KiCadGeometry((0, 0, 4.0, 4.0), 0.4, 2)
        expected_x = int((geo.grid_max_x - geo.grid_min_x) / 0.4) + 1
        expected_y = int((geo.grid_max_y - geo.grid_min_y) / 0.4) + 1
        assert geo.x_steps == expected_x
        assert geo.y_steps == expected_y

    def test_grid_dimensions_exact(self):
        """For bounds (0,0,4,4) with pitch 0.4, x_steps and y_steps should be 11."""
        geo = KiCadGeometry((0, 0, 4.0, 4.0), 0.4, 2)
        assert geo.x_steps == 11
        assert geo.y_steps == 11

    def test_pitch_alignment(self):
        """grid_min_x/y are pitch-aligned (multiples of pitch)."""
        geo = KiCadGeometry((1.1, 2.3, 10.9, 15.7), 0.4, 2)
        # After round(val/pitch)*pitch, the result should be a multiple of pitch
        remainder_x = geo.grid_min_x % 0.4
        assert remainder_x < 1e-9 or abs(remainder_x - 0.4) < 1e-9
        remainder_y = geo.grid_min_y % 0.4
        assert remainder_y < 1e-9 or abs(remainder_y - 0.4) < 1e-9

    def test_different_pitch(self):
        """Create with pitch=0.2, verify approximately doubled x_steps/y_steps."""
        geo_04 = KiCadGeometry((0, 0, 4.0, 4.0), 0.4, 2)
        geo_02 = KiCadGeometry((0, 0, 4.0, 4.0), 0.2, 2)
        # With half pitch, approximately double the steps (2N-1 where N is original)
        assert geo_02.x_steps == pytest.approx(geo_04.x_steps * 2 - 1, abs=1)
        assert geo_02.y_steps == pytest.approx(geo_04.y_steps * 2 - 1, abs=1)

    def test_bounds_stored(self):
        """Original bounds are stored as attributes."""
        geo = KiCadGeometry((1.0, 2.0, 5.0, 8.0), 0.4, 2)
        assert geo.min_x == pytest.approx(1.0)
        assert geo.min_y == pytest.approx(2.0)
        assert geo.max_x == pytest.approx(5.0)
        assert geo.max_y == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


class TestCoordinateConversion:
    """Tests for lattice_to_world and world_to_lattice methods."""

    @pytest.fixture
    def geo(self):
        """Standard geometry for coordinate tests."""
        return KiCadGeometry((0, 0, 4.0, 4.0), 0.4, 2)

    def test_lattice_to_world_origin(self, geo):
        """lattice_to_world(0,0) returns (grid_min_x, grid_min_y)."""
        wx, wy = geo.lattice_to_world(0, 0)
        assert wx == pytest.approx(geo.grid_min_x)
        assert wy == pytest.approx(geo.grid_min_y)

    def test_lattice_to_world_offset(self, geo):
        """lattice_to_world(5,3) returns (grid_min_x + 5*pitch, grid_min_y + 3*pitch)."""
        wx, wy = geo.lattice_to_world(5, 3)
        assert wx == pytest.approx(geo.grid_min_x + 5 * 0.4)
        assert wy == pytest.approx(geo.grid_min_y + 3 * 0.4)

    def test_lattice_to_world_last_point(self, geo):
        """lattice_to_world at max index returns grid_max coordinates."""
        wx, wy = geo.lattice_to_world(geo.x_steps - 1, geo.y_steps - 1)
        assert wx == pytest.approx(geo.grid_max_x)
        assert wy == pytest.approx(geo.grid_max_y)

    def test_world_to_lattice_roundtrip(self, geo):
        """world_to_lattice(lattice_to_world(x,y)) == (x,y) for all valid coords."""
        for x in range(geo.x_steps):
            for y in range(geo.y_steps):
                wx, wy = geo.lattice_to_world(x, y)
                rx, ry = geo.world_to_lattice(wx, wy)
                assert rx == x, f"x mismatch at ({x},{y})"
                assert ry == y, f"y mismatch at ({x},{y})"

    def test_world_to_lattice_snapping(self, geo):
        """world_to_lattice snaps to nearest grid point."""
        # 1.21 is close to 1.2 (index 3), 1.59 is close to 1.6 (index 4)
        lx, ly = geo.world_to_lattice(1.21, 1.59)
        assert lx == 3  # round(1.21/0.4) = round(3.025) = 3
        assert ly == 4  # round(1.59/0.4) = round(3.975) = 4

    def test_world_to_lattice_exact(self, geo):
        """world_to_lattice returns exact index for on-grid coordinates."""
        lx, ly = geo.world_to_lattice(0.8, 1.2)
        assert lx == 2
        assert ly == 3


# ---------------------------------------------------------------------------
# Node indexing
# ---------------------------------------------------------------------------


class TestNodeIndexing:
    """Tests for node_index and index_to_coords methods."""

    @pytest.fixture
    def geo(self):
        """Small geometry for index exhaustive tests."""
        return KiCadGeometry((0, 0, 2.0, 2.0), 0.4, 3)

    def test_node_index_uniqueness(self, geo):
        """All (x,y,z) combos in small grid produce unique indices."""
        indices = set()
        for z in range(geo.layer_count):
            for y in range(geo.y_steps):
                for x in range(geo.x_steps):
                    idx = geo.node_index(x, y, z)
                    assert idx not in indices, f"Duplicate index {idx} at ({x},{y},{z})"
                    indices.add(idx)

    def test_node_index_range(self, geo):
        """All indices fall in [0, x_steps*y_steps*layer_count)."""
        total = geo.x_steps * geo.y_steps * geo.layer_count
        for z in range(geo.layer_count):
            for y in range(geo.y_steps):
                for x in range(geo.x_steps):
                    idx = geo.node_index(x, y, z)
                    assert 0 <= idx < total

    def test_index_to_coords_roundtrip(self, geo):
        """index_to_coords(node_index(x,y,z)) == (x,y,z) for all valid coords."""
        for z in range(geo.layer_count):
            for y in range(geo.y_steps):
                for x in range(geo.x_steps):
                    idx = geo.node_index(x, y, z)
                    rx, ry, rz = geo.index_to_coords(idx)
                    assert (rx, ry, rz) == (x, y, z)

    def test_node_index_origin(self, geo):
        """node_index(0,0,0) returns 0."""
        assert geo.node_index(0, 0, 0) == 0

    def test_node_index_layer_offset(self, geo):
        """node_index(0,0,1) returns x_steps*y_steps (start of layer 1)."""
        layer_size = geo.x_steps * geo.y_steps
        assert geo.node_index(0, 0, 1) == layer_size

    def test_node_index_total_count(self, geo):
        """Total unique indices matches x_steps * y_steps * layer_count."""
        indices = set()
        for z in range(geo.layer_count):
            for y in range(geo.y_steps):
                for x in range(geo.x_steps):
                    indices.add(geo.node_index(x, y, z))
        assert len(indices) == geo.x_steps * geo.y_steps * geo.layer_count


# ---------------------------------------------------------------------------
# Layer directions
# ---------------------------------------------------------------------------


class TestLayerDirections:
    """Tests for layer direction assignment."""

    @pytest.fixture
    def geo(self):
        """6-layer geometry for direction tests."""
        return KiCadGeometry((0, 0, 4.0, 4.0), 0.4, 6)

    def test_layer_direction_layer0_vertical(self, geo):
        """Layer 0 (F.Cu) is vertical."""
        assert geo.layer_directions[0] == "v"

    def test_layer_direction_odd_horizontal(self, geo):
        """Layer 1 is horizontal."""
        assert geo.layer_directions[1] == "h"

    def test_layer_direction_even_vertical(self, geo):
        """Layer 2 is vertical."""
        assert geo.layer_directions[2] == "v"

    def test_layer_direction_full_pattern(self, geo):
        """Full alternating pattern: v, h, v, h, v, h."""
        expected = ["v", "h", "v", "h", "v", "h"]
        assert geo.layer_directions == expected

    def test_layer_direction_2layer(self):
        """2-layer board: v, h."""
        geo = KiCadGeometry((0, 0, 4, 4), 0.4, 2)
        assert geo.layer_directions == ["v", "h"]

    def test_layer_direction_count_matches_layers(self, geo):
        """Number of direction entries matches layer_count."""
        assert len(geo.layer_directions) == geo.layer_count


# ---------------------------------------------------------------------------
# Edge validation
# ---------------------------------------------------------------------------


class TestEdgeValidation:
    """Tests for is_valid_edge method."""

    @pytest.fixture
    def geo(self):
        """6-layer geometry for edge validation tests."""
        return KiCadGeometry((0, 0, 4.0, 4.0), 0.4, 6)

    def test_is_valid_edge_horizontal_on_h_layer(self, geo):
        """Horizontal move on H-layer (layer 1) is valid."""
        assert geo.is_valid_edge(3, 3, 1, 4, 3, 1) is True

    def test_is_valid_edge_horizontal_on_v_layer(self, geo):
        """Horizontal move on V-layer (layer 0) is invalid."""
        assert geo.is_valid_edge(3, 3, 0, 4, 3, 0) is False

    def test_is_valid_edge_vertical_on_v_layer(self, geo):
        """Vertical move on V-layer (layer 0) is valid."""
        assert geo.is_valid_edge(3, 3, 0, 3, 4, 0) is True

    def test_is_valid_edge_vertical_on_h_layer(self, geo):
        """Vertical move on H-layer (layer 1) is invalid."""
        assert geo.is_valid_edge(3, 3, 1, 3, 4, 1) is False

    def test_is_valid_edge_via_same_xy(self, geo):
        """Layer change at same (x,y) is always valid."""
        assert geo.is_valid_edge(3, 3, 0, 3, 3, 1) is True

    def test_is_valid_edge_via_skip_layers(self, geo):
        """Layer change skipping layers is still valid (via)."""
        assert geo.is_valid_edge(3, 3, 2, 3, 3, 5) is True

    def test_is_valid_edge_via_different_xy(self, geo):
        """Via with different x,y is valid (code returns True for any layer change)."""
        assert geo.is_valid_edge(3, 3, 0, 5, 5, 1) is True

    def test_is_valid_edge_diagonal_invalid(self, geo):
        """Diagonal move (dx!=0 and dy!=0) is invalid on any layer."""
        assert geo.is_valid_edge(3, 3, 0, 4, 4, 0) is False
        assert geo.is_valid_edge(3, 3, 1, 4, 4, 1) is False

    def test_is_valid_edge_multi_step_horizontal_invalid(self, geo):
        """Multi-step horizontal move (|dx|>1) is invalid."""
        assert geo.is_valid_edge(3, 3, 1, 5, 3, 1) is False

    def test_is_valid_edge_multi_step_vertical_invalid(self, geo):
        """Multi-step vertical move (|dy|>1) is invalid."""
        assert geo.is_valid_edge(3, 3, 0, 3, 5, 0) is False

    def test_is_valid_edge_stationary_invalid(self, geo):
        """Staying at same position on same layer is invalid (no move)."""
        # dx=0, dy=0, same layer => not horizontal and not vertical
        assert geo.is_valid_edge(3, 3, 0, 3, 3, 0) is False

    def test_is_valid_edge_negative_direction_h_layer(self, geo):
        """Horizontal move in negative direction on H-layer is valid."""
        assert geo.is_valid_edge(4, 3, 1, 3, 3, 1) is True

    def test_is_valid_edge_negative_direction_v_layer(self, geo):
        """Vertical move in negative direction on V-layer is valid."""
        assert geo.is_valid_edge(3, 4, 0, 3, 3, 0) is True
