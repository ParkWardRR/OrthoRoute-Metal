"""Tests for base routing grid operations."""
import pytest

from orthoroute.domain.models.board import Bounds, Coordinate
from orthoroute.algorithms.base.grid import RoutingGrid, GridCell, CellState


def test_grid_cell_default_state():
    """New grid cell starts EMPTY."""
    cell = GridCell(0, 0, 0)
    assert cell.state == CellState.EMPTY


def test_grid_cell_set_routed():
    """Setting a cell as routed changes state and records net."""
    cell = GridCell(1, 2, 0)
    cell.set_routed("N1")
    assert cell.state == CellState.ROUTED
    assert cell.net_id == "N1"


def test_grid_cell_accessible_by_own_net():
    """Routed cell is accessible by the net that routed it."""
    cell = GridCell(0, 0, 0)
    cell.set_routed("N1")
    assert cell.is_accessible_by_net("N1") is True


def test_grid_cell_not_accessible_by_other_net():
    """Routed cell is NOT accessible by a different net."""
    cell = GridCell(0, 0, 0)
    cell.set_routed("N1")
    assert cell.is_accessible_by_net("N2") is False


def test_grid_cell_empty_accessible_by_any_net():
    """Empty cell is accessible by any net."""
    cell = GridCell(0, 0, 0)
    assert cell.is_accessible_by_net("any") is True


def test_grid_cell_obstacle_with_accessibility():
    """Obstacle with accessibility mask only allows listed nets."""
    cell = GridCell(0, 0, 0)
    cell.set_obstacle({"N1"})
    assert cell.is_accessible_by_net("N1") is True
    assert cell.is_accessible_by_net("N2") is False


def test_grid_cell_blocked_not_accessible():
    """Blocked cell is not accessible by any net."""
    cell = GridCell(0, 0, 0)
    cell.state = CellState.BLOCKED
    assert cell.is_accessible_by_net("N1") is False


def test_grid_cell_clear():
    """Clearing a cell resets it to EMPTY."""
    cell = GridCell(0, 0, 0)
    cell.set_routed("N1")
    cell.clear()
    assert cell.state == CellState.EMPTY
    assert cell.net_id is None


def test_grid_dimensions():
    """Grid dimensions computed from bounds and resolution."""
    bounds = Bounds(0.0, 0.0, 10.0, 5.0)
    grid = RoutingGrid(bounds, ["F.Cu", "B.Cu"], resolution=0.5)
    assert grid.width == 20
    assert grid.height == 10
    assert grid.layer_count == 2


def test_grid_world_to_grid():
    """World-to-grid conversion maps coordinates to cell indices."""
    bounds = Bounds(0.0, 0.0, 10.0, 10.0)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    gx, gy = grid.world_to_grid(5.0, 5.0)
    assert gx == 5
    assert gy == 5


def test_grid_world_to_grid_clamped():
    """Out-of-bounds world coords are clamped to grid edges."""
    bounds = Bounds(0.0, 0.0, 10.0, 10.0)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    gx, gy = grid.world_to_grid(-5.0, 20.0)
    assert gx == 0
    assert gy == grid.height - 1


def test_grid_to_world():
    """Grid-to-world returns center of cell."""
    bounds = Bounds(0.0, 0.0, 10.0, 10.0)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    coord = grid.grid_to_world(0, 0)
    assert coord.x == pytest.approx(0.5)
    assert coord.y == pytest.approx(0.5)


def test_grid_valid_position():
    """Valid positions are within grid bounds."""
    bounds = Bounds(0.0, 0.0, 5.0, 5.0)
    grid = RoutingGrid(bounds, ["F.Cu", "B.Cu"], resolution=1.0)
    assert grid.is_valid_position(0, 0, 0) is True
    assert grid.is_valid_position(4, 4, 1) is True


def test_grid_invalid_position():
    """Invalid positions are outside grid bounds."""
    bounds = Bounds(0.0, 0.0, 5.0, 5.0)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    assert grid.is_valid_position(-1, 0, 0) is False
    assert grid.is_valid_position(0, 0, 1) is False


def test_grid_layer_name_to_index():
    """Layer name maps to correct index."""
    bounds = Bounds(0, 0, 5, 5)
    grid = RoutingGrid(bounds, ["F.Cu", "In1.Cu", "B.Cu"], resolution=1.0)
    assert grid.get_layer_index("In1.Cu") == 1


def test_grid_layer_unknown_raises():
    """Unknown layer name raises ValueError."""
    bounds = Bounds(0, 0, 5, 5)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    with pytest.raises(ValueError):
        grid.get_layer_index("Unknown")


def test_grid_cell_state_default_empty():
    """All cells start as EMPTY."""
    bounds = Bounds(0, 0, 2, 2)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    assert grid.get_cell_state(0, 0, 0) == CellState.EMPTY


def test_grid_set_and_get_cell_state():
    """Set cell state and read it back."""
    bounds = Bounds(0, 0, 5, 5)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    grid.set_cell_state(2, 2, 0, CellState.OBSTACLE)
    assert grid.get_cell_state(2, 2, 0) == CellState.OBSTACLE


def test_grid_invalid_position_returns_blocked():
    """Getting state of invalid position returns BLOCKED."""
    bounds = Bounds(0, 0, 5, 5)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    assert grid.get_cell_state(100, 100, 0) == CellState.BLOCKED


def test_grid_no_diagonal_adjacency():
    """Manhattan routing: grid is 3D (layer, y, x) confirming discrete structure."""
    bounds = Bounds(0, 0, 3, 3)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    assert grid.state_array.ndim == 3
    assert grid.state_array.shape == (1, 3, 3)


def test_grid_statistics():
    """Grid statistics dict contains expected keys."""
    bounds = Bounds(0, 0, 5, 5)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    stats = grid.get_statistics()
    assert 'total_cells' in stats
    assert stats['empty_cells'] == stats['total_cells']


def test_grid_memory_usage():
    """Grid memory usage returns a dict with expected keys."""
    bounds = Bounds(0, 0, 5, 5)
    grid = RoutingGrid(bounds, ["F.Cu"], resolution=1.0)
    mem = grid.get_memory_usage()
    assert 'total_memory' in mem
    assert mem['using_gpu'] is False
