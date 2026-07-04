"""Tests for spatial hash grid."""
import pytest

from orthoroute.algorithms.manhattan.pathfinder.spatial_hash import SpatialHash


def test_spatial_hash_creation():
    """SpatialHash can be created with a cell size."""
    sh = SpatialHash(cell_size=1.0)
    assert sh.cell_size == 1.0


def test_spatial_hash_insert_and_query():
    """Inserted segment appears in query results."""
    sh = SpatialHash(cell_size=1.0)
    sh.insert_segment((0.0, 0.0), (5.0, 0.0), radius=0.5, tag="net1")
    candidates = sh.query_segment((2.0, 0.0), (3.0, 0.0), radius=0.5)
    assert len(candidates) > 0


def test_spatial_hash_query_returns_tag():
    """Queried candidates have the correct tag."""
    sh = SpatialHash(cell_size=1.0)
    sh.insert_segment((0.0, 0.0), (5.0, 0.0), radius=0.5, tag="net1")
    candidates = sh.query_segment((2.0, 0.0), (3.0, 0.0), radius=0.5)
    tags = [c.tag for c in candidates]
    assert "net1" in tags


def test_spatial_hash_empty_query():
    """Query on empty grid returns no candidates."""
    sh = SpatialHash(cell_size=1.0)
    candidates = sh.query_segment((0.0, 0.0), (1.0, 0.0), radius=0.5)
    assert len(candidates) == 0


def test_spatial_hash_distant_segment_not_found():
    """Segment far away is not found by local query."""
    sh = SpatialHash(cell_size=1.0)
    sh.insert_segment((100.0, 100.0), (105.0, 100.0), radius=0.5, tag="far")
    candidates = sh.query_segment((0.0, 0.0), (1.0, 0.0), radius=0.5)
    assert len(candidates) == 0


def test_spatial_hash_multiple_segments():
    """Multiple segments in same area are all returned."""
    sh = SpatialHash(cell_size=2.0)
    sh.insert_segment((0.0, 0.0), (3.0, 0.0), radius=0.3, tag="a")
    sh.insert_segment((1.0, 0.0), (4.0, 0.0), radius=0.3, tag="b")
    candidates = sh.query_segment((1.5, 0.0), (2.5, 0.0), radius=0.3)
    tags = {c.tag for c in candidates}
    assert "a" in tags
    assert "b" in tags


def test_spatial_hash_nearest_distance():
    """nearest_distance returns distance to nearest foreign segment."""
    sh = SpatialHash(cell_size=1.0)
    sh.insert_segment((5.0, 0.0), (6.0, 0.0), radius=0.2, tag="net2")
    dist = sh.nearest_distance((0.0, 0.0), (1.0, 0.0), exclude_net="net1", cap=10.0)
    assert dist is not None
    assert dist > 0


def test_spatial_hash_nearest_distance_exclude_own_net():
    """nearest_distance excludes segments from the same net."""
    sh = SpatialHash(cell_size=1.0)
    sh.insert_segment((0.0, 0.0), (1.0, 0.0), radius=0.2, tag="net1")
    dist = sh.nearest_distance((0.0, 0.0), (1.0, 0.0), exclude_net="net1", cap=10.0)
    assert dist is None


def test_spatial_hash_bounding_box_cells():
    """_get_cells_for_segment returns correct cell set for a segment."""
    sh = SpatialHash(cell_size=1.0)
    cells = sh._get_cells_for_segment((0.5, 0.5), (2.5, 0.5), radius=0.0)
    assert (0, 0) in cells
    assert (1, 0) in cells
    assert (2, 0) in cells


def test_spatial_hash_hash_point():
    """_hash_point maps coordinates to integer cell IDs."""
    sh = SpatialHash(cell_size=2.0)
    cell = sh._hash_point(3.0, 5.0)
    assert cell == (1, 2)
