"""Tests for real_global_grid - ported from inline tests."""
import pytest
import numpy as np

from orthoroute.algorithms.manhattan.real_global_grid import (
    GridShape, gid, xyz_from_gid, neighbors_for_gid,
    validate_path_bounds, validate_edges_from_path,
    RadixHeap, EpochVisited, VersionedCosts, DeferQueue,
)


class TestGidRoundtrip:
    """Verify (layer, x, y) <-> gid conversions."""

    def test_origin(self):
        """Origin (0,0,0) maps to gid 0."""
        shape = GridShape(NL=4, NX=100, NY=80)
        assert gid(shape, 0, 0, 0) == 0

    def test_corner(self):
        """Max corner round-trips correctly."""
        shape = GridShape(NL=4, NX=100, NY=80)
        g = gid(shape, 3, 99, 79)
        l, x, y = xyz_from_gid(shape, g)
        assert (l, x, y) == (3, 99, 79)

    def test_middle(self):
        """Middle point round-trips correctly."""
        shape = GridShape(NL=4, NX=100, NY=80)
        g = gid(shape, 1, 50, 40)
        l, x, y = xyz_from_gid(shape, g)
        assert (l, x, y) == (1, 50, 40)

    def test_all_corners(self):
        """All corners of a small grid round-trip."""
        shape = GridShape(NL=2, NX=5, NY=5)
        for layer in [0, 1]:
            for x in [0, 4]:
                for y in [0, 4]:
                    g = gid(shape, layer, x, y)
                    assert xyz_from_gid(shape, g) == (layer, x, y)


def test_grid_shape_total_nodes():
    """total_nodes = NL * NX * NY."""
    shape = GridShape(NL=4, NX=100, NY=80)
    assert shape.total_nodes == 4 * 100 * 80


def test_grid_shape_xy():
    """XY = NX * NY."""
    shape = GridShape(NL=2, NX=10, NY=10)
    assert shape.XY == 100


def test_grid_shape_positive_assertion():
    """GridShape rejects zero or negative dimensions."""
    with pytest.raises(AssertionError):
        GridShape(NL=0, NX=10, NY=10)


def test_neighbors_center_track_count():
    """Center node on layer 1 of 4-layer grid has 4 track neighbors."""
    shape = GridShape(NL=4, NX=10, NY=10)
    transitions = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    center = gid(shape, 1, 5, 5)
    nbrs = neighbors_for_gid(shape, center, transitions, include_vias=True)
    track = [n for n in nbrs if not n[2]]
    assert len(track) == 4


def test_neighbors_center_via_count():
    """Layer-1 node has 2 via neighbors (layers 0 and 2)."""
    shape = GridShape(NL=4, NX=10, NY=10)
    transitions = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    center = gid(shape, 1, 5, 5)
    nbrs = neighbors_for_gid(shape, center, transitions, include_vias=True)
    vias = [n for n in nbrs if n[2]]
    assert len(vias) == 2


def test_neighbors_via_targets_layers():
    """Via neighbors target the correct layers."""
    shape = GridShape(NL=4, NX=10, NY=10)
    transitions = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    center = gid(shape, 1, 5, 5)
    nbrs = neighbors_for_gid(shape, center, transitions, include_vias=True)
    via_layers = {xyz_from_gid(shape, n[0])[0] for n in nbrs if n[2]}
    assert via_layers == {0, 2}


def test_neighbors_corner_fewer_track_neighbors():
    """Corner node (0,0) has only 2 track neighbors."""
    shape = GridShape(NL=2, NX=10, NY=10)
    transitions = {0: [1], 1: [0]}
    corner = gid(shape, 0, 0, 0)
    nbrs = neighbors_for_gid(shape, corner, transitions)
    track = [n for n in nbrs if not n[2]]
    assert len(track) == 2


def test_neighbors_no_vias():
    """When include_vias=False, no via neighbors are returned."""
    shape = GridShape(NL=4, NX=10, NY=10)
    transitions = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    center = gid(shape, 1, 5, 5)
    nbrs = neighbors_for_gid(shape, center, transitions, include_vias=False)
    vias = [n for n in nbrs if n[2]]
    assert len(vias) == 0


def test_path_bounds_valid():
    """Valid path passes bounds check."""
    shape = GridShape(NL=2, NX=10, NY=10)
    path = np.array([gid(shape, 0, 0, 0), gid(shape, 0, 1, 0)])
    assert validate_path_bounds(shape, path, "test") is True


def test_path_bounds_oob():
    """Path with out-of-bounds gid fails validation."""
    shape = GridShape(NL=2, NX=10, NY=10)
    oob_path = np.array([0, shape.total_nodes + 100])
    assert validate_path_bounds(shape, oob_path, "test") is False


def test_path_bounds_empty():
    """Empty path fails validation."""
    shape = GridShape(NL=2, NX=10, NY=10)
    assert validate_path_bounds(shape, np.array([]), "test") is False


def test_edges_from_valid_path():
    """Valid path produces correct number of edges."""
    path = np.array([0, 1, 2, 3])
    edges = validate_edges_from_path(path, "test")
    assert edges.shape == (3, 2)


def test_edges_from_short_path():
    """Path with < 2 nodes raises ValueError."""
    with pytest.raises(ValueError):
        validate_edges_from_path(np.array([42]), "test")


def test_edges_from_path_zero_length_edge():
    """Path with duplicate adjacent gids raises ValueError."""
    with pytest.raises(ValueError):
        validate_edges_from_path(np.array([0, 1, 1, 2]), "test")


def test_radix_heap_extraction_order():
    """Items extracted in ascending key order."""
    heap = RadixHeap()
    for key, val in [(10, 'ten'), (5, 'five'), (15, 'fifteen'), (1, 'one')]:
        heap.push(key, val)
    extracted = []
    while not heap.empty():
        extracted.append(heap.pop())
    assert [e[0] for e in extracted] == [1, 5, 10, 15]


def test_radix_heap_empty_pop_raises():
    """Popping from empty heap raises IndexError."""
    heap = RadixHeap()
    with pytest.raises(IndexError):
        heap.pop()


def test_radix_heap_size_tracking():
    """Heap size updates correctly on push/pop."""
    heap = RadixHeap()
    assert heap.size == 0
    heap.push(5, 'a')
    heap.push(3, 'b')
    assert heap.size == 2
    heap.pop()
    assert heap.size == 1


def test_radix_heap_single_element():
    """Single element can be pushed and popped."""
    heap = RadixHeap()
    heap.push(42, 'answer')
    key, val = heap.pop()
    assert key == 42
    assert val == 'answer'
    assert heap.empty()


def test_epoch_visited_mark_and_check():
    """Marked node is recognized in current epoch."""
    ev = EpochVisited(100)
    ev.mark(5)
    assert ev.is_marked(5) is True


def test_epoch_visited_new_epoch_clears():
    """New epoch makes previously-marked nodes unvisited."""
    ev = EpochVisited(100)
    ev.mark(5)
    ev.new_epoch()
    assert ev.is_marked(5) is False


def test_epoch_visited_oob_safe():
    """Out-of-bounds node is never marked."""
    ev = EpochVisited(10)
    ev.mark(20)
    assert ev.is_marked(20) is False


def test_versioned_costs_initial_version():
    """Initial version is 0."""
    vc = VersionedCosts()
    assert vc.version == 0


def test_versioned_costs_increment():
    """Version increments by 1."""
    vc = VersionedCosts()
    new = vc.increment_version()
    assert new == 1


def test_versioned_costs_edge_cost_no_pressure():
    """Edge cost with no pressure equals base cost."""
    vc = VersionedCosts()
    assert vc.get_edge_cost(0, 1, 1) == 1.0


def test_versioned_costs_add_path_usage():
    """Path usage increments edge counters."""
    vc = VersionedCosts()
    vc.add_path_usage([0, 1, 2])
    assert vc.edge_usage[(0, 1)] == 1
    assert vc.edge_usage[(1, 2)] == 1


def test_defer_queue_add_and_batch():
    """Failed nets appear in batch output."""
    dq = DeferQueue()
    dq.add_failed_net("net1", 100)
    dq.add_failed_net("net2", 50)
    batch = dq.get_next_batch(10)
    assert "net1" in batch
    assert "net2" in batch


def test_defer_queue_remove_successful():
    """Successful net is removed from queue."""
    dq = DeferQueue()
    dq.add_failed_net("net1", 100)
    dq.remove_successful("net1")
    assert dq.get_next_batch(10) == []


def test_defer_queue_priority_order():
    """Nets with more failures appear first."""
    dq = DeferQueue()
    dq.add_failed_net("net1", 100)
    dq.add_failed_net("net2", 50)
    dq.add_failed_net("net1", 100)
    batch = dq.get_next_batch(1)
    assert batch[0] == "net1"
