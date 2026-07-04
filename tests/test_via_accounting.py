"""Via pooling / accounting tests — column usage, hard-block, segment capacity."""
import pytest
import numpy as np

from orthoroute.algorithms.manhattan.pathfinder.via_kernels import (
    hard_block_via_edges_cpu,
    apply_via_penalties_cpu,
    detect_barrel_conflicts_cpu,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def via_grid_5x5():
    """5×5 grid with column capacity=2 and 3 Z-segments.

    Returns dict with all arrays needed for via kernel tests.
    """
    Nx, Ny, segZ = 5, 5, 3
    via_col_use = np.zeros((Nx, Ny), dtype=np.int32)
    via_col_cap = np.full((Nx, Ny), 2, dtype=np.int32)
    via_seg_use = np.zeros((Nx, Ny, segZ), dtype=np.int32)
    via_seg_cap = np.full((Nx, Ny, segZ), 1, dtype=np.int32)
    return {
        'Nx': Nx, 'Ny': Ny, 'segZ': segZ,
        'col_use': via_col_use, 'col_cap': via_col_cap,
        'seg_use': via_seg_use, 'seg_cap': via_seg_cap,
    }


def _make_via_metadata(edge_indices, xy_coords, z_lo, z_hi):
    """Helper to construct via metadata dict."""
    return {
        'indices': np.array(edge_indices, dtype=np.int32),
        'xy_coords': np.array(xy_coords, dtype=np.int32).reshape(-1, 2),
        'z_lo': np.array(z_lo, dtype=np.int32),
        'z_hi': np.array(z_hi, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Column usage & hard-block
# ---------------------------------------------------------------------------

class TestColumnUsage:
    def test_column_usage_increment(self, via_grid_5x5):
        """Committing a via at (2,3) should increase column usage at that coordinate."""
        g = via_grid_5x5
        assert g['col_use'][2, 3] == 0
        g['col_use'][2, 3] += 1
        assert g['col_use'][2, 3] == 1

    def test_column_hard_block(self, via_grid_5x5):
        """When column usage >= capacity, hard_block_via_edges_cpu sets cost to INFINITY."""
        g = via_grid_5x5
        # Saturate column (2,3) to capacity
        g['col_use'][2, 3] = g['col_cap'][2, 3]  # usage == capacity → blocked

        # Create a via edge at (2,3) spanning z=1..3
        total_cost = np.ones(10, dtype=np.float32)
        meta = _make_via_metadata(
            edge_indices=[5],
            xy_coords=[[2, 3]],
            z_lo=[1],
            z_hi=[3],
        )
        blocked = hard_block_via_edges_cpu(
            meta, g['col_use'], g['col_cap'],
            g['seg_use'], g['seg_cap'],
            total_cost, g['segZ'],
        )
        assert blocked == 1
        assert np.isinf(total_cost[5]), "Blocked via edge should have INF cost"

    def test_unblocked_via_keeps_cost(self, via_grid_5x5):
        """A via edge below capacity should retain its original cost."""
        g = via_grid_5x5
        g['col_use'][1, 1] = 0  # well below capacity=2
        total_cost = np.ones(10, dtype=np.float32) * 3.0
        meta = _make_via_metadata(
            edge_indices=[2],
            xy_coords=[[1, 1]],
            z_lo=[1],
            z_hi=[2],
        )
        blocked = hard_block_via_edges_cpu(
            meta, g['col_use'], g['col_cap'],
            g['seg_use'], g['seg_cap'],
            total_cost, g['segZ'],
        )
        assert blocked == 0
        assert total_cost[2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Segment capacity
# ---------------------------------------------------------------------------

class TestSegmentCapacity:
    def test_segment_capacity_blocks(self, via_grid_5x5):
        """Segment-level saturation should also trigger hard-block."""
        g = via_grid_5x5
        # Column is fine, but segment 0 at (3,3) is saturated
        g['seg_use'][3, 3, 0] = g['seg_cap'][3, 3, 0]  # segment full
        total_cost = np.ones(10, dtype=np.float32)
        meta = _make_via_metadata(
            edge_indices=[7],
            xy_coords=[[3, 3]],
            z_lo=[1],  # z_lo=1 maps to seg_idx=0
            z_hi=[2],
        )
        blocked = hard_block_via_edges_cpu(
            meta, g['col_use'], g['col_cap'],
            g['seg_use'], g['seg_cap'],
            total_cost, g['segZ'],
        )
        assert blocked == 1
        assert np.isinf(total_cost[7])


# ---------------------------------------------------------------------------
# Via cost pooling penalty
# ---------------------------------------------------------------------------

class TestViaCostPoolingPenalty:
    def test_via_cost_pooling_penalty(self, via_grid_5x5):
        """Partially used columns should get a congestion penalty added to cost."""
        g = via_grid_5x5
        Nx, Ny = g['Nx'], g['Ny']

        via_col_pres = np.zeros((Nx, Ny), dtype=np.float32)
        via_col_pres[2, 2] = 1.5  # some congestion

        via_seg_pres = np.zeros((Nx, Ny, g['segZ']), dtype=np.float32)

        total_cost = np.ones(10, dtype=np.float32)
        meta = _make_via_metadata(
            edge_indices=[4],
            xy_coords=[[2, 2]],
            z_lo=[1],
            z_hi=[3],
        )
        penalties = apply_via_penalties_cpu(
            meta, via_col_pres, via_seg_pres,
            col_weight=2.0, seg_weight=1.0,
            total_cost=total_cost, segZ=g['segZ'],
        )
        assert penalties == 1
        assert total_cost[4] > 1.0, "Cost should have increased from penalty"
        # Expected: 1.0 + 1.5*2.0 = 4.0
        assert total_cost[4] == pytest.approx(1.0 + 1.5 * 2.0)

    def test_zero_congestion_no_penalty(self, via_grid_5x5):
        """Zero congestion should yield zero penalty."""
        g = via_grid_5x5
        Nx, Ny = g['Nx'], g['Ny']
        via_col_pres = np.zeros((Nx, Ny), dtype=np.float32)
        via_seg_pres = np.zeros((Nx, Ny, g['segZ']), dtype=np.float32)
        total_cost = np.ones(10, dtype=np.float32) * 5.0
        meta = _make_via_metadata(
            edge_indices=[0],
            xy_coords=[[0, 0]],
            z_lo=[1],
            z_hi=[2],
        )
        penalties = apply_via_penalties_cpu(
            meta, via_col_pres, via_seg_pres,
            col_weight=2.0, seg_weight=1.0,
            total_cost=total_cost, segZ=g['segZ'],
        )
        assert penalties == 0
        assert total_cost[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Barrel conflict detection
# ---------------------------------------------------------------------------

class TestBarrelConflictDetection:
    def test_barrel_conflict_detected(self):
        """Edges touching nodes owned by a different net should be flagged."""
        # 5-edge scenario; edge 2 touches a node owned by net 99
        edge_indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        edge_net_ids = np.array([1, 1, 1, 1, 1], dtype=np.int32)
        # edge_src_map: edge_idx → src node
        edge_src_map = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        # graph_indices: edge_idx → dst node
        graph_indices = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        # node_owner: node → owner net (-1 = free)
        node_owner = np.full(6, -1, dtype=np.int32)
        node_owner[3] = 99  # node 3 is owned by net 99

        conflicts = detect_barrel_conflicts_cpu(
            edge_indices, edge_net_ids, edge_src_map, graph_indices, node_owner
        )
        # Edge 2 (src=2, dst=3) touches node 3 owned by net 99 → conflict
        assert conflicts >= 1

    def test_barrel_no_conflict_own_net(self):
        """Edges touching nodes owned by the same net should NOT be flagged."""
        edge_indices = np.array([0, 1], dtype=np.int32)
        edge_net_ids = np.array([5, 5], dtype=np.int32)
        edge_src_map = np.array([0, 1], dtype=np.int32)
        graph_indices = np.array([1, 2], dtype=np.int32)
        node_owner = np.full(3, -1, dtype=np.int32)
        node_owner[1] = 5  # same net
        node_owner[2] = 5  # same net

        conflicts = detect_barrel_conflicts_cpu(
            edge_indices, edge_net_ids, edge_src_map, graph_indices, node_owner
        )
        assert conflicts == 0
