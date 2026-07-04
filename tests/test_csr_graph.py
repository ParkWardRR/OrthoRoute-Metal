"""CSR matrix integrity tests — validates indptr monotonicity, index bounds, symmetry."""
import pytest
import numpy as np

from orthoroute.algorithms.manhattan.unified_pathfinder import CSRGraph, Lattice3D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lattice_and_graph():
    """Build a 5×5×4 lattice and its CSR graph on CPU."""
    bounds = (0.0, 0.0, 1.6, 1.6)
    lat = Lattice3D(bounds, pitch=0.4, layers=4)
    graph = lat.build_graph(via_cost=1.0, use_gpu=False)
    return lat, graph


@pytest.fixture
def manual_csr():
    """Manually built CSR for a 5-node directed line: 0↔1↔2↔3↔4."""
    g = CSRGraph(use_gpu=False)
    for u in range(4):
        g.add_edge(u, u + 1, 1.0)
        g.add_edge(u + 1, u, 1.0)
    g.finalize(num_nodes=5)
    return g


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

class TestIndptr:
    def test_indptr_monotonic(self, lattice_and_graph):
        """indptr must be monotonically non-decreasing."""
        _, g = lattice_and_graph
        indptr = g.indptr if isinstance(g.indptr, np.ndarray) else g.indptr.get()
        diffs = np.diff(indptr)
        assert np.all(diffs >= 0), "indptr is not monotonically non-decreasing"

    def test_indptr_length(self, lattice_and_graph):
        """len(indptr) must equal num_nodes + 1."""
        lat, g = lattice_and_graph
        indptr = g.indptr if isinstance(g.indptr, np.ndarray) else g.indptr.get()
        assert len(indptr) == lat.num_nodes + 1

    def test_indptr_starts_at_zero(self, manual_csr):
        """indptr[0] must be 0."""
        indptr = manual_csr.indptr
        assert indptr[0] == 0


class TestIndices:
    def test_indices_in_range(self, lattice_and_graph):
        """All column indices must be in [0, num_nodes)."""
        lat, g = lattice_and_graph
        indices = g.indices if isinstance(g.indices, np.ndarray) else g.indices.get()
        assert np.all(indices >= 0), "Found negative column index"
        assert np.all(indices < lat.num_nodes), "Found column index >= num_nodes"

    def test_no_self_loops(self, lattice_and_graph):
        """No edge should connect a node to itself."""
        _, g = lattice_and_graph
        indptr = g.indptr if isinstance(g.indptr, np.ndarray) else g.indptr.get()
        indices = g.indices if isinstance(g.indices, np.ndarray) else g.indices.get()
        for u in range(len(indptr) - 1):
            neighbors = indices[int(indptr[u]):int(indptr[u + 1])]
            assert u not in neighbors, f"Self-loop at node {u}"


class TestWeights:
    def test_weights_positive(self, lattice_and_graph):
        """All base edge costs must be > 0."""
        _, g = lattice_and_graph
        costs = g.base_costs if isinstance(g.base_costs, np.ndarray) else g.base_costs.get()
        assert np.all(costs > 0), "Found non-positive edge weight"

    def test_weights_finite(self, lattice_and_graph):
        """All base edge costs must be finite."""
        _, g = lattice_and_graph
        costs = g.base_costs if isinstance(g.base_costs, np.ndarray) else g.base_costs.get()
        assert np.all(np.isfinite(costs)), "Found non-finite edge weight"


# ---------------------------------------------------------------------------
# Symmetry & consistency
# ---------------------------------------------------------------------------

class TestSymmetry:
    def test_symmetric_edges(self, lattice_and_graph):
        """For every edge (u,v), the reverse edge (v,u) must exist.

        Both lateral and via edges are added bi-directionally by Lattice3D.build_graph.
        """
        _, g = lattice_and_graph
        indptr = g.indptr if isinstance(g.indptr, np.ndarray) else g.indptr.get()
        indices = g.indices if isinstance(g.indices, np.ndarray) else g.indices.get()
        num_nodes = len(indptr) - 1

        # Build adjacency sets for fast lookup (sample 500 random nodes to keep test fast)
        rng = np.random.default_rng(42)
        sample = rng.choice(num_nodes, size=min(500, num_nodes), replace=False)

        for u in sample:
            u_start, u_end = int(indptr[u]), int(indptr[u + 1])
            for ei in range(u_start, u_end):
                v = int(indices[ei])
                # Check reverse: v must have u in its neighbor list
                v_start, v_end = int(indptr[v]), int(indptr[v + 1])
                v_neighbors = indices[v_start:v_end]
                assert u in v_neighbors, (
                    f"Edge ({u},{v}) exists but reverse ({v},{u}) is missing"
                )


class TestEdgeCountConsistency:
    def test_edge_count_consistent(self, lattice_and_graph):
        """indptr[-1] must equal len(indices) and len(base_costs)."""
        _, g = lattice_and_graph
        indptr = g.indptr if isinstance(g.indptr, np.ndarray) else g.indptr.get()
        indices = g.indices if isinstance(g.indices, np.ndarray) else g.indices.get()
        costs = g.base_costs if isinstance(g.base_costs, np.ndarray) else g.base_costs.get()

        assert int(indptr[-1]) == len(indices), (
            f"indptr[-1]={int(indptr[-1])} != len(indices)={len(indices)}"
        )
        assert int(indptr[-1]) == len(costs), (
            f"indptr[-1]={int(indptr[-1])} != len(weights)={len(costs)}"
        )

    def test_manual_csr_edge_count(self, manual_csr):
        """5-node line graph should have 8 directed edges."""
        g = manual_csr
        assert int(g.indptr[-1]) == 8
