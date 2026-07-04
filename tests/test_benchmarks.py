"""
Performance Benchmark Tests

Timed benchmarks for core routing operations:
- Dijkstra on 10K-node grid (< 1 s)
- Dijkstra on 100K-node grid (< 5 s)
- Lattice3D construction for 100×100×6 (< 1 s)
- CSR finalization for 50K edges (< 1 s)
- ROI extraction timing (< 500 ms)
- EdgeAccountant cost update for 100K edges (< 100 ms)
"""

import time
import pytest
import numpy as np

from orthoroute.algorithms.manhattan.unified_pathfinder import (
    CSRGraph,
    EdgeAccountant,
    Lattice3D,
    SimpleDijkstra,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_grid_graph(width: int, height: int) -> CSRGraph:
    """Build a 2D grid graph with Manhattan connectivity.

    Total nodes = width * height.
    Each interior node has 4 neighbours; edges are bidirectional.
    """
    n = width * height
    graph = CSRGraph(use_gpu=False, edge_capacity=4 * n)

    for y in range(height):
        for x in range(width):
            node = y * width + x
            if x + 1 < width:
                right = y * width + (x + 1)
                graph.add_edge(node, right, 1.0)
                graph.add_edge(right, node, 1.0)
            if y + 1 < height:
                down = (y + 1) * width + x
                graph.add_edge(node, down, 1.0)
                graph.add_edge(down, node, 1.0)

    graph.finalize(num_nodes=n)
    return graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDijkstra10KNodes:
    """Benchmark: Dijkstra on 10K-node grid."""

    def test_dijkstra_10k_nodes_under_1s(self):
        """Build 100×100 grid (10K nodes), run SimpleDijkstra, verify < 1 s."""
        width, height = 100, 100
        n = width * height

        graph = build_grid_graph(width, height)
        dijk = SimpleDijkstra(graph, lattice=None)

        roi_nodes = np.arange(n, dtype=np.int32)
        g2r = np.arange(n, dtype=np.int32)

        t0 = time.time()
        path = dijk.find_path_roi(0, n - 1, graph.base_costs, roi_nodes, g2r)
        elapsed = time.time() - t0

        assert path is not None, "Dijkstra should find path on connected grid"
        assert path[0] == 0
        assert path[-1] == n - 1
        assert elapsed < 1.0, f"Dijkstra on 10K nodes took {elapsed:.2f}s (limit: 1s)"


@pytest.mark.slow
class TestDijkstra100KNodes:
    """Benchmark: Dijkstra on 100K-node grid."""

    def test_dijkstra_100k_nodes_under_5s(self):
        """Build ~100K-node grid (316×316), run SimpleDijkstra, verify < 5 s."""
        width, height = 316, 316
        n = width * height  # ≈ 99,856

        graph = build_grid_graph(width, height)
        dijk = SimpleDijkstra(graph, lattice=None)

        roi_nodes = np.arange(n, dtype=np.int32)
        g2r = np.arange(n, dtype=np.int32)

        t0 = time.time()
        path = dijk.find_path_roi(0, n - 1, graph.base_costs, roi_nodes, g2r)
        elapsed = time.time() - t0

        assert path is not None, "Dijkstra should find path on connected grid"
        assert path[0] == 0
        assert path[-1] == n - 1
        assert elapsed < 5.0, f"Dijkstra on 100K nodes took {elapsed:.2f}s (limit: 5s)"


@pytest.mark.slow
class TestLatticeBuild:
    """Benchmark: Lattice3D construction timing."""

    def test_lattice_build_100x100x6_under_1s(self):
        """Lattice3D(bounds=(0,0,40,40), pitch=0.4, layers=6) should build in < 1 s.

        This creates a 101×101×6 = 61,206 node lattice.
        """
        t0 = time.time()
        lattice = Lattice3D(bounds=(0, 0, 40, 40), pitch=0.4, layers=6)
        elapsed = time.time() - t0

        assert lattice.num_nodes > 0, "Lattice should have nodes"
        assert lattice.x_steps > 0
        assert lattice.y_steps > 0
        assert elapsed < 1.0, f"Lattice construction took {elapsed:.2f}s (limit: 1s)"


@pytest.mark.slow
class TestCSRFinalize:
    """Benchmark: CSR finalization for 50K edges."""

    def test_csr_finalize_under_1s(self):
        """Finalize a CSR graph with 50K edges in < 1 s."""
        n_nodes = 10000
        n_edges = 50000

        graph = CSRGraph(use_gpu=False, edge_capacity=n_edges)

        # Add random edges
        rng = np.random.RandomState(42)
        sources = rng.randint(0, n_nodes, size=n_edges)
        targets = rng.randint(0, n_nodes, size=n_edges)
        costs = rng.uniform(0.1, 2.0, size=n_edges).astype(np.float32)

        for i in range(n_edges):
            graph.add_edge(int(sources[i]), int(targets[i]), float(costs[i]))

        t0 = time.time()
        graph.finalize(num_nodes=n_nodes)
        elapsed = time.time() - t0

        assert len(graph.indices) == n_edges
        assert elapsed < 1.0, f"CSR finalize took {elapsed:.2f}s (limit: 1s)"


@pytest.mark.slow
class TestROIExtraction:
    """Benchmark: ROI extraction timing."""

    def test_roi_extraction_under_500ms(self):
        """Extract ROI from a lattice in < 500 ms."""
        lattice = Lattice3D(bounds=(0, 0, 40, 40), pitch=0.4, layers=6)

        # Simulate ROI: select nodes within central bounding box
        roi_x_min, roi_x_max = 20, 60  # lattice indices
        roi_y_min, roi_y_max = 20, 60

        t0 = time.time()
        roi_nodes = []
        for z in range(lattice.layers):
            for y in range(roi_y_min, min(roi_y_max, lattice.y_steps)):
                for x in range(roi_x_min, min(roi_x_max, lattice.x_steps)):
                    roi_nodes.append(lattice.node_idx(x, y, z))
        elapsed = time.time() - t0

        assert len(roi_nodes) > 0, "ROI should contain nodes"
        assert len(roi_nodes) < lattice.num_nodes, "ROI should be subset"
        assert elapsed < 0.5, f"ROI extraction took {elapsed:.3f}s (limit: 0.5s)"


@pytest.mark.slow
class TestEdgeAccountantUpdate:
    """Benchmark: EdgeAccountant cost update for 100K edges."""

    def test_edge_accountant_update_under_100ms(self):
        """Cost update for 100K edges should complete in < 100 ms."""
        n_edges = 100_000
        acct = EdgeAccountant(num_edges=n_edges, use_gpu=False)

        # Set up realistic state
        base_costs = np.random.uniform(0.1, 2.0, size=n_edges).astype(np.float32)
        # Simulate some overuse
        acct.present[:1000] = 2.0  # 1000 edges with usage=2, capacity=1
        acct.present_ema = acct.present.copy()

        pres_fac = 2.0
        hist_weight = 1.0

        t0 = time.time()
        acct.update_costs(base_costs, pres_fac, hist_weight, add_jitter=False)
        elapsed = time.time() - t0

        assert acct.total_cost is not None
        assert len(acct.total_cost) == n_edges
        assert elapsed < 0.1, f"Cost update took {elapsed:.4f}s (limit: 0.1s)"
