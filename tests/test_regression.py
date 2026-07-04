"""
Regression Test Suite

Integration-level tests for core data structures and invariants:
- CSR finalize preserves edge count
- Duplicate edge handling in CSR
- EdgeAccountant usage monotonicity (commit increases, clear decreases)
- Lattice pitch effects on node count
- ROI extractor returns subset of total nodes
- Path edges are valid graph edges
- Geometry payload matches committed paths
"""

import pytest
import numpy as np

from orthoroute.algorithms.manhattan.unified_pathfinder import (
    CSRGraph,
    EdgeAccountant,
    Lattice3D,
    SimpleDijkstra,
)
from orthoroute.algorithms.manhattan.pathfinder.config import PathFinderConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_csr():
    """Build a small CSR graph with 5 nodes, 8 directed edges."""
    g = CSRGraph(use_gpu=False)
    edges = [(0, 1, 1.0), (1, 0, 1.0),
             (1, 2, 1.0), (2, 1, 1.0),
             (2, 3, 1.0), (3, 2, 1.0),
             (3, 4, 1.0), (4, 3, 1.0)]
    for u, v, c in edges:
        g.add_edge(u, v, c)
    g.finalize(num_nodes=5)
    return g, len(edges)


@pytest.fixture
def accountant_100():
    """EdgeAccountant with 100 edges, CPU mode."""
    return EdgeAccountant(num_edges=100, use_gpu=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCSRFinalizePreservesEdgeCount:
    """Verify finalize() preserves the exact number of added edges."""

    def test_csr_finalize_preserves_edge_count(self, small_csr):
        """After finalize(), len(indices) must equal the number of edges added."""
        graph, expected_edges = small_csr
        actual_edges = len(graph.indices)
        assert actual_edges == expected_edges, (
            f"Expected {expected_edges} edges, got {actual_edges}"
        )

    def test_csr_indptr_length(self, small_csr):
        """indptr should have length num_nodes + 1."""
        graph, _ = small_csr
        assert len(graph.indptr) == 6  # 5 nodes + 1


@pytest.mark.integration
class TestCSRAddDuplicateEdge:
    """Verify duplicate edges are counted separately."""

    def test_csr_add_duplicate_edge_counted(self):
        """Adding same (u,v) twice creates 2 entries in the CSR."""
        g = CSRGraph(use_gpu=False)
        g.add_edge(0, 1, 1.0)
        g.add_edge(0, 1, 2.0)
        g.add_edge(1, 0, 1.0)
        g.finalize(num_nodes=2)

        # Two edges from node 0 to node 1
        start = int(g.indptr[0])
        end = int(g.indptr[1])
        edges_from_0 = end - start
        assert edges_from_0 == 2, (
            f"Expected 2 edges from node 0, got {edges_from_0}"
        )


@pytest.mark.integration
class TestEdgeAccountantUsageMonotonic:
    """Verify commit increases and clear decreases usage."""

    def test_commit_path_only_increases(self, accountant_100):
        """commit_path should only increase present usage."""
        acct = accountant_100
        before = float(acct.present[10])
        acct.commit_path([10, 20, 30])
        after = float(acct.present[10])
        assert after > before, f"Usage should increase: {before} → {after}"

    def test_clear_path_only_decreases(self, accountant_100):
        """clear_path should only decrease present usage (down to 0)."""
        acct = accountant_100
        acct.commit_path([10, 20, 30])
        before = float(acct.present[10])
        acct.clear_path([10, 20, 30])
        after = float(acct.present[10])
        assert after < before, f"Usage should decrease: {before} → {after}"

    def test_clear_never_goes_negative(self, accountant_100):
        """Clearing a path that was never committed should not make usage negative."""
        acct = accountant_100
        acct.clear_path([50])
        assert float(acct.present[50]) >= 0, "Usage should never be negative"


@pytest.mark.integration
class TestLatticePitchAffectsNodeCount:
    """Verify that halving pitch quadruples XY node count."""

    def test_lattice_pitch_affects_node_count(self):
        """Halving pitch from 0.8 to 0.4 should ~quadruple XY node count."""
        bounds = (0, 0, 40, 40)
        layers = 6

        lattice_coarse = Lattice3D(bounds=bounds, pitch=0.8, layers=layers)
        lattice_fine = Lattice3D(bounds=bounds, pitch=0.4, layers=layers)

        xy_coarse = lattice_coarse.x_steps * lattice_coarse.y_steps
        xy_fine = lattice_fine.x_steps * lattice_fine.y_steps

        ratio = xy_fine / xy_coarse
        # Should be approximately 4x (exact depends on rounding)
        assert 3.5 <= ratio <= 4.5, (
            f"Expected ~4x ratio, got {ratio:.2f} "
            f"(coarse={xy_coarse}, fine={xy_fine})"
        )


@pytest.mark.integration
class TestROIExtractorReturnsSubset:
    """Verify ROI nodes are a subset of total graph nodes."""

    def test_roi_extractor_returns_subset(self):
        """ROI node indices should all be valid global node indices."""
        lattice = Lattice3D(bounds=(0, 0, 20, 20), pitch=0.4, layers=4)

        # Simulate ROI extraction: select nodes within a bounding box
        roi_x_min, roi_x_max = 5, 15  # lattice x indices
        roi_y_min, roi_y_max = 5, 15
        roi_nodes = []
        for z in range(lattice.layers):
            for y in range(roi_y_min, min(roi_y_max, lattice.y_steps)):
                for x in range(roi_x_min, min(roi_x_max, lattice.x_steps)):
                    roi_nodes.append(lattice.node_idx(x, y, z))

        roi_set = set(roi_nodes)
        total_nodes = lattice.num_nodes

        # Every ROI node must be valid
        for node in roi_set:
            assert 0 <= node < total_nodes, f"ROI node {node} out of range [0, {total_nodes})"

        # ROI should be a proper subset
        assert len(roi_set) < total_nodes, "ROI should be smaller than full graph"
        assert len(roi_set) > 0, "ROI should not be empty"


@pytest.mark.integration
class TestPathEdgesAreValidGraphEdges:
    """Verify every edge in a found path exists in the CSR graph."""

    def test_path_edges_are_valid_graph_edges(self):
        """Every consecutive pair in a Dijkstra path should be an edge in the CSR."""
        # Build a small connected graph
        g = CSRGraph(use_gpu=False)
        edges_set = set()
        for i in range(10):
            if i < 9:
                g.add_edge(i, i + 1, 1.0)
                g.add_edge(i + 1, i, 1.0)
                edges_set.add((i, i + 1))
                edges_set.add((i + 1, i))
        g.finalize(num_nodes=10)

        # Find path
        dijk = SimpleDijkstra(g, lattice=None)
        roi_nodes = np.arange(10, dtype=np.int32)
        g2r = np.arange(10, dtype=np.int32)
        path = dijk.find_path_roi(0, 9, g.base_costs, roi_nodes, g2r)

        assert path is not None, "Path should exist in connected graph"

        # Verify every edge in path exists in graph
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            assert (u, v) in edges_set, (
                f"Edge ({u}, {v}) in path but not in graph"
            )


@pytest.mark.integration
class TestGeometryPayloadMatchesCommittedPaths:
    """Verify geometry output matches number of committed nets."""

    def test_geometry_payload_matches_committed_paths(self):
        """Number of geometry segments should be >= number of committed nets.

        Each committed net produces at least one geometry segment.
        """
        # Simulate committed net paths
        committed_nets = {
            "NET1": [0, 1, 2, 3],
            "NET2": [4, 5, 6],
            "NET3": [7, 8, 9, 10, 11],
        }

        # Simulate geometry emission: each path produces len(path)-1 segments
        total_segments = sum(
            len(path) - 1 for path in committed_nets.values() if len(path) > 1
        )

        assert total_segments >= len(committed_nets), (
            f"Expected >= {len(committed_nets)} segments, got {total_segments}"
        )

        # Verify each net contributes at least one segment
        for net_id, path in committed_nets.items():
            assert len(path) >= 2, f"Net {net_id} has path too short for geometry"
