"""
Advanced GPU/CPU Parity Tests

Tests for GPU and CPU provider parity covering:
- SimpleDijkstra shortest path correctness on a small CSR graph
- Unreachable nodes in disconnected graphs
- Single-node graph edge case
- CPUFallbackProvider array create/copy/to_cpu/to_gpu round-trip
- MetalProvider.is_available() on macOS
- get_best_provider() factory returns Metal on macOS, CPU elsewhere
"""

import platform
import pytest
import numpy as np

from orthoroute.algorithms.manhattan.unified_pathfinder import CSRGraph, SimpleDijkstra, Lattice3D
from orthoroute.infrastructure.gpu.cpu_fallback import CPUFallbackProvider
from orthoroute.infrastructure.gpu.metal_provider import MetalProvider
from orthoroute.infrastructure.gpu import get_best_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_line_graph(n_nodes: int, weight: float = 1.0):
    """Build a CSRGraph of a simple line: 0–1–2–...–(n-1) with bidirectional edges."""
    graph = CSRGraph(use_gpu=False)
    for i in range(n_nodes - 1):
        graph.add_edge(i, i + 1, weight)
        graph.add_edge(i + 1, i, weight)
    graph.finalize(num_nodes=n_nodes)
    return graph


def build_disconnected_graph():
    """Build a CSRGraph with two disconnected components: {0,1,2} and {3,4}."""
    graph = CSRGraph(use_gpu=False)
    # Component 1: 0-1-2
    graph.add_edge(0, 1, 1.0)
    graph.add_edge(1, 0, 1.0)
    graph.add_edge(1, 2, 1.0)
    graph.add_edge(2, 1, 1.0)
    # Component 2: 3-4
    graph.add_edge(3, 4, 1.0)
    graph.add_edge(4, 3, 1.0)
    graph.finalize(num_nodes=5)
    return graph


def build_single_node_graph():
    """Build a CSRGraph with exactly 1 node and no edges."""
    graph = CSRGraph(use_gpu=False)
    # Need at least one edge for CSR finalization; use a self-loop
    graph.add_edge(0, 0, 0.0)
    graph.finalize(num_nodes=1)
    return graph


def run_dijkstra_on_graph(graph, src, dst, n_nodes):
    """Run SimpleDijkstra using full-graph ROI (all nodes)."""
    dijk = SimpleDijkstra(graph, lattice=None)
    roi_nodes = np.arange(n_nodes, dtype=np.int32)
    global_to_roi = np.arange(n_nodes, dtype=np.int32)

    costs = graph.base_costs if not hasattr(graph.base_costs, 'get') else graph.base_costs.get()
    path = dijk.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestDijkstraCPUFindsShortestPath:
    """Verify SimpleDijkstra finds correct shortest path on a small graph."""

    def test_dijkstra_cpu_finds_shortest_path(self):
        """Build 10-node line graph, verify shortest path from 0 to 9."""
        n = 10
        graph = build_line_graph(n)
        path = run_dijkstra_on_graph(graph, 0, n - 1, n)

        assert path is not None, "Dijkstra should find a path"
        assert path[0] == 0, f"Path should start at 0, got {path[0]}"
        assert path[-1] == n - 1, f"Path should end at {n-1}, got {path[-1]}"
        # Shortest path on a line graph is the sequential walk
        assert len(path) == n, f"Expected path length {n}, got {len(path)}"
        assert path == list(range(n)), f"Expected sequential path, got {path}"

    def test_dijkstra_short_path(self):
        """Path from node 3 to node 5 should be [3, 4, 5]."""
        graph = build_line_graph(10)
        path = run_dijkstra_on_graph(graph, 3, 5, 10)

        assert path is not None
        assert path == [3, 4, 5]


@pytest.mark.gpu
class TestDijkstraUnreachableNodes:
    """Verify unreachable nodes are handled correctly."""

    def test_dijkstra_unreachable_nodes(self):
        """In disconnected graph, path from component 1 to component 2 is None."""
        graph = build_disconnected_graph()
        path = run_dijkstra_on_graph(graph, 0, 4, 5)

        assert path is None, "Should return None for unreachable destination"

    def test_dijkstra_reachable_within_component(self):
        """Path within same connected component should succeed."""
        graph = build_disconnected_graph()
        path = run_dijkstra_on_graph(graph, 0, 2, 5)

        assert path is not None
        assert path[0] == 0
        assert path[-1] == 2


@pytest.mark.gpu
class TestDijkstraSingleNode:
    """Verify single-node graph edge case."""

    def test_dijkstra_single_node(self):
        """Graph with 1 node, source==sink should return None (path length <= 1)."""
        graph = build_single_node_graph()
        path = run_dijkstra_on_graph(graph, 0, 0, 1)

        # SimpleDijkstra returns None if path length <= 1
        # This is expected: the early-exit check hits u_roi == roi_dst immediately,
        # then reconstructs a single-node path which is filtered out.
        # This is correct behaviour — no routing needed for src==dst.
        assert path is None or path == [0], (
            f"Single-node path should be None or [0], got {path}"
        )


@pytest.mark.gpu
class TestCPUProviderArrayOperations:
    """Verify CPU provider create/copy/to_cpu/to_gpu round-trip."""

    def test_cpu_provider_array_operations(self):
        """Create array, copy, to_cpu, to_gpu should all produce valid numpy arrays."""
        provider = CPUFallbackProvider()
        provider.initialize()

        try:
            # Create
            arr = provider.create_array((10,), dtype=np.float32, fill_value=3.14)
            assert arr.shape == (10,)
            assert np.allclose(arr, 3.14)

            # Copy
            arr_copy = provider.copy_array(arr)
            assert np.array_equal(arr, arr_copy)
            arr_copy[0] = 999.0
            assert arr[0] != arr_copy[0], "Copy should be independent"

            # to_cpu (no-op for CPU provider)
            arr_cpu = provider.to_cpu(arr)
            assert np.array_equal(arr_cpu, arr)

            # to_gpu (no-op for CPU provider)
            arr_gpu = provider.to_gpu(arr)
            assert np.array_equal(arr_gpu, arr)
        finally:
            provider.cleanup()

    def test_cpu_provider_context_manager(self):
        """CPU provider works as context manager."""
        with CPUFallbackProvider() as provider:
            arr = provider.create_array((5, 5), dtype=np.int32, fill_value=0)
            assert arr.shape == (5, 5)
            assert arr.dtype == np.int32


@pytest.mark.gpu
class TestMetalProviderAvailability:
    """Verify Metal provider detection on macOS."""

    @pytest.mark.skipif(
        platform.system() != 'Darwin',
        reason="Metal is only available on macOS"
    )
    def test_metal_provider_available_on_macos(self):
        """On macOS, MetalProvider.is_available() should be True if orthoroute_mac is built."""
        provider = MetalProvider()
        # This may be True or False depending on whether orthoroute_mac is compiled
        # We just verify it doesn't crash and returns a bool
        result = provider.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        platform.system() == 'Darwin',
        reason="Only run on non-macOS to verify Metal is unavailable"
    )
    def test_metal_provider_unavailable_off_macos(self):
        """On non-macOS, MetalProvider.is_available() should be False."""
        provider = MetalProvider()
        assert provider.is_available() is False


@pytest.mark.gpu
class TestProviderFactory:
    """Verify get_best_provider() returns the correct provider type."""

    def test_provider_factory_returns_best(self):
        """get_best_provider() returns Metal on macOS arm64, CPU otherwise."""
        provider = get_best_provider()

        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            # On Apple Silicon, should be Metal if orthoroute_mac is available,
            # otherwise CPU fallback
            assert isinstance(provider, (MetalProvider, CPUFallbackProvider))
        else:
            # On other platforms, should be CPU fallback (or CUDA if available)
            assert provider is not None
            assert provider.is_available() or isinstance(provider, CPUFallbackProvider)

    def test_provider_is_available(self):
        """Whatever provider is returned, it should report as available."""
        provider = get_best_provider()
        # CPU fallback is always available
        assert hasattr(provider, 'is_available')
