"""Performance benchmark tests for Lattice3D and CSR graph construction.

These tests enforce wall-clock limits on critical construction paths
to catch performance regressions. Marked with @pytest.mark.slow.
"""
import time

import pytest

from orthoroute.algorithms.manhattan.unified_pathfinder import Lattice3D


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestLatticeBuildTime:
    """Lattice3D for a 6-layer board builds within the time budget."""

    def test_six_layer_lattice_builds_in_5s(self):
        """Lattice3D for a realistic 6-layer board should build in <5s.

        Uses a board-sized lattice (~50×50 grid) with 6 copper layers.
        """
        bounds = (0.0, 0.0, 20.0, 20.0)  # ~20mm × 20mm board
        pitch = 0.4  # 0.4mm pitch → ~51 steps per axis

        start = time.perf_counter()
        lat = Lattice3D(bounds, pitch=pitch, layers=6)
        elapsed = time.perf_counter() - start

        assert lat.num_nodes > 0
        assert lat.layers == 6
        assert elapsed < 5.0, f"Lattice3D build took {elapsed:.2f}s (limit: 5s)"

    def test_large_lattice_still_reasonable(self):
        """Even a 100-step grid on 6 layers should build quickly."""
        bounds = (0.0, 0.0, 39.6, 39.6)  # 100 steps per axis
        pitch = 0.4

        start = time.perf_counter()
        lat = Lattice3D(bounds, pitch=pitch, layers=6)
        elapsed = time.perf_counter() - start

        assert lat.x_steps == 100
        assert lat.y_steps == 100
        assert lat.num_nodes == 100 * 100 * 6
        assert elapsed < 5.0, f"Large lattice build took {elapsed:.2f}s (limit: 5s)"


@pytest.mark.slow
class TestCSRConstructionTime:
    """CSR graph construction from a lattice within the time budget."""

    def test_csr_builds_in_5s(self):
        """CSR graph for a 50×50×6 lattice should build in <5s."""
        bounds = (0.0, 0.0, 19.6, 19.6)  # 50 steps per axis
        pitch = 0.4
        lat = Lattice3D(bounds, pitch=pitch, layers=6)

        start = time.perf_counter()
        graph = lat.build_graph(via_cost=1.0, use_gpu=False)
        elapsed = time.perf_counter() - start

        assert graph.indptr is not None
        assert graph.indices is not None
        assert len(graph.indptr) == lat.num_nodes + 1
        assert elapsed < 5.0, f"CSR build took {elapsed:.2f}s (limit: 5s)"
