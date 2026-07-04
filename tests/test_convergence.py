"""Convergence mechanism tests for PathFinder negotiated congestion routing.

Validates the core convergence dynamics:
- Pressure factor (pres_fac) escalation after conflict iterations
- History cost accumulation for persistently contested edges

Marked with @pytest.mark.integration.
"""
import pytest
import numpy as np

from orthoroute.algorithms.manhattan.unified_pathfinder import EdgeAccountant
from orthoroute.algorithms.manhattan.pathfinder.config import PathFinderConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def accountant():
    """EdgeAccountant with 20 edges, CPU mode."""
    return EdgeAccountant(num_edges=20, use_gpu=False)


@pytest.fixture
def config():
    """Default PathFinderConfig."""
    return PathFinderConfig()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPressureEscalation:
    """After an iteration with conflicts, pres_fac should increase."""

    def test_pres_fac_increases_by_mult(self, config):
        """pres_fac * pres_fac_mult should give a larger value."""
        pres_fac = config.pres_fac_init
        new_pres_fac = pres_fac * config.pres_fac_mult
        assert new_pres_fac > pres_fac

    def test_pres_fac_respects_max(self, config):
        """pres_fac should be capped at pres_fac_max."""
        pres_fac = config.pres_fac_init
        for _ in range(200):  # Simulate many iterations
            pres_fac = min(pres_fac * config.pres_fac_mult, config.pres_fac_max)
        assert pres_fac <= config.pres_fac_max

    def test_escalation_over_iterations(self, config):
        """pres_fac should monotonically increase over iterations until capped."""
        pres_fac = config.pres_fac_init
        prev = pres_fac
        for _ in range(10):
            pres_fac = min(pres_fac * config.pres_fac_mult, config.pres_fac_max)
            assert pres_fac >= prev
            prev = pres_fac

    def test_total_cost_increases_with_pres_fac(self, accountant):
        """Higher pres_fac should result in higher total_cost on overused edges."""
        E = accountant.E
        base = np.ones(E, dtype=np.float32)

        # Create overuse on edges 0-4
        for i in range(5):
            accountant.commit_path([i])
            accountant.commit_path([i])  # usage=2, capacity=1 → overuse=1
        accountant.update_present_ema(beta=1.0)  # Sync EMA to raw present

        # Low pressure
        accountant.update_costs(base, pres_fac=1.0, hist_weight=0.0, add_jitter=False)
        cost_low = accountant.total_cost[0].copy()

        # High pressure
        accountant.update_costs(base, pres_fac=10.0, hist_weight=0.0, add_jitter=False)
        cost_high = accountant.total_cost[0].copy()

        assert cost_high > cost_low


@pytest.mark.integration
class TestHistoryAccumulates:
    """History cost should grow for contested (overused) edges."""

    def test_history_starts_at_zero(self, accountant):
        """History should be zero initially."""
        assert np.all(accountant.history == 0.0)

    def test_history_grows_after_update(self, accountant):
        """update_history with overuse should increase history on affected edges."""
        E = accountant.E
        base = np.ones(E, dtype=np.float32)

        # Create overuse: edges 0-4 have usage=2 (capacity=1)
        for i in range(5):
            accountant.commit_path([i])
            accountant.commit_path([i])

        # Sync present_ema
        accountant.update_present_ema(beta=1.0)

        # Update history
        accountant.update_history(gain=0.5, base_costs=base)

        # Overused edges should have positive history
        for i in range(5):
            assert accountant.history[i] > 0.0, f"Edge {i} should have positive history"

        # Non-overused edges should have zero history
        for i in range(5, E):
            assert accountant.history[i] == 0.0, f"Edge {i} should have zero history"

    def test_history_accumulates_across_iterations(self, accountant):
        """Multiple update_history calls should accumulate history."""
        E = accountant.E
        base = np.ones(E, dtype=np.float32)

        # Overuse edge 0
        accountant.commit_path([0])
        accountant.commit_path([0])
        accountant.update_present_ema(beta=1.0)

        # First update
        accountant.update_history(gain=0.5, base_costs=base)
        hist_after_1 = accountant.history[0].copy()

        # Second update (overuse persists)
        accountant.update_history(gain=0.5, base_costs=base)
        hist_after_2 = accountant.history[0].copy()

        assert hist_after_2 > hist_after_1

    def test_history_influences_total_cost(self, accountant):
        """Edges with accumulated history should have higher total_cost."""
        E = accountant.E
        base = np.ones(E, dtype=np.float32)

        # Overuse edge 0 and accumulate history
        accountant.commit_path([0])
        accountant.commit_path([0])
        accountant.update_present_ema(beta=1.0)
        accountant.update_history(gain=1.0, base_costs=base)

        # Update costs with hist_weight
        accountant.update_costs(base, pres_fac=1.0, hist_weight=5.0, add_jitter=False)

        # Edge 0 should be more expensive than edge 10 (no history)
        assert accountant.total_cost[0] > accountant.total_cost[10]
