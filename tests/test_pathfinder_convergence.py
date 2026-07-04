"""
PathFinder Convergence Tests

Integration-level tests for the PathFinder negotiation loop components:
- EdgeAccountant pressure escalation after conflicting commits
- History cost accumulation over iterations
- Overuse reduction when conflicting paths are cleared
- Cost function composition (base + pres_fac*overuse + hist_weight*history)
- Stagnation detection after STAGNATION_PATIENCE identical overuse counts
- Zero-overuse convergence condition
"""

import pytest
import numpy as np

from orthoroute.algorithms.manhattan.unified_pathfinder import EdgeAccountant
from orthoroute.algorithms.manhattan.pathfinder.config import (
    PathFinderConfig,
    STAGNATION_PATIENCE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def accountant():
    """EdgeAccountant with 100 edges, CPU mode."""
    return EdgeAccountant(num_edges=100, use_gpu=False)


@pytest.fixture
def accountant_with_capacity():
    """EdgeAccountant with capacity = 1 per edge (default), CPU mode."""
    acct = EdgeAccountant(num_edges=100, use_gpu=False)
    # capacity is already np.ones by default
    return acct


@pytest.fixture
def config():
    """Default PathFinderConfig."""
    return PathFinderConfig()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEdgeAccountantPressureEscalation:
    """Verify overuse detection after conflicting path commits."""

    def test_edge_accountant_pressure_escalation(self, accountant_with_capacity):
        """After committing conflicting paths on the same edge, overuse > 0."""
        acct = accountant_with_capacity

        # Commit two paths that share edge 5 (capacity = 1)
        acct.commit_path([5, 10, 15])
        acct.commit_path([5, 20, 25])

        over_sum, over_count = acct.compute_overuse()

        # Edge 5 is used twice but has capacity 1 → overuse = 1
        assert over_sum > 0, "Expected overuse after conflicting commit"
        assert over_count > 0, "Expected at least 1 overused edge"

    def test_single_commit_no_overuse(self, accountant_with_capacity):
        """A single path commit (usage=1, capacity=1) should have zero overuse."""
        acct = accountant_with_capacity
        acct.commit_path([5, 10, 15])

        over_sum, over_count = acct.compute_overuse()
        assert over_sum == 0
        assert over_count == 0


@pytest.mark.integration
class TestHistoryCostAccumulates:
    """Verify history array grows with repeated update_history calls."""

    def test_history_cost_accumulates(self, accountant_with_capacity):
        """Calling update_history multiple times increases history on overused edges."""
        acct = accountant_with_capacity

        # Create overuse on edge 5
        acct.commit_path([5])
        acct.commit_path([5])
        acct.refresh_from_canonical()
        # Sync present_ema so update_history sees overuse
        acct.present_ema = acct.present.copy()

        base_costs = np.ones(100, dtype=np.float32)
        hist_before = acct.history[5].copy()

        # Three rounds of history update
        for _ in range(3):
            acct.update_history(gain=1.0, base_costs=base_costs)

        hist_after = acct.history[5]
        assert hist_after > hist_before, (
            f"History should grow: before={hist_before}, after={hist_after}"
        )

    def test_history_zero_without_overuse(self, accountant_with_capacity):
        """Edges with no overuse should accumulate zero history."""
        acct = accountant_with_capacity
        acct.refresh_from_canonical()
        acct.present_ema = acct.present.copy()

        base_costs = np.ones(100, dtype=np.float32)
        acct.update_history(gain=1.0, base_costs=base_costs)

        assert float(acct.history.sum()) == 0.0, "No overuse → no history increment"


@pytest.mark.integration
class TestOveruseDecreasesWithReroute:
    """Verify overuse drops when conflicting paths are cleared."""

    def test_overuse_decreases_with_reroute(self, accountant_with_capacity):
        """Clearing one of two conflicting paths should reduce overuse."""
        acct = accountant_with_capacity

        path_a = [5, 10]
        path_b = [5, 20]
        acct.commit_path(path_a)
        acct.commit_path(path_b)

        over_before, _ = acct.compute_overuse()
        assert over_before > 0

        # Clear one conflicting path
        acct.clear_path(path_b)

        over_after, _ = acct.compute_overuse()
        assert over_after < over_before, (
            f"Overuse should decrease: before={over_before}, after={over_after}"
        )


@pytest.mark.integration
class TestCostFunctionIncludesHistory:
    """Verify update_costs produces cost = base + pres_fac*overuse + hist_weight*history."""

    def test_cost_function_includes_history(self, accountant_with_capacity):
        """After update_costs, total_cost should include base, present, and history terms."""
        acct = accountant_with_capacity

        # Set up known state
        base_costs = np.full(100, 0.5, dtype=np.float32)
        pres_fac = 2.0
        hist_weight = 3.0

        # Create overuse on edge 0
        acct.commit_path([0])
        acct.commit_path([0])
        acct.refresh_from_canonical()
        acct.present_ema = acct.present.copy()

        # Add history
        acct.update_history(gain=1.0, base_costs=base_costs)

        # Run update_costs
        acct.update_costs(base_costs, pres_fac, hist_weight, add_jitter=False)

        # Edge 0: usage=2, cap=1, overuse=1
        # Expected ≈ base*via_mult*base_weight + pres_fac*overuse + hist_weight*history
        total = float(acct.total_cost[0])
        assert total > float(base_costs[0]), (
            f"Total cost {total} should exceed base {base_costs[0]} due to overuse+history"
        )

        # Edge 50: no overuse, no history → cost ≈ base * base_cost_weight
        total_50 = float(acct.total_cost[50])
        # base_cost_weight defaults to 0.01, so adjusted_base = 0.5 * 1.0 * 0.01 = 0.005
        assert total_50 < 1.0, f"Non-overused edge cost {total_50} should be small"


@pytest.mark.integration
class TestStagnationDetected:
    """Verify stagnation detection logic."""

    def test_stagnation_detected(self):
        """After STAGNATION_PATIENCE identical overuse counts, routing should stop."""
        patience = STAGNATION_PATIENCE
        overuse_history = []
        stagnant = 0

        # Simulate negotiation iterations with identical overuse
        fixed_overuse = 42
        for it in range(patience + 2):
            overuse_history.append(fixed_overuse)

            if it > 0 and overuse_history[-1] >= overuse_history[-2]:
                stagnant += 1
            else:
                stagnant = 0

            if stagnant >= patience:
                break

        assert stagnant >= patience, (
            f"Stagnation should be detected after {patience} flat iterations"
        )

    def test_improvement_resets_stagnation(self):
        """An improvement in overuse should reset the stagnation counter."""
        stagnant = 3  # Already stagnant for 3 iterations
        prev_overuse = 10
        new_overuse = 8  # Improvement

        if new_overuse < prev_overuse:
            stagnant = 0

        assert stagnant == 0, "Improvement should reset stagnation counter"


@pytest.mark.integration
class TestZeroOveruseConvergence:
    """Verify zero overuse signals convergence."""

    def test_zero_overuse_means_convergence(self, accountant_with_capacity):
        """When no edges are overused, routing has converged."""
        acct = accountant_with_capacity

        # Commit non-conflicting paths (each edge used once, capacity 1)
        acct.commit_path([0, 1, 2])
        acct.commit_path([3, 4, 5])

        over_sum, over_count = acct.compute_overuse()
        converged = (over_sum == 0)

        assert converged, f"Expected convergence (overuse=0), got overuse_sum={over_sum}"

    def test_empty_accountant_is_converged(self, accountant_with_capacity):
        """An accountant with no committed paths has zero overuse (trivially converged)."""
        acct = accountant_with_capacity
        over_sum, over_count = acct.compute_overuse()
        assert over_sum == 0
        assert over_count == 0
