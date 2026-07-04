"""
Convergence Manager — PathFinder Negotiation Loop Coordinator

Wraps the PathFinder negotiation loop, providing a clean interface for:
- Running the iterative rip-up-and-reroute negotiation
- Managing convergence detection (stagnation, zero-overuse)
- Detail pass execution for fine-grained improvements
- Hot-set construction and offender ripping
- Layer bias computation and application
- Logging per-layer congestion and overused channels

This class uses a delegation pattern: it holds a reference to the
PathFinderRouter instance and coordinates the negotiation methods
that remain on the router (due to deep coupling with router state).

Usage:
    manager = ConvergenceManager(router)
    result = manager.run(tasks, progress_cb, iteration_cb)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from orthoroute.algorithms.manhattan.unified_pathfinder import PathFinderRouter

logger = logging.getLogger(__name__)


class ConvergenceManager:
    """Coordinates the PathFinder negotiation loop.

    The ConvergenceManager acts as a facade over the negotiation methods
    on PathFinderRouter. It provides structured access to the iterative
    rip-up-and-reroute algorithm, convergence detection, and layer
    balancing utilities.

    Attributes:
        router: Reference to the owning PathFinderRouter instance.
            All state (lattice, graph, accountant, config, etc.) is
            accessed through this reference.

    Design Notes:
        The negotiation loop is deeply coupled to PathFinderRouter state
        (lattice, graph, accountant, committed_paths, via_usage, etc.).
        Rather than copying 1500+ lines and doing fragile self→self.router
        renaming, we use a delegation pattern where the ConvergenceManager
        coordinates calls to methods that remain on the router.

        This is the recommended pattern from the Metal codebase where
        GeometryEmitter uses the same delegation approach.
    """

    def __init__(self, router: PathFinderRouter) -> None:
        """Initialize with a reference to the owning PathFinderRouter.

        Args:
            router: The PathFinderRouter whose negotiation loop this
                    manager coordinates.
        """
        self.router = router

    def run(
        self,
        tasks: Dict[str, Tuple[int, int]],
        progress_cb=None,
        iteration_cb=None,
    ) -> Dict:
        """Execute the full PathFinder negotiation loop.

        Delegates to PathFinderRouter._pathfinder_negotiation() which
        implements the core rip-up-and-reroute algorithm with:
        - Auto-configuration from board characteristics
        - Pressure factor escalation per iteration
        - History cost accumulation for contested edges
        - Hot-set construction for targeted rerouting
        - Stagnation detection and early termination
        - Optional detail pass for fine-grained convergence

        Args:
            tasks: Net routing tasks as {net_id: (source_node, sink_node)}.
            progress_cb: Optional progress callback(iteration, total).
            iteration_cb: Optional per-iteration callback with metrics.

        Returns:
            Dictionary with routing results including:
            - 'tracks': List of track segments
            - 'vias': List of via placements
            - 'iterations': Number of iterations executed
            - 'converged': Whether zero-overuse was achieved
            - 'overflow_count': Final number of overused edges
        """
        return self.router._pathfinder_negotiation(
            tasks, progress_cb=progress_cb, iteration_cb=iteration_cb
        )

    def build_hotset(
        self,
        tasks: Dict[str, Tuple[int, int]],
        ripped: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Build the hot-set of nets to reroute in this iteration.

        Delegates to PathFinderRouter._build_hotset().

        Args:
            tasks: All routing tasks.
            ripped: Set of already-ripped net IDs to exclude.

        Returns:
            Set of net IDs selected for rerouting.
        """
        return self.router._build_hotset(tasks, ripped=ripped)

    def rip_top_k_offenders(self, k: int = 20) -> Set[str]:
        """Identify and rip the top-K congestion offenders.

        Delegates to PathFinderRouter._rip_top_k_offenders().

        Args:
            k: Number of top offenders to rip.

        Returns:
            Set of ripped net IDs.
        """
        return self.router._rip_top_k_offenders(k=k)

    def detail_pass(
        self,
        tasks: Dict[str, Tuple[int, int]],
        initial_overuse: int,
        initial_edges: int,
    ) -> Dict:
        """Run a detail pass for fine-grained convergence improvement.

        Delegates to PathFinderRouter._detail_pass().

        Args:
            tasks: Net routing tasks.
            initial_overuse: Overuse count before detail pass.
            initial_edges: Edge count before detail pass.

        Returns:
            Dictionary with detail pass results.
        """
        return self.router._detail_pass(tasks, initial_overuse, initial_edges)

    def compute_layer_bias(
        self,
        accountant,
        graph,
        num_layers: int,
        alpha: float = 0.9,
        max_boost: float = 1.8,
    ):
        """Compute per-layer routing bias based on congestion.

        Delegates to PathFinderRouter._compute_layer_bias().

        Args:
            accountant: EdgeAccountant with current usage data.
            graph: CSRGraph with edge structure.
            num_layers: Total number of routing layers.
            alpha: EWMA smoothing factor.
            max_boost: Maximum bias multiplier.
        """
        return self.router._compute_layer_bias(
            accountant, graph, num_layers, alpha=alpha, max_boost=max_boost
        )

    def update_layer_bias(
        self, overuse_by_layer: dict, layer_bias: dict
    ) -> dict:
        """Update layer bias based on per-layer overuse.

        Delegates to PathFinderRouter._update_layer_bias().

        Args:
            overuse_by_layer: Dict mapping layer index to overuse count.
            layer_bias: Current layer bias values.

        Returns:
            Updated layer bias dictionary.
        """
        return self.router._update_layer_bias(overuse_by_layer, layer_bias)

    def apply_layer_bias_to_costs(self, layer_bias: dict) -> None:
        """Apply layer bias multipliers to graph edge costs.

        Delegates to PathFinderRouter._apply_layer_bias_to_costs().

        Args:
            layer_bias: Dict mapping layer index to bias multiplier.
        """
        self.router._apply_layer_bias_to_costs(layer_bias)

    def analyze_layer_requirements(
        self, failed_nets: int, overuse_edges: int, overuse_sum: int
    ) -> Dict:
        """Analyze whether the board needs more layers.

        Delegates to PathFinderRouter._analyze_layer_requirements().

        Args:
            failed_nets: Number of nets that failed to route.
            overuse_edges: Number of overused edges.
            overuse_sum: Total overuse sum.

        Returns:
            Dictionary with layer analysis results.
        """
        return self.router._analyze_layer_requirements(
            failed_nets, overuse_edges, overuse_sum
        )

    def log_top_overused_channels(self, over, top_k: int = 10) -> None:
        """Log the top-K most overused routing channels.

        Delegates to PathFinderRouter._log_top_overused_channels().

        Args:
            over: Overuse array (per-edge).
            top_k: Number of top channels to log.
        """
        self.router._log_top_overused_channels(over, top_k=top_k)

    def log_per_layer_congestion(self, over) -> None:
        """Log per-layer congestion statistics.

        Delegates to PathFinderRouter._log_per_layer_congestion().

        Args:
            over: Overuse array (per-edge).
        """
        self.router._log_per_layer_congestion(over)
