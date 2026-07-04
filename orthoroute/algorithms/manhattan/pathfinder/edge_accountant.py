"""
═══════════════════════════════════════════════════════════════════════════════
EDGE ACCOUNTANT
═══════════════════════════════════════════════════════════════════════════════

Extracted from unified_pathfinder.py.

Provides EdgeAccountant: edge usage tracking, cost computation, and
history management for the PathFinder negotiated congestion algorithm.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np  # Fallback to numpy if cupy not available
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class EdgeAccountant:
    """Edge usage tracking"""

    def __init__(self, num_edges: int, use_gpu=False):
        self.E = num_edges
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        self.canonical: Dict[int, int] = {}
        self.present = self.xp.zeros(num_edges, dtype=self.xp.float32)
        self.present_ema = self.xp.zeros(num_edges, dtype=self.xp.float32)  # Smoothed present for stable convergence
        self.history = self.xp.zeros(num_edges, dtype=self.xp.float32)
        self.capacity = self.xp.ones(num_edges, dtype=self.xp.float32)
        self.total_cost = None

    def refresh_from_canonical(self):
        """Rebuild present"""
        self.present.fill(0)
        for idx, count in self.canonical.items():
            if 0 <= idx < self.E:
                self.present[idx] = float(count)

    def commit_path(self, edge_indices: List[int]):
        """Add path and keep present in sync"""
        for idx in edge_indices:
            self.canonical[idx] = self.canonical.get(idx, 0) + 1
            # Keep present in sync during iteration
            self.present[idx] = self.present[idx] + 1

    def clear_path(self, edge_indices: List[int]):
        """Remove path and keep present in sync"""
        for idx in edge_indices:
            if idx in self.canonical:
                self.canonical[idx] -= 1
                if self.canonical[idx] <= 0:
                    del self.canonical[idx]
            # Reflect in present
            self.present[idx] = self.xp.maximum(0, self.present[idx] - 1)

    def compute_overuse(self, router_instance=None) -> Tuple[int, int]:
        """
        Compute total overuse including edge AND via spatial overuse.

        Args:
            router_instance: Optional PathFinderRouter instance for via spatial checks

        Returns:
            (total_overuse_sum, edge_overuse_count)
        """
        # Edge overuse (existing)
        usage = self.present.get() if self.use_gpu else self.present
        cap = self.capacity.get() if self.use_gpu else self.capacity
        edge_over = np.maximum(0, usage - cap)
        edge_over_sum = int(edge_over.sum())
        edge_over_count = int((edge_over > 0).sum())

        # Via spatial overuse (NEW)
        via_col_over_sum = 0
        via_seg_over_sum = 0

        if router_instance is not None:
            # Check via column overuse
            if hasattr(router_instance, 'via_col_use') and hasattr(router_instance, 'via_col_cap'):
                via_col_over = np.maximum(0, router_instance.via_col_use - router_instance.via_col_cap)
                via_col_over_sum = int(via_col_over.sum())

            # Check via segment overuse
            if hasattr(router_instance, 'via_seg_use') and hasattr(router_instance, 'via_seg_cap'):
                via_seg_over = np.maximum(0, router_instance.via_seg_use - router_instance.via_seg_cap)
                via_seg_over_sum = int(via_seg_over.sum())

        total_over = edge_over_sum + via_col_over_sum + via_seg_over_sum

        # Log via violations if present (helps with debugging)
        if via_col_over_sum > 0 or via_seg_over_sum > 0:
            logger.info(f"[OVERUSE] edge={edge_over_sum} via_col={via_col_over_sum} via_seg={via_seg_over_sum} total={total_over}")

        return (total_over, edge_over_count)

    def verify_present_matches_canonical(self) -> bool:
        """Sanity check: verify present usage matches canonical store"""
        recomputed = self.xp.zeros(self.E, dtype=self.xp.float32)
        for idx, count in self.canonical.items():
            if 0 <= idx < self.E:
                recomputed[idx] = float(count)

        if self.use_gpu:
            present_cpu = self.present.get()
            recomputed_cpu = recomputed.get()
        else:
            present_cpu = self.present
            recomputed_cpu = recomputed

        mismatch = np.sum(np.abs(present_cpu - recomputed_cpu))
        if mismatch > 0.01:
            logger.error(f"[ACCOUNTING] Present/canonical mismatch: {mismatch:.2f}")
            return False
        return True

    def update_history(self, gain: float, base_costs=None, history_cap_multiplier=10.0, decay_factor=0.98, use_raw_present=False):
        """
        Update history with:
        - Gentle decay: history *= 0.98 before adding increment (decay_factor param)
        - Clamping: increment capped at history_cap = 10 * base_cost
        - Uses present_ema (smoothed) by default, or raw present if use_raw_present=True
        """
        import logging
        import sys
        logger = logging.getLogger(__name__)

        # DIAGNOSTIC: Log what's actually happening
        if not hasattr(self, '_hist_update_count'):
            self._hist_update_count = 0
        self._hist_update_count += 1

        # Always log first 5 calls
        if self._hist_update_count <= 5:
            # Before update
            hist_before_max = float(self.history.max()) if self.history.size > 0 else 0.0
            logger.debug(f"[UPDATE-HISTORY CALLED] Call #{self._hist_update_count} START gain={gain:.3f}")

        # Apply gentle decay before adding new history
        self.history *= decay_factor

        # Use smoothed present_ema by default, or raw present if requested
        present_for_history = self.present if use_raw_present else self.present_ema
        over = self.xp.maximum(0, present_for_history - self.capacity)
        increment = gain * over

        # Clamp per-edge history increment
        if base_costs is not None:
            history_cap = history_cap_multiplier * base_costs
            increment_before_cap = increment.copy()
            increment = self.xp.minimum(increment, history_cap)

            if self._hist_update_count <= 5:
                # Check how many edges are being capped
                capped_mask = increment_before_cap > history_cap
                capped_count = int(self.xp.sum(capped_mask))
                if capped_count > 0:
                    logger.debug(f"  [HIST-CAP] {capped_count} edges capped! avg_cap={float(history_cap.mean()):.3f}")

        self.history += increment

        if self._hist_update_count <= 5:
            # After update
            hist_after_max = float(self.history.max())
            incr_max = float(increment.max())
            over_max = float(over.max())
            over_mean = float(over[over > 0].mean()) if (over > 0).any() else 0.0
            pres_max = float(present_for_history.max())
            pres_ema_max = float(self.present_ema.max())
            pres_raw_max = float(self.present.max())

            logger.debug(f"[UPDATE-HISTORY #{self._hist_update_count}]")
            logger.debug(f"  gain={gain:.3f} decay={decay_factor:.3f} cap_mult={history_cap_multiplier:.1f}")
            logger.debug(f"  use_raw_present={use_raw_present}")
            logger.debug(f"  present_raw_max={pres_raw_max:.1f} present_ema_max={pres_ema_max:.1f}")
            logger.debug(f"  overuse: max={over_max:.2f} mean={over_mean:.3f}")
            logger.debug(f"  increment: max={incr_max:.3f}")
            logger.debug(f"  history: before={hist_before_max:.3f} → after={hist_after_max:.3f}")
            if base_costs is not None:
                logger.debug(f"  base_cost: mean={float(base_costs.mean()):.4f} max={float(base_costs.max()):.4f}")

    def update_present_ema(self, beta: float = 0.60):
        """
        Update exponential moving average of present usage for stability.
        Smooths bang-bang oscillations in overuse detection.

        Args:
            beta: EMA smoothing factor (higher = more smoothing, typically 0.6)
        """
        self.present_ema = beta * self.present + (1.0 - beta) * self.present_ema

    def update_costs(
        self,
        base_costs,
        pres_fac: float,
        hist_weight: float = 1.0,
        add_jitter: bool = True,
        via_cost_multiplier: float = 1.0,
        base_cost_weight: float = 0.01,
        *,
        edge_layer=None,          # np/cp array [E] with source layer per edge
        layer_bias_per_layer=None,  # np/cp array [L] with multiplicative bias
        edge_kind=None            # np/cp array [E] with 0=horiz/vert, 1=via
    ):
        """
        total = (base * via_multiplier * base_weight * layer_bias) + pres_fac*overuse + hist_weight*history + epsilon_jitter
        Jitter breaks ties and prevents oscillation in equal-cost paths.
        Via cost multiplier enables late-stage via annealing.
        Base cost weight controls length vs completion trade-off (lower = prefer completion over short paths).
        Layer bias: applied only to horizontal/vertical edges (not vias) to rebalance layer usage.
        Uses present_ema (smoothed) instead of raw present to prevent bang-bang oscillation.
        """
        xp = self.xp
        # Use smoothed present (EMA) to prevent oscillation - critical for convergence
        over = xp.maximum(0, self.present_ema - self.capacity)

        # Vectorized per-edge layer bias (single gather operation)
        # Only apply to horizontal/vertical edges (edge_kind==0), not vias (edge_kind==1)
        per_edge_bias = 1.0
        if (edge_layer is not None) and (layer_bias_per_layer is not None) and (edge_kind is not None):
            if self.use_gpu:
                # Ensure arrays are on GPU
                layer_bias = cp.asarray(layer_bias_per_layer) if not hasattr(layer_bias_per_layer, "get") else layer_bias_per_layer
                edge_layer_arr = cp.asarray(edge_layer) if not hasattr(edge_layer, "get") else edge_layer
                edge_kind_arr = cp.asarray(edge_kind) if not hasattr(edge_kind, "get") else edge_kind
            else:
                # NumPy arrays
                layer_bias = layer_bias_per_layer
                edge_layer_arr = edge_layer
                edge_kind_arr = edge_kind

            # Gather bias for each edge's layer
            bias_factors = layer_bias[edge_layer_arr]
            # Apply bias only to horizontal/vertical edges (edge_kind==0), set via edges to 1.0
            per_edge_bias = xp.where(edge_kind_arr == 0, bias_factors, 1.0)

        # Apply both via multiplier, base weight, and layer bias to base costs
        # base_cost_weight < 1.0 makes router prefer completion over short paths
        adjusted_base = base_costs * via_cost_multiplier * base_cost_weight * per_edge_bias

        # Apply INVERTED layer bias to present term to directly pressure hot layers
        # Base term uses per_edge_bias (hot layers cheaper for length optimization)
        # Present term uses INVERSE (hot layers more expensive for congestion avoidance)
        # For vias, keep bias at 1.0 (no layer-specific present penalty)
        if (edge_layer is not None) and (layer_bias_per_layer is not None) and (edge_kind is not None):
            # Invert bias for present: if bias=0.9 (cheap base), use 1/0.9=1.11 (expensive present)
            # Clamp to prevent extreme values
            inverted_bias = xp.where(per_edge_bias != 0, 1.0 / xp.maximum(per_edge_bias, 0.5), 1.0)
            inverted_bias = xp.where(edge_kind_arr == 0, inverted_bias, 1.0)  # Only H/V edges
            present_term = (pres_fac * inverted_bias) * over
        else:
            present_term = pres_fac * over

        self.total_cost = adjusted_base + present_term + hist_weight * self.history

        # Add per-edge epsilon jitter to break ties (stable across iterations)
        if add_jitter:
            E = len(self.total_cost)
            # Use edge index modulo prime for deterministic jitter
            jitter = xp.arange(E, dtype=xp.float32) % 9973
            jitter = jitter * 1e-6  # tiny epsilon
            self.total_cost += jitter

    def update_present_cost_only(self, pres_fac: float, base_costs):
        """
        FAST per-net cost update: Only recomputes present cost term (not history).
        Called after EACH net routes to update costs for the NEXT net.
        History cost only updates at END of iteration.

        This is critical for PathFinder convergence!

        Formula: total_cost = base_cost + pres_fac * overuse + history
        """
        # Recompute overuse with current occupancy (using smoothed present_ema for consistency)
        over = self.xp.maximum(0, self.present_ema - self.capacity)

        # Update total_cost: base + present_penalty + history
        # Don't modify history here - only update present penalty based on current occupancy
        self.total_cost = base_costs + pres_fac * over + self.history

        # Log first few updates for debugging
        if not hasattr(self, '_pernet_update_count'):
            self._pernet_update_count = 0
        self._pernet_update_count += 1
        if self._pernet_update_count <= 3:
            overuse_count = int(self.xp.sum(over > 0))
            logger.info(f"[PER-NET-UPDATE #{self._pernet_update_count}] Overuse edges: {overuse_count}, pres_fac={pres_fac:.2f}")
