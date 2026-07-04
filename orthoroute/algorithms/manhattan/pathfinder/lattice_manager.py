"""
═══════════════════════════════════════════════════════════════════════════════
3D LATTICE MANAGER
═══════════════════════════════════════════════════════════════════════════════

Extracted from unified_pathfinder.py.

Provides Lattice3D: the 3D routing lattice with H/V discipline,
coordinate conversions, and CSR graph construction.
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional, Set, Tuple

import numpy as np

from .kicad_geometry import KiCadGeometry

# Import CSRGraph from the parent module (it remains in unified_pathfinder.py)
# Use a lazy import to avoid circular dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..unified_pathfinder import CSRGraph

logger = logging.getLogger(__name__)


class Lattice3D:
    """3D routing lattice with H/V discipline"""

    def __init__(self, bounds: Tuple[float, float, float, float], pitch: float, layers: int):
        self.bounds = bounds
        self.pitch = pitch
        self.layers = layers

        self.geom = KiCadGeometry(bounds, pitch, layer_count=layers)
        self.x_steps = self.geom.x_steps
        self.y_steps = self.geom.y_steps
        self.num_nodes = self.x_steps * self.y_steps * layers

        self.layer_dir = self._assign_directions()
        logger.info(f"Lattice: {self.x_steps}×{self.y_steps}×{layers} = {self.num_nodes:,} nodes")

    def _assign_directions(self) -> List[str]:
        """F.Cu=V (vertical escape routing), internal layers alternate H/V"""
        dirs = []
        for z in range(self.layers):
            if z == 0:
                # F.Cu has vertical routing for escape stubs
                dirs.append('v')
            else:
                # Internal layers alternate: In1.Cu=H, In2.Cu=V, In3.Cu=H, etc.
                dirs.append('h' if z % 2 == 1 else 'v')
        return dirs

    def get_legal_axis(self, layer: int) -> str:
        """Return 'h' or 'v' for which axis this layer allows."""
        if layer >= len(self.layer_dir):
            return 'h' if layer % 2 == 1 else 'v'
        return self.layer_dir[layer]

    def is_legal_planar_edge(self, from_x: int, from_y: int, from_layer: int,
                              to_x: int, to_y: int, to_layer: int) -> bool:
        """Check if planar edge follows H/V discipline."""
        if from_layer != to_layer:
            return True  # Vias always legal (checked separately)

        dx = abs(to_x - from_x)
        dy = abs(to_y - from_y)

        # Must be adjacent (Manhattan distance 1)
        if dx + dy != 1:
            return False

        # Check layer direction
        direction = self.get_legal_axis(from_layer)
        if direction == 'h':
            return dy == 0 and dx == 1  # Horizontal: only ±X
        else:
            return dx == 0 and dy == 1  # Vertical: only ±Y

    def get_legal_via_pairs(self, layer_count: int) -> set:
        """
        Return set of legal (from_layer, to_layer) via pairs.

        CRITICAL: Must include F.Cu (layer 0) → internal layer transitions!
        The escape planner creates stubs on F.Cu, and PathFinder must be able
        to create vias from F.Cu to whatever internal layer it chooses.
        """
        # Check config for via policy (default to FULL blind/buried)
        allow_any = True  # ALWAYS allow full blind/buried for convergence

        # Internal routing layers (exclude B.Cu which is layer_count-1)
        routing_layers = list(range(1, layer_count - 1))

        logger.info(f"[VIA-PAIRS] layer_count={layer_count}, routing_layers={len(routing_layers)}, "
                   f"allow_any={allow_any}")

        if allow_any:
            # FULL BLIND/BURIED: Any routing layer to any other routing layer
            legal_pairs = set()
            for z1 in routing_layers:
                for z2 in routing_layers:
                    if z1 != z2:
                        legal_pairs.add((z1, z2))

            # CRITICAL: Add F.Cu (layer 0) → internal layer transitions
            # This allows PathFinder to create escape vias from F.Cu to any internal layer
            for z in routing_layers:
                legal_pairs.add((0, z))  # F.Cu → internal layer
                legal_pairs.add((z, 0))  # internal layer → F.Cu (bidirectional)

            logger.info(f"[VIA-PAIRS] Generated {len(legal_pairs)} pairs: {len(routing_layers)}×{len(routing_layers)} internal + {len(routing_layers)}×2 F.Cu transitions")
            return legal_pairs

        # FALLBACK: Adjacent routing layers only
        legal_pairs = set()
        for i in range(len(routing_layers) - 1):
            z1, z2 = routing_layers[i], routing_layers[i+1]
            legal_pairs.add((z1, z2))
            legal_pairs.add((z2, z1))
        logger.info(f"[VIA-PAIRS] Generated {len(legal_pairs)} adjacent-only pairs (fallback mode)")
        return legal_pairs

    def node_idx(self, x: int, y: int, z: int) -> int:
        """(x,y,z) → flat"""
        return self.geom.node_index(x, y, z)

    def idx_to_coord(self, idx: int) -> Tuple[int, int, int]:
        """flat → (x,y,z)"""
        return self.geom.index_to_coords(idx)

    def world_to_lattice(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """mm → lattice"""
        return self.geom.world_to_lattice(x_mm, y_mm)

    def build_graph(self, via_cost: float, allowed_via_spans: Optional[Set[Tuple[int, int]]] = None, use_gpu=False):
        """
        Build graph with H/V constraints and flexible via spans.

        Args:
            via_cost: Base cost for via transitions
            allowed_via_spans: Set of (from_layer, to_layer) pairs allowed for vias.
                              If None, all layer pairs are allowed (full blind/buried support).
                              Layers are indexed 0..N-1.
            use_gpu: Enable GPU acceleration
        """
        # Import CSRGraph at runtime to avoid circular imports
        from ..unified_pathfinder import CSRGraph

        # Count edges to pre-allocate array (avoids MemoryError with 30M edges)
        edge_count = 0

        # Count H/V edges (exclude outer layers 0 and self.layers-1)
        for z in range(1, self.layers - 1):
            if self.layer_dir[z] == 'h':
                edge_count += 2 * self.y_steps * (self.x_steps - 1)
            else:  # 'v'
                edge_count += 2 * self.x_steps * (self.y_steps - 1)

        # Count via edges using ACTUAL legal pairs (not parameter guess)
        legal_via_pairs_set = self.get_legal_via_pairs(self.layers)
        via_edge_count = 2 * self.x_steps * self.y_steps * len(legal_via_pairs_set)
        edge_count += via_edge_count

        logger.info(f"Pre-allocating for {edge_count:,} edges ({via_edge_count:,} via edges for {len(legal_via_pairs_set)} pairs)")
        graph = CSRGraph(use_gpu, edge_capacity=edge_count)

        # Build lateral edges (H/V discipline, exclude outer layers 0 and self.layers-1)
        for z in range(1, self.layers - 1):
            direction = self.layer_dir[z]

            if direction == 'h':
                for y in range(self.y_steps):
                    for x in range(self.x_steps - 1):
                        u = self.node_idx(x, y, z)
                        v = self.node_idx(x+1, y, z)

                        # MANHATTAN VALIDATION
                        if not self.is_legal_planar_edge(x, y, z, x+1, y, z):
                            logger.error(f"[MANHATTAN-VIOLATION] Illegal H edge on layer {z}: ({x},{y}) → ({x+1},{y})")
                            continue  # Skip illegal edge

                        graph.add_edge(u, v, self.pitch)
                        graph.add_edge(v, u, self.pitch)
            else:  # direction == 'v'
                for x in range(self.x_steps):
                    for y in range(self.y_steps - 1):
                        u = self.node_idx(x, y, z)
                        v = self.node_idx(x, y+1, z)

                        # MANHATTAN VALIDATION
                        if not self.is_legal_planar_edge(x, y, z, x, y+1, z):
                            logger.error(f"[MANHATTAN-VIOLATION] Illegal V edge on layer {z}: ({x},{y}) → ({x},{y+1})")
                            continue  # Skip illegal edge

                        graph.add_edge(u, v, self.pitch)
                        graph.add_edge(v, u, self.pitch)

        # Build via edges using the SAME legal pairs as pre-allocation
        via_count = 0

        for x in range(self.x_steps):
            for y in range(self.y_steps):
                for (z_from, z_to) in legal_via_pairs_set:
                    # Only add if this specific pair is legal
                    span = abs(z_to - z_from)
                    span_alpha = 0.15
                    cost = via_cost * (1.0 + span_alpha * (span - 1))

                    u = self.node_idx(x, y, z_from)
                    v = self.node_idx(x, y, z_to)
                    graph.add_edge(u, v, cost)
                    graph.add_edge(v, u, cost)
                    via_count += 2

        # LOG what was built
        logger.info(f"Vias: {via_count} edges created")
        logger.info(f"Via policy: {len(legal_via_pairs_set)} layer pairs (FULL BLIND/BURIED ENABLED!)")
        for pair in sorted(list(legal_via_pairs_set))[:10]:
            logger.info(f"  Legal via: {pair[0]} ↔ {pair[1]}")
        if len(legal_via_pairs_set) > 20:
            logger.info(f"  ... and {len(legal_via_pairs_set) - 10} more pairs (showing first 10 only)")

        # Finalize the graph before validation (converts edge list to CSR format)
        graph.finalize(self.num_nodes, num_layers=self.layers)

        # MANHATTAN VALIDATION: Sample 1000 random edges and verify they're legal
        edge_count = len(graph.indices) if hasattr(graph, 'indices') else 0
        sample_size = min(1000, edge_count)

        if sample_size > 0:
            logger.info(f"[MANHATTAN-VALIDATION] Sampling {sample_size} edges to verify H/V discipline...")
            violations = 0

            import numpy as np

            # Convert indptr to CPU for validation (if it's on GPU)
            indptr_cpu = graph.indptr if isinstance(graph.indptr, np.ndarray) else graph.indptr.get()

            for _ in range(sample_size):
                # Pick random edge from CSR structure
                edge_idx = random.randint(0, edge_count - 1)

                # Get source node (find which node this edge belongs to) using binary search
                # indptr[u] <= edge_idx < indptr[u+1], so searchsorted gives us u+1
                u = int(np.searchsorted(indptr_cpu, edge_idx, side='right')) - 1

                # Get target node
                v = int(graph.indices[edge_idx]) if isinstance(graph.indices[edge_idx], (int, np.integer)) else int(graph.indices[edge_idx].get())

                # Convert to coordinates
                x_u, y_u, z_u = self.idx_to_coord(u)
                x_v, y_v, z_v = self.idx_to_coord(v)

                # Convert to Python ints for set membership testing
                z_u, z_v = int(z_u), int(z_v)

                # Check if it's a via (different layers)
                if z_u != z_v:
                    # Via edge - check if it's in legal pairs
                    if (z_u, z_v) not in legal_via_pairs_set and (z_v, z_u) not in legal_via_pairs_set:
                        logger.error(f"[MANHATTAN-VIOLATION] Illegal via: layer {z_u} ↔ {z_v} at ({x_u},{y_u})")
                        violations += 1
                else:
                    # Planar edge - check H/V discipline
                    if not self.is_legal_planar_edge(x_u, y_u, z_u, x_v, y_v, z_v):
                        logger.error(f"[MANHATTAN-VIOLATION] Illegal planar edge on layer {z_u}: ({x_u},{y_u}) → ({x_v},{y_v})")
                        violations += 1

            if violations > 0:
                logger.error(f"[MANHATTAN-VALIDATION] Found {violations} illegal edges in graph!")
                raise RuntimeError("Graph contains non-Manhattan edges")
            else:
                logger.info(f"[MANHATTAN-VALIDATION] All {sample_size} sampled edges are legal ✓")

        return graph
