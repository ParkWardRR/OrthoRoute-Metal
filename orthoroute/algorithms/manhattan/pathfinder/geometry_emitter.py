"""
═══════════════════════════════════════════════════════════════════════════════
GEOMETRY EMITTER
═══════════════════════════════════════════════════════════════════════════════

Extracted from unified_pathfinder.py (PathFinderRouter geometry methods).

Provides GeometryEmitter: converts routed node paths into drawable segments
and vias for KiCad export and GUI display.

Uses delegation pattern — stores a reference to the PathFinderRouter.
"""
from __future__ import annotations

import logging
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..unified_pathfinder import PathFinderRouter
    from ...domain.models.board import Board

logger = logging.getLogger(__name__)


class GeometryEmitter:
    """Converts routed paths to exportable geometry (tracks + vias).

    Delegation pattern: stores a reference to the parent PathFinderRouter
    and accesses its attributes through ``self.router``.
    """

    def __init__(self, router: 'PathFinderRouter'):
        self.router = router

    # ─── helper: coordinate conversion ───────────────────────────────────

    def _segment_world(self, a_idx: int, b_idx: int, layer: int, net: str):
        router = self.router
        ax, ay, _ = router.lattice.idx_to_coord(a_idx)
        bx, by, _ = router.lattice.idx_to_coord(b_idx)
        (ax_mm, ay_mm) = router.lattice.geom.lattice_to_world(ax, ay)
        (bx_mm, by_mm) = router.lattice.geom.lattice_to_world(bx, by)

        # QUANTIZE: Round to grid to prevent float drift
        pitch = router.lattice.geom.pitch
        origin_x = router.lattice.geom.grid_min_x
        origin_y = router.lattice.geom.grid_min_y

        ax_mm = origin_x + round((ax_mm - origin_x) / pitch) * pitch
        ay_mm = origin_y + round((ay_mm - origin_y) / pitch) * pitch
        bx_mm = origin_x + round((bx_mm - origin_x) / pitch) * pitch
        by_mm = origin_y + round((by_mm - origin_y) / pitch) * pitch

        return {
            'net': net,
            'layer': router.config.layer_names[layer] if layer < len(router.config.layer_names) else f"L{layer}",
            'x1': ax_mm, 'y1': ay_mm, 'x2': bx_mm, 'y2': by_mm,
            'width': router.config.grid_pitch * 0.6,
        }

    def _via_world(self, at_idx: int, net: str, from_layer: int, to_layer: int):
        router = self.router
        x, y, _ = router.lattice.idx_to_coord(at_idx)
        (x_mm, y_mm) = router.lattice.geom.lattice_to_world(x, y)

        # CRITICAL FIX: Quantize via coordinates to grid (same as _segment_world)
        # This ensures via centers EXACTLY match track endpoints (no epsilon mismatch!)
        pitch = router.lattice.geom.pitch
        origin_x = router.lattice.geom.grid_min_x
        origin_y = router.lattice.geom.grid_min_y
        x_mm = origin_x + round((x_mm - origin_x) / pitch) * pitch
        y_mm = origin_y + round((y_mm - origin_y) / pitch) * pitch

        # Normalize layer order (consistent output, KiCad accepts either way)
        if from_layer > to_layer:
            from_layer, to_layer = to_layer, from_layer

        return {
            'net': net,
            'x': x_mm, 'y': y_mm,
            'from_layer': router.config.layer_names[from_layer] if from_layer < len(router.config.layer_names) else f"L{from_layer}",
            'to_layer': router.config.layer_names[to_layer] if to_layer < len(router.config.layer_names) else f"L{to_layer}",
            'diameter': 0.25,  # hole (0.15) + 2×annular (0.05) = 0.25mm
            'drill': 0.15,     # hole diameter
        }

    def _path_is_manhattan(self, path: List[int]) -> bool:
        """Validate that path obeys Manhattan routing discipline"""
        router = self.router
        for a, b in zip(path, path[1:]):
            x0, y0, z0 = router.lattice.idx_to_coord(a)
            x1, y1, z1 = router.lattice.idx_to_coord(b)
            if z0 == z1:
                # Planar move: must be adjacent (Manhattan distance = 1)
                if (abs(x1 - x0) + abs(y1 - y0)) != 1:
                    logger.error(f"[PATH-INVALID-DETAIL] Planar non-adjacent: ({x0},{y0},{z0}) -> ({x1},{y1},{z1}), dist={abs(x1-x0)+abs(y1-y0)}")
                    return False
            else:
                # Via move: same X,Y, any Z distance (allow multi-layer vias for portals)
                if not ((x1 == x0) and (y1 == y0)):
                    logger.error(f"[PATH-INVALID-DETAIL] Via with X/Y change: ({x0},{y0},{z0}) -> ({x1},{y1},{z1})")
                    return False
        return True

    # ─── main geometry generation ────────────────────────────────────────

    def _generate_geometry_from_paths(self) -> Tuple[List, List]:
        """Generate tracks and vias from net_paths"""
        router = self.router
        tracks, vias = [], []

        for net_id, path in router.net_paths.items():
            if not path:
                continue

            # NOTE: Escape geometry is pre-computed by PadEscapePlanner and cached.
            # It will be merged with routed geometry in emit_geometry().

            # Generate tracks/vias from main path
            run_start = path[0]
            prev = path[0]
            prev_dir = None
            prev_layer = router.lattice.idx_to_coord(prev)[2]

            for node in path[1:]:
                x0, y0, z0 = router.lattice.idx_to_coord(prev)
                x1, y1, z1 = router.lattice.idx_to_coord(node)

                # Drop any planar segment on outer layers (shouldn't happen once graph/ROI are fixed)
                if z0 == z1 and (z0 == 0 or z0 == router.lattice.layers - 1):
                    logger.error(f"[EMIT-GUARD] refusing planar segment on outer layer {z0} for net {net_id}")
                    prev = node
                    prev_layer = z1
                    run_start = node
                    continue

                # VALIDATION: Check if nodes are adjacent (Manhattan distance should be 1)
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                dz = abs(z1 - z0)

                if dz == 0:  # Same layer - enforce H/V discipline
                    # Must be adjacent
                    if (dx + dy) != 1:
                        logger.error(f"[GEOMETRY-BUG] Non-adjacent nodes in path for net {net_id}: "
                                   f"({x0},{y0},{z0}) → ({x1},{y1},{z1}), Manhattan dist = {dx+dy}")
                        logger.error(f"[GEOMETRY-BUG] Path indices: prev={prev}, node={node}")
                        logger.error(f"[GEOMETRY-BUG] This creates diagonal segment! GPU parent pointers are CORRUPT!")
                        continue  # Skip illegal segment

                    # Check layer direction discipline
                    layer_axis = router.lattice.get_legal_axis(z0)
                    if layer_axis == 'h':
                        # H layer: y must be constant (horizontal movement)
                        if dy != 0:
                            logger.error(f"[LAYER-VIOLATION] H-layer {z0} has vertical move: "
                                       f"({x0},{y0})→({x1},{y1}), dy={dy}")
                            continue
                    else:  # 'v'
                        # V layer: x must be constant (vertical movement)
                        if dx != 0:
                            logger.error(f"[LAYER-VIOLATION] V-layer {z0} has horizontal move: "
                                       f"({x0},{y0})→({x1},{y1}), dx={dx}")
                            continue

                if z1 != z0:
                    # flush any pending straight run before via
                    if prev != run_start:
                        tracks.append(self._segment_world(run_start, prev, prev_layer, net_id))
                    vias.append(self._via_world(prev, net_id, z0, z1))
                    run_start = node
                    prev_dir = None
                else:
                    dir_vec = (np.sign(x1 - x0), np.sign(y1 - y0))
                    if prev_dir is None or dir_vec == prev_dir:
                        # keep extending run
                        pass
                    else:
                        # direction changed: flush previous run
                        tracks.append(self._segment_world(run_start, prev, prev_layer, net_id))
                        run_start = prev
                    prev_dir = dir_vec

                prev = node
                prev_layer = z1

            # flush final run
            if prev != run_start:
                tracks.append(self._segment_world(run_start, prev, prev_layer, net_id))

        # FINAL VALIDATION: Check all tracks are axis-aligned
        violations = []
        for i, track in enumerate(tracks):
            x1, y1 = track['x1'], track['y1']
            x2, y2 = track['x2'], track['y2']

            # Must be axis-aligned (one coordinate must be constant)
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > 0.001 and dy > 0.001:
                violations.append((i, track, dx, dy))

        if violations:
            logger.error(f"[EMIT-VALIDATION] Found {len(violations)} diagonal segments!")
            for i, track, dx, dy in violations[:5]:  # Show first 5
                logger.error(f"  Track {i}: ({track['x1']:.2f},{track['y1']:.2f})->({track['x2']:.2f},{track['y2']:.2f}), "
                           f"Delta=({dx:.2f},{dy:.2f}) on {track['layer']}")

            # In debug mode, raise error
            if __debug__:
                raise RuntimeError(f"{len(violations)} diagonal segments detected at emission")
        else:
            logger.info(f"[EMIT-VALIDATION] All {len(tracks)} tracks are axis-aligned ✓")

        # Count tracks by layer and direction
        layer_stats = {}
        for track in tracks:
            layer = track['layer']
            x1, y1 = track['x1'], track['y1']
            x2, y2 = track['x2'], track['y2']

            is_horizontal = (abs(y1 - y2) < 0.001)
            is_vertical = (abs(x1 - x2) < 0.001)

            if layer not in layer_stats:
                layer_stats[layer] = {'h': 0, 'v': 0}

            if is_horizontal:
                layer_stats[layer]['h'] += 1
            elif is_vertical:
                layer_stats[layer]['v'] += 1

        # Log per-layer statistics and check direction discipline
        for layer in sorted(layer_stats.keys()):
            h_count = layer_stats[layer]['h']
            v_count = layer_stats[layer]['v']
            logger.info(f"[LAYER-STATS] {layer}: {h_count} horizontal, {v_count} vertical")

            # Check if layer has wrong direction (extract layer index from name)
            # Assuming layer names like "In1.Cu", "In2.Cu", etc.
            if 'In' in layer:
                try:
                    layer_str = layer.replace('In', '').replace('.Cu', '')
                    layer_num = int(layer_str)
                    # Odd layers (In1, In3) = H, Even layers (In2, In4) = V
                    expected_dir = 'h' if layer_num % 2 == 1 else 'v'

                    if expected_dir == 'h' and v_count > h_count:
                        logger.warning(f"[LAYER-DIRECTION] {layer} should be H-dominant but has more V traces!")
                    elif expected_dir == 'v' and h_count > v_count:
                        logger.warning(f"[LAYER-DIRECTION] {layer} should be V-dominant but has more H traces!")
                except (ValueError, IndexError):
                    pass  # Skip if layer name doesn't match expected pattern

        return (tracks, vias)

    # ─── public API (delegated from PathFinderRouter) ────────────────────

    def emit_geometry(self, board) -> Tuple[int, int]:
        """
        Convert routed node paths into drawable segments and vias.
        - Clean geometry (for KiCad export): only if overuse == 0
        - Provisional geometry (for GUI feedback): always generated

        CRITICAL: Escape geometry is ALWAYS merged, even with overuse.
        Escapes are the connection from pads to the routing grid and must be exported.
        """
        router = self.router
        # Import GeometryPayload from the parent module
        from ..unified_pathfinder import GeometryPayload

        # Generate provisional geometry from routing paths
        provisional_tracks, provisional_vias = self._generate_geometry_from_paths()

        # ALWAYS merge escape geometry with routed geometry
        # Deduplicate helper
        def _dedupe(items, key_fn):
            seen, out = set(), []
            for it in items:
                k = key_fn(it)
                if k in seen:
                    continue
                seen.add(k)
                out.append(it)
            return out

        final_tracks = provisional_tracks
        final_vias = provisional_vias

        if hasattr(router, '_escape_tracks') and router._escape_tracks:
            # Merge escapes first (so they're visually "underneath")
            combined_tracks = router._escape_tracks + provisional_tracks
            combined_vias = router._escape_vias + provisional_vias

            # Deduplicate by geometric signature
            final_tracks = _dedupe(
                combined_tracks,
                lambda t: (t["net"], t["layer"],
                          round(t["x1"], 3), round(t["y1"], 3),
                          round(t["x2"], 3), round(t["y2"], 3),
                          round(t["width"], 3))
            )
            final_vias = _dedupe(
                combined_vias,
                lambda v: (v["net"], round(v["x"], 3), round(v["y"], 3),
                          v.get("from_layer"), v.get("to_layer"),
                          round(v.get("drill", 0), 3),
                          round(v.get("diameter", 0), 3))
            )

            logger.info(f"[ESCAPE-MERGE] escapes={len(router._escape_tracks)} + "
                       f"routed={len(provisional_tracks)} → "
                       f"total={len(final_tracks)} tracks after dedup")
            logger.info(f"[ESCAPE-MERGE] escape_vias={len(router._escape_vias)} + "
                       f"routed_vias={len(provisional_vias)} → "
                       f"total={len(final_vias)} vias after dedup")

        # Store merged geometry as provisional (for GUI display)
        router._provisional_geometry = GeometryPayload(final_tracks, final_vias)

        # Check for overuse (include via spatial violations)
        over_sum, over_cnt = router.accounting.compute_overuse(router_instance=router)

        if over_sum > 0:
            logger.warning(f"[EMIT] Overuse={over_sum}: showing merged geometry in GUI but not exporting to KiCad")
            router._geometry_payload = GeometryPayload([], [])  # No clean geometry for export
            # Return merged counts so GUI shows escapes + routes
            return (len(final_tracks), len(final_vias))

        # No overuse: emit clean geometry for KiCad export
        logger.info("[EMIT] Routing converged! Exporting clean geometry with escapes")
        router._geometry_payload = GeometryPayload(final_tracks, final_vias)
        return (len(final_tracks), len(final_vias))

    def get_geometry_payload(self):
        """
        Get geometry payload for GUI/export.

        Returns clean geometry if available (no overuse),
        otherwise returns provisional geometry so GUI can still display/export.
        """
        router = self.router
        # If clean geometry is empty but provisional exists, return provisional
        if (not router._geometry_payload.tracks and not router._geometry_payload.vias
            and hasattr(router, '_provisional_geometry')
            and (router._provisional_geometry.tracks or router._provisional_geometry.vias)):
            return router._provisional_geometry
        return router._geometry_payload

    def get_provisional_geometry(self):
        """Get provisional geometry for GUI feedback (always available)"""
        return self.router._provisional_geometry
