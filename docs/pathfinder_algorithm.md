# PathFinder Algorithm

This document describes OrthoRoute's implementation of the **PathFinder negotiated congestion** routing algorithm, originally developed for FPGA routing by McMurchie and Ebeling (1995) and adapted here for GPU-accelerated PCB Manhattan routing.

## Overview

PathFinder is an **iterative rip-up and reroute** algorithm. It routes all nets simultaneously, allows temporary sharing of routing resources (edges), and then uses escalating congestion penalties to negotiate exclusive resource ownership. Nets that share overused edges are ripped up and rerouted with higher costs in subsequent iterations until all congestion is resolved.

### Why PathFinder for PCBs?

Traditional PCB autorouters use sequential routing (route one net at a time), which is sensitive to net ordering. PathFinder removes this ordering dependency by letting all nets compete for resources and negotiating conflicts iteratively. This produces higher-quality results, especially on dense boards.

### Key Properties

- **Global optimization** — All nets compete simultaneously, avoiding local optima from sequential ordering.
- **Guaranteed convergence** — With monotonically increasing pressure, convergence is guaranteed for routable boards (ρ < 1.0).
- **GPU-accelerable** — The inner shortest-path computation is the dominant cost and maps efficiently to GPU parallelism.

---

## Algorithm Flowchart

```
┌─────────────────────────────────────────┐
│           INITIALIZE                     │
│  • Build 3D routing lattice              │
│  • Compute congestion ratio ρ            │
│  • Set initial parameters:               │
│    pres_fac = 1.0, hist = 0              │
│  • Order nets by HPWL (longest first)    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         ITERATION LOOP                   │
│         iter = 1, 2, ..., max_iters      │
│                                          │◄──────────────────┐
│  ┌─────────────────────────────────┐     │                   │
│  │  1. SELECT NETS TO ROUTE        │     │                   │
│  │     • Iter 1: ALL nets          │     │                   │
│  │     • Iter 2+: HOT SET only     │     │                   │
│  │       (nets using overused      │     │                   │
│  │        edges)                   │     │                   │
│  └──────────┬──────────────────────┘     │                   │
│             │                            │                   │
│             ▼                            │                   │
│  ┌─────────────────────────────────┐     │                   │
│  │  2. RIP UP selected nets        │     │                   │
│  │     • Decrement edge usage      │     │                   │
│  │     • Preserve history costs    │     │                   │
│  └──────────┬──────────────────────┘     │                   │
│             │                            │                   │
│             ▼                            │                   │
│  ┌─────────────────────────────────┐     │                   │
│  │  3. UPDATE EDGE COSTS           │     │                   │
│  │     cost(e) = base(e)           │     │                   │
│  │             + pres_fac × max(0, │     │                   │
│  │               usage(e)-cap(e))  │     │                   │
│  │             + hist_weight ×     │     │                   │
│  │               history(e)        │     │                   │
│  └──────────┬──────────────────────┘     │                   │
│             │                            │                   │
│             ▼                            │                   │
│  ┌─────────────────────────────────┐     │                   │
│  │  4. REROUTE EACH NET            │     │                   │
│  │     For each net in hot set:    │     │                   │
│  │     ┌───────────────────────┐   │     │                   │
│  │     │ a. Extract ROI subgraph│   │     │                   │
│  │     │ b. Run GPU SSSP       │   │     │                   │
│  │     │    (Dijkstra/SPFA)    │   │     │                   │
│  │     │ c. Backtrace path     │   │     │                   │
│  │     │ d. Commit edges       │   │     │                   │
│  │     └───────────────────────┘   │     │                   │
│  └──────────┬──────────────────────┘     │                   │
│             │                            │                   │
│             ▼                            │                   │
│  ┌─────────────────────────────────┐     │                   │
│  │  5. UPDATE HISTORY              │     │                   │
│  │     For each edge e:            │     │                   │
│  │     history(e) += hist_gain ×   │     │                   │
│  │       max(0, usage(e) - cap(e)) │     │                   │
│  └──────────┬──────────────────────┘     │                   │
│             │                            │                   │
│             ▼                            │                   │
│  ┌─────────────────────────────────┐     │                   │
│  │  6. ESCALATE PRESSURE           │     │                   │
│  │     pres_fac *= pres_fac_mult   │     │                   │
│  │     pres_fac = min(pres_fac,    │     │                   │
│  │                    pres_fac_max)│     │                   │
│  └──────────┬──────────────────────┘     │                   │
│             │                            │                   │
│             ▼                            │                   │
│  ┌─────────────────────────────────┐     │                   │
│  │  7. CHECK CONVERGENCE           │     │  No               │
│  │     • total_overuse == 0?  ──────────────────────────────┘
│  │     • stagnation detected?      │     │
│  │     • max_iters reached?        │     │
│  └──────────┬──────────────────────┘     │
│             │ Yes                        │
└─────────────┼────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│           FINALIZE                       │
│  • Extract tracks and vias               │
│  • Emit KiCad geometry                   │
│  • Report convergence statistics         │
└─────────────────────────────────────────┘
```

---

## Cost Function

The edge cost function is the core of PathFinder's negotiation mechanism. Every edge `e` in the routing graph has a total cost computed as:

```
cost(e) = base_cost(e) + pres_fac × present_penalty(e) + hist_cost_weight × history(e)
```

### Components

| Component | Formula | Purpose |
|-----------|---------|---------|
| `base_cost(e)` | Edge weight from lattice (1 for tracks, 8 for vias) | Physical routing cost |
| `present_penalty(e)` | `max(0, usage(e) - capacity(e))` | Current overuse penalty |
| `history(e)` | Accumulated overuse across iterations | Long-term memory |
| `pres_fac` | Escalates geometrically each iteration | Urgency of resolving conflicts |
| `hist_cost_weight` | Fixed multiplier (10.0–16.0) | Relative importance of history |

### Present Factor Escalation

```
pres_fac(iter) = pres_fac_init × (pres_fac_mult)^(iter − 1)
pres_fac(iter) = min(pres_fac(iter), pres_fac_max)
```

With default values (`pres_fac_init=1.0`, `pres_fac_mult=1.10`):

```
Iter  1: pres_fac = 1.00
Iter  5: pres_fac = 1.46
Iter 10: pres_fac = 2.36
Iter 20: pres_fac = 6.12  → capped at pres_fac_max=8.0
```

### History Accumulation

After each iteration, the history array is updated:

```
history(e) += hist_gain × max(0, usage(e) − capacity(e))
```

History accumulates **permanently** — it never decreases. This creates a "long-term memory" that discourages routes from returning to historically congested areas, even if those areas are temporarily free.

### Balance: History vs. Present

The `hist/pres ratio` should stay between 0.5 and 2.0 for stable convergence. See [tuning_guide.md](tuning_guide.md) for diagnostic details.

---

## Convergence Criteria

The negotiation loop terminates when **any** of these conditions is met:

1. **Zero overuse**: `total_overuse == 0` — All edges have `usage ≤ capacity`. This is the success condition.
2. **Stagnation**: Overuse has not improved for `stagnation_patience` consecutive iterations (default: 5).
3. **Max iterations**: `iter >= max_iterations` — Give up after the configured limit.

### What Convergence Means

```
CONVERGED (overuse = 0):
  Every edge in the routing graph is used by at most one net.
  All nets have legal, non-overlapping routes.

NOT CONVERGED (overuse > 0):
  Some edges are still shared by multiple nets.
  The result may still be partially usable but contains DRC violations.
```

---

## Hot-Set Selection

Starting from iteration 2, OrthoRoute uses **hot-set routing**: only nets that are currently using overused edges are ripped up and rerouted. This dramatically reduces computation per iteration.

### Selection Algorithm

```python
hot_set = set()
for net in all_routed_nets:
    for edge in net.committed_edges:
        if edge.usage > edge.capacity:
            hot_set.add(net)
            break
```

### Hot-Set Cap

A `hotset_cap` parameter (default: 150) limits the number of nets rerouted per iteration to prevent mass decommits that can destabilize convergence.

---

## Portal Escape Architecture

OrthoRoute includes a novel **portal escape** mechanism that dramatically improves routing success (from 16% to 80%+) for pads surrounded by obstructions.

### Problem

Pad terminals on the top or bottom layer may be surrounded by other pads, creating a local congestion bottleneck. Traditional routing forces all escape routes through the same layer, causing contention.

### Solution

Portal escapes add **virtual escape edges** from pad terminals to nearby nodes on adjacent layers via short "portal" vias. This gives each pad multiple escape directions across layers.

```
                     ┌─── Portal via (discounted cost)
                     │
  Layer 0 (F.Cu)     ● ─── Pad terminal
                     │
                     ▼
  Layer 1 (In1.Cu)   ●● ─── Portal landing nodes
                    ╱  ╲
                   ╱    ╲    Free routing channels
                  ╱      ╲
                 ●        ●
```

### Portal Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `portal_discount` | 0.4 | 60% discount on escape via cost |
| `portal_delta_min` | 3 | Minimum vertical offset (1.2 mm) |
| `portal_delta_max` | 12 | Maximum vertical offset (4.8 mm) |
| `portal_delta_pref` | 6 | Preferred offset (2.4 mm) |
| `portal_via_discount` | 0.15 | Escape via cost multiplier (85% off) |

---

## GPU Acceleration

The inner loop — finding the shortest path for each net — is the computational bottleneck. OrthoRoute offloads this to the GPU using a persistent-thread SPFA (Shortest Path Faster Algorithm) solver.

### CPU–GPU Work Split

```
CPU (Python):                        GPU (Metal/CUDA):
┌────────────────────┐               ┌────────────────────┐
│ PathFinder loop    │               │ SSSP solver        │
│ Net ordering       │               │ (Dijkstra/SPFA)    │
│ Cost updates       │    CSR graph  │                    │
│ History tracking   │ ────────────▶ │ wavefront_expand   │
│ Convergence check  │               │ or                 │
│ ROI extraction     │ ◀──────────── │ wavefront_multi    │
│                    │   distances   │                    │
└────────────────────┘   + preds     └────────────────────┘
```

### Metal Backend (This Fork)

The Metal backend uses the **persistent thread SPFA** solver with:

- **8,192 persistent threads** (16 threadgroups × 512 threads)
- **SIMD block stealing** — 32 work items per atomic, reducing contention by 32×
- **Delta-stepping buckets** — Cache-optimized frontier partitioning
- **Zero-dispatch grid barrier** — Software inter-threadgroup sync without CPU round-trips
- **Zero-copy UMA** — NumPy arrays map directly to Metal buffers

See [metal_kernel_internals.md](metal_kernel_internals.md) for full kernel documentation.

### CUDA Backend (Upstream)

The original CUDA backend uses CuPy with:

- Bellman-Ford or SPFA kernel dispatched per iteration
- `cupy.RawKernel` for custom GPU code
- Explicit CPU↔GPU memory transfers

---

## Region of Interest (ROI) Routing

For large boards, running SSSP on the full graph is wasteful — most of the graph is irrelevant to a given net. OrthoRoute extracts a **Region of Interest** (ROI) subgraph around each net's terminals.

### ROI Extraction

1. Compute the bounding box of all pads in the net.
2. Expand by `BASE_ROI_MARGIN_MM` (default: 4.0 mm) in all directions.
3. Extract all lattice nodes within this bounding box.
4. If the net fails to route, expand the ROI (`roi_widen_levels` times, each by `roi_widen_factor`).
5. If all widening levels fail, fall back to the full graph.

### ROI Size Limits

```
MAX_ROI_NODES = 20,000  (default cap)
gpu_roi_min_nodes = 1,000  (below this, use CPU — GPU dispatch overhead dominates)
```

---

## Iteration Timing

Typical timing for a 512-net board on Apple M4:

| Phase | Iter 1 | Iter 2+ |
|-------|--------|---------|
| Net selection | — | 0.1 s |
| Rip-up | — | 0.5 s |
| Cost update | 1 s | 1 s |
| GPU routing (all nets) | 90 s | 15 s |
| History update | 0.5 s | 0.5 s |
| **Total** | **~120 s** | **~20 s** |

Iteration 1 is always the slowest because all nets must be routed from scratch. Subsequent iterations only reroute the hot set.

---

## Configuration Reference

All PathFinder parameters are centralized in `pathfinder/config.py`. See [tuning_guide.md](tuning_guide.md) for detailed tuning guidance.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pres_fac_init` | 1.0 | Initial congestion pressure |
| `pres_fac_mult` | 1.10 | Pressure escalation per iteration |
| `pres_fac_max` | 8.0 | Maximum pressure cap |
| `hist_gain` | 0.20 | History accumulation rate |
| `hist_cost_weight` | 10.0 | History vs. present weighting |
| `max_iterations` | 40 | Maximum negotiation iterations |
| `stagnation_patience` | 5 | Iterations without improvement to stop |
| `via_cost` | 0.7 | Via edge cost multiplier |
| `batch_size` | 32 | Nets per routing batch |
| `grid_pitch` | 0.4 | Grid spacing in mm |
| `hotset_cap` | 150 | Maximum nets rerouted per iteration |

---

## References

1. McMurchie, L. and Ebeling, C. **"PathFinder: A Negotiation-Based Performance-Driven Router for FPGAs."** ACM/SIGDA FPGA, 1995.
2. Betz, V. and Rose, J. **"VPR: A New Packing, Placement and Routing Tool for FPGA Research."** 1997.
3. Meyer, U. and Sanders, P. **"Delta-Stepping: A Parallelizable Shortest Path Algorithm."** Journal of Algorithms, 2003.
4. bbenchoff. [OrthoRoute Build Log](https://bbenchoff.github.io/pages/OrthoRoute.html).

---

## Related Documentation

- [coordinate_system.md](coordinate_system.md) — Grid structure and GID calculation
- [tuning_guide.md](tuning_guide.md) — Parameter tuning for convergence
- [congestion_ratio.md](congestion_ratio.md) — Routability prediction
- [metal_kernel_internals.md](metal_kernel_internals.md) — GPU kernel implementation
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture overview
