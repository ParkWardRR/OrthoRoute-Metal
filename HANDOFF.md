# OrthoRoute-Metal Handoff Document

**Project:** [ParkWardRR/OrthoRoute-Metal](https://github.com/ParkWardRR/OrthoRoute-Metal)
**Parent Framework:** [ParkWardRR/CUDA2Metal-Graph](https://github.com/ParkWardRR/CUDA2Metal-Graph)
**Version:** 1.0.0
**Date:** 2026-07-04
**Status:** Production release, 80% roadmap complete

---

## What This Project Is

OrthoRoute-Metal is a GPU-accelerated PCB autorouter that runs on Apple Silicon. It is the primary case study for the CUDA2Metal-Graph framework -- a real-world application that was ported from NVIDIA CUDA/CuPy to Apple Metal compute shaders.

The autorouter converts a KiCad PCB design into a 3D lattice graph, then uses the PathFinder negotiated congestion algorithm with GPU-accelerated shortest-path solvers to find conflict-free copper routing for all nets simultaneously.

---

## Repository Structure

```
OrthoRoute-Metal/
  metal/                          # Rust + Metal compute backend
    src/kernels.metal             # 7 MSL compute kernels (891 lines)
    src/lib.rs                    # MetalDijkstra PyO3 struct
    Cargo.toml                    # orthoroute-mac crate
  vulkan/                         # Linux Vulkan stubs (future)
    src/lib.rs                    # VulkanDijkstra stubs
  orthoroute/                     # Python routing engine
    domain/models/                # Board, Net, Pad, Component, Layer
    infrastructure/
      gpu/                        # GPUProvider chain (Metal/CUDA/Vulkan/CPU)
      kicad/                      # KiCad file parser, IPC, SWIG adapters
      serialization/              # ORP/ORS format exporters
    algorithms/manhattan/
      pathfinder/                 # Core routing algorithm
        unified_pathfinder.py     # Main PathFinderRouter (5847 lines)
        lattice_manager.py        # Lattice3D (extracted)
        edge_accountant.py        # EdgeAccountant (extracted)
        geometry_emitter.py       # GeometryEmitter (extracted)
        convergence_manager.py    # ConvergenceManager (extracted)
        kicad_geometry.py         # Coordinate transforms
        config.py                 # PathFinderConfig
        constants.py              # Named constants
    presentation/
      plugin/kicad_plugin.py      # KiCad Action Plugin
      gui/                        # Qt GUI (optional)
  tests/                          # 29 test files, 529 tests
  docs/                           # 14 documentation files
  ci/                             # Local CI scripts (OrbStack)
```

---

## Metal Compute Kernels

All 7 kernels are in `metal/src/kernels.metal`:

| Kernel | Lines | Purpose |
|--------|:-----:|---------|
| `clear_counters` | 5 | Reset queue state |
| `wavefront_expand_all` | 155 | Persistent-thread SPFA with delta-stepping, SIMD block stealing, SIMD prefix-sum queue compaction, zero-dispatch grid barrier |
| `negotiation_kernel` | 25 | History-based congestion pressure with SIMD reduction |
| `roi_extractor` | 35 | GPU ROI extraction with 3D bounding box + atomic compaction |
| `via_kernels` | 80 | Via cost with hard-block enforcement + pooling penalties |
| `spfa_setup_kernel` | 20 | Frontier initialization from distance arrays |
| `wavefront_expand_multi` | 120 | Multi-net parallel SPFA (2D grid dispatch) |

### Key Optimizations Implemented

1. **Persistent Thread Work-Stealing Queue** -- 8192 threads launched once, process entire algorithm in 1 command buffer
2. **SIMD Block Stealing** -- 32 frontier nodes consumed per 1 global atomic (read side)
3. **SIMD Prefix-Sum Queue Compaction** -- `simd_enqueue()` batches frontier writes, 1 atomic per 32 threads (write side)
4. **Delta-Stepping Buckets** -- Nodes processed in distance order, deferred nodes re-queued to future buckets
5. **Zero-Dispatch Grid Barrier** -- Software atomic spinlock barrier, no CPU round-trips between iterations

---

## GPU Provider Chain

```
CUDA (cupy) -> Metal (orthoroute_mac) -> Vulkan (stub) -> CPU (numpy)
```

Detection is automatic at import time in `orthoroute/infrastructure/gpu/__init__.py`. The `GPUProvider` abstract interface defines: `create_array`, `copy_array`, `to_cpu`, `to_gpu`, `dijkstra`, `is_available`.

---

## Testing

**529 tests, 13 skipped (Qt), 0.76s**

| Category | Files | Tests |
|----------|:-----:|:-----:|
| Domain models | 1 | 48 |
| CSR graph | 1 | 19 |
| Lattice builder | 1 | 22 |
| Spatial hash | 1 | 10 |
| Via accounting | 1 | 8 |
| Portal escape | 2 | 23 |
| Pad mapping | 2 | 21 |
| GPU/CPU parity | 2 | 17 |
| Performance benchmarks | 2 | 9 |
| Regression | 1 | 10 |
| PathFinder convergence | 1 | 10 |
| KiCad file parser | 1 | 21 |
| KiCad geometry | 1 | 27 |
| KiCad serialization | 2 | 38 |
| KiCad layers + colors | 1 | 35 |
| KiCad E2E integration | 1 | 45 |
| Configuration | 1 | 15 |
| Real global grid | 1 | 32 |
| Other | 5 | ~119 |
| **Total** | **29** | **529** |

---

## Known Limitations

1. **No live KiCad IPC in headless mode.** The IPC adapter requires KiCad running with the scripting console open on port 5555. Headless routing uses file parsing only.
2. **Qt dependency for GUI.** The visualization GUI requires PyQt6. 12 color scheme tests are skipped when Qt is not installed.
3. **Vulkan backend is stubs only.** The `vulkan/` directory and `VulkanProvider` exist but contain no working GPU code. Implementation requires Vulkan SDK + ash/vulkano crate.
4. **unified_pathfinder.py is still large.** Despite extracting 4 modules, the main file is still ~5000 lines. The delegation pattern preserves the public API but the internal coupling through `self.router` references means the extracted modules are not fully independent.
5. **No differential pair routing.** Single-ended routing only. Differential pairs (USB, HDMI, etc.) are a future feature.
6. **No layer compaction.** The router uses all available layers. A compaction pass to minimize layer count is planned but not implemented.

---

## What Remains (20% of Roadmap)

### Future Features (Not Started)
- Layer compaction algorithm
- Differential pair routing
- Net class-aware routing (impedance control)
- Vulkan compute backend implementation
- Platform distribution (pip wheel, Homebrew formula)
- KiCad plugin marketplace submission
- Real-time DRC violation overlay in GUI
- Board-level timing analysis
- Thermal via insertion
- Power plane flood fill

### Testing Gaps
- Integration tests with real .kicad_pcb files (requires sample boards)
- GPU metal kernel unit tests (require Metal-capable CI runner)
- Memory leak / resource exhaustion stress tests
- Multi-board batch routing tests

---

## How to Build and Run

```bash
# Clone
git clone https://github.com/ParkWardRR/OrthoRoute-Metal.git
cd OrthoRoute-Metal

# Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Build Metal extension (requires Rust + Xcode CLI tools)
cd metal && cargo build --release && cd ..
cp metal/target/release/liborthoroute_mac.dylib orthoroute_mac.so

# Run tests
pytest tests/ -q  # 529 passed, 13 skipped, 0.76s

# Route a board (headless)
python main.py --headless path/to/board.kicad_pcb

# Route with GUI
python main.py --gui path/to/board.kicad_pcb
```

---

## Key Files for New Contributors

| File | Why It Matters |
|------|---------------|
| `metal/src/kernels.metal` | All GPU compute kernels -- start here to understand the algorithm |
| `orthoroute/algorithms/manhattan/unified_pathfinder.py` | Main routing engine -- PathFinderRouter class |
| `orthoroute/infrastructure/gpu/__init__.py` | GPU auto-detection and provider factory |
| `orthoroute/domain/models/board.py` | Board domain model -- all data structures |
| `tests/conftest.py` | Test fixtures -- how to create Board objects without files |
| `docs/metal_kernel_internals.md` | Deep dive into kernel optimizations |
| `docs/pathfinder_algorithm.md` | PathFinder negotiated congestion algorithm reference |
| `ROADMAP.md` | What's done and what's left |

---

## Relationship to CUDA2Metal-Graph

OrthoRoute-Metal is **Case Study #1** for the CUDA2Metal-Graph framework (Phase 5 of the parent roadmap). The translation patterns, Rust runtime architecture, and Metal kernel primitives developed here are documented back in the parent repo:

- `CUDA_TO_MSL_MAPPING.md` -- Atomic, barrier, memory space mappings
- `docs/translation_patterns.md` -- High-level CUDA-to-Metal patterns
- `docs/LESSONS_LEARNED.md` -- 17 architectural lessons from this integration
- `docs/PARITY_TESTING.md` -- Golden tensor comparison methodology
- `M4_OPTIMIZATIONS.md` -- Apple Silicon-specific performance tuning

The parent framework provides the reusable Rust+Metal compute bridge. OrthoRoute-Metal consumes it as `orthoroute_mac` (the compiled dylib) and wraps it in `MetalProvider` for the Python routing engine.
