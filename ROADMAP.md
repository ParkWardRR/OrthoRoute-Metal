# OrthoRoute-Metal — Project Roadmap

> **Last Updated:** July 4, 2026
>
> Legend: ✅ Completed &nbsp;|&nbsp; 🔄 In Progress &nbsp;|&nbsp; ⬜ Not Started

---

## 1. Metal GPU Backend (Rust + MSL)

### Core Compute Kernels

- [x] **Persistent Thread SPFA Solver** (`wavefront_expand_all`) — Single-dispatch, work-stealing shortest-path kernel with Delta-Stepping bucket frontier
- [x] **Multi-Net Parallel Solver** (`wavefront_expand_multi`) — Simultaneous routing of multiple nets using batched distance arrays
- [x] **Frontier Initialization** (`spfa_setup_kernel`) — SPFA frontier setup from distance arrays
- [x] **Queue State Reset** (`clear_counters`) — Counter/queue cleanup between iterations
- [x] **PathFinder Negotiation Kernel** (`negotiation_kernel`) — SIMD-group reduction for history-based congestion pressure
- [x] **ROI Distance Extraction** (`roi_extractor_mixin`) — Region-of-interest distance extraction per node
- [x] **Via Cost Computation** (`via_kernels`) — Via cost initialization per via

### GPU Architecture & Optimization

- [x] **Zero-Dispatch Software Grid Barrier** — Custom inter-threadgroup sync via `threadgroup_barrier(mem_flags::mem_device)` + atomic generation counters; eliminates 38µs/dispatch overhead
- [x] **SIMD Block Stealing** — Metal SIMD-group intrinsics (`simd_broadcast_first`, `simd_is_first`) steal 32 work items per atomic op, reducing contention 32×
- [x] **Atomic Float Minimum** — CAS loop for `atomic_fetch_min` on floats (not natively supported in Metal MSL)
- [x] **Zero-Copy UMA Memory Mapping** — NumPy arrays mapped directly into Metal buffers via `MTLResourceStorageModeShared` and `new_buffer_with_bytes_no_copy`
- [x] **AMX Coprocessor Offloading** — Dense matrix ops (congestion map updates) routed to Apple AMX via Accelerate `cblas_sgemm`
- [x] **Persistent Grid Constraint** — Safe limit of 16 threadgroups × 512 threads = 8,192 total on M4 (10-core GPU)

### Rust / PyO3 Bindings

- [x] **MetalDijkstra struct** — Pipeline management, buffer allocation, command queue
- [x] **`set_graph_csr()`** — Zero-copy CSR graph import from NumPy
- [x] **`set_distances_csr()` / `get_distances()`** — Distance array exchange (zero-copy)
- [x] **`reset_predecessors()` / `get_predecessors()`** — Predecessor array management
- [x] **`setup_spfa()`** — Frontier initialization dispatch
- [x] **`execute_until_convergence()`** — Main solve loop with configurable params (max_iters, batch_size, threadgroup_size, delta)
- [x] **`amx_sgemm_py()`** — Python-callable AMX matrix multiply
- [x] **`extract_roi()`** — Full Metal kernel dispatch with ROI bounds, coordinate arrays, returns NumPy
- [x] **`process_vias()`** — Full Metal kernel dispatch with capacity/usage/base_cost
- [x] **Cargo crate** (`orthoroute-metal v1.0.0`) — `cdylib` target with PyO3, metal-rs 0.27, numpy 0.29 dependencies

### Metal Kernel Completeness

> [!NOTE]
> All Metal kernels are now fully implemented with feature parity to their CUDA equivalents.

| Kernel | Metal Status | CUDA Equivalent |
|--------|-------------|------------------|
| `wavefront_expand_all` | ✅ Full (persistent SSSP) | ~equivalent |
| `wavefront_expand_multi` | ✅ Full (multi-net) | ~equivalent |
| `negotiation_kernel` | ✅ Full (SIMD reduction) | ~equivalent |
| `spfa_setup_kernel` | ✅ Full | ~equivalent |
| `clear_counters` | ✅ Full | ~equivalent |
| `roi_extractor_mixin` | ✅ Full (3D bounding box + coordinate filtering + atomic compaction) | ~equivalent |
| `via_kernels` | ✅ Full (hard-block + pooling penalties) | ~equivalent |

### CUDA ↔ Metal Parity

- [x] **Distance parity** — Bitwise float32 match across all 36 test cases
- [x] **Path parity** — Identical node sequences across all 36 test cases
- [x] **Reachable node parity** — Exact count match across all 36 test cases
- [x] **Graph sizes tested** — 2K / 8K / 30K / 45K / 60K / 180K nodes (6 sizes × 6 tests = 36 total)
- [x] **Golden tensors captured** — From RTX 2080 Ti on Vast.ai ($0.083/hr)

---

## 2. Python Routing Engine

### PathFinder Algorithm

- [x] **UnifiedPathFinder** — Main routing engine (282KB, ~6,000 lines) with negotiated congestion
- [x] **Mixin architecture** — Separated concerns via mixins:
  - [x] `lattice_builder_mixin.py` — 3D Manhattan lattice construction
  - [x] `graph_builder_mixin.py` — CSR matrix assembly
  - [x] `pathfinding_mixin.py` — GPU/CPU shortest-path dispatch
  - [x] `negotiation_mixin.py` — PathFinder iteration + pressure escalation
  - [x] `geometry_mixin.py` — Track/via geometry extraction
  - [x] `roi_extractor_mixin.py` — Region-of-interest distance extraction
  - [x] `diagnostics_mixin.py` — Runtime diagnostics & logging
  - [x] `via_kernels.py` — Via cost management
- [x] **Pad Escape Planner** (`pad_escape_planner.py`, 48KB) — Portal escape architecture (16% → 80%+ routing success)
- [x] **Board Analyzer** — Pre-routing board analysis
- [x] **Layer Analyzer** — Layer utilization & direction analysis
- [x] **Parameter Derivation** — Automatic PathFinder parameter tuning from board characteristics
- [x] **Congestion Ratio (ρ)** — Pre-routing routability prediction

### GPU/CPU Provider Abstraction

- [x] **CUDAProvider** — CuPy-based CUDA acceleration
- [x] **CPUProvider** — NumPy-based CPU fallback
- [x] **MetalProvider** — Full `GPUProvider` interface in `orthoroute/infrastructure/gpu/metal_provider.py`, wrapping `orthoroute_mac.MetalDijkstra`
  - [x] Create `MetalProvider` class wrapping `orthoroute_mac.MetalDijkstra`
  - [x] Wire into `unified_pathfinder.py` — CUDA→Metal→CPU priority fallback
  - [x] Replace CUDA via/ROI kernels with Metal equivalents
- [x] **Auto-detection** — `get_best_provider()` in `gpu/__init__.py` with CUDA→Metal→CPU priority

### Domain Models

- [x] **Board, Component, Net, Pad, Layer** — Core PCB domain objects
- [x] **Route, Segment, Via** — Routing result models
- [x] **DRCConstraints, NetClass** — Design rule and net class models
- [x] **Coordinate** — Position data type

### Clean Architecture Layers

- [x] **Domain layer** (`orthoroute/domain/`) — Pure business logic, models, events, services
- [x] **Application layer** (`orthoroute/application/`) — Commands, queries, orchestrator, interfaces
- [x] **Infrastructure layer** (`orthoroute/infrastructure/`) — KiCad adapters, GPU providers, persistence, serialization
- [x] **Presentation layer** (`orthoroute/presentation/`) — GUI, plugin, pipeline
- [x] **Shared layer** (`orthoroute/shared/`) — Configuration, exceptions, utilities

---

## 3. KiCad Integration

### Adapters

- [x] **IPC Adapter** (`ipc_adapter.py`, 15KB) — KiCad 9.0+ IPC API integration
- [x] **SWIG Adapter** (`swig_adapter.py`, 13KB) — Legacy `pcbnew.ActionPlugin` support
- [x] **File Parser** (`file_parser.py`, 17KB) — Direct `.kicad_pcb` file parsing
- [x] **Rich KiCad Interface** (`rich_kicad_interface.py`, 33KB) — Enhanced board data extraction

### Plugin System

- [x] **KiCadPlugin class** — Main plugin entry point with GUI and non-GUI modes
- [x] **`plugin.json`** — Modern schema v1 registration for IPC plugins
- [x] **SWIG `__init__.py`** — ActionPlugin registration for PCM-compatible packages
- [x] **Icon assets** — 24px toolbar icon, 64px catalog icon, logo assets

### Build & Packaging

- [x] **Manual IPC package** (`build.py`) — ZIP package with `INSTALL.txt` for manual install
- [x] **PCM SWIG package** (`build.py --pcm`) — Plugin Content Manager compatible package
- [ ] **PCM IPC package** — Blocked by KiCad bug (Windows crash on `runtime: "ipc"`, GitLab #19465)
- [x] **Local CI pipeline** — `ci/run.sh` + Dockerfile for OrbStack (not GitHub Actions, per project preference)

---

## 4. Execution Modes

### Entry Points (all working)

- [x] **KiCad Plugin with GUI** — `python main.py plugin` or `python main.py` (default)
- [x] **KiCad Plugin without GUI** — `python main.py plugin --no-gui`
- [x] **CLI mode** — `python main.py cli board.kicad_pcb [-o output/]`
- [x] **Headless cloud routing** — `python main.py headless input.ORP [-o output.ORS]`
- [x] **Automated Manhattan test** — `python main.py --test-manhattan`
- [x] **Headless autoroute test** — `python main.py --autoroute`
- [x] **Via pathfinding test** — `python main.py --test-via`

### Headless / Cloud Features

- [x] **ORP import** — Load board from `.ORP` export file
- [x] **ORS export** — Save routing solution to `.ORS` file (gzip-compressed JSON)
- [x] **Iteration metrics** — Per-iteration convergence tracking in ORS
- [x] **Checkpoint interval** — Configurable checkpoint save interval (`--checkpoint-interval`)
- [x] **GPU/CPU mode flags** — `--use-gpu` and `--cpu-only` overrides
- [x] **Max iterations** — Configurable via `--max-iterations`
- [x] **Checkpoint resume** — `--resume-checkpoint` loads ORS, recovers iteration count
- [x] **Progress webhooks** — `--webhook-url` flag with non-blocking POST notifications

---

## 5. GUI & Visualization

### PyQt6 Interactive Viewer

- [x] **Main Window** (`main_window.py`, 181KB) — Full-featured PCB viewer with routing controls
- [x] **PathFinder Stats Widget** — Real-time convergence metrics display
- [x] **KiCad Color Theme** — Authentic KiCad layer coloring (`kicad_colors.py`)
- [x] **PCB rendering** — Board outline, pads, tracks, vias, components, keepout zones
- [x] **Auto-start routing** — `run_with_gui_autostart()` for automated testing
- [x] **Route selected nets** — `_route_selected_nets()` with rollback support
- [x] **Clear routes** — `_clear_routes()` with full state reset
- [x] **Rollback route** — `_rollback_route()` with deep-copy restore

### Visualization Tools

- [x] **Iteration video generator** (`viz/generate_iteration_video.py`) — Animate routing convergence over iterations
- [x] **Net tour video generator** (`viz/generate_net_tour_video.py`) — Visualize net-by-net routing order
- [ ] **3D layer stack viewer** — Interactive 3D view of routed layers
- [x] **Congestion heatmap overlay** — Green→yellow→red density grid overlay

---

## 6. Serialization & File Formats

- [x] **ORP format** — Board geometry + design rules export (JSON + gzip)
- [x] **ORS format** — Routing solution export (JSON + gzip)
- [x] **ORP exporter** (`orp_exporter.py`, 35KB) — Full board export with pad/net/DRC data
- [x] **ORS exporter** (`ors_exporter.py`, 17KB) — Solution export with tracks, vias, metrics
- [x] **ORP/ORS importer** (`serialization.py`, 18KB + `__init__.py`) — Import/convert functions
- [x] **File format documentation** (`docs/ORP_ORS_file_formats.md`)
- [ ] **ORP/ORS v2 format** — Binary format for faster load/save of large boards

---

## 7. Documentation

### Completed

- [x] **README.md** (14.5KB) — Comprehensive project overview, benchmarks, usage, API examples
- [x] **ARCHITECTURE.md** — Three-layer architecture, data flow, kernel execution model, design decisions
- [x] **BENCHMARK_METHODOLOGY.md** — Hardware specs, graph sizes, measurement protocol, metric definitions
- [x] **ORP_ORS_file_formats.md** — File format specification with examples
- [x] **barrel_conflicts_explained.md** (13KB) — Via barrel conflict analysis and handling
- [x] **cloud_gpu_setup.md** (19KB) — Complete Vast.ai cloud GPU setup guide
- [x] **congestion_ratio.md** — Routability prediction metric explanation
- [x] **contributing.md** (16KB) — Contributor guide with architecture overview, coding standards, PR process
- [x] **layer_compaction.md** (34KB) — Research document on post-routing layer minimization
- [x] **plugin_manager_integration.md** (13KB) — KiCad PCM integration challenges and solutions
- [x] **tuning_guide.md** (15KB) — PathFinder parameter tuning reference
- [x] **NOTICE.md** — Upstream MIT license attribution
- [x] **LICENSE.md** — Blue Oak Model License 1.0.0 (Metal backend)

### Completed (since initial roadmap)

- [x] **API reference** (`docs/api_reference.md`) — Python API documentation
- [x] **Coordinate system guide** (`docs/coordinate_system.md`) — How (x, y, z) maps to mm and layers
- [x] **PathFinder algorithm deep-dive** (`docs/pathfinder_algorithm.md`) — Algorithm explanation
- [x] **Metal kernel internals** (`docs/metal_kernel_internals.md`) — MSL code walkthrough
- [x] **CHANGELOG.md** — Version history with breaking changes

---

## 8. Testing

### Existing Tests

- [x] **CUDA ↔ Metal parity tests** — 36/36 pass (golden tensor comparison)
- [x] **Mock integration test** (`metal/mock_orthoroute.py`) — Rust/PyO3 integration smoke test
- [x] **CLI test modes** — `--test-manhattan`, `--autoroute`, `--test-via` built into main.py
- [x] **TestBackplane board** — 774KB test KiCad PCB file for integration testing

### Completed (since initial roadmap)

- [x] **Unit test framework** — `pytest.ini`, `conftest.py`, 29 test files
- [x] **Lattice builder tests** (`test_lattice.py`) — Node count, Manhattan adjacency, layer discipline
- [x] **CSR matrix integrity tests** (`test_csr_graph.py`) — Index bounds, symmetry, weight validation
- [x] **Via pooling accounting tests** (`test_via_accounting.py`) — Column usage counts, barrel conflict detection
- [x] **Board analyzer tests** — Board analysis coverage
- [x] **Layer analyzer tests** — Layer utilization coverage
- [x] **Config, data structures, spatial hash tests** — Core infrastructure coverage
- [x] **DRC, serialization, CPU fallback tests** — Integration coverage
- [x] **Domain models, grid, real_global_grid tests** — Full domain model coverage

### Completed (v1.0.0)

- [x] **Portal escape planning tests** — `test_portal_escape_advanced.py` — DRC clearance, determinism
- [x] **Pad mapping tests** — `test_pad_mapping_advanced.py` — Nearest-node finding, multi-layer mapping
- [x] **PathFinder convergence tests** — `test_pathfinder_convergence.py` — Pressure escalation, stagnation
- [x] **GPU/CPU parity tests** — `test_gpu_cpu_parity_advanced.py` — Dijkstra correctness, provider factory
- [x] **Regression test suite** — `test_regression.py` — CSR integrity, usage monotonicity
- [x] **Performance benchmarks** — `test_benchmarks.py` — Timing assertions for critical paths
- [x] **KiCad integration tests** (180 tests):
  - [x] `test_kicad_e2e.py` — Full pipeline board→geometry→serialization
  - [x] `test_kicad_file_parser.py` — S-expression parsing & board construction
  - [x] `test_kicad_geometry.py` — Coordinate transforms & edge validation
  - [x] `test_kicad_serialization.py` — ORP/ORS round-trip with compression
  - [x] `test_kicad_layers.py` — Layer normalization, regex, color schemes

---

## 9. Performance & Benchmarking

### Completed

- [x] **Traversal time benchmarks** — 6 graph sizes, 3 CUDA GPUs + M4 Metal
- [x] **Throughput benchmarks** — Edges/sec comparison (peak: 111.4B edges/sec on M4)
- [x] **Effective bandwidth analysis** — Cache amplification factor (831.5 GB/s effective vs 120 GB/s DRAM)
- [x] **Real-world PCB benchmark** — Class A Amplifier board (26 sec, 38 tracks, 79 vias)
- [x] **CUDA crossover analysis** — Metal faster above ~150K nodes

### Planned

- [ ] **M1/M2/M3 benchmarks** — Currently only M4 tested
- [ ] **Large board benchmarks** — Boards with 1,000+ nets on Metal backend
- [ ] **Memory usage profiling** — Peak GPU memory consumption across board sizes
- [ ] **Latency breakdown** — Per-phase timing (lattice build, CSR assembly, routing, geometry emit)
- [ ] **Multi-net kernel benchmarks** — `wavefront_expand_multi` vs serial routing comparison

---

## 10. Refactoring & Code Quality

### Known Technical Debt

- [x] **UnifiedPathFinder decomposition** — 282KB file decomposed into 4 standalone modules:
  - [x] `LatticeManager` (`lattice_manager.py`) — Lattice3D class for 3D grid construction & management
  - [x] `EdgeAccountant` (`edge_accountant.py`) — Edge usage tracking, overuse computation, cost updates
  - [x] `ConvergenceManager` (`convergence_manager.py`) — PathFinder negotiation loop coordinator (delegation pattern)
  - [x] `GeometryEmitter` (`geometry_emitter.py`) — Track/via geometry extraction from committed paths
- [x] **Remove `.backup` / `.bak` files** — 6 files deleted
- [x] **Consolidate configuration** — `PathFinderConfig.from_env()`, `from_json()`, `merge()` classmethods added; single source of truth
- [x] **Type hints** — `from __future__ import annotations` added to 12 files
- [x] **Remove commented-out code** — ~300 lines removed
- [x] **Extract magic numbers** — `constants.py` with named constants (EWMA_ALPHA, PRESSURE_MULTIPLIER, GPU_ROI_THRESHOLD, etc.)
- [x] **Fix `hasattr()` fragility** — 15 patterns replaced with proper state management
- [x] **Version consistency** — All packages aligned to v1.0.0 (`__init__.py`, `setup.py`, `Cargo.toml`)

---

## 11. Future Features

### Layer Compaction (Research Phase — Documented)

- [ ] **Layer utilization analysis** — Per-layer routing density stats
- [ ] **Minimal layer count estimation** — Target layer calculation
- [ ] **Conservative migration** — Direct layer reassignment without rerouting
- [ ] **Aggressive compaction** — Rip-up and reroute with layer constraints
- [ ] **PathFinder layer mask** — Constrain routing to specific layers
- [ ] **GUI integration** — "Optimize Layer Count..." menu item
- [ ] **CLI flag** — `python main.py compact solution.ORS --target-layers 20`

### Advanced Routing

- [ ] **Blind/buried via support** — Beyond current through-hole only
- [ ] **Differential pair routing** — Matched-length paired traces
- [ ] **Length-matched routing** — Signal integrity constraints
- [ ] **Bus routing** — Grouped signal routing
- [ ] **Impedance-aware routing** — Controlled impedance trace width/spacing

### Platform & Distribution

- [ ] **Linux Metal alternative** — Vulkan compute backend for non-Apple platforms
- [ ] **Windows CUDA optimization** — Equivalent persistent-thread optimizations for CUDA
- [ ] **PyPI package** — `pip install orthoroute` with pre-built Metal wheels for macOS
- [ ] **Homebrew formula** — `brew install orthoroute`
- [ ] **KiCad PCM listing** — Official PCM repository inclusion (pending KiCad IPC bug fix)

### Vulkan Compute Backend

> Stub infrastructure created in `vulkan/` — see [`vulkan/README.md`](vulkan/README.md) for full details.

- [ ] **SPIR-V compute shaders** — 7 kernels matching Metal MSL (`wavefront_expand_all`, `wavefront_expand_multi`, `spfa_setup`, `clear_counters`, `negotiation`, `roi_extractor`, `via_kernels`)
- [ ] **ash/vulkano Rust bindings with PyO3 bridge** — `VulkanDijkstra` struct in `vulkan/src/lib.rs` (stub created)
- [ ] **VulkanProvider Python wrapper** — `orthoroute/infrastructure/gpu/vulkan_provider.py` (stub created)
- [ ] **Linux GPU auto-detection** — Vulkan → CPU fallback in `gpu/__init__.py` (wiring created, `is_available()` returns False)
- [ ] **CI testing on Linux with Vulkan SDK** — GitHub Actions or OrbStack pipeline with Vulkan SDK installed

---

## Summary

| Category | Completed | In Progress | Not Started | Total |
|----------|:---------:|:-----------:|:-----------:|:-----:|
| Metal GPU Backend | 24 | 0 | 0 | **24** |
| Python Routing Engine | 23 | 0 | 0 | **23** |
| KiCad Integration | 13 | 0 | 1 | **14** |
| Execution Modes | 13 | 0 | 0 | **13** |
| GUI & Visualization | 11 | 0 | 1 | **12** |
| Serialization | 6 | 0 | 1 | **7** |
| Documentation | 18 | 0 | 0 | **18** |
| Testing | 25 | 0 | 0 | **25** |
| Performance | 5 | 0 | 5 | **10** |
| Refactoring | 8 | 0 | 0 | **8** |
| Future Features | 0 | 0 | 22 | **22** |
| **TOTAL** | **146** | **0** | **30** | **176** |

> **Overall completion: ~83%** — All stubs fully implemented. `UnifiedPathFinder` decomposed into 4 standalone modules (`LatticeManager`, `EdgeAccountant`, `GeometryEmitter`, `ConvergenceManager`). Metal GPU backend fully integrated via `MetalProvider` with CUDA→Metal→Vulkan→CPU auto-fallback. 29 test files with 529 tests passing (13 skipped — Qt) in 0.76s. Vulkan compute backend stub infrastructure in place. KiCad integration fully tested (file parser, geometry, serialization, layers, colors). Remaining work: future features (layer compaction, differential pair routing, Vulkan implementation, platform distribution) and performance benchmarks.
