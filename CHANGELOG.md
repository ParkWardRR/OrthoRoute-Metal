# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - Unreleased

### Added

- **Apple Metal GPU backend** — Complete replacement of CUDA/CuPy with native Metal Shading Language (MSL) compute kernels for Apple Silicon (M1/M2/M3/M4).
- **Persistent thread SPFA solver** (`wavefront_expand_all`) — Single-dispatch, work-stealing shortest-path kernel that runs entirely on-GPU without CPU round-trips.
- **Delta-stepping with bucket frontier** — Partitions the frontier by distance to exploit L1/L2 cache locality; converts from DRAM-bandwidth-bound to cache-bound.
- **SIMD block stealing** — Uses Metal SIMD-group intrinsics (`simd_broadcast_first`, `simd_is_first`) to steal 32 work items per atomic operation, reducing queue contention by 32×.
- **Zero-dispatch software grid barrier** — Custom inter-threadgroup sync via `threadgroup_barrier(mem_flags::mem_device)` and atomic generation counters; eliminates 38 μs per-dispatch overhead.
- **Zero-copy UMA memory mapping** — NumPy arrays mapped directly into Metal buffers via `MTLResourceStorageModeShared` and `new_buffer_with_bytes_no_copy`.
- **AMX coprocessor offloading** — Dense matrix operations routed to Apple's AMX matrix coprocessors via the Accelerate framework `cblas_sgemm`.
- **Multi-net parallel solver** (`wavefront_expand_multi`) — Simultaneous routing of multiple nets using batched distance arrays with 2D grid dispatch.
- **PathFinder negotiation kernel** — SIMD-group reduction for history-based congestion pressure using `simd_shuffle_down`.
- **SPFA setup kernel** — GPU-accelerated frontier initialization from distance arrays.
- **ROI extractor kernel** — GPU-side region-of-interest distance extraction.
- **Via processing kernel** — GPU-side via cost initialization.
- **Rust/PyO3 bridge** (`orthoroute_metal`) — `MetalDijkstra` struct with full pipeline management, buffer allocation, and Python interop.
- **36/36 parity tests pass** — Bitwise float32 match against CUDA golden tensors across all tested graph sizes (2K–180K nodes).
- **Peak throughput: 111.4 billion edges/sec** on Apple M4 at 401,800 nodes.
- Comprehensive documentation: `ARCHITECTURE.md`, `BENCHMARK_METHODOLOGY.md`, `NOTICE.md`.

### Changed

- README cleaned up with updated badges, benchmarks, and architecture diagrams.
- Build system migrated to `maturin` for Rust→Python extension builds.

## [0.2.0] - 2025

### Added

- **GPU-vectorized layer diagnostics** — Per-layer congestion analysis with fail-fast mode for early detection of unroutable boards (`45706a8`).
- **Fail-fast GPU mode** — Early termination when congestion ratio ρ > 1.0, avoiding wasted GPU cycles on impossible boards (`b449659`).
- **Vectorized conflict detection** — GPU-accelerated barrel conflict detection replacing O(n²) Python loops (`99cf844`).
- **Persistent net exclusion** — Ability to permanently exclude nets from routing (e.g., power/ground planes) to reduce congestion and improve convergence (`29b5749`).
- **New `.ORP` file format** — Board export format for headless/cloud routing workflows (`54a0e3e`).
- **Derived max iterations** — Automatic `max_iterations` calculation based on board complexity and congestion ratio (`1c5c21a`).
- **Headless/cloud mode** — QT made optional for headless operation, enabling cloud-based routing without display dependencies (`03ff5ad`).
- OpenGL rendering fixes (`c84ecf7`).

### Fixed

- OpenGL visualization pipeline rendering issues.

## [0.1.0] - 2024

### Added

- **PathFinder negotiated congestion routing** — Implementation of the McMurchie–Ebeling PathFinder algorithm adapted for PCB Manhattan routing.
- **CUDA/CuPy GPU acceleration** — Bellman-Ford and SPFA shortest-path solvers running on NVIDIA GPUs via CuPy.
- **3D Manhattan routing lattice** — Grid-based routing with configurable pitch, layer count, and H/V layer discipline.
- **Portal escape architecture** — Novel pad escape strategy improving routing success from 16% to 80%+.
- **KiCad 9.0 IPC integration** — Real-time board parsing and route injection via the KiCad IPC API.
- **PyQt6 interactive GUI** — Visualization of routing progress, congestion heatmaps, and layer views.
- **Board-adaptive parameter tuning** — Automatic adjustment of PathFinder parameters based on congestion ratio (ρ).
- **Blind/buried via support** — Routing through arbitrary via spans with 870+ via pair capacity.
- **Clean Architecture** — Four-layer architecture (Domain, Application, Infrastructure, Presentation) for separation of concerns.
- **CSR graph representation** — Compressed Sparse Row format for efficient GPU graph traversal.
- **Region-of-interest (ROI) routing** — Focused routing on net-local subgraphs to reduce GPU memory and computation.
- **CPU fallback** — Automatic fallback to SciPy `dijkstra` when GPU is unavailable.

[1.0.0]: https://github.com/ParkWardRR/OrthoRoute-Metal/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ParkWardRR/OrthoRoute-Metal/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ParkWardRR/OrthoRoute-Metal/releases/tag/v0.1.0
