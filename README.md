# OrthoRoute-Metal v1.0.0

[![License: Blue Oak 1.0.0](https://img.shields.io/badge/License-Blue_Oak_1.0.0-2D6DB5.svg?style=flat-square)](https://blueoakcouncil.org/license/1.0.0)
[![Upstream: MIT](https://img.shields.io/badge/Upstream-MIT-green.svg?style=flat-square)](https://github.com/bbenchoff/OrthoRoute)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Platform](https://img.shields.io/badge/Platform-Apple_Silicon-000000.svg?style=flat-square&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![Metal](https://img.shields.io/badge/Metal-3.2-8E8E93.svg?style=flat-square&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![GPU Backend](https://img.shields.io/badge/GPU-Metal_MSL-7B68EE.svg?style=flat-square)](https://developer.apple.com/documentation/metal/metal_shading_language_specification)
[![Build](https://img.shields.io/badge/Build-cargo_build-DEA584.svg?style=flat-square&logo=rust)](https://doc.rust-lang.org/cargo/)
[![PyO3](https://img.shields.io/badge/PyO3-0.29-blue.svg?style=flat-square)](https://pyo3.rs)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Parity](https://img.shields.io/badge/CUDA_Parity-36%2F36_tests-brightgreen.svg?style=flat-square)](#parity-verification)
[![Throughput](https://img.shields.io/badge/Peak-111.4B_edges%2Fsec-blueviolet.svg?style=flat-square)](#performance-results)

**OrthoRoute-Metal** is a native Apple Metal GPU backend for [OrthoRoute](https://github.com/bbenchoff/OrthoRoute), the GPU-accelerated PCB autorouter for KiCad. This fork replaces the CUDA/CuPy dependency with Apple Metal Shading Language (MSL) compute kernels, enabling GPU-accelerated PCB routing on Apple Silicon (M1, M2, M3, M4) without any NVIDIA hardware.

The Metal backend is **fully integrated** into the Python routing pipeline via `MetalProvider`, with automatic CUDA→Metal→Vulkan→CPU fallback. All 7 Metal compute kernels have full feature parity with their CUDA equivalents, verified by 36/36 bitwise-identical parity tests.

> Based on [OrthoRoute](https://github.com/bbenchoff/OrthoRoute) by [Brian Benchoff](https://github.com/bbenchoff), licensed under MIT. See [NOTICE.md](NOTICE.md) for full attribution.

---

## Table of Contents

- [What This Fork Does](#what-this-fork-does)
- [Performance Results](#performance-results)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Build Instructions](#build-instructions)
- [Usage](#usage)
- [Testing](#testing)
- [Local CI](#local-ci)
- [Parity Verification](#parity-verification)
- [Metal Backend Technical Details](#metal-backend-technical-details)
- [Documentation](#documentation)
- [License](#license)
- [Attribution](#attribution)

---

## What This Fork Does

This fork adds a complete Apple Metal GPU compute backend to OrthoRoute. The Metal backend is a drop-in replacement for the original CUDA/CuPy shortest-path solver. It implements:

- **Persistent Thread SPFA Solver** -- A single-dispatch, work-stealing shortest-path kernel that runs entirely on the GPU without CPU round-trips between iterations. Replaces the CUDA wavefront expansion kernel.
- **Delta-Stepping with Bucket Frontier** -- Partitions the frontier by distance to exploit L1/L2 cache locality. Converts the problem from DRAM-bandwidth-bound to cache-bound on Apple Silicon.
- **SIMD Block Stealing** -- Uses Metal SIMD-group intrinsics (`simd_broadcast_first`, `simd_is_first`) to steal 32 work items per atomic operation, reducing queue contention by 32x.
- **Zero-Dispatch Software Grid Barrier** -- A custom inter-threadgroup synchronization primitive using `threadgroup_barrier(mem_flags::mem_device)` and atomic generation counters. Eliminates the 38-microsecond-per-dispatch overhead of multiple command buffer submissions.
- **Zero-Copy UMA Memory Mapping** -- NumPy arrays from Python are mapped directly into Metal buffers via `MTLResourceStorageModeShared` and `new_buffer_with_bytes_no_copy`. No data copies occur between CPU and GPU at any point.
- **AMX Coprocessor Offloading** -- Dense matrix operations (congestion map updates) are routed to Apple's AMX matrix coprocessors via the Accelerate framework `cblas_sgemm`.
- **Multi-Net Parallel Solver** -- A second kernel (`wavefront_expand_multi`) supports simultaneous routing of multiple nets using batched distance arrays.
- **PathFinder Negotiation Kernel** -- SIMD-group reduction for history-based congestion pressure, using `simd_shuffle_down` for efficient warp-level summation.
- **SIMD Prefix-Sum Queue Compaction** -- `simd_enqueue()` utility batches bucket-deferred node enqueues via `simd_prefix_exclusive_sum`, reducing global queue atomic contention by up to 32×.

---

## Performance Results

Benchmarks compare the Metal backend (Apple M4, local) against CUDA (NVIDIA GPUs on Vast.ai cloud instances). All measurements use corner-to-corner shortest-path on CSR-format graphs derived from real PCB routing lattices. See [docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) for full methodology.

### Traversal Time

Lower is better. The Metal backend crosses over CUDA at approximately 150,000 nodes.

```
Traversal Time (microseconds)
                                                                         
  Nodes   | RTX 2080 Ti | RTX 3060 |  RTX 3060 Ti | Apple M4 Metal
  --------|-------------|----------|--------------|---------------
    2,000 |         856 |      541 |          --- |          1,200
    8,000 |       1,786 |    1,054 |          --- |          2,100
   30,000 |       3,241 |    2,179 |          --- |          4,800
   45,000 |       3,420 |    2,391 |          --- |          5,200
   60,000 |       5,051 |    3,781 |          --- |          6,100
  180,000 |      10,294 |    9,406 |          --- |          8,500  <-- crossover
  401,800 |         --- |   15,486 |          --- |          4,130  <-- 3.7x faster
```

### Throughput

Higher is better. The M4 Metal backend reaches 111.4 billion edges per second on the largest graph.

```
Throughput (edges/sec)

  Nodes   | RTX 3060      | Apple M4 Metal   | M4 Speedup
  --------|---------------|------------------|----------
    2,000 |    20 Million |     257 Million  |     12.9x
   30,000 |    73 Million |   4,817 Million  |     66.0x
  180,000 |   111 Million |  30,034 Million  |    270.6x
  401,800 |   124 Million | 111,400 Million  |    898.4x
```

### Effective Memory Bandwidth

The M4 achieves 831.5 GB/s effective bandwidth on the 401,800-node graph. The M4 DRAM ceiling is 120 GB/s. The 6.9x amplification factor is entirely due to L1/L2 cache hits from Delta-Stepping's bucket-based frontier.

```
Effective Bandwidth (GB/s)

  Nodes   | RTX 2080 Ti | RTX 3060 | Apple M4 Metal
  --------|-------------|----------|---------------
    8,000 |          27 |       41 |             48
   30,000 |         100 |      146 |            195
   60,000 |         172 |      245 |            340
  180,000 |         349 |      418 |            620
  401,800 |         --- |      418 |          831.5
```

### Real-World PCB Routing

The Class A Amplifier board routes in 26 seconds on the M4 with the following results:

| Metric | Value |
|--------|------:|
| Routed Tracks | 38 |
| Vias Placed | 79 |
| GPU Utilization | 63% |
| Routing Time | 26 seconds |
| GPU Paths | 343 |
| CPU Paths | 203 |

---

## Architecture

```
Python (KiCad / CLI)           Rust (PyO3)              Metal GPU
+------------------+    +---------------------+    +-------------------+
| MetalProvider    | -> | MetalDijkstra       | -> | wavefront_expand  |
| LatticeManager   |    | Buffer management   |    | SPFA + Delta-Step |
| EdgeAccountant   |    | Pipeline caching    |    | Grid barrier      |
| GeometryEmitter  |    | AMX SGEMM (Accel.)  |    | SIMD block steal  |
| ConvergenceManager    +---------------------+    | SIMD prefix-sum   |
+------------------+                               +-------------------+
        |                        |                          |
  CUDA -> Metal -> Vulkan -> CPU +--- UMA (zero-copy) -----+
  (auto-fallback)
```

The `MetalProvider` in `orthoroute/infrastructure/gpu/metal_provider.py` implements the `GPUProvider` interface and is automatically selected via `get_best_provider()` with CUDA→Metal→Vulkan→CPU priority. It wraps the Rust/PyO3 `MetalDijkstra` backend for shortest-path dispatch, ROI extraction, and via cost computation. The `UnifiedPathFinder` has been decomposed into 4 standalone modules: `LatticeManager`, `EdgeAccountant`, `GeometryEmitter`, and `ConvergenceManager`.

For a detailed architecture description, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Getting Started

### Prerequisites

| Requirement | Version |
|-------------|---------|
| macOS | Sonoma 14+ or Sequoia 15+ |
| Apple Silicon | M1, M2, M3, or M4 |
| Rust | 1.70+ |
| Python | 3.10+ |
| KiCad | 9.0+ (for plugin mode) |

### Build Instructions

#### 1. Clone the repository

```bash
git clone https://github.com/ParkWardRR/OrthoRoute-Metal.git
cd OrthoRoute-Metal
```

#### 2. Build the Metal backend

```bash
cd metal
cargo build --release
```

#### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install maturin
```

#### 4. Build and install the Python extension

```bash
cd metal
maturin develop --release
```

---

## Usage

### As a KiCad Plugin

1. Open your PCB in KiCad 9.0+ with the IPC API enabled.
2. Run OrthoRoute. The Metal backend is automatically selected on macOS with Apple Silicon.

### As a Python Library

```python
import orthoroute_metal
import numpy as np

# Initialize the Metal GPU backend
dijkstra = orthoroute_metal.MetalDijkstra()

# Load your CSR graph (from SciPy or manual construction)
dijkstra.set_graph_csr(row_ptr, col_indices, weights)

# Set initial distances (source node = 0.0, all others = inf)
distances = np.full(num_nodes, np.inf, dtype=np.float32)
distances[source] = 0.0
dijkstra.set_distances_csr(distances)
dijkstra.reset_predecessors()

# Initialize the frontier
dijkstra.setup_spfa()

# Run GPU-accelerated shortest path
iters, converged = dijkstra.execute_until_convergence(
    max_iters=500,
    batch_size=1024,
    threadgroup_size=512,
    delta=0.0
)

# Retrieve results (zero-copy from GPU memory)
final_distances = dijkstra.get_distances()
final_predecessors = dijkstra.get_predecessors()
```

### CLI Routing

```bash
# Route a .kicad_pcb file from the command line
python main.py cli board.kicad_pcb -o board_routed.kicad_pcb
```

### Demo Board: F5 Turbo v2

The `f5_turbo_generator/` directory contains a complete demo board — a 44-component
Class-A power amplifier. A Go-based placer generates the `.kicad_pcb` from a SKiDL netlist:

```bash
cd f5_turbo_generator
go build -o place_f5_turbo cmd/place_f5_turbo.go
./place_f5_turbo                    # → f5_turbo_v2.kicad_pcb
kicad-cli pcb export pdf f5_turbo_v2.kicad_pcb -o f5_turbo_v2.pdf \
  --layers F.Cu,B.Cu,F.SilkS,F.Fab,Edge.Cuts --mode-single
```

See [f5_turbo_generator/README.md](f5_turbo_generator/README.md) for details.

### Headless / Cloud Mode

OrthoRoute supports headless routing via `.ORP` export files. See the upstream
[OrthoRoute documentation](https://github.com/bbenchoff/OrthoRoute) for details
on the headless workflow.

### AMX Matrix Multiplication

```python
import orthoroute_metal
import numpy as np

# Accelerate framework SGEMM (routed to AMX coprocessors)
orthoroute_metal.amx_sgemm_py(
    m=1024, n=1024, k=1024,
    alpha=1.0,
    a_array=np.random.randn(1024*1024).astype(np.float32),
    b_array=np.random.randn(1024*1024).astype(np.float32),
    beta=0.0,
    c_array=np.zeros(1024*1024, dtype=np.float32)
)
```

---

## Testing

OrthoRoute-Metal includes a comprehensive test suite with 29 test files and **529 tests** (13 skipped — Qt) covering core infrastructure, domain models, GPU backend integration, and KiCad integration.

### Run all tests

```bash
python -m pytest tests/ -v
# 529 passed, 13 skipped in 0.76s
```

### Test coverage includes

| Area | Test File(s) |
|------|-------------|
| Lattice construction | `test_lattice.py` |
| CSR graph integrity | `test_csr_graph.py` |
| Via pooling accounting | `test_via_accounting.py` |
| Board/layer analysis | `test_board_analyzer.py`, `test_layer_analyzer.py` |
| Domain models | `test_domain_models.py` |
| DRC constraints | `test_drc_checker.py` |
| Serialization (ORP/ORS) | `test_serialization.py` |
| CPU fallback | `test_cpu_fallback.py` |
| Configuration | `test_config.py` |
| Spatial hash | `test_spatial_hash.py` |
| Grid/real global grid | `test_grid.py`, `test_real_global_grid.py` |
| Data structures | `test_data_structures.py` |
| Parameter derivation | `test_parameter_derivation.py` |
| Portal escape | `test_portal_escape.py`, `test_portal_escape_advanced.py` |
| Pad mapping | `test_pad_mapping.py`, `test_pad_mapping_advanced.py` |
| GPU/CPU parity | `test_gpu_cpu_parity.py`, `test_gpu_cpu_parity_advanced.py` |
| PathFinder convergence | `test_convergence.py`, `test_pathfinder_convergence.py` |
| Performance benchmarks | `test_performance.py`, `test_benchmarks.py` |
| Regression suite | `test_regression.py` |
| KiCad end-to-end | `test_kicad_e2e.py` |
| KiCad file parsing | `test_kicad_file_parser.py` |
| KiCad geometry | `test_kicad_geometry.py` |
| KiCad serialization | `test_kicad_serialization.py` |
| KiCad layers/colors | `test_kicad_layers.py` |
| CUDA ↔ Metal parity | 36/36 golden tensor tests |

---

## Local CI

The project uses a local CI pipeline via [OrbStack](https://orbstack.dev/) (not GitHub Actions).

```bash
# Run the full CI pipeline locally
./ci/run.sh
```

The pipeline runs inside a Docker container and executes linting, type checking, and the full test suite.

---

## Parity Verification

The Metal backend produces identical outputs to the CUDA reference across all tested graph sizes. 36 out of 36 parity tests pass.

| Verification | Status |
|-------------|--------|
| Distances (bitwise float32 match) | PASS (36/36) |
| Paths (identical node sequences) | PASS (36/36) |
| Reachable nodes (exact count match) | PASS (36/36) |

Graph sizes tested: 2,000 / 8,000 / 30,000 / 45,000 / 60,000 / 180,000 nodes.

CUDA reference outputs were captured on an RTX 2080 Ti (Vast.ai, $0.083/hr) and stored as golden tensors.

---

## Metal Backend Technical Details

### Compute Kernels

| Kernel | Purpose | Threading Model |
|--------|---------|-----------------|
| `wavefront_expand_all` | Persistent SPFA with Delta-Stepping | 8,192 threads, SIMD block stealing |
| `wavefront_expand_multi` | Multi-net parallel SPFA | 2D grid (nodes x nets) |
| `spfa_setup_kernel` | Frontier initialization from distance array | 1 thread per node |
| `clear_counters` | Queue state reset between iterations | Single thread |
| `negotiation_kernel` | PathFinder history pressure reduction | SIMD shuffle down |
| `roi_extractor_mixin` | Region-of-interest distance extraction | 1 thread per node |
| `via_kernels` | Via cost initialization | 1 thread per via |

### Atomic Float Minimum

Metal MSL does not support `atomic_fetch_min` for floating-point types. A compare-and-swap (CAS) loop is used:

```metal
inline void atomic_fetch_min_float(device atomic_uint* dest, float val) {
    uint old_val_uint = atomic_load_explicit(dest, memory_order_relaxed);
    float old_val = as_type<float>(old_val_uint);
    while (val < old_val) {
        uint desired = as_type<uint>(val);
        if (atomic_compare_exchange_weak_explicit(
                dest, &old_val_uint, desired,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
        old_val = as_type<float>(old_val_uint);
    }
}
```

### Persistent Grid Constraint

Apple Silicon GPU schedulers do not preempt compute threadgroups. The persistent grid must not exceed the hardware's concurrent execution capacity. On the M4 (10-core GPU), the safe limit is 16 threadgroups of 512 threads (8,192 total). Exceeding this causes an irrecoverable OS watchdog timeout.

### Zero-Dispatch Grid Barrier

A software grid barrier using `device atomic_uint` and `threadgroup_barrier(mem_flags::mem_device)` synchronizes all threadgroups between SPFA iterations without returning to the CPU. The full SSSP computation executes inside a single `commandBuffer.commit()` call.

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Three-layer architecture overview |
| [BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) | Hardware, graphs, measurement protocol |
| [coordinate_system.md](docs/coordinate_system.md) | How (x, y, z) maps to mm and layers |
| [pathfinder_algorithm.md](docs/pathfinder_algorithm.md) | PathFinder algorithm deep-dive |
| [metal_kernel_internals.md](docs/metal_kernel_internals.md) | MSL code walkthrough |
| [api_reference.md](docs/api_reference.md) | Python API reference |
| [tuning_guide.md](docs/tuning_guide.md) | PathFinder parameter tuning |
| [barrel_conflicts_explained.md](docs/barrel_conflicts_explained.md) | Via barrel conflict analysis |
| [cloud_gpu_setup.md](docs/cloud_gpu_setup.md) | Vast.ai cloud GPU setup |
| [congestion_ratio.md](docs/congestion_ratio.md) | Routability prediction metric |
| [contributing.md](docs/contributing.md) | Contributor guide |
| [layer_compaction.md](docs/layer_compaction.md) | Post-routing layer minimization |
| [plugin_manager_integration.md](docs/plugin_manager_integration.md) | KiCad PCM integration |
| [ORP_ORS_file_formats.md](docs/ORP_ORS_file_formats.md) | File format specification |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [f5_turbo_generator/README.md](f5_turbo_generator/README.md) | Demo board generator (Go) |
| [NOTICE.md](NOTICE.md) | Upstream attribution and license |
| [Upstream OrthoRoute](https://github.com/bbenchoff/OrthoRoute) | Original CUDA-based OrthoRoute |

---

## License

The Metal backend (all files under `metal/`) is licensed under the [Blue Oak Model License 1.0.0](https://blueoakcouncil.org/license/1.0.0). See [LICENSE.md](LICENSE.md).

The upstream OrthoRoute code is licensed under the [MIT License](https://opensource.org/licenses/MIT) by [Brian Benchoff](https://github.com/bbenchoff). See [NOTICE.md](NOTICE.md) for the full upstream license text.

---

## Attribution

This project is a fork of [OrthoRoute](https://github.com/bbenchoff/OrthoRoute) by [Brian Benchoff](https://github.com/bbenchoff). OrthoRoute is a GPU-accelerated PCB autorouter for KiCad that implements the PathFinder negotiation-based routing algorithm with CUDA/CuPy GPU acceleration.

The Metal backend was developed independently to bring GPU-accelerated PCB routing to Apple Silicon hardware. The PathFinder algorithm, board parsing, net ordering, KiCad IPC integration, and visualization code originate from the upstream OrthoRoute project.

### References

- McMurchie, L. and Ebeling, C. "PathFinder: A Negotiation-Based Performance-Driven Router for FPGAs." ACM/SIGDA FPGA, 1995.
- Meyer, U. and Sanders, P. "Delta-Stepping: A Parallelizable Shortest Path Algorithm." Journal of Algorithms, 2003.
- Apple. "Metal Shading Language Specification." Version 3.2.
- bbenchoff. [OrthoRoute Build Log](https://bbenchoff.github.io/pages/OrthoRoute.html).
