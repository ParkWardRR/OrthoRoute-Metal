# Architecture

## Overview

OrthoRoute-Metal replaces the CUDA/CuPy GPU backend of OrthoRoute with a native
Apple Metal compute pipeline. The architecture has three layers:

```
+---------------------------------------------------------+
|  Python (KiCad Plugin / CLI)                            |
|  - Board parsing, net ordering, PathFinder iteration    |
|  - NumPy arrays for graph data (CSR format)             |
+---------------------------------------------------------+
         |                              ^
         | PyO3 FFI (zero-copy)         | Results (zero-copy)
         v                              |
+---------------------------------------------------------+
|  Rust Layer (orthoroute_metal)                          |
|  - MetalDijkstra: pipeline management, buffer alloc    |
|  - AMX SGEMM: Accelerate framework for dense math      |
|  - PyO3 bindings for Python interop                     |
+---------------------------------------------------------+
         |                              ^
         | Metal API (metal-rs)         | UMA shared memory
         v                              |
+---------------------------------------------------------+
|  Metal Compute Kernels (MSL)                            |
|  - wavefront_expand_all: persistent SPFA solver         |
|  - wavefront_expand_multi: multi-net parallel solver    |
|  - spfa_setup_kernel: frontier initialization           |
|  - clear_counters: queue state reset                    |
|  - negotiation_kernel: PathFinder history pressure      |
|  - roi_extractor_mixin: region-of-interest extraction   |
|  - via_kernels: via cost computation                    |
+---------------------------------------------------------+
         |
         v
+---------------------------------------------------------+
|  Apple Silicon GPU Hardware                             |
|  - 10 GPU cores (M4), 32-wide SIMD groups              |
|  - 32KB threadgroup shared memory per core              |
|  - Unified Memory Architecture (UMA)                    |
|  - AMX matrix coprocessors (via Accelerate)             |
+---------------------------------------------------------+
```

## Data Flow

### Graph Representation

The PCB routing lattice is stored as a Compressed Sparse Row (CSR) matrix:

- `indptr[N+1]`: Row pointer array. `indptr[i]` is the index into `indices`
  where node `i`'s neighbors begin.
- `indices[E]`: Column index array. Contains the target node ID for each edge.
- `weights[E]`: Edge weight array. Contains the routing cost for each edge.

where `N` is the number of nodes and `E` is the number of edges.

### Memory Model

All graph buffers use `MTLResourceStorageModeShared`, which maps to the same
physical DRAM addresses accessible by both the CPU and GPU on Apple Silicon.
NumPy arrays from the Python layer are passed through PyO3 as
`PyReadonlyArray1` and mapped directly into Metal buffers via
`new_buffer_with_bytes_no_copy`. No data copy occurs at any point in the
pipeline.

### Kernel Execution Model

The main SSSP solver (`wavefront_expand_all`) uses a persistent thread model:

1. A fixed grid of 8,192 threads (16 threadgroups of 512) is launched once.
2. Threads cooperatively process a work-stealing frontier queue.
3. Each SIMD-group steals 32 work items at a time to minimize atomic contention.
4. A software grid barrier synchronizes all threadgroups between iterations.
5. The kernel runs until the frontier is empty or the iteration limit is reached.
6. All iterations execute inside a single `commandBuffer.commit()` call.

### AMX Coprocessor Offloading

Dense matrix operations (such as congestion map updates) are offloaded to
Apple's AMX matrix coprocessors via the Accelerate framework. This is accessed
through a direct FFI call to `cblas_sgemm`, which the Accelerate framework
automatically routes to the AMX hardware for matrices above a size threshold.

## File Layout

```
metal/
  Cargo.toml          -- Rust crate configuration
  mock_orthoroute.py  -- Integration test script
  src/
    lib.rs            -- MetalDijkstra struct and PyO3 bindings
    kernels.metal     -- All Metal Shading Language compute kernels
    accelerate_ops.rs -- Apple Accelerate/AMX SGEMM bindings
```

## Key Design Decisions

### Why Rust Instead of Python Metal Bindings

Python Metal bindings (pyobjc, metalcompute) lack the low-level control needed
for persistent thread kernels, zero-copy buffer management, and pipeline
caching. The Rust `metal-rs` crate provides direct access to the full Metal API
while PyO3 handles the Python-Rust bridge with minimal overhead.

### Why Persistent Threads Instead of Per-Iteration Dispatch

Each Metal command buffer dispatch incurs approximately 38 microseconds of
latency on the M4. For algorithms requiring hundreds of iterations, this
overhead dominates execution time. The persistent thread model with a software
grid barrier eliminates this entirely.

### Why SIMD Block Stealing Instead of Per-Thread Atomics

With 8,192 threads contending on a single atomic queue index, per-thread
atomics create severe serialization. SIMD block stealing (32 items per atomic
operation) reduces contention by 32x.
