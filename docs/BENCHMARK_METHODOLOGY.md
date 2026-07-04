# Benchmark Methodology

## Hardware

### Apple M4

- Chip: Apple M4 (base configuration)
- GPU: 10-core Apple GPU
- Memory: Unified Memory Architecture (UMA), 120 GB/s bandwidth
- OS: macOS Sequoia
- Metal: Metal 3.2
- Rust: 1.70+
- Build: `cargo build --release`

### NVIDIA RTX 2080 Ti

- GPU: TU102 (Turing), 68 SMs, 4,352 CUDA cores
- VRAM: 23.1 GB GDDR6, 616 GB/s bandwidth
- Host: Vast.ai cloud instance ($0.083/hr)
- CUDA: 12.6
- Driver: 565.x
- Python: CuPy 13.x

### NVIDIA RTX 3060

- GPU: GA106 (Ampere), 28 SMs, 3,584 CUDA cores
- VRAM: 12.5 GB GDDR6, 360 GB/s bandwidth
- Host: Vast.ai cloud instance ($0.055/hr)
- CUDA: 12.6
- Driver: 565.x
- Python: CuPy 13.x

### NVIDIA RTX 3060 Ti

- GPU: GA104 (Ampere), 38 SMs, 4,864 CUDA cores
- VRAM: 8.2 GB GDDR6X, 448 GB/s bandwidth
- Host: Vast.ai cloud instance ($0.049/hr)
- CUDA: 12.6
- Driver: 565.x
- Python: CuPy 13.x

## Graphs

All benchmark graphs are derived from real PCB routing lattices in CSR
(Compressed Sparse Row) format. The graphs represent Manhattan routing grids
where nodes are grid intersections and edges are trace segments.

| Label | Nodes | Edges | Description |
|-------|------:|------:|-------------|
| Class A Amp | 2,000 | ~9,000 | Small Class A amplifier board |
| Medium PCB | 8,000 | ~36,000 | Medium-complexity 2-layer board |
| Large PCB | 30,000 | ~135,000 | 4-layer board with BGA |
| Large 6-layer | 45,000 | ~202,000 | 6-layer high-density board |
| XL PCB | 60,000 | ~270,000 | Large 6-layer with dense routing |
| XXL PCB | 180,000 | ~810,000 | Very large multi-layer board |
| ClassAB MOSFET | 401,800 | 2,005,052 | Prydin ClassAB MOSFET, 700x287x2 |

## Measurement Protocol

### CUDA (Vast.ai)

1. Graph loaded into CPU memory via NumPy/SciPy.
2. CSR arrays transferred to GPU via CuPy (`cupy.array()`).
3. Bellman-Ford kernel dispatched with optimal threadgroup size (512).
4. Total time measured from first kernel dispatch to final
   `cupy.cuda.Stream.synchronize()`.
5. Transfer time (CPU to GPU) measured separately.
6. Each benchmark repeated 3 times; median reported.

### Metal (Local M4)

1. Graph loaded into CPU memory via NumPy/SciPy.
2. CSR arrays mapped to Metal buffers via `new_buffer_with_bytes_no_copy`
   (zero-copy UMA).
3. Persistent SPFA kernel dispatched with Delta-Stepping.
4. Total time measured from `commandBuffer.commit()` to
   `commandBuffer.waitUntilCompleted()`.
5. No transfer time (UMA eliminates copies).
6. Each benchmark repeated 3 times; median reported.

## Metrics

- **Traversal Time**: Wall-clock time for the shortest-path computation only,
  excluding graph construction and result extraction.
- **Throughput**: `total_edges / traversal_time`. For Delta-Stepping, this
  counts all edges in the graph, not just the edges examined, which inflates the
  metric relative to dense Bellman-Ford.
- **Effective Bandwidth**: `total_bytes_accessed / traversal_time`. For
  Delta-Stepping, this is amplified by cache hits (L1/L2 serve most reads).
- **Per-Iteration Cost**: `traversal_time / num_iterations`. Reflects kernel
  dispatch overhead on CUDA and iteration-loop overhead on Metal.

## Parity Verification

To ensure the Metal backend produces correct results, 36 parity tests were run
across 6 graph sizes (2,000 to 180,000 nodes). Each test verifies:

1. **Distance arrays**: Bitwise float32 match between Metal and CUDA outputs.
2. **Path sequences**: Identical node-by-node paths from source to target.
3. **Reachable node counts**: Exact match of the number of nodes with
   finite distance.

The CUDA reference outputs were captured on an RTX 2080 Ti and stored as golden
tensors. The Metal outputs are compared against these golden tensors on every
test run.
