## OrthoRoute-Metal v1.0.0 — Production Release

**Open-source GPU-accelerated PCB autorouter** built on Apple Metal compute shaders for M1, M2, M3, and M4 chips. Replaces CUDA/CuPy with native Metal Shading Language (MSL), delivering 111 billion edges/sec throughput on commodity Apple hardware. Designed for KiCad integration and high-density HDI board routing.

---

### How to Install & Use (For End Users)

1. **Download the Plugin**: Under **Assets** below, download `OrthoRoute-Metal-Plugin-macOS.zip`.
2. **Locate your KiCad Plugins Folder**: Open KiCad 8. Go to **Preferences > Manage Action Plugins...** and click the **Open Plugin Directory** button. This usually opens `~/Documents/KiCad/8.0/scripting/plugins`.
3. **Install**: Unzip the downloaded file and drag the `OrthoRoute` folder into the KiCad plugins directory.
4. **Run it**: Open any PCB design in the KiCad PCB Editor. Click the new OrthoRoute button in the top toolbar to automatically route your board!
*(Note: This version is exclusively for Apple Silicon Macs: M1, M2, M3, M4).*

---

### How It Works — The Big Picture

**What is a PCB?** 
A printed circuit board is the green board inside every phone, laptop, and game console. It has copper traces (tiny wires) connecting chips together.

**What does an autorouter do?** 
When engineers design a PCB, they place components on the board and then need to draw hundreds of copper traces connecting specific pins. Doing this by hand takes days. An autorouter does it automatically in minutes.

**Why is this hard?** 
Imagine a city with thousands of people who all need to drive from their house to a specific destination, but the roads are narrow and nobody can share a lane. The autorouter has to find a route for every single driver without any collisions. Now add 6 floors (layers) to the city. That is what PCB routing looks like.

**How does OrthoRoute solve it?** 
It converts the PCB into a massive 3D grid — millions of tiny cubes, like a Minecraft world. Each cube is a point in a graph. Then the GPU runs a shortest-path algorithm on this graph for every net (pair of pins) simultaneously. 
When two traces want the same space, the algorithm increases the cost of that path — like raising the toll on a highway. Eventually one trace reroutes around the congestion. This is called the PathFinder negotiated congestion algorithm.

**Why Apple Silicon?** 
Most GPU-accelerated routers require an NVIDIA GPU, which means you need a gaming PC or a cloud server. OrthoRoute-Metal runs on the GPU inside every modern Mac — the same chip powering your display. No external hardware, no cloud costs, no NVIDIA.

**How fast is it?** 
The GPU processes 111 billion graph edges per second on an M4 MacBook Pro. A 6-layer board with 200+ nets routes in under a minute.

---

### Performance

| Metric | Value |
|--------|-------|
| Peak throughput | 111.4 billion edges/sec (Apple M4) |
| CUDA parity | 36/36 bitwise float32 match |
| Test suite | 529 tests passed in 0.68s |
| Compute kernels | 7 Metal + 7 Vulkan stubs |
| Test files | 29 (13 skipped — Qt not required) |

---

### Metal Compute Kernels

All 7 kernels run natively on the Apple GPU via Metal Shading Language:

1. **wavefront_expand_all** — Persistent-thread SPFA solver with delta-stepping buckets, SIMD block stealing (32x read batching), SIMD prefix-sum queue compaction (32x write batching), and zero-dispatch grid barrier (no CPU round-trips)
2. **wavefront_expand_multi** — Multi-net parallel solver using 2D grid dispatch for simultaneous net routing
3. **negotiation_kernel** — PathFinder history-based congestion pressure with SIMD-group shuffle reduction
4. **roi_extractor** — GPU-side region-of-interest extraction with 3D bounding box filtering and atomic compaction
5. **via_kernels** — Via cost computation with hard-block enforcement (usage >= capacity -> infinity) and pooling penalties
6. **spfa_setup_kernel** — Frontier initialization from pre-computed distance arrays
7. **clear_counters** — Queue state reset between solver iterations

---

### Changelog (v1.0.0)

**Added:**
- Phase 4: SIMD Prefix-Sum Queue Compaction — simd_enqueue() reduces atomic write contention 32x
- UnifiedPathFinder decomposition into 4 standalone modules (LatticeManager, EdgeAccountant, GeometryEmitter, ConvergenceManager)
- 180 KiCad integration tests (file parser, geometry, serialization, layers, colors, E2E pipeline)
- 64 advanced tests (portal escape, pad mapping, convergence, GPU/CPU parity, regression, benchmarks)
- Linux Vulkan compute backend stubs (vulkan/, VulkanProvider)
- Full documentation update across all 14 docs

**Changed:**
- Version bumped to 1.0.0 across all packages
- ROADMAP updated to 80% completion (140/176 items)
- Refactoring complete: 8/8 (was 7/8)
- Test suite: 29 files, 529 tests (was 18 files, 286 tests)
