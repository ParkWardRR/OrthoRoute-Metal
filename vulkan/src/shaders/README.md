# Vulkan SPIR-V Compute Shaders

> **Status:** ⬜ Not yet implemented

This directory will contain **GLSL compute shaders** (`.comp` files) that are compiled
to **SPIR-V** (`.spv` bytecode) for execution on Vulkan compute pipelines.

## Planned Shaders

These 7 shaders match the Metal MSL kernels in [`metal/src/kernels.metal`](../../metal/src/kernels.metal):

### 1. `wavefront_expand_all.comp` — Persistent Thread SPFA Solver

The core shortest-path kernel. Implements Δ-stepping SPFA with persistent threads
(all workgroups stay resident) and a software grid barrier via atomics.

**Key features:**
- Subgroup-level work stealing (`subgroupBroadcastFirst`, `subgroupElect`)
- Atomic float minimum via compare-and-swap loop
- Software barrier using `atomicAdd` on a generation counter
- Delta-stepping bucket frontier for edge relaxation

**Metal equivalent:** `wavefront_expand_all` kernel

### 2. `wavefront_expand_multi.comp` — Multi-Net Parallel Solver

Simultaneous routing of multiple nets using batched distance arrays.
Each net has its own distance/predecessor slice within a large buffer.

**Metal equivalent:** `wavefront_expand_multi` kernel

### 3. `spfa_setup.comp` — Frontier Initialization

Scans the distance array and marks nodes with `distance < infinity` as active
in the SPFA frontier queue. Initializes queue counters and generation flags.

**Metal equivalent:** `spfa_setup_kernel`

### 4. `clear_counters.comp` — Queue State Reset

Resets queue counters, generation flags, and convergence indicators between
SPFA iterations. Lightweight kernel dispatched before each solver iteration.

**Metal equivalent:** `clear_counters`

### 5. `negotiation.comp` — PathFinder Negotiation with Subgroup Reduction

Computes history-based congestion pressure using subgroup reduction operations.
Updates edge costs based on negotiation iteration count and congestion history.

**Key features:**
- Subgroup reduction for parallel pressure accumulation
- History cost escalation across PathFinder iterations

**Metal equivalent:** `negotiation_kernel` (uses SIMD-group reduction)

### 6. `roi_extractor.comp` — ROI Distance Extraction

Filters distances by a 3D spatial bounding box (Region of Interest).
Uses atomic counters for output compaction — only nodes within the ROI
are written to the output buffer.

**Key features:**
- 3D bounding box test per node
- Atomic output compaction
- Coordinate array inputs (x, y, z per node)

**Metal equivalent:** `roi_extractor_mixin`

### 7. `via_kernels.comp` — Via Cost Computation

Computes per-via routing costs based on capacity and usage:
- `usage >= capacity` → `INFINITY` (hard-block)
- `0 < usage < capacity` → `base_cost × (1 + usage/capacity)`
- `usage == 0` → `base_cost`

Also handles via pooling penalties and barrel conflict detection.

**Metal equivalent:** `via_kernels`

## Compilation

GLSL compute shaders must be compiled to SPIR-V before use:

```bash
# Using glslc (from Vulkan SDK)
glslc wavefront_expand_all.comp -o wavefront_expand_all.spv
glslc wavefront_expand_multi.comp -o wavefront_expand_multi.spv
glslc spfa_setup.comp -o spfa_setup.spv
glslc clear_counters.comp -o clear_counters.spv
glslc negotiation.comp -o negotiation.spv
glslc roi_extractor.comp -o roi_extractor.spv
glslc via_kernels.comp -o via_kernels.spv

# Or using glslangValidator
glslangValidator -V wavefront_expand_all.comp -o wavefront_expand_all.spv
```

Alternatively, the `vulkano-shaders` crate can compile GLSL at Rust build time.

## Vulkan vs Metal: Compute Shader Differences

### Subgroup Operations

| Metal (SIMD) | Vulkan (Subgroup) | Notes |
|-------------|-------------------|-------|
| `simd_broadcast_first(val)` | `subgroupBroadcastFirst(val)` | Broadcast from first active lane |
| `simd_is_first()` | `subgroupElect()` | True for first active invocation |
| `simd_sum(val)` | `subgroupAdd(val)` | Subgroup-wide sum reduction |
| `simd_min(val)` | `subgroupMin(val)` | Subgroup-wide minimum |
| SIMD width = 32 (Apple GPU) | Subgroup size varies: 32 (NVIDIA), 64 (AMD), 8–32 (Intel) | Must query at runtime |

### Memory Model

| Metal | Vulkan | Notes |
|-------|--------|-------|
| `threadgroup_barrier(mem_flags::mem_device)` | `barrier(); memoryBarrierBuffer()` | Inter-invocation sync within workgroup |
| `device` address space | SSBO (Storage Buffer) | Read/write buffer binding |
| `threadgroup` address space | `shared` qualifier | Workgroup-local memory |
| UMA (zero-copy) | Host-visible or staging buffers | Discrete GPUs need explicit transfers |

### Descriptor Binding

| Metal | Vulkan | Notes |
|-------|--------|-------|
| `[[buffer(N)]]` attribute | `layout(set=S, binding=B)` | Buffer binding declaration |
| Argument buffers | Descriptor sets | Grouped buffer bindings |
| `setBuffer(buf, offset, index)` | `vkUpdateDescriptorSets` | Runtime binding |

### Atomic Operations

| Metal | Vulkan | Notes |
|-------|--------|-------|
| `atomic_fetch_add_explicit` | `atomicAdd` | GLSL built-in |
| Custom CAS loop for float min | `atomicMin` (int) or CAS loop (float) | `VK_EXT_shader_atomic_float` adds native float atomics |
| `atomic_store_explicit` | `atomicExchange` or direct store | Memory ordering differences |

### Required Vulkan Extensions

- `VK_KHR_shader_subgroup` — Subgroup operations (core in Vulkan 1.1+)
- `VK_EXT_shader_atomic_float` — Native atomic float operations (optional, CAS fallback available)
- `VK_KHR_storage_buffer_storage_class` — SSBO support (core in Vulkan 1.1+)
- `VK_KHR_8bit_storage` — 8-bit integer storage for segment capacity arrays

## File Organization

```
shaders/
├── README.md                      ← You are here
├── wavefront_expand_all.comp      ← Main SPFA solver (future)
├── wavefront_expand_all.spv       ← Compiled SPIR-V (future, gitignored)
├── wavefront_expand_multi.comp    ← Multi-net solver (future)
├── spfa_setup.comp                ← Frontier init (future)
├── clear_counters.comp            ← Counter reset (future)
├── negotiation.comp               ← Congestion negotiation (future)
├── roi_extractor.comp             ← ROI extraction (future)
└── via_kernels.comp               ← Via cost computation (future)
```
