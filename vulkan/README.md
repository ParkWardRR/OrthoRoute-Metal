# OrthoRoute-Vulkan — Vulkan Compute Backend

> **Status:** ⬜ Stub — Not yet implemented

## Overview

This directory contains the planned **Vulkan compute backend** for OrthoRoute, enabling
GPU-accelerated PCB autorouting on **Linux** (and potentially Windows) systems with
Vulkan-capable GPUs. It mirrors the fully implemented [Metal backend](../metal/) and
provides the same `GPUProvider` interface via PyO3/Rust bindings.

The Vulkan backend will use **SPIR-V compute shaders** compiled from GLSL, with Rust
bindings provided by the [`ash`](https://github.com/ash-rs/ash) crate (low-level) or
[`vulkano`](https://github.com/vulkano-rs/vulkano) (safe wrapper). The Python interface
is exposed via [PyO3](https://pyo3.rs/).

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Python (VulkanProvider)                         │
│  orthoroute/infrastructure/gpu/vulkan_provider.py│
├──────────────────────────────────────────────────┤
│  PyO3 Bridge (VulkanDijkstra)                    │
│  vulkan/src/lib.rs                               │
├──────────────────────────────────────────────────┤
│  Vulkan API (ash / vulkano)                      │
│  vulkan/src/lib.rs                               │
├──────────────────────────────────────────────────┤
│  SPIR-V Compute Shaders                          │
│  vulkan/src/shaders/*.comp → *.spv               │
└──────────────────────────────────────────────────┘
```

### Key Differences from Metal Backend

| Aspect | Metal | Vulkan |
|--------|-------|--------|
| Shader language | MSL (Metal Shading Language) | GLSL → SPIR-V |
| Subgroup ops | `simd_broadcast_first`, `simd_is_first` | `subgroupBroadcastFirst`, `subgroupElect` |
| Memory model | Unified Memory (UMA, zero-copy) | Explicit host/device transfers (non-UMA GPUs) |
| Sync | `threadgroup_barrier` + atomic counters | `memoryBarrier` + `barrier()` + descriptor sets |
| Buffer binding | Argument buffers | Descriptor sets / push constants |
| Command dispatch | `MTLComputeCommandEncoder` | `VkComputePipeline` + command buffers |
| Atomic float min | CAS loop (custom) | `VK_EXT_shader_atomic_float` or CAS loop |

## Planned Compute Shaders (7 kernels)

These match the 7 Metal MSL kernels in `metal/src/kernels.metal`:

| # | Shader | Description | Metal Equivalent |
|---|--------|-------------|------------------|
| 1 | `wavefront_expand_all.comp` | Persistent thread SPFA solver with Δ-stepping | `wavefront_expand_all` |
| 2 | `wavefront_expand_multi.comp` | Multi-net parallel solver (batched distance arrays) | `wavefront_expand_multi` |
| 3 | `spfa_setup.comp` | Frontier initialization from distance arrays | `spfa_setup_kernel` |
| 4 | `clear_counters.comp` | Queue state / counter reset between iterations | `clear_counters` |
| 5 | `negotiation.comp` | PathFinder negotiation with subgroup reduction | `negotiation_kernel` |
| 6 | `roi_extractor.comp` | ROI distance extraction per spatial bounding box | `roi_extractor_mixin` |
| 7 | `via_kernels.comp` | Via cost computation (hard-block + pooling penalties) | `via_kernels` |

## Required Dependencies

- **Vulkan SDK** (≥ 1.3) — [LunarG Vulkan SDK](https://vulkan.lunarg.com/)
- **Rust toolchain** (stable, ≥ 1.70)
- **ash** crate (low-level Vulkan bindings) *or* **vulkano** crate (safe wrapper)
- **PyO3** (≥ 0.29) — Rust ↔ Python bindings
- **glslc** or **shaderc** — GLSL → SPIR-V compiler (bundled with Vulkan SDK)
- **numpy** (≥ 0.29 crate) — NumPy array interop via PyO3

## Build Instructions

> ⚠️ **Not yet buildable** — these are placeholder instructions for when implementation begins.

```bash
# 1. Install Vulkan SDK
#    Ubuntu/Debian:
#      sudo apt install vulkan-sdk libvulkan-dev
#    Fedora:
#      sudo dnf install vulkan-loader-devel vulkan-headers
#    Arch:
#      sudo pacman -S vulkan-devel

# 2. Install Rust + maturin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin

# 3. Compile GLSL shaders to SPIR-V
cd vulkan/src/shaders/
glslc wavefront_expand_all.comp -o wavefront_expand_all.spv
glslc wavefront_expand_multi.comp -o wavefront_expand_multi.spv
glslc spfa_setup.comp -o spfa_setup.spv
glslc clear_counters.comp -o clear_counters.spv
glslc negotiation.comp -o negotiation.spv
glslc roi_extractor.comp -o roi_extractor.spv
glslc via_kernels.comp -o via_kernels.spv

# 4. Build the Python extension
cd vulkan/
maturin develop --release

# 5. Verify
python -c "import orthoroute_vulkan; print('Vulkan backend loaded')"
```

## Reference Implementation

The Metal backend (`metal/`) serves as the blueprint for this Vulkan backend:

- **Rust entry point:** [`metal/src/lib.rs`](../metal/src/lib.rs) — `MetalDijkstra` struct with PyO3 bindings
- **Compute kernels:** [`metal/src/kernels.metal`](../metal/src/kernels.metal) — 7 MSL kernels
- **Python wrapper:** [`orthoroute/infrastructure/gpu/metal_provider.py`](../orthoroute/infrastructure/gpu/metal_provider.py)
- **Provider factory:** [`orthoroute/infrastructure/gpu/__init__.py`](../orthoroute/infrastructure/gpu/__init__.py)

## Contributing

See [`../docs/contributing.md`](../docs/contributing.md) for general contribution guidelines.
When implementing the Vulkan backend, follow the Metal backend's architecture as closely
as possible to maintain consistency across GPU providers.
