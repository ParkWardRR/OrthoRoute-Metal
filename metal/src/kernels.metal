#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
using namespace metal;

// Helper: Atomic float Min via CAS loop
inline void atomic_fetch_min_float(device atomic_uint* dest, float val) {
    uint old_val_uint = atomic_load_explicit(dest, memory_order_relaxed);
    float old_val = as_type<float>(old_val_uint);
    
    while (val < old_val) {
        uint desired = as_type<uint>(val);
        if (atomic_compare_exchange_weak_explicit(dest, &old_val_uint, desired, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
        old_val = as_type<float>(old_val_uint);
    }
}
// -----------------------------------------------------------------------------
// 1. Clear Counters
// -----------------------------------------------------------------------------
kernel void clear_counters(
    device atomic_uint* q_next [[buffer(0)]],
    device atomic_uint* steal_idx [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        atomic_store_explicit(q_next, 0, memory_order_relaxed);
        atomic_store_explicit(steal_idx, 0, memory_order_relaxed);
    }
}

// -----------------------------------------------------------------------------
// 2. Persistent Thread Work-Stealing Queue
// -----------------------------------------------------------------------------
kernel void wavefront_expand_all(
    constant uint& num_nodes [[buffer(0)]],
    device int* predecessors [[buffer(4)]],
    device const int* indptr [[buffer(5)]],
    device const int* indices [[buffer(6)]],
    device const float* weights [[buffer(7)]],
    device atomic_uint* distances [[buffer(8)]],
    device atomic_uint* f0 [[buffer(9)]],
    device atomic_uint* qs0 [[buffer(10)]],
    device atomic_uint* f1 [[buffer(11)]],
    device atomic_uint* qs1 [[buffer(12)]],
    device atomic_uint* node_active [[buffer(13)]],
    device atomic_uint* steal_index [[buffer(14)]],
    device atomic_uint* grid_barrier [[buffer(15)]],
    constant float& delta [[buffer(16)]],
    constant uint& max_iters [[buffer(17)]],
    constant uint& num_threadgroups [[buffer(18)]],
    uint tid [[thread_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]
) {
    uint iter = 0;
    while (iter < max_iters) {
        device atomic_uint* queue_size_current = (iter % 2 == 0) ? qs0 : qs1;
        device atomic_uint* queue_size_next = (iter % 2 == 0) ? qs1 : qs0;
        device atomic_uint* frontier_current = (iter % 2 == 0) ? f0 : f1;
        device atomic_uint* frontier_next = (iter % 2 == 0) ? f1 : f0;
        
        uint current_size = atomic_load_explicit(queue_size_current, memory_order_relaxed);
        uint current_bucket = atomic_load_explicit(&grid_barrier[2], memory_order_relaxed);
        float bucket_max = (float)(current_bucket + 1) * delta;
        
        if (current_size == 0) break;
        
        while (true) {
            uint base_idx = 0;
            if (simd_is_first()) {
                base_idx = atomic_fetch_add_explicit(steal_index, 32, memory_order_relaxed);
            }
            base_idx = simd_broadcast_first(base_idx);
            
            if (base_idx >= current_size) break;
            
            uint idx = base_idx + lane_id;
            bool is_valid = true;
            uint u = 0xFFFFFFFF;
            if (idx < current_size) {
                u = atomic_load_explicit(&frontier_current[idx], memory_order_relaxed);
            } else {
                is_valid = false;
            }
            if (u >= num_nodes) is_valid = false;
            
            float dist_u = 1e38f;
            int start = 0;
            int end = 0;
            
            if (is_valid) {
                dist_u = as_type<float>(atomic_load_explicit(&distances[u], memory_order_relaxed));
                if (dist_u > bucket_max) {
                    // Push back to next queue and skip processing
                    uint q_idx = atomic_fetch_add_explicit(queue_size_next, 1, memory_order_relaxed);
                    atomic_store_explicit(&frontier_next[q_idx], u, memory_order_relaxed);
                    
                    atomic_fetch_min_explicit(&grid_barrier[4], as_type<uint>(dist_u), memory_order_relaxed);
                    is_valid = false;
                } else if (dist_u >= 1e30f) {
                    is_valid = false;
                } else {
                    atomic_store_explicit(&node_active[u], 0, memory_order_relaxed);
                    atomic_fetch_add_explicit(&grid_barrier[3], 1, memory_order_relaxed);
                    start = indptr[u];
                    end = indptr[u+1];
                }
            }
            
            if (is_valid) {
                for (int i = start; i < end; ++i) {
                    int v = indices[i];
                    float w = weights[i];
                    float new_dist = dist_u + w;
                    
                    uint old_val_uint = atomic_load_explicit(&distances[v], memory_order_relaxed);
                    float old_val = as_type<float>(old_val_uint);
                    
                    while (new_dist < old_val) {
                        uint desired = as_type<uint>(new_dist);
                        if (atomic_compare_exchange_weak_explicit(&distances[v], &old_val_uint, desired, memory_order_relaxed, memory_order_relaxed)) {
                            predecessors[v] = int(u);
                            
                            uint was_active = atomic_exchange_explicit(&node_active[v], 1, memory_order_relaxed);
                            if (was_active == 0) {
                                uint q_idx = atomic_fetch_add_explicit(queue_size_next, 1, memory_order_relaxed);
                                atomic_store_explicit(&frontier_next[q_idx], (uint)v, memory_order_relaxed);
                            }
                            break;
                        }
                        old_val = as_type<float>(old_val_uint);
                    }
                }
            }
        }
        
        // --- Grid Barrier (Zero-Dispatch) ---
        threadgroup_barrier(mem_flags::mem_device);
        
        bool is_last = false;
        if (lane_id == 0 && simdgroup_id == 0) {
            uint old_count = atomic_fetch_add_explicit(&grid_barrier[0], 1, memory_order_relaxed);
            if (old_count == num_threadgroups - 1) {
                uint qs_next = atomic_load_explicit(queue_size_next, memory_order_relaxed);
                if (qs_next > 0) {
                    uint processed = atomic_load_explicit(&grid_barrier[3], memory_order_relaxed);
                    if (processed == 0) {
                        float min_dist = as_type<float>(atomic_load_explicit(&grid_barrier[4], memory_order_relaxed));
                        uint new_bucket = (uint)(min_dist / delta);
                        atomic_store_explicit(&grid_barrier[2], new_bucket, memory_order_relaxed);
                    }
                    atomic_store_explicit(&grid_barrier[3], 0, memory_order_relaxed);
                    atomic_store_explicit(&grid_barrier[4], 0x7F7FFFFF, memory_order_relaxed); // 1e38f approx
                }
                
                atomic_store_explicit(&grid_barrier[0], 0, memory_order_relaxed);
                atomic_store_explicit(queue_size_current, 0, memory_order_relaxed);
                atomic_store_explicit(steal_index, 0, memory_order_relaxed);
                is_last = true;
            }
        }
        
        // Flush the stores to L2 cache before incrementing generation
        threadgroup_barrier(mem_flags::mem_device);
        
        if (is_last) {
            if (delta < 0.1f) {
                atomic_store_explicit(&grid_barrier[1], 9999999, memory_order_relaxed);
            } else {
                atomic_fetch_add_explicit(&grid_barrier[1], 1, memory_order_relaxed);
            }
        }
        
        uint target_gen = iter + 1;
        while (true) {
            uint current_gen = atomic_load_explicit(&grid_barrier[1], memory_order_relaxed);
            if (current_gen >= target_gen) break;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        iter++;
    }
}

// -----------------------------------------------------------------------------
// 2. Negotiation Kernel (History Pressure)
// -----------------------------------------------------------------------------
kernel void negotiation_kernel(
    device atomic_uint* history_costs [[buffer(0)]],
    device const uint* node_congestion [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]]
) {
    // Example of SIMD shuffle down for reductions
    uint val = node_congestion[tid];
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    
    if (tid == 0) {
        atomic_fetch_add_explicit(history_costs, val, memory_order_relaxed);
    }
}

// -----------------------------------------------------------------------------
// 3. ROI Extractor Mixin
//
// Extracts distances for nodes whose (x, y, z) coordinates fall within
// the specified axis-aligned bounding box [x_min..x_max, y_min..y_max,
// z_min..z_max]. Matched distances and node IDs are compacted into
// output arrays using an atomic counter.
//
// Buffer layout:
//   [0] distances     — per-node shortest-path distances (float)
//   [1] roi_distances — output: filtered distances (float), sized to num_nodes
//   [2] roi_node_ids  — output: matched node IDs (uint), sized to num_nodes
//   [3] roi_count     — output: atomic counter of matched nodes (uint)
//   [4] node_x        — per-node X coordinates (float)
//   [5] node_y        — per-node Y coordinates (float)
//   [6] node_z        — per-node Z coordinates (float)
//   [7] roi_bounds    — [x_min, y_min, z_min, x_max, y_max, z_max] (float)
// -----------------------------------------------------------------------------
kernel void roi_extractor_mixin(
    device const float* distances      [[buffer(0)]],
    device float*       roi_distances  [[buffer(1)]],
    device uint*        roi_node_ids   [[buffer(2)]],
    device atomic_uint* roi_count      [[buffer(3)]],
    device const float* node_x         [[buffer(4)]],
    device const float* node_y         [[buffer(5)]],
    device const float* node_z         [[buffer(6)]],
    device const float* roi_bounds     [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    float x = node_x[tid];
    float y = node_y[tid];
    float z = node_z[tid];

    float x_min = roi_bounds[0];
    float y_min = roi_bounds[1];
    float z_min = roi_bounds[2];
    float x_max = roi_bounds[3];
    float y_max = roi_bounds[4];
    float z_max = roi_bounds[5];

    // Check if the node falls within the 3D bounding box
    bool in_roi = (x >= x_min && x <= x_max &&
                   y >= y_min && y <= y_max &&
                   z >= z_min && z <= z_max);

    // Only output nodes inside the ROI with finite distances
    if (in_roi && distances[tid] < 1e30f) {
        uint slot = atomic_fetch_add_explicit(roi_count, 1, memory_order_relaxed);
        roi_distances[slot] = distances[tid];
        roi_node_ids[slot]  = tid;
    }
}

// -----------------------------------------------------------------------------
// 4. Via Processing Kernels
//
// Computes via costs based on capacity and current usage:
//   - usage >= capacity → cost = INFINITY (hard-block, via is full)
//   - 0 < usage < capacity → cost = base_cost * (1.0 + usage/capacity)
//     (pooling penalty — congested vias become more expensive)
//   - usage == 0 → cost = base_cost (no penalty)
//
// Buffer layout:
//   [0] via_costs    — output: computed cost per via (float)
//   [1] via_capacity — per-via capacity limit (float)
//   [2] via_usage    — per-via current usage count (float)
//   [3] base_cost    — scalar base cost multiplier (float)
// -----------------------------------------------------------------------------
kernel void via_kernels(
    device float*       via_costs    [[buffer(0)]],
    device const float* via_capacity [[buffer(1)]],
    device const float* via_usage    [[buffer(2)]],
    constant float&     base_cost    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    float capacity = via_capacity[tid];
    float usage    = via_usage[tid];

    if (capacity <= 0.0f || usage >= capacity) {
        // Hard-block: via is at or over capacity (or has zero/negative capacity)
        via_costs[tid] = INFINITY;
    } else if (usage > 0.0f) {
        // Pooling penalty: linearly increasing cost as usage approaches capacity
        via_costs[tid] = base_cost * (1.0f + usage / capacity);
    } else {
        // No congestion — base cost only
        via_costs[tid] = base_cost;
    }
}
// -----------------------------------------------------------------------------
// 5. SPFA Setup Kernel
// -----------------------------------------------------------------------------
kernel void spfa_setup_kernel(
    device const float* distances [[buffer(0)]],
    device atomic_uint* active [[buffer(1)]],
    device atomic_uint* frontier [[buffer(2)]],
    device atomic_uint* queue_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (distances[tid] < 1e30f) {
        atomic_store_explicit(&active[tid], 1, memory_order_relaxed);
        uint idx = atomic_fetch_add_explicit(queue_size, 1, memory_order_relaxed);
        atomic_store_explicit(&frontier[idx], tid, memory_order_relaxed);
    } else {
        atomic_store_explicit(&active[tid], 0, memory_order_relaxed);
    }
}

// -----------------------------------------------------------------------------
// 6. Multi-Net SPFA Kernel (Phase 9)
// -----------------------------------------------------------------------------
kernel void wavefront_expand_multi(
    constant uint& num_nodes [[buffer(0)]],
    device int* predecessors [[buffer(4)]],
    device const int* indptr [[buffer(5)]],
    device const int* indices [[buffer(6)]],
    device const float* weights [[buffer(7)]],
    device atomic_uint* distances [[buffer(8)]],
    device atomic_uint* global_changed [[buffer(9)]],
    device atomic_uint* node_active [[buffer(10)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint u = tid.x;
    uint net_id = tid.y;
    
    if (u >= num_nodes) return;
    
    // Calculate the offset for this specific net
    uint offset = net_id * num_nodes;
    uint u_idx = offset + u;
    
    int start = indptr[u];
    int end = indptr[u+1];
    
    uint active = atomic_exchange_explicit(&node_active[u_idx], 0, memory_order_relaxed);
    
    bool local_changed = false;

    if (active != 0) {
        float dist_u = as_type<float>(atomic_load_explicit(&distances[u_idx], memory_order_relaxed));
        if (dist_u < 1e30f) {
            for (int i = start; i < end; ++i) {
                int v = indices[i];
                float w = weights[i];
                float new_dist = dist_u + w;
                
                uint v_idx = offset + v;
                uint old_val_uint = atomic_load_explicit(&distances[v_idx], memory_order_relaxed);
                float old_val = as_type<float>(old_val_uint);
                
                while (new_dist < old_val) {
                    uint desired = as_type<uint>(new_dist);
                    if (atomic_compare_exchange_weak_explicit(&distances[v_idx], &old_val_uint, desired, memory_order_relaxed, memory_order_relaxed)) {
                        predecessors[v_idx] = int(u);  
                        local_changed = true;
                        atomic_store_explicit(&node_active[v_idx], 1, memory_order_relaxed);
                        break;
                    }
                    old_val = as_type<float>(old_val_uint);
                }
            }
        }
    }
    
    // SIMD group reduction
    if (simd_any(local_changed)) {
        if (simd_is_first()) {
            atomic_store_explicit(&global_changed[net_id], 1, memory_order_relaxed);
        }
    }
}
