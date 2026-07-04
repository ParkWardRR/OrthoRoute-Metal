#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
using namespace metal;

// =============================================================================
// Utility: Atomic float add via CAS loop (IEEE-754 bit-cast)
// =============================================================================
inline void atomic_fetch_add_float(device atomic_uint* dest, float val) {
    uint old_val_uint = atomic_load_explicit(dest, memory_order_relaxed);
    while (true) {
        float old_val = as_type<float>(old_val_uint);
        float new_val = old_val + val;
        uint desired = as_type<uint>(new_val);
        if (atomic_compare_exchange_weak_explicit(dest, &old_val_uint, desired,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

// =============================================================================
// Utility: Atomic float Min via CAS loop
// =============================================================================
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

// =============================================================================
// Utility: SIMD Prefix-Sum Queue Compaction
// =============================================================================
// Instead of each thread doing atomic_fetch_add(queue_size, 1) to push to the
// frontier queue (32 global atomics per SIMD-group), we use SIMD prefix-sum
// to compute local offsets within the group, then do ONE global atomic to
// reserve a contiguous block for the entire SIMD-group.
//
// This reduces global memory contention by up to 32× on the queue counter.
//
// Usage:
//   bool wants_to_push = (new_dist < old_dist);
//   uint local_offset = simd_prefix_exclusive_sum(wants_to_push ? 1u : 0u);
//   uint group_total  = simd_sum(wants_to_push ? 1u : 0u);
//   uint base;
//   if (simd_is_first()) base = atomic_fetch_add_explicit(queue_size, group_total, memory_order_relaxed);
//   base = simd_broadcast_first(base);
//   if (wants_to_push) frontier[base + local_offset] = node_id;
//
inline uint simd_enqueue(
    device atomic_uint* frontier,
    device atomic_uint* queue_size,
    uint node_id,
    bool wants_to_push
) {
    // Compute per-thread offset within SIMD-group via prefix-sum
    uint contribution = wants_to_push ? 1u : 0u;
    uint local_offset = simd_prefix_exclusive_sum(contribution);
    uint group_total  = simd_sum(contribution);
    
    // Only the first active lane does the single global atomic
    uint base = 0;
    if (group_total > 0) {
        if (simd_is_first()) {
            base = atomic_fetch_add_explicit(queue_size, group_total, memory_order_relaxed);
        }
        base = simd_broadcast_first(base);
    }
    
    // Each thread writes to its reserved slot
    if (wants_to_push) {
        atomic_store_explicit(&frontier[base + local_offset], node_id, memory_order_relaxed);
    }
    
    return base + local_offset;
}

// =============================================================================
// 1. Clear Counters
// =============================================================================
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

// =============================================================================
// 2. Persistent Thread Work-Stealing Queue (Δ-Stepping SPFA)
// =============================================================================
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
                    // SIMD Prefix-Sum Queue Compaction: batch deferred nodes
                    // Instead of 32 per-thread atomic_fetch_add, one SIMD-group
                    // atomic reserves a contiguous block for all deferred lanes.
                    simd_enqueue(frontier_next, queue_size_next, u, true);
                    
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

// =============================================================================
// 3. Negotiation Kernel (History Pressure — SIMD Reduction)
// =============================================================================
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

// =============================================================================
// 4. ROI Extractor — Legacy Distance-Filtering Kernel
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
// =============================================================================
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

// =============================================================================
// 5. Legacy Via Processing Kernel (simple capacity/usage model)
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
// =============================================================================
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

// =============================================================================
// 6. SPFA Setup Kernel
// =============================================================================
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

// =============================================================================
// 7. Multi-Net SPFA Kernel (Phase 9)
// =============================================================================
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


// #############################################################################
// #############################################################################
// ##                                                                         ##
// ##  CUDA-PARITY VIA KERNELS (matching via_kernels.py exactly)              ##
// ##                                                                         ##
// #############################################################################
// #############################################################################

// =============================================================================
// 8. Hard-Block Via Capacity Kernel (CUDA Kernel #1 parity)
//
// Checks BOTH column capacity AND per-segment capacity for each via edge.
// If the column is at capacity OR any spanned segment is at capacity,
// the edge cost is set to INFINITY and the blocked counter is incremented.
//
// Buffer layout:
//   [0] edge_indices    — [num_via_edges] Edge indices into the cost array
//   [1] xy_coords       — [num_via_edges * 2] Interleaved (x,y) per via edge
//   [2] z_lo            — [num_via_edges] Lower z bound per via edge
//   [3] z_hi            — [num_via_edges] Upper z bound per via edge
//   [4] via_col_use     — [Nx * Ny] Column usage (short/int16)
//   [5] via_col_cap     — [Nx * Ny] Column capacity (short/int16)
//   [6] via_seg_use     — [Nx * Ny * segZ] Segment usage (int8)
//   [7] via_seg_cap     — [Nx * Ny * segZ] Segment capacity (int8)
//   [8] total_cost      — [num_edges] Cost array (float), modified in-place
//   [9] blocked_count   — [1] Atomic counter of blocked edges
//   [10] params         — [3] = {num_via_edges, Ny, segZ}
// =============================================================================
kernel void hard_block_via_capacity(
    device const int*    edge_indices   [[buffer(0)]],   // [num_via_edges]
    device const int*    xy_coords      [[buffer(1)]],   // [num_via_edges * 2] interleaved (x,y)
    device const int*    z_lo           [[buffer(2)]],   // [num_via_edges]
    device const int*    z_hi           [[buffer(3)]],   // [num_via_edges]
    device const short*  via_col_use    [[buffer(4)]],   // [Nx * Ny] flattened
    device const short*  via_col_cap    [[buffer(5)]],   // [Nx * Ny] flattened
    device const char*   via_seg_use    [[buffer(6)]],   // [Nx * Ny * segZ] flattened
    device const char*   via_seg_cap    [[buffer(7)]],   // [Nx * Ny * segZ] flattened
    device atomic_uint*  total_cost     [[buffer(8)]],   // [num_edges] as atomic_uint for CAS float ops
    device atomic_uint*  blocked_count  [[buffer(9)]],   // [1] atomic counter
    constant int*        params         [[buffer(10)]],  // [3] = {num_via_edges, Ny, segZ}
    uint tid [[thread_position_in_grid]]
) {
    int num_via_edges = params[0];
    int Ny            = params[1];
    int segZ          = params[2];

    int idx = (int)tid;
    if (idx >= num_via_edges) return;

    // Get via location and span
    int xu      = xy_coords[idx * 2 + 0];
    int yu      = xy_coords[idx * 2 + 1];
    int z_start = z_lo[idx];
    int z_end   = z_hi[idx];

    // Check column capacity (single global memory access)
    int col_idx     = xu * Ny + yu;
    bool col_blocked = (via_col_use[col_idx] >= via_col_cap[col_idx]);

    // Check segment capacity (loop over spanned segments)
    bool seg_blocked = false;
    if (!col_blocked) {  // Skip if already blocked by column
        for (int z = z_start; z < z_end; z++) {
            int seg_idx = z - 1;
            if (seg_idx >= 0 && seg_idx < segZ) {
                int seg_offset = col_idx * segZ + seg_idx;
                if (via_seg_use[seg_offset] >= via_seg_cap[seg_offset]) {
                    seg_blocked = true;
                    break;
                }
            }
        }
    }

    // Hard-block if at capacity: set cost to INFINITY
    if (col_blocked || seg_blocked) {
        int edge_idx = edge_indices[idx];
        // Write INFINITY (0x7f800000) via atomic store (IEEE-754 +inf)
        atomic_store_explicit(&total_cost[edge_idx], 0x7f800000u, memory_order_relaxed);
        atomic_fetch_add_explicit(blocked_count, 1, memory_order_relaxed);
    }
}

// =============================================================================
// 9. Apply Via Pooling Penalties Kernel (CUDA Kernel #2 parity)
//
// Calculates column penalty + sum of segment penalties for each via edge,
// and atomically adds the penalty to the edge's total cost.
//
// Buffer layout:
//   [0] edge_indices    — [num_via_edges] Edge indices into cost array
//   [1] xy_coords       — [num_via_edges * 2] Interleaved (x,y) per via edge
//   [2] z_lo            — [num_via_edges] Lower z bound
//   [3] z_hi            — [num_via_edges] Upper z bound
//   [4] via_col_pres    — [Nx * Ny] Column present congestion (float)
//   [5] via_seg_pres    — [Nx * Ny * segZ] Segment present congestion (float)
//   [6] total_cost      — [num_edges] Cost array (float), modified via atomic add
//   [7] penalty_count   — [1] Atomic counter of penalties applied
//   [8] params          — [3] = {num_via_edges, Ny, segZ}
//   [9] weights         — [2] = {col_weight, seg_weight} (float)
// =============================================================================
kernel void apply_via_pooling_penalties(
    device const int*    edge_indices   [[buffer(0)]],   // [num_via_edges]
    device const int*    xy_coords      [[buffer(1)]],   // [num_via_edges * 2]
    device const int*    z_lo           [[buffer(2)]],   // [num_via_edges]
    device const int*    z_hi           [[buffer(3)]],   // [num_via_edges]
    device const float*  via_col_pres   [[buffer(4)]],   // [Nx * Ny]
    device const float*  via_seg_pres   [[buffer(5)]],   // [Nx * Ny * segZ]
    device atomic_uint*  total_cost     [[buffer(6)]],   // [num_edges] as atomic_uint for CAS float add
    device atomic_uint*  penalty_count  [[buffer(7)]],   // [1] atomic counter
    constant int*        params         [[buffer(8)]],   // [3] = {num_via_edges, Ny, segZ}
    constant float*      weights        [[buffer(9)]],   // [2] = {col_weight, seg_weight}
    uint tid [[thread_position_in_grid]]
) {
    int num_via_edges = params[0];
    int Ny            = params[1];
    int segZ          = params[2];
    float col_weight  = weights[0];
    float seg_weight  = weights[1];

    int idx = (int)tid;
    if (idx >= num_via_edges) return;

    // Get via location and span
    int xu      = xy_coords[idx * 2 + 0];
    int yu      = xy_coords[idx * 2 + 1];
    int z_start = z_lo[idx];
    int z_end   = z_hi[idx];

    // Calculate column penalty
    int col_idx  = xu * Ny + yu;
    float penalty = via_col_pres[col_idx] * col_weight;

    // Calculate segment penalties (sum over spanned segments)
    for (int z = z_start; z < z_end; z++) {
        int seg_idx = z - 1;
        if (seg_idx >= 0 && seg_idx < segZ) {
            int seg_offset = col_idx * segZ + seg_idx;
            penalty += via_seg_pres[seg_offset] * seg_weight;
        }
    }

    // Apply penalty to edge cost (atomic float add for thread safety)
    if (penalty > 0.0f) {
        int edge_idx = edge_indices[idx];
        atomic_fetch_add_float(&total_cost[edge_idx], penalty);
        atomic_fetch_add_explicit(penalty_count, 1, memory_order_relaxed);
    }
}

// =============================================================================
// 10. Via Barrel Conflict Detection Kernel (CUDA Kernel #4 parity)
//
// Detects when committed edges touch via barrel nodes owned by other nets.
// For each edge, checks if either the source or destination node is owned
// by a different net — if so, it's a barrel conflict.
//
// Buffer layout:
//   [0] edge_indices         — [num_edges_to_check] Edge indices to check
//   [1] edge_net_ids         — [num_edges_to_check] Net ID per edge
//   [2] edge_src_map         — [total_edges] Precomputed src node for each edge idx
//   [3] graph_indices        — [total_edges] CSR indices (destination nodes)
//   [4] node_owner           — [num_nodes] Node ownership (-1 = free, else net_id)
//   [5] conflict_count       — [1] Atomic counter of total conflicts detected
//   [6] conflict_edge_flags  — [num_edges_to_check] 1 if conflict, 0 otherwise (optional)
//   [7] params               — [1] = {num_edges_to_check}
// =============================================================================
kernel void detect_barrel_conflicts(
    device const int*    edge_indices         [[buffer(0)]],
    device const int*    edge_net_ids         [[buffer(1)]],
    device const int*    edge_src_map         [[buffer(2)]],
    device const int*    graph_indices        [[buffer(3)]],
    device const int*    node_owner           [[buffer(4)]],
    device atomic_uint*  conflict_count       [[buffer(5)]],
    device int*          conflict_edge_flags  [[buffer(6)]],
    constant int*        params               [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    int num_edges_to_check = params[0];
    int has_flags          = params[1];  // 1 if conflict_edge_flags is valid, 0 if not

    int idx = (int)tid;
    if (idx >= num_edges_to_check) return;

    int edge_idx = edge_indices[idx];
    int net_id   = edge_net_ids[idx];

    // Get source and destination nodes for this edge
    int src_node = edge_src_map[edge_idx];
    int dst_node = graph_indices[edge_idx];

    // Check if either endpoint is owned by a different net (via barrel conflict!)
    int src_owner = node_owner[src_node];
    int dst_owner = node_owner[dst_node];

    bool conflict = false;

    // Conflict if src is owned by another net
    if (src_owner != -1 && src_owner != net_id) {
        conflict = true;
    }

    // Conflict if dst is owned by another net
    if (dst_owner != -1 && dst_owner != net_id) {
        conflict = true;
    }

    // Record conflict
    if (conflict) {
        atomic_fetch_add_explicit(conflict_count, 1, memory_order_relaxed);
        if (has_flags != 0) {
            conflict_edge_flags[idx] = 1;
        }
    } else {
        if (has_flags != 0) {
            conflict_edge_flags[idx] = 0;
        }
    }
}

// =============================================================================
// 11. Owner-Aware Via Keepout Blocking Kernel (CUDA Kernel #3 parity)
//
// Blocks outgoing planar edges from via keepout nodes owned by other nets.
// For each keepout node, if it's not owned by the current net, blocks all
// planar outgoing edges (where neighbor has same z-coordinate).
//
// Buffer layout:
//   [0] via_keepout_nodes  — [num_keepouts] Node indices occupied by vias
//   [1] via_keepout_owners — [num_keepouts] Owner net IDs (int)
//   [2] indptr             — Graph CSR indptr
//   [3] indices            — Graph CSR indices (destinations)
//   [4] node_coords_z      — [num_nodes] Z coordinate for each node (int)
//   [5] costs              — [num_edges] Cost array (float), modified in-place
//   [6] blocked_count      — [1] Atomic counter of blocked edges
//   [7] params             — [3] = {num_keepouts, current_net_id, num_nodes}
//   [8] block_cost_buf     — [1] = {block_cost} (float)
// =============================================================================
kernel void block_via_keepouts_owner_aware(
    device const int*    via_keepout_nodes  [[buffer(0)]],
    device const int*    via_keepout_owners [[buffer(1)]],
    device const int*    indptr             [[buffer(2)]],
    device const int*    indices            [[buffer(3)]],
    device const int*    node_coords_z      [[buffer(4)]],
    device float*        costs              [[buffer(5)]],
    device atomic_uint*  blocked_count      [[buffer(6)]],
    constant int*        params             [[buffer(7)]],
    constant float*      block_cost_buf     [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    int num_keepouts   = params[0];
    int current_net_id = params[1];
    int num_nodes      = params[2];
    float block_cost   = block_cost_buf[0];

    int kid = (int)tid;
    if (kid >= num_keepouts) return;

    // Skip vias owned by current net (owner-aware!)
    if (via_keepout_owners[kid] == current_net_id) return;

    int node_idx = via_keepout_nodes[kid];
    if (node_idx < 0 || node_idx >= num_nodes) return;

    int node_z = node_coords_z[node_idx];

    // Block all outgoing planar edges from this via node
    int edge_start = indptr[node_idx];
    int edge_end   = indptr[node_idx + 1];

    for (int eid = edge_start; eid < edge_end; eid++) {
        int neighbor = indices[eid];
        if (neighbor < 0 || neighbor >= num_nodes) continue;

        int neighbor_z = node_coords_z[neighbor];

        // Block planar edges (same z-coordinate)
        if (neighbor_z == node_z) {
            costs[eid] = block_cost;
            atomic_fetch_add_explicit(blocked_count, 1, memory_order_relaxed);
        }
    }
}


// #############################################################################
// #############################################################################
// ##                                                                         ##
// ##  CUDA-PARITY ROI EXTRACTION KERNELS                                     ##
// ##  (matching roi_extractor_mixin.py spatial-index approach)                ##
// ##                                                                         ##
// #############################################################################
// #############################################################################

// =============================================================================
// 12. ROI Mark Nodes Kernel (Spatial-Index Based)
//
// Uses a pre-built spatial grid index (spatial_indptr + spatial_node_ids)
// to mark all nodes within the ROI bounding box. Each thread processes
// one (layer, cell_x, cell_y) combination — O(1) cell lookup.
//
// This replaces coordinate-comparison with grid-cell enumeration,
// matching the CUDA extract_roi_nodes kernel behavior exactly.
//
// Buffer layout:
//   [0] spatial_indptr    — [max_cell_id + 1] CSR-style indptr for spatial grid
//   [1] spatial_node_ids  — [total_entries] Node IDs stored in spatial grid cells
//   [2] roi_node_mask     — [total_nodes] Output: 1 if node is in ROI, 0 otherwise
//   [3] params            — [8] = {grid_x0, grid_y0, grid_x1, grid_y1,
//                                   grid_width, grid_height, max_layers, max_cell_id}
//   [4] total_nodes_buf   — [1] = {total_nodes}
// =============================================================================
kernel void roi_mark_nodes(
    device const int*    spatial_indptr    [[buffer(0)]],
    device const int*    spatial_node_ids  [[buffer(1)]],
    device atomic_uint*  roi_node_mask     [[buffer(2)]],  // uint: 1=in ROI, 0=not
    constant int*        params            [[buffer(3)]],
    constant int*        total_nodes_buf   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int grid_x0      = params[0];
    int grid_y0      = params[1];
    int grid_x1      = params[2];
    int grid_y1      = params[3];
    int grid_width   = params[4];
    int grid_height  = params[5];
    int max_layers   = params[6];
    int max_cell_id  = params[7];
    int total_nodes  = total_nodes_buf[0];

    // Thread configuration: each thread processes one (layer, cell) combination
    int roi_width  = grid_x1 - grid_x0;
    int roi_height = grid_y1 - grid_y0;
    int cells_per_layer = roi_width * roi_height;
    int total_cells     = max_layers * cells_per_layer;

    int idx = (int)tid;
    if (idx >= total_cells) return;

    // Decompose thread index into (layer, cell_y, cell_x)
    int layer        = idx / cells_per_layer;
    int cell_in_layer = idx % cells_per_layer;
    int cell_y       = cell_in_layer / roi_width + grid_y0;
    int cell_x       = cell_in_layer % roi_width + grid_x0;

    // Calculate global cell ID: layer_offset + cell_y * grid_width + cell_x
    int layer_offset = layer * grid_width * grid_height;
    int cell_id      = layer_offset + cell_y * grid_width + cell_x;

    // Bounds check
    if (cell_id < 0 || cell_id >= max_cell_id) return;

    // Get node range for this cell from spatial index
    int start_idx = spatial_indptr[cell_id];
    int end_idx   = spatial_indptr[cell_id + 1];

    // Mark all nodes in this cell as part of ROI
    for (int i = start_idx; i < end_idx; i++) {
        int node_id = spatial_node_ids[i];
        if (node_id >= 0 && node_id < total_nodes) {
            atomic_store_explicit(&roi_node_mask[node_id], 1u, memory_order_relaxed);
        }
    }
}

// =============================================================================
// 13. ROI Extract Subgraph Kernel (CSR Subgraph Extraction)
//
// Given a node mask (from roi_mark_nodes) and global→local mapping,
// extracts the CSR subgraph for nodes within the ROI.
//
// Phase 1 (count_only=1): Count edges per ROI node (for building indptr)
// Phase 2 (count_only=0): Write edges into output arrays using offsets
//
// Buffer layout:
//   [0] roi_nodes        — [num_roi_nodes] Global node IDs in ROI (sorted)
//   [1] global_to_local  — [max_global_id] Maps global ID → local index (-1 if not in ROI)
//   [2] csr_indptr       — [num_total_nodes + 1] Global CSR indptr
//   [3] csr_indices      — [num_total_edges] Global CSR column indices
//   [4] csr_weights      — [num_total_edges] Global CSR edge weights (float)
//   [5] out_edge_counts  — [num_roi_nodes] Output: edge count per ROI node (Phase 1)
//                           OR output indptr for offset lookup (Phase 2)
//   [6] out_indices      — [max_output_edges] Output: local dest indices (Phase 2)
//   [7] out_weights      — [max_output_edges] Output: edge weights (Phase 2)
//   [8] params           — [3] = {num_roi_nodes, max_global_id, count_only}
// =============================================================================
kernel void roi_extract_subgraph(
    device const int*    roi_nodes        [[buffer(0)]],
    device const int*    global_to_local  [[buffer(1)]],
    device const int*    csr_indptr       [[buffer(2)]],
    device const int*    csr_indices      [[buffer(3)]],
    device const float*  csr_weights      [[buffer(4)]],
    device int*          out_edge_counts  [[buffer(5)]],  // Phase 1: counts; Phase 2: prefix-sum offsets
    device int*          out_indices      [[buffer(6)]],
    device float*        out_weights      [[buffer(7)]],
    constant int*        params           [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    int num_roi_nodes = params[0];
    int max_global_id = params[1];
    int count_only    = params[2];  // 1 = Phase 1 (count), 0 = Phase 2 (extract)

    int local_src = (int)tid;
    if (local_src >= num_roi_nodes) return;

    int global_src = roi_nodes[local_src];

    // Get edges for this source node from global CSR
    int edge_start = csr_indptr[global_src];
    int edge_end   = csr_indptr[global_src + 1];

    if (count_only == 1) {
        // Phase 1: Just count how many edges have both endpoints in ROI
        int count = 0;
        for (int e = edge_start; e < edge_end; e++) {
            int global_dst = csr_indices[e];
            if (global_dst >= 0 && global_dst < max_global_id) {
                int local_dst = global_to_local[global_dst];
                if (local_dst >= 0) {
                    count++;
                }
            }
        }
        out_edge_counts[local_src] = count;
    } else {
        // Phase 2: Write edges using precomputed offsets (out_edge_counts = prefix sum)
        int write_offset = out_edge_counts[local_src];
        for (int e = edge_start; e < edge_end; e++) {
            int global_dst = csr_indices[e];
            if (global_dst >= 0 && global_dst < max_global_id) {
                int local_dst = global_to_local[global_dst];
                if (local_dst >= 0) {
                    out_indices[write_offset] = local_dst;
                    out_weights[write_offset] = csr_weights[e];
                    write_offset++;
                }
            }
        }
    }
}
