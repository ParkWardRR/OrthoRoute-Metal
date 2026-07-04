use pyo3::prelude::*;
mod accelerate_ops;

use metal::*;
use std::mem;
use std::env;
use numpy::{PyReadonlyArray1, PyArray1};

const KERNEL_SRC: &str = include_str!("kernels.metal");

/// Metal-accelerated Dijkstra shortest path finder,
/// replacing CUDADijkstra from the original OrthoRoute.
#[pyclass]
#[allow(dead_code)]
pub struct MetalDijkstra {
    device: Device,
    command_queue: CommandQueue,
    // --- Existing pipeline states ---
    wavefront_pipeline: ComputePipelineState,
    wavefront_multi_pipeline: ComputePipelineState,
    roi_pipeline: ComputePipelineState,
    via_pipeline: ComputePipelineState,
    spfa_pipeline: ComputePipelineState,
    clear_counters_pipeline: ComputePipelineState,
    // --- New CUDA-parity pipeline states ---
    hard_block_pipeline: ComputePipelineState,
    via_penalty_pipeline: ComputePipelineState,
    barrel_conflict_pipeline: ComputePipelineState,
    keepout_blocking_pipeline: ComputePipelineState,
    roi_mark_nodes_pipeline: ComputePipelineState,
    roi_extract_subgraph_pipeline: ComputePipelineState,
    // --- Persistent buffers ---
    indptr_buf: Option<Buffer>,
    indices_buf: Option<Buffer>,
    weights_buf: Option<Buffer>,
    distances_buf: Option<Buffer>,
    predecessors_buf: Option<Buffer>,
    frontier_buf_0: Option<Buffer>,
    frontier_buf_1: Option<Buffer>,
    queue_size_buf_0: Option<Buffer>,
    queue_size_buf_1: Option<Buffer>,
    steal_index_buf: Option<Buffer>,
    active_buf: Option<Buffer>,
    grid_barrier_state_buf: Option<Buffer>,
    node_count: usize,
}

/// Helper: create a zero-copy Metal buffer from a NumPy slice.
/// Returns the buffer and a phantom reference to keep the slice alive.
fn make_nocopy_buffer(device: &Device, ptr: *const u8, len: usize) -> Buffer {
    let options = MTLResourceOptions::StorageModeShared;
    device.new_buffer_with_bytes_no_copy(
        ptr as *const _,
        len as u64,
        options,
        None,
    )
}

/// Helper: create a Metal buffer by copying data.
fn make_buffer_with_data(device: &Device, data: &[u8]) -> Buffer {
    let options = MTLResourceOptions::StorageModeShared;
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        data.len() as u64,
        options,
    )
}

/// Helper: create a zeroed Metal buffer of given byte size.
fn make_zeroed_buffer(device: &Device, byte_size: usize) -> Buffer {
    let options = MTLResourceOptions::StorageModeShared;
    let buf = device.new_buffer(byte_size as u64, options);
    unsafe {
        std::ptr::write_bytes(buf.contents() as *mut u8, 0, byte_size);
    }
    buf
}


#[pymethods]
impl MetalDijkstra {
    #[new]
    fn new() -> PyResult<Self> {
        let device = Device::system_default().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device found!"))?;
        let command_queue = device.new_command_queue();
        
        let compile_options = CompileOptions::new();
        let library = device.new_library_with_source(KERNEL_SRC, &compile_options).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to compile MSL: {}", e))
        })?;

        // Existing kernels
        let wavefront_func = library.get_function("wavefront_expand_all", None).unwrap();
        let wavefront_multi_func = library.get_function("wavefront_expand_multi", None).unwrap();
        let roi_func = library.get_function("roi_extractor_mixin", None).unwrap();
        let via_func = library.get_function("via_kernels", None).unwrap();
        let spfa_func = library.get_function("spfa_setup_kernel", None).unwrap();
        let clear_func = library.get_function("clear_counters", None).unwrap();

        // New CUDA-parity kernels
        let hard_block_func = library.get_function("hard_block_via_capacity", None).unwrap();
        let via_penalty_func = library.get_function("apply_via_pooling_penalties", None).unwrap();
        let barrel_conflict_func = library.get_function("detect_barrel_conflicts", None).unwrap();
        let keepout_blocking_func = library.get_function("block_via_keepouts_owner_aware", None).unwrap();
        let roi_mark_nodes_func = library.get_function("roi_mark_nodes", None).unwrap();
        let roi_extract_subgraph_func = library.get_function("roi_extract_subgraph", None).unwrap();

        // Create pipeline states
        let wavefront_pipeline = device.new_compute_pipeline_state_with_function(&wavefront_func).unwrap();
        let wavefront_multi_pipeline = device.new_compute_pipeline_state_with_function(&wavefront_multi_func).unwrap();
        let roi_pipeline = device.new_compute_pipeline_state_with_function(&roi_func).unwrap();
        let via_pipeline = device.new_compute_pipeline_state_with_function(&via_func).unwrap();
        let spfa_pipeline = device.new_compute_pipeline_state_with_function(&spfa_func).unwrap();
        let clear_counters_pipeline = device.new_compute_pipeline_state_with_function(&clear_func).unwrap();

        let hard_block_pipeline = device.new_compute_pipeline_state_with_function(&hard_block_func).unwrap();
        let via_penalty_pipeline = device.new_compute_pipeline_state_with_function(&via_penalty_func).unwrap();
        let barrel_conflict_pipeline = device.new_compute_pipeline_state_with_function(&barrel_conflict_func).unwrap();
        let keepout_blocking_pipeline = device.new_compute_pipeline_state_with_function(&keepout_blocking_func).unwrap();
        let roi_mark_nodes_pipeline = device.new_compute_pipeline_state_with_function(&roi_mark_nodes_func).unwrap();
        let roi_extract_subgraph_pipeline = device.new_compute_pipeline_state_with_function(&roi_extract_subgraph_func).unwrap();
        
        println!("[Metal-Init] Initialized MetalDijkstra on device: {:?}", device.name());
        println!("[Metal-Init] CUDA-parity kernels compiled: hard_block, via_penalty, barrel_conflict, keepout, roi_mark, roi_extract");
        
        Ok(MetalDijkstra {
            device,
            command_queue,
            wavefront_pipeline,
            wavefront_multi_pipeline,
            roi_pipeline,
            via_pipeline,
            spfa_pipeline,
            clear_counters_pipeline,
            hard_block_pipeline,
            via_penalty_pipeline,
            barrel_conflict_pipeline,
            keepout_blocking_pipeline,
            roi_mark_nodes_pipeline,
            roi_extract_subgraph_pipeline,
            indptr_buf: None,
            indices_buf: None,
            weights_buf: None,
            distances_buf: None,
            predecessors_buf: None,
            frontier_buf_0: None,
            frontier_buf_1: None,
            queue_size_buf_0: None,
            queue_size_buf_1: None,
            steal_index_buf: None,
            active_buf: None,
            grid_barrier_state_buf: None,
            node_count: 0,
        })
    }

    /// Receives CSR Graph arrays from Python (SciPy/CuPy fallback to NumPy)
    /// without copying, and maps them to Metal UMA buffers.
    pub fn set_graph_csr<'py>(
        &mut self,
        _py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i32>,
        indices: PyReadonlyArray1<'py, i32>,
        weights: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<String> {
        let indptr_slice = indptr.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("indptr not contiguous: {}", e)))?;
        let indices_slice = indices.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("indices not contiguous: {}", e)))?;
        let weights_slice = weights.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("weights not contiguous: {}", e)))?;
            
        let options = MTLResourceOptions::StorageModeShared;
        
        self.indptr_buf = Some(self.device.new_buffer_with_bytes_no_copy(
            indptr_slice.as_ptr() as *const _,
            (indptr_slice.len() * mem::size_of::<i32>()) as u64,
            options,
            None,
        ));
        self.indices_buf = Some(self.device.new_buffer_with_bytes_no_copy(
            indices_slice.as_ptr() as *const _,
            (indices_slice.len() * mem::size_of::<i32>()) as u64,
            options,
            None,
        ));
        self.weights_buf = Some(self.device.new_buffer_with_bytes_no_copy(
            weights_slice.as_ptr() as *const _,
            (weights_slice.len() * mem::size_of::<f32>()) as u64,
            options,
            None,
        ));
        
        self.node_count = indptr_slice.len().saturating_sub(1);
        
        // Setup Frontier Queues for Work-efficient BF
        let frontier_size = (self.node_count * mem::size_of::<u32>()) as u64;
        self.frontier_buf_0 = Some(self.device.new_buffer(frontier_size, options));
        self.frontier_buf_1 = Some(self.device.new_buffer(frontier_size, options));
        self.queue_size_buf_0 = Some(self.device.new_buffer(mem::size_of::<u32>() as u64, options));
        self.queue_size_buf_1 = Some(self.device.new_buffer(mem::size_of::<u32>() as u64, options));
        
        // The index threads use to steal work from the frontier
        self.steal_index_buf = Some(self.device.new_buffer(mem::size_of::<u32>() as u64, options));
        
        self.active_buf = Some(self.device.new_buffer(
            (self.node_count * mem::size_of::<u32>()) as u64,
            options,
        ));
        self.grid_barrier_state_buf = Some(self.device.new_buffer(
            (5 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        
        // Predecessor buffer for path reconstruction (initialized to -1)
        let pred_size = self.node_count * mem::size_of::<i32>();
        let pred_buf = self.device.new_buffer(pred_size as u64, MTLResourceOptions::StorageModeShared);
        let pred_ptr = pred_buf.contents() as *mut i32;
        unsafe {
            for i in 0..self.node_count {
                *pred_ptr.add(i) = -1;
            }
        }
        self.predecessors_buf = Some(pred_buf);
        
        Ok(format!(
            "Mapped UMA CSR Buffers. indptr: {}, indices: {}, weights: {}",
            indptr_slice.len(), indices_slice.len(), weights_slice.len()
        ))
    }

    /// Receives Graph data and expands wavefronts
    pub fn set_distances_csr<'py>(&mut self, _py: Python<'py>, dists: PyReadonlyArray1<'py, f32>) -> PyResult<()> {
        let slice = dists.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("dists not contiguous: {}", e)))?;
        self.distances_buf = Some(self.device.new_buffer_with_bytes_no_copy(
            slice.as_ptr() as *const _,
            (slice.len() * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        ));
        Ok(())
    }

    /// Setup SPFA frontier based on current distances
    pub fn setup_spfa(&self) -> PyResult<()> {
        if let (Some(distances), Some(active), Some(frontier_0), Some(queue_size_0), Some(qs1), Some(steal_index)) = 
            (&self.distances_buf, &self.active_buf, &self.frontier_buf_0, &self.queue_size_buf_0, &self.queue_size_buf_1, &self.steal_index_buf) 
        {
            unsafe { *(queue_size_0.contents() as *mut u32) = 0; }
            
            let command_buffer_init = self.command_queue.new_command_buffer();
            let encoder_init = command_buffer_init.new_compute_command_encoder();
            encoder_init.push_debug_group("Clear Counters Init");
            encoder_init.set_compute_pipeline_state(&self.clear_counters_pipeline);
            encoder_init.set_buffer(0, Some(qs1), 0); // MUST be qs1, qs0 has the sources!
            encoder_init.set_buffer(1, Some(steal_index), 0);
            encoder_init.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder_init.end_encoding();
            
            let encoder = command_buffer_init.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.spfa_pipeline);
            encoder.set_buffer(0, Some(distances), 0);
            encoder.set_buffer(1, Some(active), 0);
            encoder.set_buffer(2, Some(frontier_0), 0);
            encoder.set_buffer(3, Some(queue_size_0), 0);
            
            let grid_size = MTLSize::new(self.node_count.max(1) as u64, 1, 1);
            let threadgroup_size = MTLSize::new(512u64.min(self.spfa_pipeline.max_total_threads_per_threadgroup()), 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
            
            encoder.end_encoding();
            command_buffer_init.commit();
            command_buffer_init.wait_until_completed();
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err("Buffers not initialized for SPFA"))
        }
    }

    pub fn execute_until_convergence(&self, max_iters: u32, _batch_size: u32, tg_size: u32, delta: f32) -> PyResult<(u32, bool)> {
        let _device = self.device.clone();
        let _command_queue = self.command_queue.clone();
        let is_capturing = env::var("METAL_CAPTURE_TRACE").unwrap_or_default() == "1";
        if is_capturing {
            let capture_manager = CaptureManager::shared();
            capture_manager.start_capture_with_command_queue(&self.command_queue);
        }
        
        if let (
            Some(indptr), Some(indices), Some(weights), Some(distances), Some(predecessors), 
            Some(active), Some(f0), Some(f1), Some(qs0), Some(qs1), Some(steal_index),
            Some(grid_barrier_state)
        ) = (
            &self.indptr_buf, &self.indices_buf, &self.weights_buf, &self.distances_buf, 
            &self.predecessors_buf, &self.active_buf, &self.frontier_buf_0, &self.frontier_buf_1, 
            &self.queue_size_buf_0, &self.queue_size_buf_1, &self.steal_index_buf,
            &self.grid_barrier_state_buf
        ) {
            let total_iters;
            let converged;
            let num_nodes: u32 = self.node_count as u32;
            
            let persistent_threads = 8192u64;
            let optimal_threadgroup = (tg_size as u64).min(self.wavefront_pipeline.max_total_threads_per_threadgroup());
            let threadgroup_size = MTLSize::new(optimal_threadgroup, 1, 1);
            let persistent_grid = MTLSize::new(persistent_threads, 1, 1);
            
            let num_threadgroups = persistent_threads / optimal_threadgroup;
            
            // Clear the Grid Barrier State (5 elements)
            unsafe {
                let ptr = self.grid_barrier_state_buf.as_ref().unwrap().contents() as *mut u32;
                *ptr.add(0) = 0; // count
                *ptr.add(1) = 0; // generation
                *ptr.add(2) = 0; // current_bucket
                *ptr.add(3) = 0; // nodes_processed
                *ptr.add(4) = 0x7F7FFFFF; // min_skipped_dist (approx 1e38f)
            }
            
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.push_debug_group("Persistent Zero-Dispatch SPFA");
            
            encoder.set_compute_pipeline_state(&self.wavefront_pipeline);
            encoder.set_bytes(0, mem::size_of::<u32>() as u64, &num_nodes as *const u32 as *const _);
            encoder.set_buffer(4, Some(predecessors), 0);
            encoder.set_buffer(5, Some(indptr), 0);
            encoder.set_buffer(6, Some(indices), 0);
            encoder.set_buffer(7, Some(weights), 0);
            encoder.set_buffer(8, Some(distances), 0);
            encoder.set_buffer(9, Some(f0), 0);
            encoder.set_buffer(10, Some(qs0), 0);
            encoder.set_buffer(11, Some(f1), 0);
            encoder.set_buffer(12, Some(qs1), 0);
            encoder.set_buffer(13, Some(active), 0);
            encoder.set_buffer(14, Some(steal_index), 0);
            encoder.set_buffer(15, Some(grid_barrier_state), 0);
            encoder.set_bytes(16, std::mem::size_of::<f32>() as u64, &delta as *const f32 as *const _);
            encoder.set_bytes(17, std::mem::size_of::<u32>() as u64, &max_iters as *const u32 as *const _);
            encoder.set_bytes(18, std::mem::size_of::<u32>() as u64, &(num_threadgroups as u32) as *const u32 as *const _);
            
            encoder.dispatch_threads(persistent_grid, threadgroup_size);
            
            encoder.pop_debug_group();
            encoder.end_encoding();
            
            let val0 = unsafe { *(qs0.contents() as *const u32) };
            let val1 = unsafe { *(qs1.contents() as *const u32) };
            println!("[Metal] BEFORE DISPATCH: qs0 = {}, qs1 = {}", val0, val1);
            
            command_buffer.commit();
            command_buffer.wait_until_completed();
            
            let val0_after = unsafe { *(qs0.contents() as *const u32) };
            let val1_after = unsafe { *(qs1.contents() as *const u32) };
            println!("[Metal] AFTER DISPATCH: qs0 = {}, qs1 = {}", val0_after, val1_after);
            
            let gb_ptr = grid_barrier_state.contents() as *const u32;
            let final_iter = unsafe { *gb_ptr.add(1) };
            
            total_iters = final_iter;
            converged = (val0_after == 0) && (val1_after == 0);
            if is_capturing {
                let capture_manager = CaptureManager::shared();
                capture_manager.stop_capture();
            }
            
            return Ok((total_iters, converged));
        } else {
            println!("[Metal-Exec] Warning: Buffers not initialized.");
            return Ok((0, false));
        }
    }

    pub fn get_distances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        if let Some(buf) = &self.distances_buf {
            let ptr = buf.contents() as *mut f32;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.node_count) };
            let py_array = PyArray1::from_slice(py, slice);
            Ok(py_array.into_any())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err("Distances buffer not initialized"))
        }
    }

    pub fn get_predecessors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        if let Some(buf) = &self.predecessors_buf {
            let ptr = buf.contents() as *mut i32;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.node_count) };
            let py_array = PyArray1::from_slice(py, slice);
            Ok(py_array.into_any())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err("Predecessors buffer not initialized"))
        }
    }

    /// Reset predecessors to -1 before a new SSSP run
    pub fn reset_predecessors(&self) -> PyResult<()> {
        if let Some(buf) = &self.predecessors_buf {
            let ptr = buf.contents() as *mut i32;
            unsafe {
                for i in 0..self.node_count {
                    *ptr.add(i) = -1;
                }
            }
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err("Predecessors buffer not initialized"))
        }
    }

    /// Dispatch the ROI extractor kernel on the Metal GPU (legacy distance-filtering).
    ///
    /// Accepts ROI bounds and per-node coordinate arrays, dispatches the
    /// `roi_extractor_mixin` kernel to filter distances by spatial region,
    /// and returns (filtered_distances, matched_node_ids) as NumPy arrays.
    pub fn extract_roi<'py>(
        &self,
        py: Python<'py>,
        x_min: f32,
        y_min: f32,
        z_min: f32,
        x_max: f32,
        y_max: f32,
        z_max: f32,
        node_x: PyReadonlyArray1<'py, f32>,
        node_y: PyReadonlyArray1<'py, f32>,
        node_z: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(Bound<'py, pyo3::types::PyAny>, Bound<'py, pyo3::types::PyAny>)> {
        let distances_buf = self.distances_buf.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Distances buffer not initialized")
        })?;

        let node_x_slice = node_x.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("node_x not contiguous: {}", e)))?;
        let node_y_slice = node_y.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("node_y not contiguous: {}", e)))?;
        let node_z_slice = node_z.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("node_z not contiguous: {}", e)))?;

        let num_nodes = self.node_count;
        if node_x_slice.len() != num_nodes || node_y_slice.len() != num_nodes || node_z_slice.len() != num_nodes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Coordinate arrays must have length {} (node_count), got x={}, y={}, z={}",
                num_nodes, node_x_slice.len(), node_y_slice.len(), node_z_slice.len()
            )));
        }

        let options = MTLResourceOptions::StorageModeShared;

        // Output buffers: worst case all nodes match
        let roi_distances_buf = self.device.new_buffer(
            (num_nodes * mem::size_of::<f32>()) as u64, options,
        );
        let roi_node_ids_buf = self.device.new_buffer(
            (num_nodes * mem::size_of::<u32>()) as u64, options,
        );
        // Atomic counter for matched nodes (initialized to 0)
        let roi_count_buf = self.device.new_buffer(
            mem::size_of::<u32>() as u64, options,
        );
        unsafe { *(roi_count_buf.contents() as *mut u32) = 0; }

        // Coordinate buffers (zero-copy via UMA)
        let node_x_buf = self.device.new_buffer_with_bytes_no_copy(
            node_x_slice.as_ptr() as *const _, (num_nodes * mem::size_of::<f32>()) as u64, options, None,
        );
        let node_y_buf = self.device.new_buffer_with_bytes_no_copy(
            node_y_slice.as_ptr() as *const _, (num_nodes * mem::size_of::<f32>()) as u64, options, None,
        );
        let node_z_buf = self.device.new_buffer_with_bytes_no_copy(
            node_z_slice.as_ptr() as *const _, (num_nodes * mem::size_of::<f32>()) as u64, options, None,
        );

        // ROI bounds buffer: [x_min, y_min, z_min, x_max, y_max, z_max]
        let bounds: [f32; 6] = [x_min, y_min, z_min, x_max, y_max, z_max];
        let roi_bounds_buf = self.device.new_buffer_with_data(
            bounds.as_ptr() as *const _, (6 * mem::size_of::<f32>()) as u64, options,
        );

        // Dispatch the roi_extractor_mixin kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.push_debug_group("ROI Extraction");
        encoder.set_compute_pipeline_state(&self.roi_pipeline);
        encoder.set_buffer(0, Some(distances_buf), 0);
        encoder.set_buffer(1, Some(&roi_distances_buf), 0);
        encoder.set_buffer(2, Some(&roi_node_ids_buf), 0);
        encoder.set_buffer(3, Some(&roi_count_buf), 0);
        encoder.set_buffer(4, Some(&node_x_buf), 0);
        encoder.set_buffer(5, Some(&node_y_buf), 0);
        encoder.set_buffer(6, Some(&node_z_buf), 0);
        encoder.set_buffer(7, Some(&roi_bounds_buf), 0);

        let grid_size = MTLSize::new(num_nodes.max(1) as u64, 1, 1);
        let tg_size = MTLSize::new(
            512u64.min(self.roi_pipeline.max_total_threads_per_threadgroup()), 1, 1,
        );
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.pop_debug_group();
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back the count of matched nodes
        let match_count = unsafe { *(roi_count_buf.contents() as *const u32) } as usize;

        // Copy matched results into NumPy arrays
        let dist_ptr = roi_distances_buf.contents() as *const f32;
        let ids_ptr = roi_node_ids_buf.contents() as *const u32;

        let dist_slice = unsafe { std::slice::from_raw_parts(dist_ptr, match_count) };
        let ids_slice = unsafe { std::slice::from_raw_parts(ids_ptr, match_count) };

        // Convert u32 node IDs to i32 for Python compatibility
        let ids_i32: Vec<i32> = ids_slice.iter().map(|&id| id as i32).collect();

        let py_distances = PyArray1::from_slice(py, dist_slice);
        let py_node_ids = PyArray1::from_slice(py, &ids_i32);

        println!("[Metal-Exec] ROI extraction complete: {} nodes matched", match_count);

        Ok((py_distances.into_any(), py_node_ids.into_any()))
    }

    /// Dispatch the via cost computation kernel on the Metal GPU (legacy simple model).
    pub fn process_vias<'py>(
        &self,
        py: Python<'py>,
        via_capacity: PyReadonlyArray1<'py, f32>,
        via_usage: PyReadonlyArray1<'py, f32>,
        base_cost: f32,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let cap_slice = via_capacity.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("via_capacity not contiguous: {}", e)))?;
        let usage_slice = via_usage.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("via_usage not contiguous: {}", e)))?;

        let num_vias = cap_slice.len();
        if usage_slice.len() != num_vias {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "via_capacity ({}) and via_usage ({}) must have the same length",
                num_vias, usage_slice.len()
            )));
        }

        let options = MTLResourceOptions::StorageModeShared;

        // Output buffer for computed via costs
        let via_costs_buf = self.device.new_buffer(
            (num_vias * mem::size_of::<f32>()) as u64, options,
        );

        // Input buffers (zero-copy via UMA)
        let cap_buf = self.device.new_buffer_with_bytes_no_copy(
            cap_slice.as_ptr() as *const _,
            (num_vias * mem::size_of::<f32>()) as u64,
            options,
            None,
        );
        let usage_buf = self.device.new_buffer_with_bytes_no_copy(
            usage_slice.as_ptr() as *const _,
            (num_vias * mem::size_of::<f32>()) as u64,
            options,
            None,
        );

        // Dispatch the via_kernels kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.push_debug_group("Via Cost Computation");
        encoder.set_compute_pipeline_state(&self.via_pipeline);
        encoder.set_buffer(0, Some(&via_costs_buf), 0);
        encoder.set_buffer(1, Some(&cap_buf), 0);
        encoder.set_buffer(2, Some(&usage_buf), 0);
        encoder.set_bytes(3, mem::size_of::<f32>() as u64, &base_cost as *const f32 as *const _);

        let grid_size = MTLSize::new(num_vias.max(1) as u64, 1, 1);
        let tg_size = MTLSize::new(
            512u64.min(self.via_pipeline.max_total_threads_per_threadgroup()), 1, 1,
        );
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.pop_debug_group();
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back computed via costs into a NumPy array
        let costs_ptr = via_costs_buf.contents() as *const f32;
        let costs_slice = unsafe { std::slice::from_raw_parts(costs_ptr, num_vias) };
        let py_costs = PyArray1::from_slice(py, costs_slice);

        println!("[Metal-Exec] Via processing complete: {} vias computed", num_vias);

        Ok(py_costs.into_any())
    }

    // =========================================================================
    // CUDA-PARITY: Hard-Block Via Edges at Capacity
    //
    // Matches ViaKernelManager.hard_block_via_edges() from via_kernels.py.
    // Checks column capacity AND per-segment capacity, sets cost to INFINITY
    // for blocked edges.
    //
    // Args:
    //   edge_indices: [N] i32 — Edge indices into cost array
    //   xy_coords:    [N*2] i32 — Interleaved (x,y) per via edge
    //   z_lo:         [N] i32 — Lower z bound per via edge
    //   z_hi:         [N] i32 — Upper z bound per via edge
    //   via_col_use:  [Nx*Ny] i16 — Column usage
    //   via_col_cap:  [Nx*Ny] i16 — Column capacity
    //   via_seg_use:  [Nx*Ny*segZ] i8 — Segment usage
    //   via_seg_cap:  [Nx*Ny*segZ] i8 — Segment capacity
    //   total_cost:   [E] f32 — Cost array (modified in-place)
    //   ny:           i32 — Y grid dimension
    //   seg_z:        i32 — Number of segments
    //
    // Returns: number of edges blocked
    // =========================================================================
    pub fn hard_block_via_edges<'py>(
        &self,
        _py: Python<'py>,
        edge_indices: PyReadonlyArray1<'py, i32>,
        xy_coords: PyReadonlyArray1<'py, i32>,
        z_lo: PyReadonlyArray1<'py, i32>,
        z_hi: PyReadonlyArray1<'py, i32>,
        via_col_use: PyReadonlyArray1<'py, i16>,
        via_col_cap: PyReadonlyArray1<'py, i16>,
        via_seg_use: PyReadonlyArray1<'py, i8>,
        via_seg_cap: PyReadonlyArray1<'py, i8>,
        total_cost: PyReadonlyArray1<'py, f32>,
        ny: i32,
        seg_z: i32,
    ) -> PyResult<i32> {
        let edge_indices_s = edge_indices.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let xy_coords_s = xy_coords.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let z_lo_s = z_lo.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let z_hi_s = z_hi.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let col_use_s = via_col_use.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let col_cap_s = via_col_cap.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let seg_use_s = via_seg_use.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let seg_cap_s = via_seg_cap.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let cost_s = total_cost.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        let num_via_edges = edge_indices_s.len();
        if num_via_edges == 0 {
            return Ok(0);
        }

        let options = MTLResourceOptions::StorageModeShared;

        // Zero-copy input buffers
        let edge_buf = make_nocopy_buffer(&self.device, edge_indices_s.as_ptr() as *const u8, num_via_edges * mem::size_of::<i32>());
        let xy_buf = make_nocopy_buffer(&self.device, xy_coords_s.as_ptr() as *const u8, xy_coords_s.len() * mem::size_of::<i32>());
        let zlo_buf = make_nocopy_buffer(&self.device, z_lo_s.as_ptr() as *const u8, num_via_edges * mem::size_of::<i32>());
        let zhi_buf = make_nocopy_buffer(&self.device, z_hi_s.as_ptr() as *const u8, num_via_edges * mem::size_of::<i32>());
        let col_use_buf = make_nocopy_buffer(&self.device, col_use_s.as_ptr() as *const u8, col_use_s.len() * mem::size_of::<i16>());
        let col_cap_buf = make_nocopy_buffer(&self.device, col_cap_s.as_ptr() as *const u8, col_cap_s.len() * mem::size_of::<i16>());
        let seg_use_buf = make_nocopy_buffer(&self.device, seg_use_s.as_ptr() as *const u8, seg_use_s.len() * mem::size_of::<i8>());
        let seg_cap_buf = make_nocopy_buffer(&self.device, seg_cap_s.as_ptr() as *const u8, seg_cap_s.len() * mem::size_of::<i8>());
        // total_cost is read-write (modified in-place via atomic store)
        let cost_buf = make_nocopy_buffer(&self.device, cost_s.as_ptr() as *const u8, cost_s.len() * mem::size_of::<f32>());

        // Atomic blocked_count (initialized to 0)
        let blocked_count_buf = make_zeroed_buffer(&self.device, mem::size_of::<u32>());

        // Params buffer: [num_via_edges, Ny, segZ]
        let params: [i32; 3] = [num_via_edges as i32, ny, seg_z];
        let params_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * mem::size_of::<i32>())
        });

        // Dispatch kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.push_debug_group("Hard-Block Via Capacity");
        encoder.set_compute_pipeline_state(&self.hard_block_pipeline);
        encoder.set_buffer(0, Some(&edge_buf), 0);
        encoder.set_buffer(1, Some(&xy_buf), 0);
        encoder.set_buffer(2, Some(&zlo_buf), 0);
        encoder.set_buffer(3, Some(&zhi_buf), 0);
        encoder.set_buffer(4, Some(&col_use_buf), 0);
        encoder.set_buffer(5, Some(&col_cap_buf), 0);
        encoder.set_buffer(6, Some(&seg_use_buf), 0);
        encoder.set_buffer(7, Some(&seg_cap_buf), 0);
        encoder.set_buffer(8, Some(&cost_buf), 0);
        encoder.set_buffer(9, Some(&blocked_count_buf), 0);
        encoder.set_buffer(10, Some(&params_buf), 0);

        let grid_size = MTLSize::new(num_via_edges as u64, 1, 1);
        let tg_size = MTLSize::new(
            256u64.min(self.hard_block_pipeline.max_total_threads_per_threadgroup()), 1, 1,
        );
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.pop_debug_group();
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let blocked = unsafe { *(blocked_count_buf.contents() as *const u32) } as i32;
        println!("[Metal-Exec] Hard-block via capacity: {} edges blocked out of {}", blocked, num_via_edges);

        Ok(blocked)
    }

    // =========================================================================
    // CUDA-PARITY: Apply Via Pooling Penalties
    //
    // Matches ViaKernelManager.apply_via_penalties() from via_kernels.py.
    // Calculates column + segment penalties and atomically adds to edge costs.
    //
    // Returns: number of penalties applied
    // =========================================================================
    pub fn apply_via_pooling_penalties<'py>(
        &self,
        _py: Python<'py>,
        edge_indices: PyReadonlyArray1<'py, i32>,
        xy_coords: PyReadonlyArray1<'py, i32>,
        z_lo: PyReadonlyArray1<'py, i32>,
        z_hi: PyReadonlyArray1<'py, i32>,
        via_col_pres: PyReadonlyArray1<'py, f32>,
        via_seg_pres: PyReadonlyArray1<'py, f32>,
        total_cost: PyReadonlyArray1<'py, f32>,
        col_weight: f32,
        seg_weight: f32,
        ny: i32,
        seg_z: i32,
    ) -> PyResult<i32> {
        let edge_indices_s = edge_indices.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let xy_coords_s = xy_coords.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let z_lo_s = z_lo.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let z_hi_s = z_hi.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let col_pres_s = via_col_pres.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let seg_pres_s = via_seg_pres.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let cost_s = total_cost.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        let num_via_edges = edge_indices_s.len();
        if num_via_edges == 0 {
            return Ok(0);
        }

        let options = MTLResourceOptions::StorageModeShared;

        // Zero-copy input buffers
        let edge_buf = make_nocopy_buffer(&self.device, edge_indices_s.as_ptr() as *const u8, num_via_edges * mem::size_of::<i32>());
        let xy_buf = make_nocopy_buffer(&self.device, xy_coords_s.as_ptr() as *const u8, xy_coords_s.len() * mem::size_of::<i32>());
        let zlo_buf = make_nocopy_buffer(&self.device, z_lo_s.as_ptr() as *const u8, num_via_edges * mem::size_of::<i32>());
        let zhi_buf = make_nocopy_buffer(&self.device, z_hi_s.as_ptr() as *const u8, num_via_edges * mem::size_of::<i32>());
        let col_pres_buf = make_nocopy_buffer(&self.device, col_pres_s.as_ptr() as *const u8, col_pres_s.len() * mem::size_of::<f32>());
        let seg_pres_buf = make_nocopy_buffer(&self.device, seg_pres_s.as_ptr() as *const u8, seg_pres_s.len() * mem::size_of::<f32>());
        let cost_buf = make_nocopy_buffer(&self.device, cost_s.as_ptr() as *const u8, cost_s.len() * mem::size_of::<f32>());

        // Atomic penalty_count (initialized to 0)
        let penalty_count_buf = make_zeroed_buffer(&self.device, mem::size_of::<u32>());

        // Params buffer: [num_via_edges, Ny, segZ]
        let params: [i32; 3] = [num_via_edges as i32, ny, seg_z];
        let params_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * mem::size_of::<i32>())
        });

        // Weights buffer: [col_weight, seg_weight]
        let weights: [f32; 2] = [col_weight, seg_weight];
        let weights_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len() * mem::size_of::<f32>())
        });

        // Dispatch kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.push_debug_group("Apply Via Pooling Penalties");
        encoder.set_compute_pipeline_state(&self.via_penalty_pipeline);
        encoder.set_buffer(0, Some(&edge_buf), 0);
        encoder.set_buffer(1, Some(&xy_buf), 0);
        encoder.set_buffer(2, Some(&zlo_buf), 0);
        encoder.set_buffer(3, Some(&zhi_buf), 0);
        encoder.set_buffer(4, Some(&col_pres_buf), 0);
        encoder.set_buffer(5, Some(&seg_pres_buf), 0);
        encoder.set_buffer(6, Some(&cost_buf), 0);
        encoder.set_buffer(7, Some(&penalty_count_buf), 0);
        encoder.set_buffer(8, Some(&params_buf), 0);
        encoder.set_buffer(9, Some(&weights_buf), 0);

        let grid_size = MTLSize::new(num_via_edges as u64, 1, 1);
        let tg_size = MTLSize::new(
            256u64.min(self.via_penalty_pipeline.max_total_threads_per_threadgroup()), 1, 1,
        );
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.pop_debug_group();
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let penalties = unsafe { *(penalty_count_buf.contents() as *const u32) } as i32;
        println!("[Metal-Exec] Via pooling penalties: {} applied across {} edges", penalties, num_via_edges);

        Ok(penalties)
    }

    // =========================================================================
    // CUDA-PARITY: Detect Barrel Conflicts
    //
    // Matches ViaKernelManager.detect_barrel_conflicts_gpu() from via_kernels.py.
    // Checks each edge to see if endpoints are owned by different nets.
    //
    // Returns: number of conflicts detected
    // =========================================================================
    pub fn detect_barrel_conflicts<'py>(
        &self,
        _py: Python<'py>,
        edge_indices: PyReadonlyArray1<'py, i32>,
        edge_net_ids: PyReadonlyArray1<'py, i32>,
        edge_src_map: PyReadonlyArray1<'py, i32>,
        graph_indices: PyReadonlyArray1<'py, i32>,
        node_owner: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<i32> {
        let edge_indices_s = edge_indices.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let net_ids_s = edge_net_ids.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let src_map_s = edge_src_map.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let graph_indices_s = graph_indices.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let node_owner_s = node_owner.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        let num_edges = edge_indices_s.len();
        if num_edges == 0 {
            return Ok(0);
        }

        // Zero-copy input buffers
        let edge_buf = make_nocopy_buffer(&self.device, edge_indices_s.as_ptr() as *const u8, num_edges * mem::size_of::<i32>());
        let net_buf = make_nocopy_buffer(&self.device, net_ids_s.as_ptr() as *const u8, num_edges * mem::size_of::<i32>());
        let src_buf = make_nocopy_buffer(&self.device, src_map_s.as_ptr() as *const u8, src_map_s.len() * mem::size_of::<i32>());
        let gidx_buf = make_nocopy_buffer(&self.device, graph_indices_s.as_ptr() as *const u8, graph_indices_s.len() * mem::size_of::<i32>());
        let owner_buf = make_nocopy_buffer(&self.device, node_owner_s.as_ptr() as *const u8, node_owner_s.len() * mem::size_of::<i32>());

        // Atomic conflict_count (initialized to 0)
        let conflict_count_buf = make_zeroed_buffer(&self.device, mem::size_of::<u32>());

        // Dummy flags buffer (we only need count, not per-edge flags)
        let flags_buf = make_zeroed_buffer(&self.device, mem::size_of::<i32>());

        // Params buffer: [num_edges_to_check, has_flags=0]
        let params: [i32; 2] = [num_edges as i32, 0];
        let params_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * mem::size_of::<i32>())
        });

        // Dispatch kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.push_debug_group("Detect Barrel Conflicts");
        encoder.set_compute_pipeline_state(&self.barrel_conflict_pipeline);
        encoder.set_buffer(0, Some(&edge_buf), 0);
        encoder.set_buffer(1, Some(&net_buf), 0);
        encoder.set_buffer(2, Some(&src_buf), 0);
        encoder.set_buffer(3, Some(&gidx_buf), 0);
        encoder.set_buffer(4, Some(&owner_buf), 0);
        encoder.set_buffer(5, Some(&conflict_count_buf), 0);
        encoder.set_buffer(6, Some(&flags_buf), 0);
        encoder.set_buffer(7, Some(&params_buf), 0);

        let grid_size = MTLSize::new(num_edges as u64, 1, 1);
        let tg_size = MTLSize::new(
            256u64.min(self.barrel_conflict_pipeline.max_total_threads_per_threadgroup()), 1, 1,
        );
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.pop_debug_group();
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let conflicts = unsafe { *(conflict_count_buf.contents() as *const u32) } as i32;
        println!("[Metal-Exec] Barrel conflicts: {} detected in {} edges", conflicts, num_edges);

        Ok(conflicts)
    }

    // =========================================================================
    // CUDA-PARITY: ROI Subgraph Extraction (Spatial-Index Based)
    //
    // Two-phase extraction matching roi_extractor_mixin.py:
    //   Phase 1: roi_mark_nodes — marks nodes in ROI using spatial grid index
    //   Phase 2: roi_extract_subgraph — extracts CSR subgraph for marked nodes
    //
    // Returns: (roi_node_ids, roi_indptr, roi_indices, roi_weights) as NumPy arrays
    // =========================================================================
    pub fn extract_roi_subgraph<'py>(
        &self,
        py: Python<'py>,
        spatial_indptr: PyReadonlyArray1<'py, i32>,
        spatial_node_ids: PyReadonlyArray1<'py, i32>,
        csr_indptr: PyReadonlyArray1<'py, i32>,
        csr_indices: PyReadonlyArray1<'py, i32>,
        csr_weights: PyReadonlyArray1<'py, f32>,
        grid_x0: i32, grid_y0: i32,
        grid_x1: i32, grid_y1: i32,
        grid_width: i32, grid_height: i32,
        max_layers: i32, max_cell_id: i32,
        total_nodes: i32,
    ) -> PyResult<(
        Bound<'py, pyo3::types::PyAny>,  // roi_node_ids
        Bound<'py, pyo3::types::PyAny>,  // roi_indptr
        Bound<'py, pyo3::types::PyAny>,  // roi_indices
        Bound<'py, pyo3::types::PyAny>,  // roi_weights
    )> {
        let sp_indptr_s = spatial_indptr.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let sp_nodes_s = spatial_node_ids.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let csr_indptr_s = csr_indptr.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let csr_indices_s = csr_indices.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let csr_weights_s = csr_weights.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        let total_nodes_usize = total_nodes as usize;
        let options = MTLResourceOptions::StorageModeShared;

        // =====================================================================
        // Phase 1: Mark nodes in ROI using spatial index
        // =====================================================================
        let sp_indptr_buf = make_nocopy_buffer(&self.device, sp_indptr_s.as_ptr() as *const u8, sp_indptr_s.len() * mem::size_of::<i32>());
        let sp_nodes_buf = make_nocopy_buffer(&self.device, sp_nodes_s.as_ptr() as *const u8, sp_nodes_s.len() * mem::size_of::<i32>());

        // ROI node mask: 1 element per node (uint32 for atomic ops), initialized to 0
        let roi_mask_buf = make_zeroed_buffer(&self.device, total_nodes_usize * mem::size_of::<u32>());

        // Params: [grid_x0, grid_y0, grid_x1, grid_y1, grid_width, grid_height, max_layers, max_cell_id]
        let roi_params: [i32; 8] = [grid_x0, grid_y0, grid_x1, grid_y1, grid_width, grid_height, max_layers, max_cell_id];
        let roi_params_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(roi_params.as_ptr() as *const u8, roi_params.len() * mem::size_of::<i32>())
        });
        let total_nodes_arr: [i32; 1] = [total_nodes];
        let total_nodes_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(total_nodes_arr.as_ptr() as *const u8, mem::size_of::<i32>())
        });

        // Calculate total cells to dispatch
        let roi_width = (grid_x1 - grid_x0) as u64;
        let roi_height = (grid_y1 - grid_y0) as u64;
        let cells_per_layer = roi_width * roi_height;
        let total_cells = (max_layers as u64) * cells_per_layer;

        if total_cells > 0 {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.push_debug_group("ROI Mark Nodes");
            encoder.set_compute_pipeline_state(&self.roi_mark_nodes_pipeline);
            encoder.set_buffer(0, Some(&sp_indptr_buf), 0);
            encoder.set_buffer(1, Some(&sp_nodes_buf), 0);
            encoder.set_buffer(2, Some(&roi_mask_buf), 0);
            encoder.set_buffer(3, Some(&roi_params_buf), 0);
            encoder.set_buffer(4, Some(&total_nodes_buf), 0);

            let grid_size = MTLSize::new(total_cells.max(1), 1, 1);
            let tg_size = MTLSize::new(
                256u64.min(self.roi_mark_nodes_pipeline.max_total_threads_per_threadgroup()), 1, 1,
            );
            encoder.dispatch_threads(grid_size, tg_size);
            encoder.pop_debug_group();
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // =====================================================================
        // Read mask back to CPU to get ROI node list and build global-to-local
        // =====================================================================
        let mask_ptr = roi_mask_buf.contents() as *const u32;
        let mask_slice = unsafe { std::slice::from_raw_parts(mask_ptr, total_nodes_usize) };

        // Collect ROI node IDs
        let mut roi_node_ids: Vec<i32> = Vec::new();
        for i in 0..total_nodes_usize {
            if mask_slice[i] != 0 {
                roi_node_ids.push(i as i32);
            }
        }

        let num_roi_nodes = roi_node_ids.len();
        if num_roi_nodes == 0 {
            // Return empty arrays
            let empty_i32 = PyArray1::<i32>::from_slice(py, &[]);
            let empty_f32 = PyArray1::<f32>::from_slice(py, &[]);
            let indptr_out = PyArray1::<i32>::from_slice(py, &[0i32]);
            return Ok((
                empty_i32.clone().into_any(),
                indptr_out.into_any(),
                empty_i32.into_any(),
                empty_f32.into_any(),
            ));
        }

        // Build global-to-local mapping (dense array, -1 = not in ROI)
        let max_global = total_nodes_usize;
        let mut global_to_local: Vec<i32> = vec![-1i32; max_global];
        for (local_idx, &global_id) in roi_node_ids.iter().enumerate() {
            global_to_local[global_id as usize] = local_idx as i32;
        }

        // =====================================================================
        // Phase 2a: Count edges per ROI node (kernel, count_only=1)
        // =====================================================================
        let roi_nodes_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(roi_node_ids.as_ptr() as *const u8, num_roi_nodes * mem::size_of::<i32>())
        });
        let g2l_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(global_to_local.as_ptr() as *const u8, max_global * mem::size_of::<i32>())
        });
        let csr_indptr_buf = make_nocopy_buffer(&self.device, csr_indptr_s.as_ptr() as *const u8, csr_indptr_s.len() * mem::size_of::<i32>());
        let csr_indices_buf = make_nocopy_buffer(&self.device, csr_indices_s.as_ptr() as *const u8, csr_indices_s.len() * mem::size_of::<i32>());
        let csr_weights_buf = make_nocopy_buffer(&self.device, csr_weights_s.as_ptr() as *const u8, csr_weights_s.len() * mem::size_of::<f32>());

        // Edge counts buffer (one per ROI node)
        let edge_counts_buf = make_zeroed_buffer(&self.device, num_roi_nodes * mem::size_of::<i32>());

        // Dummy output buffers for Phase 1 (count only — not written to)
        let dummy_indices_buf = self.device.new_buffer(mem::size_of::<i32>() as u64, options);
        let dummy_weights_buf = self.device.new_buffer(mem::size_of::<f32>() as u64, options);

        // Phase 1 params: [num_roi_nodes, max_global_id, count_only=1]
        let phase1_params: [i32; 3] = [num_roi_nodes as i32, max_global as i32, 1];
        let phase1_params_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(phase1_params.as_ptr() as *const u8, phase1_params.len() * mem::size_of::<i32>())
        });

        {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.push_debug_group("ROI Extract Subgraph - Phase 1 (Count)");
            encoder.set_compute_pipeline_state(&self.roi_extract_subgraph_pipeline);
            encoder.set_buffer(0, Some(&roi_nodes_buf), 0);
            encoder.set_buffer(1, Some(&g2l_buf), 0);
            encoder.set_buffer(2, Some(&csr_indptr_buf), 0);
            encoder.set_buffer(3, Some(&csr_indices_buf), 0);
            encoder.set_buffer(4, Some(&csr_weights_buf), 0);
            encoder.set_buffer(5, Some(&edge_counts_buf), 0);
            encoder.set_buffer(6, Some(&dummy_indices_buf), 0);
            encoder.set_buffer(7, Some(&dummy_weights_buf), 0);
            encoder.set_buffer(8, Some(&phase1_params_buf), 0);

            let grid_size = MTLSize::new(num_roi_nodes.max(1) as u64, 1, 1);
            let tg_size = MTLSize::new(
                256u64.min(self.roi_extract_subgraph_pipeline.max_total_threads_per_threadgroup()), 1, 1,
            );
            encoder.dispatch_threads(grid_size, tg_size);
            encoder.pop_debug_group();
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // =====================================================================
        // CPU prefix sum on edge counts to build indptr + get total edges
        // =====================================================================
        let counts_ptr = edge_counts_buf.contents() as *mut i32;
        let counts_slice = unsafe { std::slice::from_raw_parts_mut(counts_ptr, num_roi_nodes) };

        let mut roi_indptr: Vec<i32> = vec![0i32; num_roi_nodes + 1];
        for i in 0..num_roi_nodes {
            roi_indptr[i + 1] = roi_indptr[i] + counts_slice[i];
        }
        let total_edges = roi_indptr[num_roi_nodes] as usize;

        if total_edges == 0 {
            // Return node IDs with empty edge data
            let py_nodes = PyArray1::from_slice(py, &roi_node_ids);
            let py_indptr = PyArray1::from_slice(py, &roi_indptr);
            let empty_i32 = PyArray1::<i32>::from_slice(py, &[]);
            let empty_f32 = PyArray1::<f32>::from_slice(py, &[]);
            return Ok((
                py_nodes.into_any(),
                py_indptr.into_any(),
                empty_i32.into_any(),
                empty_f32.into_any(),
            ));
        }

        // =====================================================================
        // Phase 2b: Extract edges using prefix-sum offsets (count_only=0)
        // =====================================================================
        // Write prefix-sum offsets back into edge_counts_buf (reuse as offset buffer)
        for i in 0..num_roi_nodes {
            unsafe { *(counts_ptr.add(i)) = roi_indptr[i]; }
        }

        // Allocate output edge buffers
        let out_indices_buf = self.device.new_buffer((total_edges * mem::size_of::<i32>()) as u64, options);
        let out_weights_buf = self.device.new_buffer((total_edges * mem::size_of::<f32>()) as u64, options);

        // Phase 2 params: [num_roi_nodes, max_global_id, count_only=0]
        let phase2_params: [i32; 3] = [num_roi_nodes as i32, max_global as i32, 0];
        let phase2_params_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(phase2_params.as_ptr() as *const u8, phase2_params.len() * mem::size_of::<i32>())
        });

        {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.push_debug_group("ROI Extract Subgraph - Phase 2 (Extract)");
            encoder.set_compute_pipeline_state(&self.roi_extract_subgraph_pipeline);
            encoder.set_buffer(0, Some(&roi_nodes_buf), 0);
            encoder.set_buffer(1, Some(&g2l_buf), 0);
            encoder.set_buffer(2, Some(&csr_indptr_buf), 0);
            encoder.set_buffer(3, Some(&csr_indices_buf), 0);
            encoder.set_buffer(4, Some(&csr_weights_buf), 0);
            encoder.set_buffer(5, Some(&edge_counts_buf), 0);  // Now contains prefix-sum offsets
            encoder.set_buffer(6, Some(&out_indices_buf), 0);
            encoder.set_buffer(7, Some(&out_weights_buf), 0);
            encoder.set_buffer(8, Some(&phase2_params_buf), 0);

            let grid_size = MTLSize::new(num_roi_nodes.max(1) as u64, 1, 1);
            let tg_size = MTLSize::new(
                256u64.min(self.roi_extract_subgraph_pipeline.max_total_threads_per_threadgroup()), 1, 1,
            );
            encoder.dispatch_threads(grid_size, tg_size);
            encoder.pop_debug_group();
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // =====================================================================
        // Read results back to NumPy arrays
        // =====================================================================
        let out_idx_ptr = out_indices_buf.contents() as *const i32;
        let out_wt_ptr = out_weights_buf.contents() as *const f32;

        let out_idx_slice = unsafe { std::slice::from_raw_parts(out_idx_ptr, total_edges) };
        let out_wt_slice = unsafe { std::slice::from_raw_parts(out_wt_ptr, total_edges) };

        let py_nodes = PyArray1::from_slice(py, &roi_node_ids);
        let py_indptr = PyArray1::from_slice(py, &roi_indptr);
        let py_indices = PyArray1::from_slice(py, out_idx_slice);
        let py_weights = PyArray1::from_slice(py, out_wt_slice);

        println!("[Metal-Exec] ROI subgraph extraction: {} nodes, {} edges", num_roi_nodes, total_edges);

        Ok((
            py_nodes.into_any(),
            py_indptr.into_any(),
            py_indices.into_any(),
            py_weights.into_any(),
        ))
    }

    // =========================================================================
    // CUDA-PARITY: Owner-Aware Via Keepout Blocking
    //
    // Blocks outgoing planar edges from via keepout nodes owned by other nets.
    //
    // Returns: number of edges blocked
    // =========================================================================
    pub fn block_via_keepouts<'py>(
        &self,
        _py: Python<'py>,
        via_keepout_nodes: PyReadonlyArray1<'py, i32>,
        via_keepout_owners: PyReadonlyArray1<'py, i32>,
        indptr: PyReadonlyArray1<'py, i32>,
        indices: PyReadonlyArray1<'py, i32>,
        node_coords_z: PyReadonlyArray1<'py, i32>,
        costs: PyReadonlyArray1<'py, f32>,
        current_net_id: i32,
        block_cost: f32,
    ) -> PyResult<i32> {
        let nodes_s = via_keepout_nodes.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let owners_s = via_keepout_owners.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let indptr_s = indptr.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let indices_s = indices.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let z_s = node_coords_z.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let costs_s = costs.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        let num_keepouts = nodes_s.len();
        if num_keepouts == 0 {
            return Ok(0);
        }

        let num_nodes = indptr_s.len().saturating_sub(1);

        // Zero-copy input buffers
        let nodes_buf = make_nocopy_buffer(&self.device, nodes_s.as_ptr() as *const u8, num_keepouts * mem::size_of::<i32>());
        let owners_buf = make_nocopy_buffer(&self.device, owners_s.as_ptr() as *const u8, num_keepouts * mem::size_of::<i32>());
        let indptr_buf = make_nocopy_buffer(&self.device, indptr_s.as_ptr() as *const u8, indptr_s.len() * mem::size_of::<i32>());
        let indices_buf = make_nocopy_buffer(&self.device, indices_s.as_ptr() as *const u8, indices_s.len() * mem::size_of::<i32>());
        let z_buf = make_nocopy_buffer(&self.device, z_s.as_ptr() as *const u8, z_s.len() * mem::size_of::<i32>());
        let costs_buf = make_nocopy_buffer(&self.device, costs_s.as_ptr() as *const u8, costs_s.len() * mem::size_of::<f32>());

        // Atomic blocked_count
        let blocked_count_buf = make_zeroed_buffer(&self.device, mem::size_of::<u32>());

        // Params: [num_keepouts, current_net_id, num_nodes]
        let params: [i32; 3] = [num_keepouts as i32, current_net_id, num_nodes as i32];
        let params_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * mem::size_of::<i32>())
        });

        // Block cost
        let bc_arr: [f32; 1] = [block_cost];
        let bc_buf = make_buffer_with_data(&self.device, unsafe {
            std::slice::from_raw_parts(bc_arr.as_ptr() as *const u8, mem::size_of::<f32>())
        });

        // Dispatch kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.push_debug_group("Block Via Keepouts Owner-Aware");
        encoder.set_compute_pipeline_state(&self.keepout_blocking_pipeline);
        encoder.set_buffer(0, Some(&nodes_buf), 0);
        encoder.set_buffer(1, Some(&owners_buf), 0);
        encoder.set_buffer(2, Some(&indptr_buf), 0);
        encoder.set_buffer(3, Some(&indices_buf), 0);
        encoder.set_buffer(4, Some(&z_buf), 0);
        encoder.set_buffer(5, Some(&costs_buf), 0);
        encoder.set_buffer(6, Some(&blocked_count_buf), 0);
        encoder.set_buffer(7, Some(&params_buf), 0);
        encoder.set_buffer(8, Some(&bc_buf), 0);

        let grid_size = MTLSize::new(num_keepouts as u64, 1, 1);
        let tg_size = MTLSize::new(
            256u64.min(self.keepout_blocking_pipeline.max_total_threads_per_threadgroup()), 1, 1,
        );
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.pop_debug_group();
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let blocked = unsafe { *(blocked_count_buf.contents() as *const u32) } as i32;
        println!("[Metal-Exec] Via keepout blocking: {} edges blocked for {} keepouts", blocked, num_keepouts);

        Ok(blocked)
    }
}

#[pyfunction]
fn amx_sgemm_py(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a_array: numpy::PyReadonlyArray1<f32>,
    b_array: numpy::PyReadonlyArray1<f32>,
    beta: f32,
    mut c_array: numpy::PyReadwriteArray1<f32>,
) -> PyResult<()> {
    let a_slice = a_array.as_slice().unwrap();
    let b_slice = b_array.as_slice().unwrap();
    let c_slice = c_array.as_slice_mut().unwrap();
    
    accelerate_ops::amx_sgemm(m, n, k, alpha, a_slice, b_slice, beta, c_slice);
    Ok(())
}

#[pymodule]
fn orthoroute_mac(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MetalDijkstra>()?;
    m.add_function(wrap_pyfunction!(amx_sgemm_py, m)?)?;
    Ok(())
}
