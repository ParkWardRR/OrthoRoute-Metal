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
    wavefront_pipeline: ComputePipelineState,
    wavefront_multi_pipeline: ComputePipelineState,
    roi_pipeline: ComputePipelineState,
    via_pipeline: ComputePipelineState,
    spfa_pipeline: ComputePipelineState,
    clear_counters_pipeline: ComputePipelineState,
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

        let wavefront_func = library.get_function("wavefront_expand_all", None).unwrap();
        let wavefront_multi_func = library.get_function("wavefront_expand_multi", None).unwrap();
        let roi_func = library.get_function("roi_extractor_mixin", None).unwrap();
        let via_func = library.get_function("via_kernels", None).unwrap();
        let spfa_func = library.get_function("spfa_setup_kernel", None).unwrap();
        let clear_func = library.get_function("clear_counters", None).unwrap();

        let wavefront_pipeline = device.new_compute_pipeline_state_with_function(&wavefront_func).unwrap();
        let wavefront_multi_pipeline = device.new_compute_pipeline_state_with_function(&wavefront_multi_func).unwrap();
        let roi_pipeline = device.new_compute_pipeline_state_with_function(&roi_func).unwrap();
        let via_pipeline = device.new_compute_pipeline_state_with_function(&via_func).unwrap();
        let spfa_pipeline = device.new_compute_pipeline_state_with_function(&spfa_func).unwrap();
        let clear_counters_pipeline = device.new_compute_pipeline_state_with_function(&clear_func).unwrap();
        
        println!("[Metal-Init] Initialized MetalDijkstra on device: {:?}", device.name());
        
        Ok(MetalDijkstra {
            device,
            command_queue,
            wavefront_pipeline,
            wavefront_multi_pipeline,
            roi_pipeline,
            via_pipeline,
            spfa_pipeline,
            clear_counters_pipeline,
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

    pub fn extract_roi(&self) -> PyResult<()> {
        println!("[Metal-Exec] Dispatching ROI extraction...");
        Ok(())
    }

    pub fn process_vias(&self) -> PyResult<()> {
        println!("[Metal-Exec] Dispatching Via processing...");
        Ok(())
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
