"""Metal GPU provider implementation for Apple Silicon."""
import logging
import platform
import psutil
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ...application.interfaces.gpu_provider import GPUProvider

logger = logging.getLogger(__name__)

# Attempt to import the PyO3 Metal module (built via maturin)
try:
    import orthoroute_mac
    _METAL_AVAILABLE = True
except ImportError:
    orthoroute_mac = None
    _METAL_AVAILABLE = False


class MetalProvider(GPUProvider):
    """Apple Metal GPU provider using orthoroute_mac (PyO3/Rust).

    Wraps the MetalDijkstra class from the orthoroute_mac native module
    to provide the same interface as CUDAProvider and CPUFallbackProvider.

    On Apple Silicon Macs with the orthoroute_mac module compiled, this
    provider enables GPU-accelerated shortest-path computation via Metal
    compute shaders. If the module is unavailable, is_available() returns
    False and the system falls back to CPUFallbackProvider.
    """

    def __init__(self):
        """Initialize Metal provider."""
        self._metal_dijkstra = None
        self._device_info = {}
        self._initialized = False
        self._memory_limit = None
        self._allocated_arrays = []

    def is_available(self) -> bool:
        """Check if Metal GPU acceleration is available.

        Requires:
        - macOS on Apple Silicon (arm64)
        - orthoroute_mac native module compiled and importable
        - A Metal-capable GPU device
        """
        if not _METAL_AVAILABLE:
            logger.debug("orthoroute_mac module not available — Metal provider disabled")
            return False

        if platform.system() != 'Darwin':
            logger.debug("Not running on macOS — Metal provider disabled")
            return False

        try:
            # Attempt to create a MetalDijkstra instance to verify Metal device access
            test_dijkstra = orthoroute_mac.MetalDijkstra()
            del test_dijkstra
            logger.info("Metal GPU detected and working")
            return True
        except Exception as e:
            logger.warning(f"Metal device error: {e} — Metal provider disabled")
            return False

    def initialize(self) -> bool:
        """Initialize Metal resources and the MetalDijkstra engine."""
        if self._initialized:
            return True

        try:
            if not _METAL_AVAILABLE:
                logger.error("Cannot initialize Metal: orthoroute_mac module not available")
                return False

            # Create the MetalDijkstra instance (compiles MSL kernels on init)
            self._metal_dijkstra = orthoroute_mac.MetalDijkstra()

            # On Apple Silicon, GPU and CPU share unified memory (UMA).
            # Use 50% of available system memory as a soft limit, matching
            # the CPUFallbackProvider approach.
            available_memory = psutil.virtual_memory().available
            total_memory = psutil.virtual_memory().total
            self._memory_limit = int(available_memory * 0.5)

            # Gather device info — on Apple Silicon the Metal device name
            # (e.g. "Apple M1 Max") is printed by lib.rs during construction.
            arch = platform.machine()  # 'arm64' on Apple Silicon
            self._device_info = {
                'name': f'Metal GPU ({arch})',
                'compute_capability': 'Metal 3' if arch == 'arm64' else 'Metal',
                'total_memory': total_memory,
                'free_memory': available_memory,
                'memory_limit': self._memory_limit,
                'device_id': 'metal-0',
                'architecture': arch,
                'unified_memory': True,
            }

            self._initialized = True
            logger.info(
                f"Metal provider initialized: {self._device_info['name']}"
            )
            logger.info(
                f"Unified Memory: {total_memory / 1024**3:.1f}GB total, "
                f"{available_memory / 1024**3:.1f}GB free"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Metal provider: {e}")
            self._metal_dijkstra = None
            return False

    def cleanup(self) -> None:
        """Cleanup Metal resources."""
        self._metal_dijkstra = None
        self._allocated_arrays.clear()
        self._initialized = False
        logger.debug("Metal provider cleaned up")

    def get_device_info(self) -> Dict[str, Any]:
        """Get Metal device information."""
        return self._device_info.copy()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get Metal/UMA memory usage information.

        Since Apple Silicon uses Unified Memory Architecture, GPU memory
        stats mirror system memory stats.
        """
        try:
            vm = psutil.virtual_memory()

            # Estimate memory used by tracked arrays
            array_memory = sum(
                arr.nbytes for arr in self._allocated_arrays
                if hasattr(arr, 'nbytes')
            )

            return {
                'total_memory': vm.total,
                'free_memory': vm.available,
                'used_memory': vm.used,
                'memory_pool_used': array_memory,
                'memory_pool_total': array_memory,
            }

        except Exception as e:
            logger.error(f"Error getting Metal memory info: {e}")
            return {
                'total_memory': 0,
                'free_memory': 0,
                'used_memory': 0,
                'memory_pool_used': 0,
                'memory_pool_total': 0,
            }

    def create_array(self, shape: Tuple[int, ...], dtype=None, fill_value=None) -> np.ndarray:
        """Create array in unified memory.

        On Apple Silicon, NumPy arrays in system memory are directly
        accessible to Metal via UMA — no explicit GPU upload is needed.
        """
        if not self._initialized:
            raise RuntimeError("Metal provider not initialized")

        if dtype is None:
            dtype = np.float32

        try:
            # Check memory limit
            estimated_size = np.prod(shape) * np.dtype(dtype).itemsize
            if self._memory_limit and estimated_size > self._memory_limit:
                raise MemoryError(
                    f"Array size {estimated_size} exceeds memory limit {self._memory_limit}"
                )

            # Create array — directly usable by Metal via UMA
            if fill_value is None:
                array = np.empty(shape, dtype=dtype)
            elif fill_value == 0:
                array = np.zeros(shape, dtype=dtype)
            elif fill_value == 1:
                array = np.ones(shape, dtype=dtype)
            else:
                array = np.full(shape, fill_value, dtype=dtype)

            # Track allocated arrays
            self._allocated_arrays.append(array)

            # Clean up references to deleted arrays
            self._allocated_arrays = [
                arr for arr in self._allocated_arrays
                if hasattr(arr, 'nbytes')
            ]

            return array

        except Exception as e:
            logger.error(f"Error creating Metal array: {e}")
            raise

    def copy_array(self, array: np.ndarray) -> np.ndarray:
        """Create copy of array in unified memory."""
        if not self._initialized:
            raise RuntimeError("Metal provider not initialized")

        try:
            copied = np.copy(array)
            self._allocated_arrays.append(copied)
            return copied

        except Exception as e:
            logger.error(f"Error copying Metal array: {e}")
            raise

    def to_cpu(self, array: np.ndarray) -> np.ndarray:
        """Convert array to CPU.

        No-op on Apple Silicon — UMA means arrays are already shared.
        """
        return array

    def to_gpu(self, array: np.ndarray) -> np.ndarray:
        """Convert CPU array for GPU use.

        No-op on Apple Silicon — UMA means arrays are already shared.
        """
        return array

    def synchronize(self) -> None:
        """Synchronize Metal operations.

        Metal command buffer wait_until_completed() is called internally
        by the Rust backend after each dispatch. This is a no-op at the
        Python provider level.
        """
        pass

    # -------------------------------------------------------------------------
    # Metal-specific routing operations (delegated to MetalDijkstra)
    # -------------------------------------------------------------------------

    def set_graph(self, indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> str:
        """Set the CSR graph on the Metal device.

        Maps CSR arrays to Metal UMA buffers via zero-copy where possible.

        Args:
            indptr: CSR row pointer array (int32).
            indices: CSR column indices array (int32).
            weights: CSR edge weights array (float32).

        Returns:
            Status string from the Metal backend.
        """
        if not self._initialized or self._metal_dijkstra is None:
            raise RuntimeError("Metal provider not initialized")

        # Ensure correct dtypes for the Rust/Metal backend
        indptr = np.ascontiguousarray(indptr, dtype=np.int32)
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)

        return self._metal_dijkstra.set_graph_csr(indptr, indices, weights)

    def set_distances(self, distances: np.ndarray) -> None:
        """Set initial distance values on the Metal device.

        Args:
            distances: Distance array (float32). Source node(s) should
                       have distance 0.0, all others should be inf.
        """
        if not self._initialized or self._metal_dijkstra is None:
            raise RuntimeError("Metal provider not initialized")

        distances = np.ascontiguousarray(distances, dtype=np.float32)
        self._metal_dijkstra.set_distances_csr(distances)

    def setup_spfa(self) -> None:
        """Initialize SPFA frontier based on current distances."""
        if not self._initialized or self._metal_dijkstra is None:
            raise RuntimeError("Metal provider not initialized")

        self._metal_dijkstra.setup_spfa()

    def shortest_path(
        self,
        max_iters: int = 2000,
        batch_size: int = 0,
        threadgroup_size: int = 512,
        delta: float = 1.0,
    ) -> Tuple[int, bool]:
        """Run shortest-path (Δ-stepping SPFA) on the Metal GPU.

        Args:
            max_iters: Maximum number of relaxation iterations.
            batch_size: Batch size hint (unused in persistent-thread mode).
            threadgroup_size: Threads per threadgroup (default 512).
            delta: Δ-stepping bucket width.

        Returns:
            Tuple of (iterations_run, converged).
        """
        if not self._initialized or self._metal_dijkstra is None:
            raise RuntimeError("Metal provider not initialized")

        return self._metal_dijkstra.execute_until_convergence(
            max_iters, batch_size, threadgroup_size, delta
        )

    def get_distances(self) -> np.ndarray:
        """Retrieve computed distances from the Metal device.

        Returns:
            NumPy float32 array of shortest-path distances.
        """
        if not self._initialized or self._metal_dijkstra is None:
            raise RuntimeError("Metal provider not initialized")

        return np.asarray(self._metal_dijkstra.get_distances())

    def get_predecessors(self) -> np.ndarray:
        """Retrieve predecessor array from the Metal device.

        Returns:
            NumPy int32 array of predecessor node IDs (-1 if unreachable).
        """
        if not self._initialized or self._metal_dijkstra is None:
            raise RuntimeError("Metal provider not initialized")

        return np.asarray(self._metal_dijkstra.get_predecessors())

    def reset_predecessors(self) -> None:
        """Reset predecessor buffer to -1 for a new SSSP run."""
        if not self._initialized or self._metal_dijkstra is None:
            raise RuntimeError("Metal provider not initialized")

        self._metal_dijkstra.reset_predecessors()

    def extract_roi(
        self,
        roi_bounds: Tuple[float, float, float, float, float, float],
        node_x: np.ndarray,
        node_y: np.ndarray,
        node_z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract distances for nodes within an ROI bounding box.

        Dispatches the roi_extractor_mixin Metal kernel to filter distances
        by spatial region of interest.

        Args:
            roi_bounds: (x_min, y_min, z_min, x_max, y_max, z_max) bounds.
            node_x: X coordinates of each node (float32).
            node_y: Y coordinates of each node (float32).
            node_z: Z coordinates of each node (float32).

        Returns:
            Tuple of (filtered_distances, matched_node_ids) as NumPy arrays.
        """
        if not self._initialized or self._metal_dijkstra is None:
            raise RuntimeError("Metal provider not initialized")

        node_x = np.ascontiguousarray(node_x, dtype=np.float32)
        node_y = np.ascontiguousarray(node_y, dtype=np.float32)
        node_z = np.ascontiguousarray(node_z, dtype=np.float32)

        return self._metal_dijkstra.extract_roi(
            roi_bounds[0], roi_bounds[1], roi_bounds[2],
            roi_bounds[3], roi_bounds[4], roi_bounds[5],
            node_x, node_y, node_z,
        )

    def process_vias(
        self,
        via_capacity: np.ndarray,
        via_usage: np.ndarray,
        base_cost: float = 1.0,
    ) -> np.ndarray:
        """Compute via costs on the Metal GPU.

        Dispatches the via_kernels Metal kernel to compute costs based on
        capacity and current usage:
        - usage >= capacity → INFINITY (hard-block)
        - 0 < usage < capacity → base_cost * (1 + usage/capacity)
        - usage == 0 → base_cost

        Args:
            via_capacity: Per-via capacity limits (float32).
            via_usage: Per-via current usage counts (float32).
            base_cost: Base cost multiplier.

        Returns:
            NumPy float32 array of computed via costs.
        """
        if not self._initialized or self._metal_dijkstra is None:
            raise RuntimeError("Metal provider not initialized")

        via_capacity = np.ascontiguousarray(via_capacity, dtype=np.float32)
        via_usage = np.ascontiguousarray(via_usage, dtype=np.float32)

        return self._metal_dijkstra.process_vias(via_capacity, via_usage, base_cost)

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
