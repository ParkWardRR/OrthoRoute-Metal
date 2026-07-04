"""CPU fallback provider implementation."""
from __future__ import annotations

import logging
from types import TracebackType
from typing import Optional, Dict, Any, Tuple, Type

import numpy as np
import numpy.typing as npt
import psutil

from ...application.interfaces.gpu_provider import GPUProvider

logger = logging.getLogger(__name__)


class CPUFallbackProvider(GPUProvider):
    """CPU-only fallback implementation of GPU provider."""
    
    def __init__(self) -> None:
        """Initialize CPU fallback provider."""
        self._initialized: bool = False
        self._memory_limit: Optional[int] = None
        self._allocated_arrays: list[npt.NDArray[np.generic]] = []
    
    def is_available(self) -> bool:
        """CPU is always available."""
        return True
    
    def initialize(self) -> bool:
        """Initialize CPU resources."""
        try:
            # Set memory limit based on available system memory
            available_memory = psutil.virtual_memory().available
            self._memory_limit = int(available_memory * 0.5)  # Use 50% of available memory
            
            self._initialized = True
            logger.info(f"CPU fallback provider initialized with {self._memory_limit / 1024**3:.1f}GB limit")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CPU fallback: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup CPU resources."""
        # Clear array references to help garbage collection
        self._allocated_arrays.clear()
        self._initialized = False
        logger.debug("CPU fallback provider cleaned up")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get CPU device information."""
        try:
            cpu_info = {
                'name': f'{psutil.cpu_count()} CPU cores',
                'compute_capability': 'N/A',
                'total_memory': psutil.virtual_memory().total,
                'free_memory': psutil.virtual_memory().available,
                'memory_limit': self._memory_limit or 0,
                'device_id': 'cpu'
            }
            return cpu_info
        except Exception as e:
            logger.error(f"Error getting CPU device info: {e}")
            return {
                'name': 'CPU',
                'compute_capability': 'N/A',
                'total_memory': 0,
                'free_memory': 0,
                'memory_limit': 0,
                'device_id': 'cpu'
            }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get CPU memory usage information."""
        try:
            vm = psutil.virtual_memory()
            
            # Estimate memory used by our arrays
            array_memory = sum(arr.nbytes for arr in self._allocated_arrays if hasattr(arr, 'nbytes'))
            
            return {
                'total_memory': vm.total,
                'free_memory': vm.available,
                'used_memory': vm.used,
                'memory_pool_used': array_memory,
                'memory_pool_total': array_memory
            }
            
        except Exception as e:
            logger.error(f"Error getting CPU memory info: {e}")
            return {
                'total_memory': 0,
                'free_memory': 0,
                'used_memory': 0,
                'memory_pool_used': 0,
                'memory_pool_total': 0
            }
    
    def create_array(self, shape: Tuple[int, ...], dtype: Optional[np.dtype[Any]] = None, fill_value: Optional[float] = None) -> npt.NDArray[Any]:
        """Create array in CPU memory."""
        if not self._initialized:
            raise RuntimeError("CPU provider not initialized")
        
        if dtype is None:
            dtype = np.float32
        
        try:
            # Check memory limit
            estimated_size = np.prod(shape) * np.dtype(dtype).itemsize
            if self._memory_limit and estimated_size > self._memory_limit:
                raise MemoryError(f"Array size {estimated_size} exceeds memory limit {self._memory_limit}")
            
            # Create array
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
            self._allocated_arrays = [arr for arr in self._allocated_arrays 
                                   if hasattr(arr, 'nbytes')]
            
            return array
            
        except Exception as e:
            logger.error(f"Error creating CPU array: {e}")
            raise
    
    def copy_array(self, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Create copy of array in CPU memory."""
        if not self._initialized:
            raise RuntimeError("CPU provider not initialized")
        
        try:
            copied = np.copy(array)
            self._allocated_arrays.append(copied)
            return copied
            
        except Exception as e:
            logger.error(f"Error copying CPU array: {e}")
            raise
    
    def to_cpu(self, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Array is already on CPU."""
        return array
    
    def to_gpu(self, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """No-op for CPU fallback."""
        return array
    
    def synchronize(self) -> None:
        """No-op for CPU (operations are synchronous)."""
        pass
    
    def __enter__(self) -> CPUFallbackProvider:
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit."""
        self.cleanup()


# Alias for backward compatibility
CPUProvider = CPUFallbackProvider