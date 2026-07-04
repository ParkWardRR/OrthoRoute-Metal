"""GPU infrastructure adapters."""
import logging
import platform

from .cuda_provider import CUDAProvider
from .cpu_fallback import CPUFallbackProvider, CPUProvider
from .metal_provider import MetalProvider

logger = logging.getLogger(__name__)

__all__ = [
    'CUDAProvider',
    'CPUFallbackProvider',
    'CPUProvider',
    'MetalProvider',
    'get_best_provider',
]


def get_best_provider():
    """Auto-detect and return the best available GPU provider.

    Selection priority:
    1. CUDA (NVIDIA GPUs via CuPy) — if CuPy is installed and a CUDA GPU is present
    2. Metal (Apple Silicon GPUs) — if on macOS arm64 with orthoroute_mac compiled
    3. CPU fallback — always available

    Returns:
        An instance of the best available GPUProvider subclass.
    """
    # 1. Try CUDA first (NVIDIA GPUs)
    try:
        cuda = CUDAProvider()
        if cuda.is_available():
            logger.info("Auto-detected CUDA GPU — using CUDAProvider")
            return cuda
    except Exception as e:
        logger.debug(f"CUDA detection failed: {e}")

    # 2. On macOS + Apple Silicon, try Metal
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            metal = MetalProvider()
            if metal.is_available():
                logger.info("Auto-detected Apple Silicon Metal GPU — using MetalProvider")
                return metal
        except Exception as e:
            logger.debug(f"Metal detection failed: {e}")

    # 3. CPU fallback (always available)
    logger.info("No GPU acceleration available — using CPUFallbackProvider")
    return CPUFallbackProvider()