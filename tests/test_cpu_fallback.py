"""Tests for CPU fallback provider."""
import pytest
import numpy as np

from orthoroute.infrastructure.gpu.cpu_fallback import CPUFallbackProvider, CPUProvider


@pytest.fixture
def provider():
    """Initialized CPU provider for tests."""
    p = CPUFallbackProvider()
    p.initialize()
    yield p
    p.cleanup()


def test_cpu_provider_is_available():
    """CPU provider is always available."""
    p = CPUFallbackProvider()
    assert p.is_available() is True


def test_cpu_provider_initialize():
    """CPU provider initializes successfully."""
    p = CPUFallbackProvider()
    assert p.initialize() is True
    p.cleanup()


def test_cpu_provider_create_array_zeros(provider):
    """Create array filled with zeros."""
    arr = provider.create_array((3, 4), dtype=np.float32, fill_value=0)
    assert arr.shape == (3, 4)
    assert arr.dtype == np.float32
    assert np.all(arr == 0)


def test_cpu_provider_create_array_ones(provider):
    """Create array filled with ones."""
    arr = provider.create_array((2, 2), dtype=np.float32, fill_value=1)
    assert np.all(arr == 1)


def test_cpu_provider_create_array_custom_fill(provider):
    """Create array filled with custom value."""
    arr = provider.create_array((5,), dtype=np.int32, fill_value=42)
    assert np.all(arr == 42)


def test_cpu_provider_create_array_empty(provider):
    """Create empty (uninitialized) array."""
    arr = provider.create_array((3, 3), dtype=np.float32)
    assert arr.shape == (3, 3)


def test_cpu_provider_copy_array(provider):
    """Copy array produces independent copy."""
    original = provider.create_array((3,), dtype=np.float32, fill_value=1)
    copied = provider.copy_array(original)
    copied[0] = 99
    assert original[0] == 1


def test_cpu_provider_to_cpu_identity(provider):
    """to_cpu returns the same array (no-op on CPU)."""
    arr = np.zeros(5)
    result = provider.to_cpu(arr)
    assert result is arr


def test_cpu_provider_to_gpu_identity(provider):
    """to_gpu returns the same array (no-op on CPU)."""
    arr = np.zeros(5)
    result = provider.to_gpu(arr)
    assert result is arr


def test_cpu_provider_synchronize(provider):
    """synchronize is a no-op that doesn't raise."""
    provider.synchronize()


def test_cpu_provider_device_info(provider):
    """Device info returns dict with expected keys."""
    info = provider.get_device_info()
    assert 'name' in info
    assert info['device_id'] == 'cpu'


def test_cpu_provider_memory_info(provider):
    """Memory info returns dict with expected keys."""
    info = provider.get_memory_info()
    assert 'total_memory' in info
    assert 'free_memory' in info


def test_cpu_provider_context_manager():
    """CPU provider works as context manager."""
    with CPUFallbackProvider() as p:
        assert p._initialized is True
        arr = p.create_array((2, 2), fill_value=0)
        assert arr.shape == (2, 2)
    assert p._initialized is False


def test_cpu_provider_not_initialized_raises():
    """Creating array without initialization raises RuntimeError."""
    p = CPUFallbackProvider()
    with pytest.raises(RuntimeError):
        p.create_array((3,))


def test_cpu_provider_alias():
    """CPUProvider is an alias for CPUFallbackProvider."""
    assert CPUProvider is CPUFallbackProvider


def test_cpu_provider_cleanup_resets_state():
    """Cleanup resets initialized state."""
    p = CPUFallbackProvider()
    p.initialize()
    p.cleanup()
    assert p._initialized is False
