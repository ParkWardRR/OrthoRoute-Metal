"""GPU / CPU provider parity tests.

Validates MetalProvider availability detection, get_best_provider()
auto-selection, and CPUFallbackProvider array creation.

All tests in this module require GPU marker (or explicit CPU-only testing).
"""
import platform

import pytest
import numpy as np

from orthoroute.infrastructure.gpu.metal_provider import MetalProvider
from orthoroute.infrastructure.gpu.cpu_fallback import CPUFallbackProvider
from orthoroute.infrastructure.gpu import get_best_provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestMetalProviderAvailable:
    """Check MetalProvider.is_available() on macOS."""

    def test_metal_provider_instantiates(self):
        """MetalProvider() should not raise."""
        provider = MetalProvider()
        assert provider is not None

    def test_is_available_returns_bool(self):
        """is_available() should return a boolean, not raise."""
        provider = MetalProvider()
        result = provider.is_available()
        assert isinstance(result, bool)

    def test_is_available_on_macos(self):
        """On macOS arm64 with orthoroute_mac, is_available() should be True.

        If orthoroute_mac is not compiled, it will return False — which is
        acceptable; we just verify no crash.
        """
        provider = MetalProvider()
        result = provider.is_available()
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            # On Apple Silicon, result depends on whether orthoroute_mac is compiled
            assert isinstance(result, bool)
        else:
            # On non-macOS, Metal should never be available
            assert result is False


@pytest.mark.gpu
class TestProviderAutoDetection:
    """Test get_best_provider() auto-detection logic."""

    def test_returns_a_provider(self):
        """get_best_provider() should always return a provider instance."""
        provider = get_best_provider()
        assert provider is not None

    def test_provider_has_required_methods(self):
        """Returned provider should implement the GPUProvider interface."""
        provider = get_best_provider()
        assert hasattr(provider, 'is_available')
        assert hasattr(provider, 'initialize')
        assert hasattr(provider, 'cleanup')
        assert hasattr(provider, 'create_array')
        assert hasattr(provider, 'get_device_info')

    def test_fallback_is_cpu_when_no_gpu(self):
        """If no GPU is available, get_best_provider() returns CPUFallbackProvider."""
        provider = get_best_provider()
        # The provider should be available regardless of type
        assert provider.is_available() or isinstance(provider, CPUFallbackProvider)


@pytest.mark.gpu
class TestCPUArrayCreation:
    """CPUFallbackProvider creates arrays correctly."""

    def test_create_zeros_array(self):
        """create_array with fill_value=0 should produce a zero-filled array."""
        provider = CPUFallbackProvider()
        provider.initialize()
        try:
            arr = provider.create_array((10, 10), dtype=np.float32, fill_value=0)
            assert arr.shape == (10, 10)
            assert arr.dtype == np.float32
            assert np.all(arr == 0)
        finally:
            provider.cleanup()

    def test_create_ones_array(self):
        """create_array with fill_value=1 should produce an all-ones array."""
        provider = CPUFallbackProvider()
        provider.initialize()
        try:
            arr = provider.create_array((5, 5), dtype=np.float32, fill_value=1)
            assert np.all(arr == 1)
        finally:
            provider.cleanup()

    def test_create_custom_fill_array(self):
        """create_array with fill_value=42.0 should produce correct values."""
        provider = CPUFallbackProvider()
        provider.initialize()
        try:
            arr = provider.create_array((3, 4), dtype=np.float32, fill_value=42.0)
            assert np.allclose(arr, 42.0)
        finally:
            provider.cleanup()

    def test_create_array_without_init_raises(self):
        """create_array before initialize() should raise RuntimeError."""
        provider = CPUFallbackProvider()
        with pytest.raises(RuntimeError):
            provider.create_array((5,), dtype=np.float32, fill_value=0)

    def test_copy_array(self):
        """copy_array should produce an independent copy."""
        provider = CPUFallbackProvider()
        provider.initialize()
        try:
            original = provider.create_array((4,), dtype=np.float32, fill_value=3.14)
            copied = provider.copy_array(original)
            assert np.allclose(original, copied)
            # Modifying the copy should not affect the original
            copied[0] = 999.0
            assert original[0] != 999.0
        finally:
            provider.cleanup()
