# OrthoRoute-Metal Integration Lessons

These lessons were learned during the full integration of the CUDA2Metal-Graph-Research framework into the OrthoRoute PCB autorouter.

## Python-Rust-Metal Integration

1. **PyO3 buffer protocol is the fastest Python-Metal bridge.** Converting NumPy arrays to Metal buffers via `pyo3::buffer::PyBuffer` is zero-copy on UMA. The alternative (serializing to bytes and copying) adds 15-30ms for large CSR graphs. Always use `MTLResourceStorageModeShared` and pass the raw pointer from NumPy's buffer protocol.

2. **GPU provider abstraction must be a runtime decision, not a compile-time one.** The `GPUProvider` abstract interface with `CUDA -> Metal -> Vulkan -> CPU` fallback chain was critical. Users install the package once; the provider factory auto-detects available hardware at import time via `importlib` try/except chains. Never use conditional compilation for GPU backend selection in Python.

3. **CPU fallback must produce identical results.** Our `SimpleDijkstra` CPU fallback uses the exact same CSR graph format and float32 precision as the GPU kernels. This enabled automated parity testing: run both backends on the same input and assert bitwise equality. Without this, debugging GPU kernel bugs is nearly impossible.

## Large Codebase Management

4. **Decompose 5000+ line files using the delegation pattern, not inheritance.** The original `unified_pathfinder.py` (5847 lines, 282KB) was unmaintainable. We extracted 4 modules (`LatticeManager`, `EdgeAccountant`, `GeometryEmitter`, `ConvergenceManager`) using delegation -- each holds a reference to the parent router and replaces `self.x` with `self.router.x`. This preserves the public API while enabling independent testing of each subsystem.

5. **Mixin inheritance creates fragile implicit coupling.** The original architecture used 7 mixins (`LatticeBuilderMixin`, `NegotiationMixin`, etc.) that all assumed `self` had specific attributes set by other mixins. This made testing impossible -- you could not instantiate one mixin without all 7. Standalone delegation classes with explicit constructor arguments are far more testable.

6. **Magic numbers are the #1 source of configuration bugs.** Extracting 15 magic numbers (EWMA_ALPHA, PRESSURE_MULTIPLIER, GPU_ROI_THRESHOLD, etc.) into `constants.py` with named constants and docstrings eliminated an entire class of bugs where different files used subtly different values for the same parameter.

## Testing Strategy

7. **Test the domain model without any I/O.** All Board/Net/Pad/Component objects are plain dataclasses that can be constructed in-memory. This means 100% of KiCad integration tests run without any .kicad_pcb files on disk. The `conftest.py` fixtures create realistic 6-layer BGA boards with 10 pads and 3 nets in <1ms.

8. **Serialization round-trip tests catch format drift.** Every ORP/ORS format change was caught by `export -> import -> assert_equal` tests. The round-trip pattern is: create Board object, export to file, import from file, reconstruct Board, assert pad/net/layer counts match. This caught 3 silent regressions where field names changed between export and import code paths.

9. **Performance benchmark tests need generous thresholds.** Initial benchmark tests with tight timing thresholds (e.g., "Dijkstra 10K nodes < 100ms") caused flaky CI on loaded machines. Using 3-5x headroom (< 1s) keeps tests deterministic while still catching O(n^2) regressions.

10. **529 tests in 0.76s is achievable with proper isolation.** By avoiding database, network, and filesystem I/O in unit tests, and using in-memory fixtures, the entire test suite runs in under 1 second. The only I/O tests are serialization round-trips using pytest's `tmp_path` fixture.

## KiCad Integration

11. **KiCad's internal units are nanometers, not millimeters.** The SWIG adapter must convert `nm -> mm` on read and `mm -> nm` on write. Getting this wrong produces boards that are 1,000,000x too large or too small. Always validate with: `assert 0.1 < track_width_mm < 10.0` after conversion.

12. **KiCad layer names must match exactly.** `F.Cu` is not `FCu`, `F_Cu`, or `front_copper`. The regex `^(F\.Cu|B\.Cu|In\d+\.Cu)$` validates all standard copper layer names. Non-copper layers (`F.SilkS`, `Edge.Cuts`) must be excluded from routing layer counts, but `Edge.Cuts` sneakily contains "Cu" as a substring -- match on `\.Cu$`, not just `Cu`.

13. **Adapter priority matters: IPC > SWIG > File.** KiCad's IPC API (HTTP on port 5555) is the fastest for live editing. The SWIG API (`pcbnew.GetBoard()`) works when running inside KiCad's Python console. The file parser is always available as a fallback but produces a snapshot, not a live connection.
