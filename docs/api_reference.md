# API Reference

This is a manual API reference for OrthoRoute-Metal's public Python API, domain models, configuration options, and CLI interface.

---

## Table of Contents

- [orthoroute\_metal.MetalDijkstra](#orthoroute_metalmetaldijkstra)
- [orthoroute\_metal.amx\_sgemm\_py](#orthoroute_metalamx_sgemm_py)
- [Domain Models](#domain-models)
  - [Board](#board)
  - [Component](#component)
  - [Net](#net)
  - [Pad](#pad)
  - [Layer](#layer)
  - [Route](#route)
  - [Segment](#segment)
  - [Via](#via)
  - [DRCConstraints](#drcconstraints)
  - [NetClass](#netclass)
- [Configuration](#configuration)
  - [PathFinderConfig](#pathfinderconfig)
  - [orthoroute.json](#orthoroutejson)
- [CLI Reference](#cli-reference)

---

## `orthoroute_metal.MetalDijkstra`

The GPU-accelerated shortest-path solver. This is the primary interface to the Metal compute backend, exposed to Python via PyO3.

### Constructor

```python
dijkstra = orthoroute_metal.MetalDijkstra()
```

Creates a new Metal compute context:
- Acquires the system default Metal device.
- Creates a command queue.
- Compiles all 7 MSL compute kernels and caches their pipeline states.

**Raises**: `RuntimeError` if no Metal device is found (e.g., running on non-Apple hardware).

---

### `set_graph_csr(indptr, indices, weights)`

Maps a CSR-format graph into Metal GPU buffers using zero-copy UMA.

```python
result_str = dijkstra.set_graph_csr(indptr, indices, weights)
```

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `indptr` | `numpy.ndarray[int32]` | CSR row pointer array of shape `(N+1,)` |
| `indices` | `numpy.ndarray[int32]` | CSR column indices of shape `(E,)` |
| `weights` | `numpy.ndarray[float32]` | CSR edge weights of shape `(E,)` |

**Returns**: `str` — Summary string with buffer sizes.

**Side effects**:
- Allocates frontier buffers (2× `N` elements), queue size buffers, steal index, active flags, grid barrier state, and predecessor buffer (initialized to −1).
- Sets `node_count = len(indptr) - 1`.

**Raises**: `ValueError` if any input array is not C-contiguous.

> **Important**: The NumPy arrays **must** remain alive for the lifetime of the `MetalDijkstra` instance. The Metal buffers point directly to their memory (zero-copy).

---

### `set_distances_csr(distances)`

Sets the initial distance array for the SSSP solver.

```python
dijkstra.set_distances_csr(distances)
```

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `distances` | `numpy.ndarray[float32]` | Distance array of shape `(N,)`. Source node(s) should be set to `0.0`, all others to `np.inf`. |

**Returns**: `None`

> **Important**: Same lifetime requirement as `set_graph_csr` — the NumPy array must remain alive.

---

### `reset_predecessors()`

Resets all predecessor values to −1 (no predecessor). Call this before each new SSSP run.

```python
dijkstra.reset_predecessors()
```

**Returns**: `None`

**Raises**: `RuntimeError` if predecessors buffer is not initialized.

---

### `setup_spfa()`

Initializes the SPFA frontier from the current distance array. This dispatches two GPU kernels:

1. `clear_counters` — resets queue counters
2. `spfa_setup_kernel` — scans distances, marks nodes with finite distance as active, and adds them to the frontier

```python
dijkstra.setup_spfa()
```

**Returns**: `None`

**Raises**: `RuntimeError` if buffers are not initialized.

> **Must be called** after `set_distances_csr()` and before `execute_until_convergence()`.

---

### `execute_until_convergence(max_iters, batch_size, threadgroup_size, delta)`

Runs the persistent-thread SPFA solver on the GPU until convergence or the iteration limit.

```python
total_iters, converged = dijkstra.execute_until_convergence(
    max_iters=500,
    batch_size=1024,
    threadgroup_size=512,
    delta=0.0
)
```

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `max_iters` | `int` | — | Maximum SPFA iterations |
| `batch_size` | `int` | — | Reserved (not currently used in persistent model) |
| `threadgroup_size` | `int` | — | Threads per threadgroup (capped at pipeline max) |
| `delta` | `float` | — | Delta-stepping bucket width. Use `0.0` to disable delta-stepping (processes all frontier nodes per iteration). |

**Returns**: `tuple[int, bool]`
- `total_iters` — Number of iterations executed.
- `converged` — `True` if both frontier queues are empty (SSSP completed).

**Notes**:
- Launches 8,192 persistent threads (16 threadgroups).
- The entire computation runs inside a single `commandBuffer.commit()` call.
- Set `METAL_CAPTURE_TRACE=1` environment variable to enable GPU frame capture for profiling.

---

### `get_distances()`

Returns the computed distance array. Zero-copy — the returned NumPy array points to GPU memory.

```python
distances = dijkstra.get_distances()  # numpy.ndarray[float32] of shape (N,)
```

**Returns**: `numpy.ndarray[float32]` — Final shortest-path distances.

**Raises**: `RuntimeError` if distances buffer is not initialized.

---

### `get_predecessors()`

Returns the predecessor array for path reconstruction.

```python
predecessors = dijkstra.get_predecessors()  # numpy.ndarray[int32] of shape (N,)
```

**Returns**: `numpy.ndarray[int32]` — Predecessor node IDs. A value of −1 means no predecessor (unreachable or source node).

**Raises**: `RuntimeError` if predecessors buffer is not initialized.

---

### `extract_roi()`

Placeholder for ROI extraction dispatch. Currently logs and returns.

```python
dijkstra.extract_roi()
```

---

### `process_vias()`

Placeholder for via processing dispatch. Currently logs and returns.

```python
dijkstra.process_vias()
```

---

### Complete Example

```python
import orthoroute_metal
import numpy as np
from scipy.sparse import random as sparse_random

# Create a MetalDijkstra instance
dijkstra = orthoroute_metal.MetalDijkstra()

# Build a random sparse graph in CSR format
graph = sparse_random(1000, 1000, density=0.01, format='csr', dtype=np.float32)
graph.data = np.abs(graph.data) + 0.1  # Ensure positive weights

# Map graph to GPU
dijkstra.set_graph_csr(
    graph.indptr.astype(np.int32),
    graph.indices.astype(np.int32),
    graph.data.astype(np.float32)
)

# Set source node 0
distances = np.full(1000, np.inf, dtype=np.float32)
distances[0] = 0.0
dijkstra.set_distances_csr(distances)
dijkstra.reset_predecessors()

# Initialize frontier
dijkstra.setup_spfa()

# Solve
iters, converged = dijkstra.execute_until_convergence(
    max_iters=500,
    batch_size=1024,
    threadgroup_size=512,
    delta=0.0
)

print(f"Converged: {converged} in {iters} iterations")
print(f"Distance to node 999: {dijkstra.get_distances()[999]:.2f}")

# Reconstruct path from 999 to 0
preds = dijkstra.get_predecessors()
path = []
node = 999
while node != -1:
    path.append(node)
    node = preds[node]
path.reverse()
print(f"Path: {path[:10]}...")
```

---

## `orthoroute_metal.amx_sgemm_py`

Single-precision general matrix multiply (SGEMM) offloaded to Apple's AMX coprocessors via the Accelerate framework.

```python
orthoroute_metal.amx_sgemm_py(m, n, k, alpha, a_array, b_array, beta, c_array)
```

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `m` | `int` | Rows of matrix A and C |
| `n` | `int` | Columns of matrix B and C |
| `k` | `int` | Columns of A / rows of B |
| `alpha` | `float` | Scalar multiplier for A × B |
| `a_array` | `numpy.ndarray[float32]` | Matrix A, flattened row-major `(m × k,)` |
| `b_array` | `numpy.ndarray[float32]` | Matrix B, flattened row-major `(k × n,)` |
| `beta` | `float` | Scalar multiplier for C |
| `c_array` | `numpy.ndarray[float32]` | Matrix C (output), flattened row-major `(m × n,)` |

**Computes**: `C = alpha × A × B + beta × C`

**Returns**: `None` (C is modified in-place).

---

## Domain Models

All domain models are defined in `orthoroute/domain/models/`.

### Board

**Module**: `orthoroute.domain.models.board.Board`

The aggregate root representing a PCB board.

```python
@dataclass
class Board:
    id: str
    name: str
    components: List[Component]
    nets: List[Net]
    layers: List[Layer]
    thickness: float = 1.6       # mm
    layer_count: int = 2
```

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `add_component(component)` | `None` | Add a component to the board |
| `add_net(net)` | `None` | Add a net to the board |
| `add_layer(layer)` | `None` | Add a layer to the board |
| `get_component(id)` | `Component \| None` | Get component by ID |
| `get_net(net_id)` | `Net \| None` | Get net by ID |
| `get_net_by_name(name)` | `Net \| None` | Get net by name |
| `get_layer(name)` | `Layer \| None` | Get layer by name |
| `get_routable_nets()` | `List[Net]` | Nets with 2+ pads |
| `get_routing_layers()` | `List[Layer]` | Layers usable for routing |
| `get_bounds()` | `Bounds` | Board bounding box from components |
| `get_all_pads()` | `List[Pad]` | All pads from all components |
| `validate_integrity()` | `List[str]` | List of integrity issues found |

---

### Component

**Module**: `orthoroute.domain.models.board.Component`

```python
@dataclass
class Component:
    id: str
    reference: str           # e.g. "R1", "U3"
    value: str               # e.g. "10k", "ATmega328P"
    footprint: str           # e.g. "Resistor_SMD:R_0402"
    position: Coordinate     # (x, y) in mm
    angle: float = 0.0       # Rotation in degrees
    layer: str = "F.Cu"      # Placement layer
    pads: List[Pad] = []
```

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `get_bounds()` | `Bounds` | Bounding box from pad positions |

---

### Net

**Module**: `orthoroute.domain.models.board.Net`

```python
@dataclass
class Net:
    id: str
    name: str                   # e.g. "GND", "VCC", "Net-(R1-Pad1)"
    netclass: str = "Default"
    pads: List[Pad] = []
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `is_routable` | `bool` | `True` if net has 2+ pads |

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `get_bounds()` | `Bounds` | Bounding box of all pad positions |
| `calculate_min_distance()` | `float` | Minimum pairwise pad distance |

---

### Pad

**Module**: `orthoroute.domain.models.board.Pad`

```python
@dataclass
class Pad:
    id: str
    component_id: str
    net_id: Optional[str]
    position: Coordinate      # (x, y) in mm
    size: Tuple[float, float] # (width, height) in mm
    drill_size: Optional[float] = None
    layer: str = "F.Cu"
    shape: str = "circle"     # "circle", "rect", "roundrect", "oval"
    angle: float = 0.0
```

---

### Layer

**Module**: `orthoroute.domain.models.board.Layer`

```python
@dataclass
class Layer:
    name: str                  # e.g. "F.Cu", "In1.Cu", "B.Cu"
    type: str                  # "copper", "signal", "power", "ground"
    stackup_position: int
    thickness: float = 0.035   # mm
    material: str = "copper"
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `is_routing_layer` | `bool` | `True` if type is signal/power/ground and name contains "Cu" |

---

### Route

**Module**: `orthoroute.domain.models.routing.Route`

```python
@dataclass
class Route:
    id: str
    net_id: str
    segments: List[Segment] = []
    vias: List[Via] = []
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `total_length` | `float` | Sum of all segment lengths |
| `layers_used` | `Set[str]` | Set of layer names used |
| `via_count` | `int` | Number of vias |

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `add_segment(segment)` | `None` | Add a routing segment |
| `add_via(via)` | `None` | Add a via |
| `is_manhattan_compliant()` | `bool` | All segments are horizontal or vertical |
| `validate_connectivity()` | `List[str]` | List of connectivity issues |
| `get_route_statistics()` | `Dict[str, Any]` | Length, vias, layers, etc. |

---

### Segment

**Module**: `orthoroute.domain.models.routing.Segment`

```python
@dataclass(frozen=True)
class Segment:
    type: SegmentType        # TRACK, ARC, VIA
    start: Coordinate
    end: Coordinate
    width: float             # mm
    layer: str
    net_id: str
```

**Enums**:

```python
class SegmentType(Enum):
    TRACK = "track"
    ARC = "arc"
    VIA = "via"
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `length` | `float` | Euclidean length (0 for vias) |
| `is_horizontal` | `bool` | Y coordinates match (within 0.001 mm) |
| `is_vertical` | `bool` | X coordinates match (within 0.001 mm) |
| `is_manhattan` | `bool` | Either horizontal or vertical |

---

### Via

**Module**: `orthoroute.domain.models.routing.Via`

```python
@dataclass(frozen=True)
class Via:
    position: Coordinate
    diameter: float          # mm
    drill_size: float        # mm
    from_layer: str
    to_layer: str
    net_id: str
    via_type: ViaType = ViaType.THROUGH
```

**Enums**:

```python
class ViaType(Enum):
    THROUGH = "through"
    BLIND = "blind"
    BURIED = "buried"
    MICRO = "micro"
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `aspect_ratio` | `float` | Board thickness / drill size (assumes 1.6 mm) |

---

### Coordinate

**Module**: `orthoroute.domain.models.board.Coordinate`

```python
@dataclass(frozen=True)
class Coordinate:
    x: float   # mm
    y: float   # mm
```

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `distance_to(other)` | `float` | Euclidean distance in mm |

---

### Bounds

**Module**: `orthoroute.domain.models.board.Bounds`

```python
@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `width` | `float` | `max_x - min_x` |
| `height` | `float` | `max_y - min_y` |
| `center` | `Coordinate` | Center point |

---

### DRCConstraints

**Module**: `orthoroute.domain.models.constraints.DRCConstraints`

```python
@dataclass
class DRCConstraints:
    # Global constraints (mm)
    min_track_width: float = 0.15
    min_clearance: float = 0.15
    min_via_diameter: float = 0.6
    min_via_drill: float = 0.3
    min_annular_ring: float = 0.13
    min_hole_clearance: float = 0.25
    board_edge_clearance: float = 0.25
    # Netclasses
    netclasses: Dict[str, NetClass] = {}
```

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `add_netclass(netclass)` | `None` | Add or update a netclass |
| `get_netclass(name)` | `NetClass` | Get by name, fallback to "Default" |
| `get_clearance_for_nets(nc1, nc2)` | `float` | Max clearance between two netclasses |
| `get_clearance(type)` | `float` | Get clearance by type enum |

---

### NetClass

**Module**: `orthoroute.domain.models.constraints.NetClass`

```python
@dataclass(frozen=True)
class NetClass:
    name: str
    track_width: float = 0.25      # mm
    clearance: float = 0.2         # mm
    via_diameter: float = 0.8      # mm
    via_drill: float = 0.4         # mm
    min_track_width: float = 0.15  # mm
    max_track_width: float = 2.0   # mm
```

---

## Configuration

### PathFinderConfig

The `PathFinderConfig` dataclass in `orthoroute/algorithms/manhattan/pathfinder/config.py` centralizes all tunable parameters. See [tuning_guide.md](tuning_guide.md) for detailed guidance.

**Key sections**:

#### Grid & Geometry

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_pitch` | `float` | `0.4` | Grid spacing in mm |
| `layer_count` | `int` | `0` | Set from `board.layer_count` |
| `layer_names` | `List[str]` | `['F.Cu', ...]` | Layer names from KiCad |

#### PathFinder Algorithm

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pres_fac_init` | `float` | `1.0` | Initial congestion pressure |
| `pres_fac_mult` | `float` | `1.10` | Pressure multiplier per iteration |
| `pres_fac_max` | `float` | `8.0` | Maximum pressure cap |
| `hist_gain` | `float` | `0.20` | History accumulation rate |
| `hist_cost_weight` | `float` | `10.0` | History vs. present weight |
| `hist_accum_gain` | `float` | `1.2` | History accumulation gain |
| `max_iterations` | `int` | `40` | Maximum PathFinder iterations |
| `batch_size` | `int` | `32` | Nets per routing batch |
| `stagnation_patience` | `int` | `5` | Iterations without improvement |
| `hotset_cap` | `int` | `150` | Max nets rerouted per iteration |

#### Via Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `via_cost` | `float` | `0.7` | Base via penalty |
| `via_capacity_per_net` | `int` | `8` | Via column capacity |
| `allow_any_layer_via` | `bool` | `True` | Allow blind/buried vias |
| `via_span_alpha` | `float` | `0.08` | Penalty for long via spans |

#### Portal Escape

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `portal_enabled` | `bool` | `True` | Enable portal escapes |
| `portal_discount` | `float` | `0.4` | Escape via discount (60% off) |
| `portal_delta_min` | `int` | `3` | Min escape offset in grid steps |
| `portal_delta_max` | `int` | `12` | Max escape offset in grid steps |
| `portal_via_discount` | `float` | `0.15` | Escape via cost multiplier |

#### GPU Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gpu` | `bool` | `False` | Enable GPU acceleration |
| `use_gpu_sequential` | `bool` | `True` | Use GPU for sequential routing |
| `gpu_roi_min_nodes` | `int` | `1000` | Minimum ROI size for GPU |

#### ROI Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_roi_nodes` | `int` | `20000` | Maximum ROI subgraph size |
| `roi_widen_levels` | `int` | `4` | ROI widening retry levels |
| `roi_widen_factor` | `float` | `2.0` | Margin multiplier per level |
| `BASE_ROI_MARGIN_MM` | `float` | `4.0` | Base ROI margin in mm |

---

### `orthoroute.json`

Project-level configuration file (optional). Placed at the project root.

```json
{
  "routing": {
    "grid_pitch": 0.4,
    "max_iterations": 100,
    "via_cost": 0.7,
    "pres_fac_mult": 1.35
  },
  "gpu": {
    "enabled": true,
    "backend": "metal",
    "threadgroup_size": 512
  },
  "logging": {
    "level": "INFO",
    "file": "orthoroute.log"
  }
}
```

---

## CLI Reference

### Usage

```
python main.py [mode] [options]
```

### Modes

#### `plugin` — KiCad Plugin

```bash
python main.py plugin [--no-gui] [--min-run-sec N]
```

| Option | Description |
|--------|-------------|
| `--no-gui` | Run without PyQt6 GUI |
| `--min-run-sec N` | Keep process alive for N seconds (CI/agents) |

#### `cli` — Command Line

```bash
python main.py cli board.kicad_pcb [-o OUTPUT] [-c CONFIG]
```

| Option | Description |
|--------|-------------|
| `board_file` | KiCad board file (`.kicad_pcb`) |
| `-o, --output` | Output directory (default: `.`) |
| `-c, --config` | Configuration file path |

#### `headless` — Cloud Routing

```bash
python main.py headless input.ORP [-o output.ORS] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `orp_file` | — | Input `.ORP` board export file |
| `-o, --output` | (derived) | Output `.ORS` filepath |
| `--max-iterations` | `250` | Override iteration limit |
| `--checkpoint-interval` | `30` | Checkpoint save interval (minutes) |
| `--resume-checkpoint` | — | Resume from checkpoint file |
| `--use-gpu` | auto | Enable GPU acceleration |
| `--cpu-only` | `False` | Force CPU-only mode |

### Global Options

| Option | Description |
|--------|-------------|
| `--version` | Show version number |
| `--test-manhattan` | Run automated Manhattan routing test |
| `--autoroute` | Run headless autoroute test |
| `--test-via` | Run tiny 2-layer via test |
| `--min-run-sec N` | Keep process alive for N seconds |

### Examples

```bash
# Run as KiCad plugin with GUI
python main.py plugin

# Run without GUI
python main.py plugin --no-gui

# Route a board via CLI
python main.py cli myboard.kicad_pcb -o output/

# Headless cloud routing with checkpoints
python main.py headless board.ORP -o result.ORS --max-iterations 500 --checkpoint-interval 60

# Quick test
python main.py --test-manhattan
```

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture
- [pathfinder_algorithm.md](pathfinder_algorithm.md) — PathFinder algorithm details
- [tuning_guide.md](tuning_guide.md) — Parameter tuning guide
- [coordinate_system.md](coordinate_system.md) — Coordinate system reference
- [metal_kernel_internals.md](metal_kernel_internals.md) — GPU kernel internals
- [contributing.md](contributing.md) — Contributing guidelines
