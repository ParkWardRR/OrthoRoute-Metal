# Coordinate System

This document describes OrthoRoute's 3D routing lattice coordinate system — how physical millimeter positions and copper layers map to grid indices and node IDs used internally by the PathFinder and GPU solvers.

## Overview

OrthoRoute models the PCB routing space as a **3D Manhattan lattice**: a regular grid of nodes connected only by orthogonal (horizontal/vertical) edges within each layer, plus vertical via edges between layers. Every node has an integer `(x, y, z)` coordinate and a unique scalar **Global ID (GID)** used to index into CSR graph arrays and GPU buffers.

```
         z (layers)
         ▲
         │
  Nz-1   ┊  ┌───────────────────────────┐  B.Cu
         ┊  │  ·   ·   ·   ·   ·   ·   │
         ┊  │  ·   ·   ·   ·   ·   ·   │
  ...    ┊  │        internal layers      │
         ┊  │  ·   ·   ·   ·   ·   ·   │
   1     ┊  ├───────────────────────────┤  In1.Cu
         ┊  │  ·   ·   ·   ·   ·   ·   │
   0     ┊  └───────────────────────────┘  F.Cu
         └──────────────────────────────────▶  x (columns)
        ╱
       ╱
      ▼  y (rows)
```

---

## Grid Dimensions

The lattice is defined by three integers:

| Symbol | Name | Description |
|--------|------|-------------|
| `Nx` | Columns | Number of grid columns along the X axis |
| `Ny` | Rows | Number of grid rows along the Y axis |
| `Nz` | Layers | Number of copper layers (from KiCad stackup) |

The total number of lattice nodes is:

```
total_nodes = Nx × Ny × Nz
```

These dimensions are stored in the `GridShape` dataclass (see `real_global_grid.py`):

```python
@dataclass(frozen=True)
class GridShape:
    NL: int  # Number of layers (= Nz)
    NX: int  # X tracks (= Nx)
    NY: int  # Y tracks (= Ny)

    @property
    def total_nodes(self) -> int:
        return self.NL * self.NX * self.NY
```

---

## Global ID (GID) Calculation

Every node `(x, y, z)` maps to a unique scalar GID used to index into all flat arrays (distances, predecessors, CSR graph, GPU buffers):

```
gid = z × (Nx × Ny) + y × Nx + x
```

Or equivalently, using `XY = Nx × Ny`:

```
gid = z × XY + y × Nx + x
```

### Inverse: GID → (z, x, y)

```python
def xyz_from_gid(shape, gid):
    z, remainder = divmod(gid, shape.NX * shape.NY)
    y, x = divmod(remainder, shape.NX)
    return z, x, y
```

### Example

For a 10×8 grid with 6 layers (`Nx=10, Ny=8, Nz=6`):

```
XY = 10 × 8 = 80
total_nodes = 6 × 80 = 480

Node (x=3, y=5, z=2):
  gid = 2 × 80 + 5 × 10 + 3 = 160 + 50 + 3 = 213

Node (x=0, y=0, z=0) = gid 0    (first node, F.Cu origin)
Node (x=9, y=7, z=5) = gid 479  (last node, B.Cu corner)
```

### Memory Layout

GIDs increase in **x-major, y-minor, z-slowest** order. This means nodes on the same layer are contiguous in memory, which is important for GPU cache efficiency:

```
Layer z=0 (F.Cu):   gid 0   .. gid 79
Layer z=1 (In1.Cu): gid 80  .. gid 159
Layer z=2 (In2.Cu): gid 160 .. gid 239
...
Layer z=5 (B.Cu):   gid 400 .. gid 479
```

Within each layer, rows are contiguous:

```
Layer z=0, row y=0: gid 0..9
Layer z=0, row y=1: gid 10..19
...
Layer z=0, row y=7: gid 70..79
```

---

## Layer Numbering

Layers are numbered from 0 (top) to Nz−1 (bottom), matching the KiCad copper stackup order:

| z-index | KiCad Layer | Description |
|---------|-------------|-------------|
| 0 | `F.Cu` | Front copper (top) |
| 1 | `In1.Cu` | First internal layer |
| 2 | `In2.Cu` | Second internal layer |
| ... | ... | ... |
| Nz−2 | `In(Nz-2).Cu` | Last internal layer |
| Nz−1 | `B.Cu` | Back copper (bottom) |

Standard layer name generation:

```python
def get_standard_layer_names(layer_count: int) -> list[str]:
    names = ['F.Cu']
    for i in range(1, layer_count - 1):
        names.append(f'In{i}.Cu')
    names.append('B.Cu')
    return names

# 6 layers: ['F.Cu', 'In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'B.Cu']
```

---

## Grid Pitch and Physical Coordinates

### Pitch

The **grid pitch** (default `0.4 mm`) defines the spacing between adjacent grid nodes. This is set in `PathFinderConfig`:

```python
GRID_PITCH = 0.4  # millimeters
```

### Physical → Grid Mapping

Given a physical position `(mm_x, mm_y)` and the board bounds `(min_x, min_y)`:

```
grid_x = round((mm_x - min_x) / pitch)
grid_y = round((mm_y - min_y) / pitch)
```

Grid dimensions are derived from the board bounding box:

```
Nx = floor((max_x - min_x) / pitch) + 1
Ny = floor((max_y - min_y) / pitch) + 1
```

### Grid → Physical Mapping

```
mm_x = min_x + grid_x × pitch
mm_y = min_y + grid_y × pitch
```

### Example

For a board spanning `(10.0, 20.0)` to `(50.0, 60.0)` mm with 0.4 mm pitch:

```
Nx = floor((50.0 - 10.0) / 0.4) + 1 = 100 + 1 = 101
Ny = floor((60.0 - 20.0) / 0.4) + 1 = 100 + 1 = 101

A pad at (25.2, 35.6) mm maps to:
  grid_x = round((25.2 - 10.0) / 0.4) = round(38.0) = 38
  grid_y = round((35.6 - 20.0) / 0.4) = round(39.0) = 39
```

---

## H/V Layer Discipline

OrthoRoute enforces **alternating preferred routing directions** per layer to reduce via count and improve routability:

| Layer z-index | Parity | Preferred Direction | Routing |
|---------------|--------|---------------------|---------|
| 0 (F.Cu) | Even | **Vertical** | Traces run primarily North↔South |
| 1 (In1.Cu) | Odd | **Horizontal** | Traces run primarily East↔West |
| 2 (In2.Cu) | Even | **Vertical** | Traces run primarily North↔South |
| 3 (In3.Cu) | Odd | **Horizontal** | Traces run primarily East↔West |
| ... | ... | ... | ... |

**Rule**: **Odd layers are horizontal, even layers are vertical.**

This discipline is enforced through cost weighting, not hard blocking. Cross-direction routes incur a penalty but are not forbidden, especially during iteration 1 where `iter1_relax_hv_discipline` allows violations with soft penalties.

### 2D Layer View

```
     Even Layer (z=0, 2, 4, ...)           Odd Layer (z=1, 3, 5, ...)
     Preferred: VERTICAL                    Preferred: HORIZONTAL

     ·   ·   ·   ·   ·   ·                 ·───·───·───·───·───·
     │   │   │   │   │   │
     ·   ·   ·   ·   ·   ·                 ·───·───·───·───·───·
     │   │   │   │   │   │
     ·   ·   ·   ·   ·   ·                 ·───·───·───·───·───·
     │   │   │   │   │   │
     ·   ·   ·   ·   ·   ·                 ·───·───·───·───·───·
```

---

## KiCad Coordinate Mapping

### KiCad's Coordinate System

KiCad uses a **millimeter-based** coordinate system with the **Y-axis pointing downward** (screen coordinates):

```
  (0,0) ──────────────────▶ +X (mm)
    │
    │
    │
    │
    ▼
   +Y (mm)
```

### OrthoRoute's Internal Grid

OrthoRoute preserves KiCad's axis orientation. The board bounds `(min_x, min_y, max_x, max_y)` are computed from component pad positions in KiCad millimeters, and the grid origin `(0, 0)` maps to `(min_x, min_y)`:

```
  Grid (0,0) = Board (min_x, min_y)          ──▶ +grid_x
    │
    │
    ▼
   +grid_y

  Grid (Nx-1, Ny-1) = Board (max_x, max_y)
```

> **Note**: OrthoRoute does **not** flip the Y-axis. Grid `y=0` corresponds to `min_y` (top of board in KiCad), and `y=Ny-1` corresponds to `max_y` (bottom of board).

---

## 3D Lattice Structure

### Node Adjacency

Each interior node has up to **4 in-plane neighbors** (Manhattan adjacency) plus **via neighbors** to other layers:

```
                  (x, y-1, z)
                      │
                      │ North
                      │
  (x-1, y, z) ──── (x,y,z) ──── (x+1, y, z)
      West            │              East
                      │ South
                      │
                  (x, y+1, z)

                  Via connections:
                  (x, y, z-1)  ↕  (x, y, z+1)
```

Edge nodes and corner nodes have fewer neighbors (2 or 3 in-plane).

### Edge Costs

| Edge Type | Default Cost | Description |
|-----------|-------------|-------------|
| In-plane track | 1 | Manhattan step within a layer |
| Via (layer change) | 8 | Through-hole or blind/buried via |

Via costs are tunable via `PathFinderConfig.via_cost` (default 0.7 for the PathFinder cost function, which multiplies by the via edge weight).

### Cross-Section View

A vertical cross-section through `(x, y)` showing via connections:

```
  Layer 0 (F.Cu)     ● ←── node (x, y, 0)
                     │
                     │ via (cost=8)
                     │
  Layer 1 (In1.Cu)   ● ←── node (x, y, 1)
                     │
                     │ via (cost=8)
                     │
  Layer 2 (In2.Cu)   ● ←── node (x, y, 2)
                     │
                     │ via (cost=8)
                     │
  Layer 3 (In3.Cu)   ● ←── node (x, y, 3)
                     │
                     │ via (cost=8)
                     │
  Layer 4 (In4.Cu)   ● ←── node (x, y, 4)
                     │
                     │ via (cost=8)
                     │
  Layer 5 (B.Cu)     ● ←── node (x, y, 5)
```

### Full 3D Lattice Visualization

A small 4×3×3 lattice (`Nx=4, Ny=3, Nz=3`):

```
  Layer z=2 (In2.Cu)
  ┌───┬───┬───┐
  │ 24│ 25│ 26│ 27
  ├───┼───┼───┤
  │ 28│ 29│ 30│ 31
  ├───┼───┼───┤
  │ 32│ 33│ 34│ 35
  └───┴───┴───┘
       ╎ vias ╎
  Layer z=1 (In1.Cu)
  ┌───┬───┬───┐
  │ 12│ 13│ 14│ 15
  ├───┼───┼───┤
  │ 16│ 17│ 18│ 19
  ├───┼───┼───┤
  │ 20│ 21│ 22│ 23
  └───┴───┴───┘
       ╎ vias ╎
  Layer z=0 (F.Cu)
  ┌───┬───┬───┐
  │  0│  1│  2│  3
  ├───┼───┼───┤
  │  4│  5│  6│  7
  ├───┼───┼───┤
  │  8│  9│ 10│ 11
  └───┴───┴───┘
  x→  0   1   2   3
  y: 0=top row, 2=bottom row
```

---

## CSR Graph Storage

The 3D lattice is flattened into a **Compressed Sparse Row (CSR)** format for GPU processing:

```
indptr[N+1]  — Row pointer: indptr[gid] = start of node gid's neighbors in indices[]
indices[E]   — Column indices: neighbor GIDs
weights[E]   — Edge weights: cost of each edge
```

where `N = total_nodes` and `E = total_edges`.

### Example

For node `gid=17` (at `x=1, y=1, z=1` in the 4×3×3 lattice above):

```
In-plane neighbors:  gid 16 (West), gid 18 (East), gid 13 (North), gid 21 (South)
Via neighbors:       gid 5 (layer 0), gid 29 (layer 2)

indptr[17] = 42   (neighbors start at index 42 in indices[])
indptr[18] = 48   (next node's neighbors start at index 48)

indices[42..47] = [16, 18, 13, 21, 5, 29]
weights[42..47] = [1,  1,  1,  1,  8, 8 ]
```

---

## Coordinate Bounds Checking

All coordinate conversions include bounds checking to prevent out-of-bounds access to GPU buffers:

```python
def gid(shape, layer, x, y):
    assert 0 <= layer < shape.NL, f"Layer {layer} OOB [0, {shape.NL})"
    assert 0 <= x < shape.NX, f"X {x} OOB [0, {shape.NX})"
    assert 0 <= y < shape.NY, f"Y {y} OOB [0, {shape.NY})"
    return layer * shape.XY + y * shape.NX + x
```

Path validation checks that all GIDs in a routed path fall within `[0, total_nodes)`:

```python
def validate_path_bounds(shape, path, net_name=""):
    if not (0 <= path.min() and path.max() < shape.total_nodes):
        logger.error(f"[OOB-BOUNDS] {net_name}: gids [{path.min()}, {path.max()}] "
                     f"exceed [0, {shape.total_nodes})")
        return False
    return True
```

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture and data flow
- [pathfinder_algorithm.md](pathfinder_algorithm.md) — PathFinder routing algorithm
- [tuning_guide.md](tuning_guide.md) — Parameter tuning (grid pitch, via costs)
- [metal_kernel_internals.md](metal_kernel_internals.md) — GPU kernel buffer layout
