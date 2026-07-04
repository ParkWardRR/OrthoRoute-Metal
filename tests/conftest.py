"""Shared pytest fixtures for OrthoRoute test suite."""
import pytest
import numpy as np

from orthoroute.domain.models.board import Board, Component, Net, Pad, Layer, Coordinate, Bounds
from orthoroute.domain.models.constraints import DRCConstraints, NetClass
from orthoroute.algorithms.manhattan.pathfinder.config import PathFinderConfig
from orthoroute.algorithms.manhattan.real_global_grid import GridShape


# ---------------------------------------------------------------------------
# Board fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_board():
    """Create a minimal Board with 2 layers, 2 components, 2 pads, 1 net.

    Layout (all dimensions in mm):
        - Board: ~10mm x 10mm implied by component positions
        - C1 at (1, 1) with pad P1 connected to net N1
        - C2 at (8, 8) with pad P2 connected to net N1
    """
    pad1 = Pad(
        id="P1",
        component_id="C1",
        net_id="N1",
        position=Coordinate(1.0, 1.0),
        size=(0.5, 0.5),
        layer="F.Cu",
    )
    pad2 = Pad(
        id="P2",
        component_id="C2",
        net_id="N1",
        position=Coordinate(8.0, 8.0),
        size=(0.5, 0.5),
        layer="F.Cu",
    )
    comp1 = Component(
        id="C1",
        reference="U1",
        value="IC1",
        footprint="QFP-32",
        position=Coordinate(1.0, 1.0),
        pads=[pad1],
    )
    comp2 = Component(
        id="C2",
        reference="R1",
        value="10k",
        footprint="0402",
        position=Coordinate(8.0, 8.0),
        pads=[pad2],
    )
    net1 = Net(id="N1", name="VCC", pads=[pad1, pad2])

    layer_f = Layer(name="F.Cu", type="signal", stackup_position=0)
    layer_b = Layer(name="B.Cu", type="signal", stackup_position=1)

    board = Board(
        id="test-board",
        name="TestBoard",
        components=[comp1, comp2],
        nets=[net1],
        layers=[layer_f, layer_b],
        layer_count=2,
    )
    return board


@pytest.fixture
def sample_board_multilayer():
    """Create a 6-layer board with multiple nets for integration-style tests.

    Layers: F.Cu, In1.Cu, In2.Cu, In3.Cu, In4.Cu, B.Cu
    Components: 4 components, 8 pads, 3 nets
    """
    layers = [
        Layer(name="F.Cu", type="signal", stackup_position=0),
        Layer(name="In1.Cu", type="signal", stackup_position=1),
        Layer(name="In2.Cu", type="signal", stackup_position=2),
        Layer(name="In3.Cu", type="signal", stackup_position=3),
        Layer(name="In4.Cu", type="signal", stackup_position=4),
        Layer(name="B.Cu", type="signal", stackup_position=5),
    ]

    pads = []
    for i in range(8):
        pads.append(Pad(
            id=f"P{i}",
            component_id=f"C{i // 2}",
            net_id=None,  # Will be set by Net.__post_init__
            position=Coordinate(2.0 * (i % 4), 2.0 * (i // 4)),
            size=(0.6, 0.6),
            layer="F.Cu",
        ))

    components = [
        Component(
            id="C0", reference="U1", value="MCU",
            footprint="BGA-256",
            position=Coordinate(1.0, 1.0),
            pads=[pads[0], pads[1]],
        ),
        Component(
            id="C1", reference="U2", value="FPGA",
            footprint="BGA-484",
            position=Coordinate(4.0, 1.0),
            pads=[pads[2], pads[3]],
        ),
        Component(
            id="C2", reference="C1", value="100nF",
            footprint="0402",
            position=Coordinate(1.0, 4.0),
            pads=[pads[4], pads[5]],
        ),
        Component(
            id="C3", reference="R1", value="4k7",
            footprint="0402",
            position=Coordinate(4.0, 4.0),
            pads=[pads[6], pads[7]],
        ),
    ]

    nets = [
        Net(id="N1", name="VCC", pads=[pads[0], pads[4]]),
        Net(id="N2", name="GND", pads=[pads[1], pads[5]]),
        Net(id="N3", name="DATA", pads=[pads[2], pads[6], pads[3], pads[7]]),
    ]

    board = Board(
        id="test-multilayer",
        name="MultiLayerBoard",
        components=components,
        nets=nets,
        layers=layers,
        layer_count=6,
    )
    return board


# ---------------------------------------------------------------------------
# Graph / grid fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_grid_shape():
    """A small GridShape (2 layers, 10x10) for unit tests."""
    return GridShape(NL=2, NX=10, NY=10)


@pytest.fixture
def simple_csr_graph():
    """Build a small CSR graph with 10 nodes arranged in a line.

    Node connectivity: 0-1-2-3-4-5-6-7-8-9
    All edge weights = 1.
    """
    n = 10
    row_ptr = [0]
    col_idx = []
    weights = []

    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < n - 1:
            neighbors.append(i + 1)
        col_idx.extend(neighbors)
        weights.extend([1.0] * len(neighbors))
        row_ptr.append(len(col_idx))

    return {
        "row_ptr": np.array(row_ptr, dtype=np.int32),
        "col_idx": np.array(col_idx, dtype=np.int32),
        "weights": np.array(weights, dtype=np.float32),
        "n_nodes": n,
    }


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pathfinder_config():
    """Default PathFinderConfig for unit tests."""
    return PathFinderConfig()


@pytest.fixture
def drc_constraints():
    """Default DRCConstraints for unit tests."""
    return DRCConstraints()
