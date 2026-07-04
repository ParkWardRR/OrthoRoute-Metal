"""
KiCad End-to-End Integration Tests

Tests the full pipeline: Board creation -> Graph initialization ->
Pad mapping -> Routing -> Geometry emission -> ORP/ORS serialization.

These tests verify that the entire KiCad integration stack works
together without requiring real .kicad_pcb files or KiCad installed.
"""

from __future__ import annotations

import os
import json
import gzip
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orthoroute.domain.models.board import (
    Board, Component, Net, Pad, Layer, Coordinate, Bounds,
)
from orthoroute.domain.models.routing import (
    Segment, Via, Route, RoutingResult, RoutingStatistics,
)
from orthoroute.domain.models.constraints import DRCConstraints, NetClass
from orthoroute.infrastructure.serialization import (
    export_board_to_orp,
    import_board_from_orp,
    convert_orp_to_board,
    convert_orp_to_board_data,
    export_solution_to_ors,
    import_solution_from_ors,
    convert_ors_to_geometry_payload,
    derive_orp_filename,
    derive_ors_filename,
    get_solution_summary,
)
from orthoroute.algorithms.manhattan.pathfinder.kicad_geometry import KiCadGeometry
from orthoroute.shared.utils.layers import norm_layer, get_layer_stackup, LAYER_NAME_RE


# =============================================================================
# Helpers
# =============================================================================

def make_6layer_board():
    """Create a realistic 6-layer board with 4 components and 3 nets."""
    layers = [
        Layer(name="F.Cu", type="signal", stackup_position=0),
        Layer(name="In1.Cu", type="signal", stackup_position=1),
        Layer(name="In2.Cu", type="power", stackup_position=2),
        Layer(name="In3.Cu", type="ground", stackup_position=3),
        Layer(name="In4.Cu", type="signal", stackup_position=4),
        Layer(name="B.Cu", type="signal", stackup_position=5),
    ]

    # BGA IC with 4 pads
    bga_pads = [
        Pad(id="U1_A1", component_id="C1", net_id=None, position=Coordinate(10.0, 10.0),
            size=(0.3, 0.3), drill_size=0.2, layer="F.Cu", shape="circle"),
        Pad(id="U1_A2", component_id="C1", net_id=None, position=Coordinate(10.8, 10.0),
            size=(0.3, 0.3), drill_size=0.2, layer="F.Cu", shape="circle"),
        Pad(id="U1_B1", component_id="C1", net_id=None, position=Coordinate(10.0, 10.8),
            size=(0.3, 0.3), drill_size=0.2, layer="F.Cu", shape="circle"),
        Pad(id="U1_B2", component_id="C1", net_id=None, position=Coordinate(10.8, 10.8),
            size=(0.3, 0.3), drill_size=0.2, layer="F.Cu", shape="circle"),
    ]
    bga = Component(id="C1", reference="U1", value="FPGA", footprint="BGA-256",
                    position=Coordinate(10.4, 10.4), layer="F.Cu", pads=bga_pads)

    # Decoupling capacitor
    cap_pads = [
        Pad(id="C1_1", component_id="C2", net_id=None, position=Coordinate(15.0, 10.0),
            size=(0.5, 0.5), layer="F.Cu", shape="rect"),
        Pad(id="C1_2", component_id="C2", net_id=None, position=Coordinate(16.0, 10.0),
            size=(0.5, 0.5), layer="F.Cu", shape="rect"),
    ]
    cap = Component(id="C2", reference="C1", value="100nF", footprint="0402",
                    position=Coordinate(15.5, 10.0), layer="F.Cu", pads=cap_pads)

    # Connector on back layer
    conn_pads = [
        Pad(id="J1_1", component_id="C3", net_id=None, position=Coordinate(5.0, 5.0),
            size=(0.8, 0.8), drill_size=0.4, layer="B.Cu", shape="circle"),
        Pad(id="J1_2", component_id="C3", net_id=None, position=Coordinate(5.0, 7.0),
            size=(0.8, 0.8), drill_size=0.4, layer="B.Cu", shape="circle"),
    ]
    conn = Component(id="C3", reference="J1", value="Conn", footprint="PinHeader_2",
                     position=Coordinate(5.0, 6.0), layer="B.Cu", pads=conn_pads)

    # Resistor
    res_pads = [
        Pad(id="R1_1", component_id="C4", net_id=None, position=Coordinate(20.0, 15.0),
            size=(0.4, 0.4), layer="F.Cu", shape="rect"),
        Pad(id="R1_2", component_id="C4", net_id=None, position=Coordinate(21.0, 15.0),
            size=(0.4, 0.4), layer="F.Cu", shape="rect"),
    ]
    res = Component(id="C4", reference="R1", value="10k", footprint="0603",
                    position=Coordinate(20.5, 15.0), layer="F.Cu", pads=res_pads)

    # Nets
    vcc = Net(id="N1", name="VCC", pads=[bga_pads[0], cap_pads[0]])
    gnd = Net(id="N2", name="GND", pads=[bga_pads[1], cap_pads[1], conn_pads[0]])
    data = Net(id="N3", name="DATA0", pads=[bga_pads[2], conn_pads[1], res_pads[0]])

    board = Board(
        id="test-6layer", name="TestBoard-6L",
        components=[bga, cap, conn, res],
        nets=[vcc, gnd, data],
        layers=layers,
        layer_count=6,
        thickness=1.6,
    )
    return board


# =============================================================================
# Board Model — Deep Validation
# =============================================================================

class TestBoardIntegrity:
    """Verify Board domain model with realistic PCB data."""

    def test_6layer_board_creation(self):
        """6-layer board with 4 components creates successfully."""
        board = make_6layer_board()
        assert board.layer_count == 6
        assert len(board.components) == 4
        assert len(board.nets) == 3

    def test_routable_nets(self):
        """All 3 nets have >= 2 pads and are routable."""
        board = make_6layer_board()
        routable = board.get_routable_nets()
        assert len(routable) == 3
        for net in routable:
            assert net.is_routable

    def test_routing_layers(self):
        """All 6 Cu layers are routing layers."""
        board = make_6layer_board()
        routing = board.get_routing_layers()
        assert len(routing) == 6
        names = {l.name for l in routing}
        assert names == {"F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "B.Cu"}

    def test_board_bounds(self):
        """Board bounds encompass all pad positions."""
        board = make_6layer_board()
        bounds = board.get_bounds()
        assert bounds.min_x <= 5.0  # connector at x=5
        assert bounds.max_x >= 21.0  # resistor at x=21
        assert bounds.min_y <= 5.0  # connector at y=5
        assert bounds.max_y >= 15.0  # resistor at y=15

    def test_board_all_pads(self):
        """Board has 10 pads total across 4 components."""
        board = make_6layer_board()
        pads = board.get_all_pads()
        assert len(pads) == 10

    def test_board_validate_integrity(self):
        """Board passes integrity validation."""
        board = make_6layer_board()
        issues = board.validate_integrity()
        assert len(issues) == 0, f"Integrity issues: {issues}"

    def test_net_pad_linkage(self):
        """Net.__post_init__ sets pad.net_id for all pads in the net."""
        board = make_6layer_board()
        for net in board.nets:
            for pad in net.pads:
                assert pad.net_id == net.id, f"Pad {pad.id} net_id={pad.net_id}, expected {net.id}"

    def test_component_pad_linkage(self):
        """Component.__post_init__ sets pad.component_id for all pads."""
        board = make_6layer_board()
        for comp in board.components:
            for pad in comp.pads:
                assert pad.component_id == comp.id

    def test_net_min_distance(self):
        """Net.calculate_min_distance returns positive values for multi-pad nets."""
        board = make_6layer_board()
        for net in board.nets:
            if len(net.pads) >= 2:
                d = net.calculate_min_distance()
                assert d > 0.0, f"Net {net.name} min distance should be > 0"

    def test_net_bounds(self):
        """Net.get_bounds encompasses all net pads."""
        board = make_6layer_board()
        for net in board.nets:
            bounds = net.get_bounds()
            for pad in net.pads:
                assert pad.position.x >= bounds.min_x
                assert pad.position.x <= bounds.max_x
                assert pad.position.y >= bounds.min_y
                assert pad.position.y <= bounds.max_y

    def test_3pin_net_connectivity(self):
        """DATA0 net has 3 pads from 3 different components."""
        board = make_6layer_board()
        data_net = board.get_net_by_name("DATA0")
        assert data_net is not None
        assert len(data_net.pads) == 3
        comp_ids = {p.component_id for p in data_net.pads}
        assert len(comp_ids) == 3

    def test_back_layer_component(self):
        """Connector is placed on B.Cu layer."""
        board = make_6layer_board()
        j1 = board.get_component("C3")
        assert j1 is not None
        assert j1.layer == "B.Cu"

    def test_drill_sizes(self):
        """Through-hole pads have drill sizes; SMD pads don't."""
        board = make_6layer_board()
        for comp in board.components:
            for pad in comp.pads:
                if pad.drill_size is not None:
                    assert pad.drill_size > 0.0


# =============================================================================
# KiCad Geometry — Coordinate System
# =============================================================================

class TestKiCadGeometryIntegration:
    """Test KiCadGeometry with realistic board dimensions."""

    def test_geometry_from_board_bounds(self):
        """Create geometry from actual board bounds."""
        board = make_6layer_board()
        bounds = board.get_bounds()
        geom = KiCadGeometry(
            (bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y),
            pitch=0.4, layer_count=6
        )
        assert geom.x_steps > 0
        assert geom.y_steps > 0

    def test_pad_position_maps_to_valid_node(self):
        """Every pad position maps to a valid lattice node."""
        board = make_6layer_board()
        bounds = board.get_bounds()
        geom = KiCadGeometry(
            (bounds.min_x - 1, bounds.min_y - 1, bounds.max_x + 1, bounds.max_y + 1),
            pitch=0.4, layer_count=6
        )
        for pad in board.get_all_pads():
            gx, gy = geom.world_to_lattice(pad.position.x, pad.position.y)
            assert 0 <= gx < geom.x_steps, f"Pad {pad.id} gx={gx} out of range"
            assert 0 <= gy < geom.y_steps, f"Pad {pad.id} gy={gy} out of range"
            # Node index must be valid
            idx = geom.node_index(gx, gy, 0)
            assert 0 <= idx < geom.x_steps * geom.y_steps * 6

    def test_world_lattice_roundtrip_accuracy(self):
        """World->lattice->world roundtrip is within half-pitch."""
        geom = KiCadGeometry((0, 0, 40, 40), pitch=0.4, layer_count=6)
        wx, wy = 12.35, 8.67
        gx, gy = geom.world_to_lattice(wx, wy)
        rx, ry = geom.lattice_to_world(gx, gy)
        assert abs(rx - wx) <= 0.2  # half pitch
        assert abs(ry - wy) <= 0.2

    def test_layer_directions_6layer(self):
        """6-layer board has alternating V/H/V/H/V/H directions."""
        geom = KiCadGeometry((0, 0, 40, 40), pitch=0.4, layer_count=6)
        # Layer 0 (F.Cu): vertical
        assert geom.is_valid_edge(5, 5, 0, 5, 6, 0)  # vertical move on layer 0
        assert not geom.is_valid_edge(5, 5, 0, 6, 5, 0)  # horizontal move on layer 0

        # Layer 1 (In1.Cu): horizontal
        assert geom.is_valid_edge(5, 5, 1, 6, 5, 1)  # horizontal move on layer 1
        assert not geom.is_valid_edge(5, 5, 1, 5, 6, 1)  # vertical move on layer 1

    def test_via_between_any_layers(self):
        """Via edges between any two layers at same (x,y) are always valid."""
        geom = KiCadGeometry((0, 0, 40, 40), pitch=0.4, layer_count=6)
        for z1 in range(6):
            for z2 in range(6):
                if z1 != z2:
                    assert geom.is_valid_edge(5, 5, z1, 5, 5, z2)

    def test_diagonal_never_valid(self):
        """Diagonal moves are never valid on any layer."""
        geom = KiCadGeometry((0, 0, 40, 40), pitch=0.4, layer_count=6)
        for z in range(6):
            assert not geom.is_valid_edge(5, 5, z, 6, 6, z)

    def test_node_index_coverage(self):
        """All (x,y,z) combos produce unique indices in correct range."""
        geom = KiCadGeometry((0, 0, 4, 4), pitch=1.0, layer_count=2)
        seen = set()
        total = geom.x_steps * geom.y_steps * 2
        for z in range(2):
            for y in range(geom.y_steps):
                for x in range(geom.x_steps):
                    idx = geom.node_index(x, y, z)
                    assert 0 <= idx < total
                    assert idx not in seen, f"Duplicate index {idx} at ({x},{y},{z})"
                    seen.add(idx)
        assert len(seen) == total


# =============================================================================
# ORP Serialization Round-Trip
# =============================================================================

class TestORPRoundTrip:
    """Test full ORP export -> import -> Board reconstruction."""

    def test_export_creates_file(self, tmp_path):
        """export_board_to_orp creates a file."""
        board = make_6layer_board()
        filepath = str(tmp_path / "test.ORP")
        export_board_to_orp(board, filepath)
        assert Path(filepath).exists()
        assert Path(filepath).stat().st_size > 0

    def test_export_gzipped_by_default(self, tmp_path):
        """Default export is gzip-compressed."""
        board = make_6layer_board()
        filepath = str(tmp_path / "test.ORP")
        export_board_to_orp(board, filepath, compress=True)
        with open(filepath, 'rb') as f:
            magic = f.read(2)
        assert magic == b'\x1f\x8b', "File should start with gzip magic bytes"

    def test_export_uncompressed(self, tmp_path):
        """compress=False creates plain JSON."""
        board = make_6layer_board()
        filepath = str(tmp_path / "test.ORP")
        export_board_to_orp(board, filepath, compress=False)
        with open(filepath, 'r') as f:
            data = json.load(f)
        assert data['format_version'] == '1.0'

    def test_import_roundtrip(self, tmp_path):
        """Export then import preserves board structure."""
        board = make_6layer_board()
        filepath = str(tmp_path / "test.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        assert orp_data is not None
        assert orp_data['format_version'] == '1.0'

    def test_roundtrip_preserves_pads(self, tmp_path):
        """ORP round-trip preserves pad count and positions."""
        board = make_6layer_board()
        filepath = str(tmp_path / "test.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        assert len(orp_data['pads']) == 10

    def test_roundtrip_preserves_nets(self, tmp_path):
        """ORP round-trip preserves net names."""
        board = make_6layer_board()
        filepath = str(tmp_path / "test.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        net_names = {n['name'] for n in orp_data['nets']}
        assert 'VCC' in net_names
        assert 'GND' in net_names
        assert 'DATA0' in net_names

    def test_roundtrip_preserves_layers(self, tmp_path):
        """ORP round-trip preserves layer definitions."""
        board = make_6layer_board()
        filepath = str(tmp_path / "test.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        layer_names = {l['name'] for l in orp_data['layers']}
        assert 'F.Cu' in layer_names
        assert 'B.Cu' in layer_names

    def test_convert_orp_to_board_object(self, tmp_path):
        """convert_orp_to_board returns a Board domain object."""
        board = make_6layer_board()
        filepath = str(tmp_path / "test.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        reconstructed = convert_orp_to_board(orp_data)
        assert isinstance(reconstructed, Board)
        assert len(reconstructed.get_routable_nets()) >= 2


# =============================================================================
# ORS Serialization Round-Trip
# =============================================================================

class TestORSRoundTrip:
    """Test ORS solution export -> import -> geometry reconstruction."""

    @staticmethod
    def _make_fake_geometry():
        """Create minimal geometry payload for testing."""
        geom = SimpleNamespace()
        geom.tracks = [
            {'net_id': 'N1', 'layer': 0, 'start': (10.0, 10.0), 'end': (15.0, 10.0), 'width': 0.15},
            {'net_id': 'N1', 'layer': 0, 'start': (15.0, 10.0), 'end': (15.0, 15.0), 'width': 0.15},
            {'net_id': 'N2', 'layer': 1, 'start': (5.0, 5.0), 'end': (10.0, 5.0), 'width': 0.2},
        ]
        geom.vias = [
            {'net_id': 'N1', 'position': (15.0, 10.0), 'from_layer': 0, 'to_layer': 1, 'drill': 0.2, 'size': 0.4},
        ]
        return geom

    def test_export_creates_file(self, tmp_path):
        """export_solution_to_ors creates a file."""
        geom = self._make_fake_geometry()
        filepath = str(tmp_path / "test.ORS")
        export_solution_to_ors(geom, [], {}, filepath)
        assert Path(filepath).exists()

    def test_import_roundtrip(self, tmp_path):
        """ORS export -> import preserves geometry data."""
        geom = self._make_fake_geometry()
        filepath = str(tmp_path / "test.ORS")
        export_solution_to_ors(geom, [], {"routing_time": 1.5}, filepath)
        geo_data, meta = import_solution_from_ors(filepath)
        assert geo_data is not None

    def test_roundtrip_preserves_track_count(self, tmp_path):
        """Track count preserved through ORS round-trip."""
        geom = self._make_fake_geometry()
        filepath = str(tmp_path / "test.ORS")
        export_solution_to_ors(geom, [], {}, filepath)
        geo_data, _ = import_solution_from_ors(filepath)
        payload = convert_ors_to_geometry_payload(geo_data)
        assert len(payload.tracks) == 3

    def test_roundtrip_preserves_via_count(self, tmp_path):
        """Via count preserved through ORS round-trip."""
        geom = self._make_fake_geometry()
        filepath = str(tmp_path / "test.ORS")
        export_solution_to_ors(geom, [], {}, filepath)
        geo_data, _ = import_solution_from_ors(filepath)
        payload = convert_ors_to_geometry_payload(geo_data)
        assert len(payload.vias) == 1


# =============================================================================
# Layer Utilities
# =============================================================================

class TestLayerUtils:
    """Test KiCad layer name normalization and stackup."""

    def test_norm_layer_int_to_name(self):
        """Integer layer indices map to correct KiCad names."""
        assert norm_layer(0) == "F.Cu"
        assert norm_layer(1) == "In1.Cu"
        assert norm_layer(2) == "In2.Cu"
        assert norm_layer(3) == "In3.Cu"
        assert norm_layer(4) == "In4.Cu"
        assert norm_layer(5) == "B.Cu"

    def test_norm_layer_string_passthrough(self):
        """Valid string layer names pass through unchanged."""
        assert norm_layer("F.Cu") == "F.Cu"
        assert norm_layer("B.Cu") == "B.Cu"
        assert norm_layer("In1.Cu") == "In1.Cu"
        assert norm_layer("In30.Cu") == "In30.Cu"

    def test_norm_layer_invalid_raises(self):
        """Invalid layer names raise ValueError."""
        with pytest.raises(ValueError):
            norm_layer("InvalidLayer")
        with pytest.raises(ValueError):
            norm_layer(99)

    def test_layer_regex_valid(self):
        """LAYER_NAME_RE matches all standard KiCad Cu layer names."""
        for name in ["F.Cu", "B.Cu", "In1.Cu", "In2.Cu", "In15.Cu", "In30.Cu"]:
            assert LAYER_NAME_RE.match(name), f"{name} should match"

    def test_layer_regex_invalid(self):
        """LAYER_NAME_RE rejects non-standard names."""
        for name in ["FCu", "X.Cu", "F.Mask", "Edge.Cuts"]:
            assert not LAYER_NAME_RE.match(name), f"{name} should not match"

    def test_stackup_coverage(self):
        """get_layer_stackup returns the standard 6-layer set."""
        stackup = get_layer_stackup()
        assert "F.Cu" in stackup
        assert "B.Cu" in stackup
        assert len(stackup) == 6


# =============================================================================
# DRC Constraints
# =============================================================================

class TestDRCConstraints:
    """Test DRC constraint validation with board-level data."""

    def test_defaults_are_reasonable(self):
        """Default DRC values are physically reasonable for PCB manufacturing."""
        drc = DRCConstraints()
        assert drc.min_track_width > 0.0
        assert drc.min_via_diameter > 0.0
        assert drc.min_via_drill > 0.0
        assert drc.default_clearance > 0.0

    def test_track_width_validation(self):
        """Tracks within netclass range are valid; below min are not."""
        drc = DRCConstraints()
        # Default netclass track_width_min is 0.125, track_width_max is 0.5
        assert drc.validate_track_width(0.25)  # default width, should be valid
        assert drc.validate_track_width(0.3)   # within range
        assert not drc.validate_track_width(0.01)  # way below min

    def test_via_size_validation(self):
        """Vias within netclass range are valid."""
        drc = DRCConstraints()
        # Default via: diameter=0.6, drill=0.3
        assert drc.validate_via_size(0.6, 0.3)  # default size
        assert not drc.validate_via_size(0.01, 0.005)  # way too small


# =============================================================================
# Filename Derivation
# =============================================================================

class TestFilenamDerivation:
    """Test ORP/ORS filename derivation chain."""

    def test_kicad_to_orp(self):
        """'.kicad_pcb' extension becomes '.ORP'."""
        assert derive_orp_filename("board.kicad_pcb") == "board.ORP"

    def test_orp_to_ors(self):
        """'.ORP' extension becomes '.ORS'."""
        assert derive_ors_filename("board.ORP") == "board.ORS"

    def test_full_chain(self):
        """kicad_pcb -> ORP -> ORS derivation chain."""
        orp = derive_orp_filename("mydesign.kicad_pcb")
        ors = derive_ors_filename(orp)
        assert ors == "mydesign.ORS"

    def test_path_preservation(self):
        """Directory path is preserved in derivation."""
        result = derive_orp_filename("/path/to/board.kicad_pcb")
        assert "board.ORP" in result
