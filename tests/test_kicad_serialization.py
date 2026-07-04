"""Tests for ORP and ORS serialization round-trips.

Covers: ORP export/import (gzipped and plain), format version validation,
pad/net/layer preservation, Board domain object conversion, ORS export/import
with geometry and metrics, geometry payload conversion, utility functions
(filename derivation, solution summary generation).
"""
import gzip
import json

import pytest
from pathlib import Path

from orthoroute.domain.models.board import (
    Board, Component, Net, Pad, Layer, Coordinate,
)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeGeometry:
    """Fake geometry payload for ORS tests."""

    def __init__(self):
        self.tracks = [
            {
                "net_id": "N1", "layer": 0,
                "start": (1.0, 1.0), "end": (5.0, 1.0), "width": 0.15,
            },
            {
                "net_id": "N1", "layer": 0,
                "start": (5.0, 1.0), "end": (5.0, 5.0), "width": 0.15,
            },
            {
                "net_id": "N2", "layer": 1,
                "start": (2.0, 2.0), "end": (8.0, 2.0), "width": 0.25,
            },
        ]
        self.vias = [
            {
                "net_id": "N1", "position": (5.0, 1.0),
                "from_layer": 0, "to_layer": 1, "drill": 0.2, "size": 0.4,
            },
        ]


def _make_sample_board():
    """Create a sample Board for ORP tests."""
    pad1 = Pad(
        id="U1_1", component_id="U1", net_id=None,
        position=Coordinate(1.0, 1.0), size=(0.6, 0.3),
        layer="F.Cu", shape="rect",
    )
    pad2 = Pad(
        id="U1_2", component_id="U1", net_id=None,
        position=Coordinate(2.0, 1.0), size=(0.6, 0.3),
        layer="F.Cu", shape="rect",
    )
    pad3 = Pad(
        id="R1_1", component_id="R1", net_id=None,
        position=Coordinate(5.0, 5.0), size=(0.4, 0.3),
        layer="F.Cu", shape="rect",
    )
    pad4 = Pad(
        id="R1_2", component_id="R1", net_id=None,
        position=Coordinate(6.0, 5.0), size=(0.4, 0.3),
        layer="F.Cu", shape="rect",
    )

    comp1 = Component(
        id="U1", reference="U1", value="IC1", footprint="QFP-32",
        position=Coordinate(1.5, 1.0), pads=[pad1, pad2],
    )
    comp2 = Component(
        id="R1", reference="R1", value="10k", footprint="0402",
        position=Coordinate(5.5, 5.0), pads=[pad3, pad4],
    )

    net1 = Net(id="N1", name="VCC", pads=[pad1, pad3])
    net2 = Net(id="N2", name="GND", pads=[pad2, pad4])

    layer_f = Layer(name="F.Cu", type="signal", stackup_position=0)
    layer_b = Layer(name="B.Cu", type="signal", stackup_position=1)

    return Board(
        id="test-board", name="TestBoard",
        components=[comp1, comp2],
        nets=[net1, net2],
        layers=[layer_f, layer_b],
        layer_count=2,
    )


def _make_sample_metadata():
    """Sample metadata dict for ORS export."""
    return {
        "total_iterations": 10,
        "converged": True,
        "total_time": 5.0,
        "final_wirelength": 100.0,
        "final_via_count": 3,
        "final_overflow": 0,
        "board_name": "TestBoard",
    }


def _make_sample_metrics():
    """Sample per-iteration metrics list for ORS export."""
    return [
        {
            "iteration": 1, "overuse_count": 5, "nets_routed": 2,
            "overflow_cost": 10.0, "wirelength": 80.0,
            "via_count": 2, "iteration_time": 0.5,
        },
        {
            "iteration": 2, "overuse_count": 0, "nets_routed": 2,
            "overflow_cost": 0.0, "wirelength": 100.0,
            "via_count": 3, "iteration_time": 0.3,
        },
    ]


# ---------------------------------------------------------------------------
# ORP Export Tests
# ---------------------------------------------------------------------------


class TestORPExport:
    """Tests for export_board_to_orp."""

    def test_export_board_to_orp_creates_file(self, tmp_path):
        """Export creates .ORP file on disk."""
        board = _make_sample_board()
        filepath = str(tmp_path / "test.ORP")
        export_board_to_orp(board, filepath)
        assert Path(filepath).exists()

    def test_export_board_to_orp_gzipped(self, tmp_path):
        """Default export is gzip-compressed."""
        board = _make_sample_board()
        filepath = str(tmp_path / "compressed.ORP")
        export_board_to_orp(board, filepath, compress=True)
        with gzip.open(filepath, "rt") as f:
            data = json.load(f)
        assert "format_version" in data

    def test_export_board_to_orp_uncompressed(self, tmp_path):
        """compress=False creates plain JSON readable without gzip."""
        board = _make_sample_board()
        filepath = str(tmp_path / "plain.ORP")
        export_board_to_orp(board, filepath, compress=False)
        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["format_version"] == "1.0"

    def test_export_board_to_orp_creates_parent_dirs(self, tmp_path):
        """Export creates parent directories if they don't exist."""
        board = _make_sample_board()
        filepath = str(tmp_path / "sub" / "dir" / "test.ORP")
        export_board_to_orp(board, filepath)
        assert Path(filepath).exists()


# ---------------------------------------------------------------------------
# ORP Import Tests
# ---------------------------------------------------------------------------


class TestORPImport:
    """Tests for import_board_from_orp."""

    def test_import_board_from_orp_roundtrip(self, tmp_path):
        """export → import yields valid ORP data dict."""
        board = _make_sample_board()
        filepath = str(tmp_path / "roundtrip.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        assert isinstance(orp_data, dict)
        assert "pads" in orp_data
        assert "nets" in orp_data

    def test_import_board_from_orp_format_version(self, tmp_path):
        """Imported data has format_version '1.0'."""
        board = _make_sample_board()
        filepath = str(tmp_path / "version.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        assert orp_data["format_version"] == "1.0"

    def test_import_board_from_orp_preserves_pads(self, tmp_path):
        """Pad positions, sizes, and layers are preserved."""
        board = _make_sample_board()
        filepath = str(tmp_path / "pads.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        pads = orp_data["pads"]
        assert len(pads) == 4
        pad_u1_1 = next(p for p in pads if p["id"] == "U1_1")
        assert pad_u1_1["position"]["x"] == pytest.approx(1.0)
        assert pad_u1_1["position"]["y"] == pytest.approx(1.0)
        assert pad_u1_1["size"]["width"] == pytest.approx(0.6)
        assert pad_u1_1["layer"] == "F.Cu"

    def test_import_board_from_orp_preserves_nets(self, tmp_path):
        """Net names and terminal pad IDs are preserved."""
        board = _make_sample_board()
        filepath = str(tmp_path / "nets.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        nets = orp_data["nets"]
        assert len(nets) == 2
        net_names = {n["name"] for n in nets}
        assert "VCC" in net_names
        assert "GND" in net_names

    def test_import_board_from_orp_preserves_layers(self, tmp_path):
        """Layer names and types are preserved."""
        board = _make_sample_board()
        filepath = str(tmp_path / "layers.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        layers = orp_data["layers"]
        assert len(layers) == 2
        layer_names = {lay["name"] for lay in layers}
        assert "F.Cu" in layer_names
        assert "B.Cu" in layer_names

    def test_import_board_from_orp_nonexistent_raises(self, tmp_path):
        """import_board_from_orp raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            import_board_from_orp(str(tmp_path / "missing.ORP"))

    def test_import_board_from_orp_auto_detects_compression(self, tmp_path):
        """import_board_from_orp auto-detects both gzip and plain JSON."""
        board = _make_sample_board()
        # Test gzipped
        gz_path = str(tmp_path / "gz.ORP")
        export_board_to_orp(board, gz_path, compress=True)
        data_gz = import_board_from_orp(gz_path)
        # Test plain
        plain_path = str(tmp_path / "plain.ORP")
        export_board_to_orp(board, plain_path, compress=False)
        data_plain = import_board_from_orp(plain_path)
        # Both should have the same structure
        assert data_gz["format_version"] == data_plain["format_version"]
        assert len(data_gz["pads"]) == len(data_plain["pads"])


# ---------------------------------------------------------------------------
# ORP Conversion Tests
# ---------------------------------------------------------------------------


class TestORPConversion:
    """Tests for convert_orp_to_board and convert_orp_to_board_data."""

    def test_convert_orp_to_board(self, tmp_path):
        """convert_orp_to_board() returns Board domain object."""
        board = _make_sample_board()
        filepath = str(tmp_path / "convert.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        result = convert_orp_to_board(orp_data)
        assert isinstance(result, Board)
        assert result.name == "TestBoard"

    def test_convert_orp_to_board_pad_count(self, tmp_path):
        """Converted board has correct pad count."""
        board = _make_sample_board()
        filepath = str(tmp_path / "padcount.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        result = convert_orp_to_board(orp_data)
        all_pads = result.get_all_pads()
        assert len(all_pads) == 4

    def test_convert_orp_to_board_net_count(self, tmp_path):
        """Converted board has correct routable net count."""
        board = _make_sample_board()
        filepath = str(tmp_path / "netcount.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        result = convert_orp_to_board(orp_data)
        routable = result.get_routable_nets()
        assert len(routable) == 2

    def test_convert_orp_to_board_data(self, tmp_path):
        """convert_orp_to_board_data returns dict with expected keys."""
        board = _make_sample_board()
        filepath = str(tmp_path / "boarddata.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        bd = convert_orp_to_board_data(orp_data)
        assert isinstance(bd, dict)
        for key in ["filename", "pads", "nets", "layers", "layer_names",
                     "clearance", "track_width", "via_diameter", "via_drill",
                     "grid_resolution", "bounds"]:
            assert key in bd, f"Missing key: {key}"

    def test_convert_orp_to_board_data_pad_format(self, tmp_path):
        """Converted board_data pads have flat x, y, width, height keys."""
        board = _make_sample_board()
        filepath = str(tmp_path / "padfmt.ORP")
        export_board_to_orp(board, filepath)
        orp_data = import_board_from_orp(filepath)
        bd = convert_orp_to_board_data(orp_data)
        for pad in bd["pads"]:
            assert "x" in pad
            assert "y" in pad
            assert "width" in pad
            assert "height" in pad


# ---------------------------------------------------------------------------
# ORS Export Tests
# ---------------------------------------------------------------------------


class TestORSExport:
    """Tests for export_solution_to_ors."""

    def test_export_solution_to_ors_creates_file(self, tmp_path):
        """Export creates .ORS file on disk."""
        filepath = str(tmp_path / "solution.ORS")
        export_solution_to_ors(
            FakeGeometry(), _make_sample_metrics(),
            _make_sample_metadata(), filepath,
        )
        assert Path(filepath).exists()

    def test_export_solution_to_ors_gzipped(self, tmp_path):
        """Default export is gzip-compressed."""
        filepath = str(tmp_path / "compressed.ORS")
        export_solution_to_ors(
            FakeGeometry(), _make_sample_metrics(),
            _make_sample_metadata(), filepath,
        )
        with gzip.open(filepath, "rt") as f:
            data = json.load(f)
        assert data["format_version"] == "1.0"

    def test_export_solution_to_ors_uncompressed(self, tmp_path):
        """compress=False creates plain JSON."""
        filepath = str(tmp_path / "plain.ORS")
        export_solution_to_ors(
            FakeGeometry(), _make_sample_metrics(),
            _make_sample_metadata(), filepath, compress=False,
        )
        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["format_version"] == "1.0"

    def test_export_solution_to_ors_none_geometry_raises(self, tmp_path):
        """Passing None geometry raises ValueError."""
        filepath = str(tmp_path / "none.ORS")
        with pytest.raises(ValueError, match="Geometry cannot be None"):
            export_solution_to_ors(None, [], {}, filepath)

    def test_export_solution_to_ors_missing_attrs_raises(self, tmp_path):
        """Geometry without tracks/vias attributes raises ValueError."""
        filepath = str(tmp_path / "bad.ORS")
        with pytest.raises(ValueError, match="tracks.*vias"):
            export_solution_to_ors(object(), [], {}, filepath)


# ---------------------------------------------------------------------------
# ORS Import Tests
# ---------------------------------------------------------------------------


class TestORSImport:
    """Tests for import_solution_from_ors."""

    def test_import_solution_from_ors_roundtrip(self, tmp_path):
        """export → import yields geometry data structure."""
        filepath = str(tmp_path / "roundtrip.ORS")
        export_solution_to_ors(
            FakeGeometry(), _make_sample_metrics(),
            _make_sample_metadata(), filepath,
        )
        geometry_data, metadata = import_solution_from_ors(filepath)
        assert "by_net" in geometry_data
        assert "all_tracks" in geometry_data
        assert "all_vias" in geometry_data

    def test_import_solution_preserves_tracks(self, tmp_path):
        """Track coordinates preserved after round-trip."""
        filepath = str(tmp_path / "tracks.ORS")
        export_solution_to_ors(
            FakeGeometry(), _make_sample_metrics(),
            _make_sample_metadata(), filepath,
        )
        geometry_data, _ = import_solution_from_ors(filepath)
        all_tracks = geometry_data["all_tracks"]
        assert len(all_tracks) == 3
        n1_tracks = [t for t in all_tracks if t["net_id"] == "N1"]
        assert len(n1_tracks) == 2

    def test_import_solution_preserves_vias(self, tmp_path):
        """Via positions and layers preserved after round-trip."""
        filepath = str(tmp_path / "vias.ORS")
        export_solution_to_ors(
            FakeGeometry(), _make_sample_metrics(),
            _make_sample_metadata(), filepath,
        )
        geometry_data, _ = import_solution_from_ors(filepath)
        all_vias = geometry_data["all_vias"]
        assert len(all_vias) == 1
        via = all_vias[0]
        assert via["position"]["x"] == pytest.approx(5.0)
        assert via["position"]["y"] == pytest.approx(1.0)
        assert via["from_layer"] == 0
        assert via["to_layer"] == 1

    def test_import_solution_metadata(self, tmp_path):
        """Imported metadata contains statistics and iteration_metrics."""
        filepath = str(tmp_path / "meta.ORS")
        export_solution_to_ors(
            FakeGeometry(), _make_sample_metrics(),
            _make_sample_metadata(), filepath,
        )
        _, metadata = import_solution_from_ors(filepath)
        assert "metadata" in metadata
        assert "statistics" in metadata
        assert "iteration_metrics" in metadata

    def test_import_solution_from_ors_nonexistent_raises(self, tmp_path):
        """import_solution_from_ors raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            import_solution_from_ors(str(tmp_path / "missing.ORS"))


# ---------------------------------------------------------------------------
# ORS Geometry Payload Conversion
# ---------------------------------------------------------------------------


class TestORSGeometryPayload:
    """Tests for convert_ors_to_geometry_payload."""

    def test_convert_ors_to_geometry_payload(self, tmp_path):
        """Returns object with .tracks and .vias attributes."""
        filepath = str(tmp_path / "payload.ORS")
        export_solution_to_ors(
            FakeGeometry(), _make_sample_metrics(),
            _make_sample_metadata(), filepath,
        )
        geometry_data, _ = import_solution_from_ors(filepath)
        payload = convert_ors_to_geometry_payload(geometry_data)
        assert hasattr(payload, "tracks")
        assert hasattr(payload, "vias")
        assert len(payload.tracks) == 3
        assert len(payload.vias) == 1

    def test_convert_ors_tracks_have_tuple_coords(self, tmp_path):
        """Converted tracks have tuple start/end coordinates."""
        filepath = str(tmp_path / "tuples.ORS")
        export_solution_to_ors(
            FakeGeometry(), _make_sample_metrics(),
            _make_sample_metadata(), filepath,
        )
        geometry_data, _ = import_solution_from_ors(filepath)
        payload = convert_ors_to_geometry_payload(geometry_data)
        for track in payload.tracks:
            assert isinstance(track["start"], tuple)
            assert isinstance(track["end"], tuple)

    def test_convert_ors_vias_have_tuple_position(self, tmp_path):
        """Converted vias have tuple position."""
        filepath = str(tmp_path / "via_tuples.ORS")
        export_solution_to_ors(
            FakeGeometry(), _make_sample_metrics(),
            _make_sample_metadata(), filepath,
        )
        geometry_data, _ = import_solution_from_ors(filepath)
        payload = convert_ors_to_geometry_payload(geometry_data)
        for via in payload.vias:
            assert isinstance(via["position"], tuple)

    def test_convert_ors_empty_geometry(self):
        """Empty geometry data returns payload with empty lists."""
        geometry_data = {"all_tracks": [], "all_vias": []}
        payload = convert_ors_to_geometry_payload(geometry_data)
        assert len(payload.tracks) == 0
        assert len(payload.vias) == 0


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """Tests for derive_orp_filename, derive_ors_filename, get_solution_summary."""

    def test_derive_orp_filename(self):
        """'board.kicad_pcb' → 'board.ORP'."""
        result = derive_orp_filename("board.kicad_pcb")
        assert result == "board.ORP"

    def test_derive_orp_filename_with_path(self):
        """Full path derivation changes only the suffix."""
        result = derive_orp_filename("/path/to/board.kicad_pcb")
        assert result.endswith("board.ORP")
        assert "/path/to/" in result

    def test_derive_ors_filename(self):
        """'board.ORP' → 'board.ORS'."""
        result = derive_ors_filename("board.ORP")
        assert result == "board.ORS"

    def test_derive_ors_from_orp_chain(self):
        """Chain derivation: kicad_pcb → ORP → ORS works."""
        orp = derive_orp_filename("design.kicad_pcb")
        ors = derive_ors_filename(orp)
        assert ors == "design.ORS"

    def test_get_solution_summary_new_format(self):
        """New ORS format produces readable summary."""
        ors_data = {
            "metadata": {
                "export_timestamp": "2025-01-01T00:00:00Z",
                "orthoroute_version": "0.1.0",
                "converged": True,
                "total_iterations": 10,
                "total_time_seconds": 5.0,
            },
            "statistics": {
                "nets_routed": 5,
                "total_wirelength_mm": 100.0,
                "total_vias": 3,
                "total_tracks": 20,
                "final_overflow_cost": 0,
            },
            "geometry": {"by_net": {"N1": {}, "N2": {}}},
        }
        summary = get_solution_summary(ors_data)
        assert isinstance(summary, str)
        assert "Convergence" in summary
        assert "Nets Routed" in summary
        assert "Wirelength" in summary

    def test_get_solution_summary_old_format(self):
        """Old ORS format produces readable summary."""
        ors_data = {
            "metadata": {"timestamp": "2025-01-01", "orthoroute_version": "0.0.1"},
            "metrics": {
                "final": {
                    "converged": False,
                    "iterations": 5,
                    "total_time": 3.0,
                    "nets_routed": 2,
                    "wirelength": 50.0,
                    "via_count": 1,
                    "overflow": 2,
                }
            },
            "nets": {"N1": {"traces": [1, 2], "vias": [1]}},
        }
        summary = get_solution_summary(ors_data)
        assert isinstance(summary, str)
        assert "Convergence" in summary

    def test_get_solution_summary_empty(self):
        """Empty data doesn't crash, returns a string."""
        summary = get_solution_summary({})
        assert isinstance(summary, str)

    def test_get_solution_summary_partial_new_format(self):
        """Partial new format data still produces a summary without crashing."""
        ors_data = {
            "metadata": {"converged": False},
            "statistics": {"nets_routed": 0},
        }
        summary = get_solution_summary(ors_data)
        assert isinstance(summary, str)
