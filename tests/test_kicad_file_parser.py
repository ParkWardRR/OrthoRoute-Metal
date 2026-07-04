"""Tests for KiCadFileParser – board file parsing and domain model creation.

Covers: instantiation, parse_file error paths, create_board_from_data with
minimal / full / multilayer dictionaries, pad-net linkage, layer counting,
and various edge cases.
"""
import pytest

from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser
from orthoroute.domain.models.board import Board, Component, Net, Pad, Layer, Coordinate


# ---------------------------------------------------------------------------
# Parser creation
# ---------------------------------------------------------------------------


class TestKiCadFileParserCreation:
    """Tests for KiCadFileParser instantiation."""

    def test_parser_creation(self):
        """KiCadFileParser() instantiates with no required arguments."""
        parser = KiCadFileParser()
        assert parser is not None

    def test_parser_is_correct_type(self):
        """KiCadFileParser() returns an instance of KiCadFileParser."""
        parser = KiCadFileParser()
        assert isinstance(parser, KiCadFileParser)


# ---------------------------------------------------------------------------
# parse_file error handling
# ---------------------------------------------------------------------------


class TestKiCadFileParserParseFile:
    """Tests for parse_file method error paths."""

    def test_parse_file_nonexistent(self):
        """parse_file raises FileNotFoundError for a nonexistent file."""
        parser = KiCadFileParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent.kicad_pcb")

    def test_parse_file_wrong_extension(self, tmp_path):
        """parse_file raises ValueError for non-.kicad_pcb extension."""
        # File must exist to get past the existence check
        txt_file = tmp_path / "board.txt"
        txt_file.write_text("dummy")
        parser = KiCadFileParser()
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse_file(str(txt_file))

    def test_parse_file_wrong_extension_json(self, tmp_path):
        """parse_file raises ValueError for .json extension."""
        json_file = tmp_path / "board.json"
        json_file.write_text("{}")
        parser = KiCadFileParser()
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse_file(str(json_file))

    def test_load_board_nonexistent_returns_none(self):
        """load_board returns None for nonexistent file (catches exception)."""
        parser = KiCadFileParser()
        result = parser.load_board("nonexistent.kicad_pcb")
        assert result is None

    def test_load_board_wrong_extension_returns_none(self):
        """load_board returns None for wrong extension (catches ValueError)."""
        parser = KiCadFileParser()
        result = parser.load_board("board.txt")
        assert result is None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parser():
    """Fresh KiCadFileParser instance."""
    return KiCadFileParser()


@pytest.fixture
def minimal_board_data():
    """Minimal valid board data dict."""
    return {
        "title": "Minimal Board",
        "layers": [
            {"name": "F.Cu", "type": "signal", "stackup_position": 0},
            {"name": "B.Cu", "type": "signal", "stackup_position": 1},
        ],
        "components": [],
        "nets": [],
    }


@pytest.fixture
def full_board_data():
    """Full board data dict with components, pads, and nets."""
    return {
        "title": "Full Test Board",
        "layers": [
            {"name": "F.Cu", "type": "signal", "stackup_position": 0},
            {"name": "B.Cu", "type": "signal", "stackup_position": 1},
        ],
        "components": [
            {
                "reference": "U1",
                "value": "IC1",
                "footprint": "QFP-32",
                "x": 10.0,
                "y": 20.0,
                "angle": 0.0,
                "layer": "F.Cu",
                "pads": [
                    {
                        "id": "U1_1",
                        "number": "1",
                        "type": "smd",
                        "shape": "rect",
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.6,
                        "height": 0.3,
                        "layer": "F.Cu",
                        "net_id": "1",
                        "drill_size": None,
                    },
                    {
                        "id": "U1_2",
                        "number": "2",
                        "type": "smd",
                        "shape": "rect",
                        "x": 1.0,
                        "y": 0.5,
                        "width": 0.6,
                        "height": 0.3,
                        "layer": "F.Cu",
                        "net_id": "2",
                        "drill_size": None,
                    },
                ],
            },
            {
                "reference": "R1",
                "value": "10k",
                "footprint": "0402",
                "x": 30.0,
                "y": 40.0,
                "angle": 90.0,
                "layer": "F.Cu",
                "pads": [
                    {
                        "id": "R1_1",
                        "number": "1",
                        "type": "smd",
                        "shape": "rect",
                        "x": 0.0,
                        "y": 0.0,
                        "width": 0.4,
                        "height": 0.3,
                        "layer": "F.Cu",
                        "net_id": "1",
                        "drill_size": None,
                    },
                    {
                        "id": "R1_2",
                        "number": "2",
                        "type": "smd",
                        "shape": "rect",
                        "x": 1.0,
                        "y": 0.0,
                        "width": 0.4,
                        "height": 0.3,
                        "layer": "F.Cu",
                        "net_id": "2",
                        "drill_size": None,
                    },
                ],
            },
        ],
        "nets": [
            {"id": "1", "name": "VCC", "netclass": "Default"},
            {"id": "2", "name": "GND", "netclass": "Power"},
        ],
        "design_rules": {
            "min_track_width": 0.15,
            "min_via_diameter": 0.6,
            "min_via_drill": 0.3,
        },
    }


# ---------------------------------------------------------------------------
# create_board_from_data
# ---------------------------------------------------------------------------


class TestCreateBoardFromData:
    """Tests for create_board_from_data method."""

    def test_create_board_from_data_minimal(self, parser, minimal_board_data):
        """create_board_from_data with minimal dict returns a Board object."""
        board = parser.create_board_from_data(minimal_board_data)
        assert isinstance(board, Board)
        assert board.name == "Minimal Board"
        assert board.id == "parsed_board"

    def test_create_board_from_data_full(self, parser, full_board_data):
        """Full dict returns complete Board with components, nets, layers."""
        board = parser.create_board_from_data(full_board_data)
        assert isinstance(board, Board)
        assert board.name == "Full Test Board"
        assert len(board.components) == 2
        assert len(board.nets) == 2
        assert len(board.layers) == 2

    def test_create_board_from_data_empty_nets(self, parser):
        """Empty nets list still creates a valid Board."""
        data = {
            "title": "No Nets Board",
            "layers": [{"name": "F.Cu", "type": "signal", "stackup_position": 0}],
            "components": [],
            "nets": [],
        }
        board = parser.create_board_from_data(data)
        assert isinstance(board, Board)
        assert len(board.nets) == 0

    def test_create_board_from_data_component_with_pads(self, parser, full_board_data):
        """Component with nested pads creates Pad objects attached to component."""
        board = parser.create_board_from_data(full_board_data)
        comp = board.components[0]
        assert len(comp.pads) == 2
        assert all(isinstance(p, Pad) for p in comp.pads)

    def test_create_board_from_data_net_pad_linkage(self, parser, full_board_data):
        """Pads in nets get correct net_id assigned via Net.__post_init__."""
        board = parser.create_board_from_data(full_board_data)
        net_vcc = board.get_net("1")
        assert net_vcc is not None
        assert net_vcc.name == "VCC"
        assert len(net_vcc.pads) == 2
        # Net.__post_init__ sets pad.net_id = self.id
        for pad in net_vcc.pads:
            assert pad.net_id == "1"

    def test_create_board_from_data_layer_count(self, parser, full_board_data):
        """Board.layer_count matches number of Cu layers in data."""
        board = parser.create_board_from_data(full_board_data)
        assert board.layer_count == 2

    def test_create_board_from_data_pad_id_format(self, parser, full_board_data):
        """Pad IDs follow 'REF_PADNUM' convention."""
        board = parser.create_board_from_data(full_board_data)
        pad_ids = [p.id for comp in board.components for p in comp.pads]
        assert "U1_1" in pad_ids
        assert "U1_2" in pad_ids
        assert "R1_1" in pad_ids
        assert "R1_2" in pad_ids

    def test_create_board_from_data_no_components(self, parser):
        """Empty components list still creates a valid Board."""
        data = {
            "title": "Empty Board",
            "layers": [{"name": "F.Cu", "type": "signal"}],
            "components": [],
            "nets": [{"id": "1", "name": "GND"}],
        }
        board = parser.create_board_from_data(data)
        assert isinstance(board, Board)
        assert len(board.components) == 0
        # Net '1' has no pads with net_id=='1', so it is NOT added
        assert len(board.nets) == 0

    def test_create_board_from_data_multilayer(self, parser):
        """6-layer board with In1-In4.Cu layers are all created."""
        data = {
            "title": "6-Layer Board",
            "layers": [
                {"name": "F.Cu", "type": "signal", "stackup_position": 0},
                {"name": "In1.Cu", "type": "signal", "stackup_position": 1},
                {"name": "In2.Cu", "type": "signal", "stackup_position": 2},
                {"name": "In3.Cu", "type": "signal", "stackup_position": 3},
                {"name": "In4.Cu", "type": "signal", "stackup_position": 4},
                {"name": "B.Cu", "type": "signal", "stackup_position": 5},
            ],
            "components": [],
            "nets": [],
        }
        board = parser.create_board_from_data(data)
        assert board.layer_count == 6
        layer_names = [layer.name for layer in board.layers]
        for name in ["F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "B.Cu"]:
            assert name in layer_names

    def test_extract_layers_fallback(self, parser):
        """When layers list is empty, layer_count is 0 and no layers added."""
        data = {
            "title": "No Layers Board",
            "layers": [],
            "components": [],
            "nets": [],
        }
        board = parser.create_board_from_data(data)
        # Empty layers => 0 Cu layers counted
        assert board.layer_count == 0
        assert len(board.layers) == 0

    def test_create_board_from_data_pad_position(self, parser):
        """Pad position is set from x, y in pad data."""
        data = {
            "title": "Pad Pos Test",
            "layers": [{"name": "F.Cu", "type": "signal"}],
            "components": [
                {
                    "reference": "J1",
                    "value": "Conn",
                    "footprint": "HDR-2",
                    "x": 5.0,
                    "y": 10.0,
                    "angle": 0.0,
                    "layer": "F.Cu",
                    "pads": [
                        {
                            "id": "J1_1",
                            "number": "1",
                            "type": "thru_hole",
                            "shape": "circle",
                            "x": 2.54,
                            "y": 3.81,
                            "width": 1.0,
                            "height": 1.0,
                            "layer": "F.Cu",
                            "net_id": None,
                            "drill_size": 0.8,
                        }
                    ],
                }
            ],
            "nets": [],
        }
        board = parser.create_board_from_data(data)
        pad = board.components[0].pads[0]
        assert pad.position.x == pytest.approx(2.54)
        assert pad.position.y == pytest.approx(3.81)
        assert pad.drill_size == pytest.approx(0.8)
        assert pad.shape == "circle"

    def test_create_board_from_data_pad_size(self, parser):
        """Pad size tuple is set from width/height in pad data."""
        data = {
            "title": "Pad Size Test",
            "layers": [{"name": "F.Cu", "type": "signal"}],
            "components": [
                {
                    "reference": "R1",
                    "value": "1k",
                    "footprint": "0402",
                    "x": 0.0,
                    "y": 0.0,
                    "pads": [
                        {
                            "id": "R1_1",
                            "number": "1",
                            "type": "smd",
                            "shape": "rect",
                            "x": 0,
                            "y": 0,
                            "width": 0.6,
                            "height": 0.3,
                            "layer": "F.Cu",
                            "net_id": None,
                        }
                    ],
                }
            ],
            "nets": [],
        }
        board = parser.create_board_from_data(data)
        pad = board.components[0].pads[0]
        assert pad.size == (0.6, 0.3)

    def test_create_board_from_data_net_without_matching_pads(self, parser):
        """Net with no matching pads is NOT added to board."""
        data = {
            "title": "Unmatched Net",
            "layers": [{"name": "F.Cu", "type": "signal"}],
            "components": [
                {
                    "reference": "U1",
                    "value": "IC",
                    "footprint": "SOT23",
                    "x": 0,
                    "y": 0,
                    "pads": [
                        {
                            "id": "U1_1",
                            "number": "1",
                            "type": "smd",
                            "shape": "rect",
                            "x": 0,
                            "y": 0,
                            "width": 0.5,
                            "height": 0.5,
                            "layer": "F.Cu",
                            "net_id": "99",
                        }
                    ],
                }
            ],
            "nets": [{"id": "1", "name": "ORPHAN_NET"}],
        }
        board = parser.create_board_from_data(data)
        assert len(board.nets) == 0

    def test_create_board_from_data_non_cu_layers_counted_by_substring(self, parser):
        """layer_count uses 'Cu' in name, so Edge.Cuts is counted (contains 'Cu')."""
        data = {
            "title": "Mixed Layers",
            "layers": [
                {"name": "F.Cu", "type": "signal", "stackup_position": 0},
                {"name": "B.Cu", "type": "signal", "stackup_position": 1},
                {"name": "F.SilkS", "type": "silk", "stackup_position": 2},
                {"name": "Edge.Cuts", "type": "edge", "stackup_position": 3},
            ],
            "components": [],
            "nets": [],
        }
        board = parser.create_board_from_data(data)
        # F.Cu, B.Cu, Edge.Cuts all contain 'Cu' substring
        assert board.layer_count == 3
        # All 4 layers are added
        assert len(board.layers) == 4

    def test_create_board_from_data_component_angle(self, parser):
        """Component angle is preserved from data."""
        data = {
            "title": "Angle Test",
            "layers": [{"name": "F.Cu", "type": "signal"}],
            "components": [
                {
                    "reference": "U1",
                    "value": "IC",
                    "footprint": "QFP",
                    "x": 5.0,
                    "y": 5.0,
                    "angle": 45.0,
                    "layer": "B.Cu",
                    "pads": [],
                }
            ],
            "nets": [],
        }
        board = parser.create_board_from_data(data)
        comp = board.components[0]
        assert comp.angle == pytest.approx(45.0)
        assert comp.layer == "B.Cu"
        assert comp.position.x == pytest.approx(5.0)
        assert comp.position.y == pytest.approx(5.0)

    def test_create_board_from_data_board_thickness_default(self, parser, minimal_board_data):
        """Board thickness defaults to 1.6 mm."""
        board = parser.create_board_from_data(minimal_board_data)
        assert board.thickness == pytest.approx(1.6)

    def test_create_board_from_data_missing_title(self, parser):
        """Missing title defaults to 'Parsed Board'."""
        data = {"layers": [], "components": [], "nets": []}
        board = parser.create_board_from_data(data)
        assert board.name == "Parsed Board"
