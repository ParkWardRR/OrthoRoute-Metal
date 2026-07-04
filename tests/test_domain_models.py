"""Tests for domain models: Board, Component, Net, Pad, Layer, Via, Route, Segment, Coordinate, DRCConstraints."""
import pytest
import math

from orthoroute.domain.models.board import Board, Component, Net, Pad, Layer, Coordinate, Bounds
from orthoroute.domain.models.routing import (
    Route, Segment, SegmentType, Via, ViaType,
    RoutingResult, RoutingStatistics,
)
from orthoroute.domain.models.constraints import (
    DRCConstraints, NetClass, ClearanceType,
)


def test_coordinate_creation():
    """Coordinate stores x and y as floats."""
    c = Coordinate(1.5, 2.5)
    assert c.x == 1.5
    assert c.y == 2.5


def test_coordinate_distance_same_point():
    """Distance from a coordinate to itself is zero."""
    c = Coordinate(3.0, 4.0)
    assert c.distance_to(c) == 0.0


def test_coordinate_distance_known():
    """3-4-5 right triangle gives distance 5."""
    a = Coordinate(0.0, 0.0)
    b = Coordinate(3.0, 4.0)
    assert abs(a.distance_to(b) - 5.0) < 1e-9


def test_coordinate_is_frozen():
    """Coordinate is immutable (frozen dataclass)."""
    c = Coordinate(1.0, 2.0)
    with pytest.raises(AttributeError):
        c.x = 99.0


def test_bounds_width_and_height():
    """Bounds computes width and height correctly."""
    b = Bounds(0.0, 0.0, 10.0, 5.0)
    assert b.width == 10.0
    assert b.height == 5.0


def test_bounds_center():
    """Bounds center is the midpoint."""
    b = Bounds(0.0, 0.0, 10.0, 10.0)
    assert b.center.x == 5.0
    assert b.center.y == 5.0


def test_bounds_is_frozen():
    """Bounds is immutable."""
    b = Bounds(0.0, 0.0, 10.0, 10.0)
    with pytest.raises(AttributeError):
        b.min_x = -1.0


def test_pad_creation():
    """Pad stores all fields correctly."""
    p = Pad(id="P1", component_id="C1", net_id="N1",
            position=Coordinate(1.0, 2.0), size=(0.5, 0.5))
    assert p.id == "P1"
    assert p.component_id == "C1"
    assert p.net_id == "N1"
    assert p.layer == "F.Cu"


def test_pad_default_shape():
    """Pad default shape is circle."""
    p = Pad(id="P1", component_id="C1", net_id=None,
            position=Coordinate(0, 0), size=(0.3, 0.3))
    assert p.shape == "circle"


def test_pad_auto_id():
    """Pad generates a UUID if id is empty."""
    p = Pad(id="", component_id="C1", net_id=None,
            position=Coordinate(0, 0), size=(0.3, 0.3))
    assert len(p.id) > 0


def test_component_creation():
    """Component stores reference, value, footprint."""
    c = Component(id="C1", reference="U1", value="IC1",
                  footprint="QFP-32", position=Coordinate(0, 0))
    assert c.reference == "U1"
    assert c.value == "IC1"


def test_component_pads_reference_parent():
    """All pads in a component should reference that component's id."""
    pad = Pad(id="P1", component_id="X", net_id=None,
              position=Coordinate(0, 0), size=(0.3, 0.3))
    comp = Component(id="C1", reference="U1", value="v",
                     footprint="fp", position=Coordinate(0, 0), pads=[pad])
    assert pad.component_id == comp.id


def test_component_bounds_no_pads():
    """Component with no pads returns point bounds at position."""
    c = Component(id="C1", reference="U1", value="v",
                  footprint="fp", position=Coordinate(5.0, 5.0))
    b = c.get_bounds()
    assert b.min_x == 5.0
    assert b.max_x == 5.0


def test_component_bounds_with_pads():
    """Component bounds envelope all pad positions."""
    p1 = Pad(id="P1", component_id="C1", net_id=None,
             position=Coordinate(1.0, 1.0), size=(0.4, 0.4))
    p2 = Pad(id="P2", component_id="C1", net_id=None,
             position=Coordinate(3.0, 3.0), size=(0.4, 0.4))
    c = Component(id="C1", reference="U1", value="v",
                  footprint="fp", position=Coordinate(2.0, 2.0), pads=[p1, p2])
    b = c.get_bounds()
    assert b.min_x == pytest.approx(0.8)
    assert b.max_x == pytest.approx(3.2)


def test_net_is_routable_two_pads():
    """Net with 2+ pads is routable."""
    p1 = Pad(id="P1", component_id="C1", net_id=None,
             position=Coordinate(0, 0), size=(0.3, 0.3))
    p2 = Pad(id="P2", component_id="C2", net_id=None,
             position=Coordinate(1, 1), size=(0.3, 0.3))
    net = Net(id="N1", name="VCC", pads=[p1, p2])
    assert net.is_routable is True


def test_net_not_routable_single_pad():
    """Net with a single pad is NOT routable."""
    p = Pad(id="P1", component_id="C1", net_id=None,
            position=Coordinate(0, 0), size=(0.3, 0.3))
    net = Net(id="N1", name="solo", pads=[p])
    assert net.is_routable is False


def test_net_sets_pad_net_id():
    """Net.__post_init__ assigns its id to each pad.net_id."""
    p = Pad(id="P1", component_id="C1", net_id=None,
            position=Coordinate(0, 0), size=(0.3, 0.3))
    net = Net(id="N1", name="test", pads=[p])
    assert p.net_id == "N1"


def test_net_min_distance():
    """Net calculates minimum distance between any two pads."""
    p1 = Pad(id="P1", component_id="C1", net_id=None,
             position=Coordinate(0, 0), size=(0.3, 0.3))
    p2 = Pad(id="P2", component_id="C2", net_id=None,
             position=Coordinate(3, 4), size=(0.3, 0.3))
    net = Net(id="N1", name="d", pads=[p1, p2])
    assert abs(net.calculate_min_distance() - 5.0) < 1e-9


def test_net_bounds():
    """Net bounds envelope all pad positions."""
    p1 = Pad(id="P1", component_id="C1", net_id=None,
             position=Coordinate(2.0, 3.0), size=(0.3, 0.3))
    p2 = Pad(id="P2", component_id="C2", net_id=None,
             position=Coordinate(8.0, 9.0), size=(0.3, 0.3))
    net = Net(id="N1", name="b", pads=[p1, p2])
    b = net.get_bounds()
    assert b.min_x == 2.0
    assert b.max_y == 9.0


def test_layer_is_routing_signal():
    """Signal layer with Cu in name is a routing layer."""
    layer = Layer(name="F.Cu", type="signal", stackup_position=0)
    assert layer.is_routing_layer is True


def test_layer_is_not_routing_mask():
    """Mask layer is not a routing layer."""
    layer = Layer(name="F.Mask", type="mask", stackup_position=0)
    assert layer.is_routing_layer is False


def test_board_creation(sample_board):
    """Board is created with correct name and component count."""
    assert sample_board.name == "TestBoard"
    assert len(sample_board.components) == 2


def test_board_get_component(sample_board):
    """Board can look up component by ID."""
    comp = sample_board.get_component("C1")
    assert comp is not None
    assert comp.reference == "U1"


def test_board_get_net_by_name(sample_board):
    """Board can look up net by name."""
    net = sample_board.get_net_by_name("VCC")
    assert net is not None
    assert net.id == "N1"


def test_board_routable_nets(sample_board):
    """Board returns nets with 2+ pads."""
    routable = sample_board.get_routable_nets()
    assert len(routable) == 1


def test_board_routing_layers(sample_board):
    """Board returns signal Cu layers as routing layers."""
    routing = sample_board.get_routing_layers()
    assert len(routing) == 2


def test_board_get_bounds(sample_board):
    """Board bounds envelope all components."""
    b = sample_board.get_bounds()
    assert b.min_x <= 1.0
    assert b.max_x >= 8.0


def test_board_get_all_pads(sample_board):
    """Board collects pads from all components."""
    pads = sample_board.get_all_pads()
    assert len(pads) == 2


def test_board_validate_integrity_clean(sample_board):
    """Clean board should have no integrity issues."""
    issues = sample_board.validate_integrity()
    assert len(issues) == 0


def test_board_add_component():
    """Board.add_component updates the components list and index."""
    board = Board(id="b", name="empty")
    comp = Component(id="C1", reference="U1", value="v",
                     footprint="fp", position=Coordinate(0, 0))
    board.add_component(comp)
    assert board.get_component("C1") is comp


def test_board_add_net():
    """Board.add_net updates nets list and name index."""
    board = Board(id="b", name="empty")
    net = Net(id="N1", name="VCC")
    board.add_net(net)
    assert board.get_net_by_name("VCC") is net


def test_segment_horizontal():
    """Horizontal segment detected correctly."""
    seg = Segment(type=SegmentType.TRACK,
                  start=Coordinate(0, 0), end=Coordinate(5, 0),
                  width=0.2, layer="F.Cu", net_id="N1")
    assert seg.is_horizontal is True
    assert seg.is_vertical is False


def test_segment_vertical():
    """Vertical segment detected correctly."""
    seg = Segment(type=SegmentType.TRACK,
                  start=Coordinate(0, 0), end=Coordinate(0, 5),
                  width=0.2, layer="F.Cu", net_id="N1")
    assert seg.is_vertical is True


def test_segment_is_manhattan():
    """Manhattan segment is either horizontal or vertical."""
    seg = Segment(type=SegmentType.TRACK,
                  start=Coordinate(0, 0), end=Coordinate(5, 0),
                  width=0.2, layer="F.Cu", net_id="N1")
    assert seg.is_manhattan is True


def test_segment_not_manhattan():
    """Diagonal segment is not Manhattan."""
    seg = Segment(type=SegmentType.TRACK,
                  start=Coordinate(0, 0), end=Coordinate(3, 4),
                  width=0.2, layer="F.Cu", net_id="N1")
    assert seg.is_manhattan is False


def test_segment_length():
    """Segment length calculation for a track."""
    seg = Segment(type=SegmentType.TRACK,
                  start=Coordinate(0, 0), end=Coordinate(3, 4),
                  width=0.2, layer="F.Cu", net_id="N1")
    assert abs(seg.length - 5.0) < 1e-9


def test_via_creation():
    """Via stores layer transition."""
    via = Via(position=Coordinate(5, 5), diameter=0.6, drill_size=0.3,
             from_layer="F.Cu", to_layer="B.Cu", net_id="N1")
    assert via.from_layer == "F.Cu"
    assert via.to_layer == "B.Cu"


def test_via_aspect_ratio():
    """Via aspect ratio = board_thickness / drill_size (assumes 1.6mm)."""
    via = Via(position=Coordinate(0, 0), diameter=0.6, drill_size=0.4,
             from_layer="F.Cu", to_layer="B.Cu", net_id="N1")
    assert abs(via.aspect_ratio - 4.0) < 1e-9


def test_via_type_default():
    """Via defaults to THROUGH type."""
    via = Via(position=Coordinate(0, 0), diameter=0.6, drill_size=0.3,
             from_layer="F.Cu", to_layer="B.Cu", net_id="N1")
    assert via.via_type == ViaType.THROUGH


def test_route_total_length():
    """Route total length sums all segment lengths."""
    seg1 = Segment(type=SegmentType.TRACK,
                   start=Coordinate(0, 0), end=Coordinate(3, 0),
                   width=0.2, layer="F.Cu", net_id="N1")
    seg2 = Segment(type=SegmentType.TRACK,
                   start=Coordinate(3, 0), end=Coordinate(3, 4),
                   width=0.2, layer="F.Cu", net_id="N1")
    route = Route(id="R1", net_id="N1", segments=[seg1, seg2])
    assert abs(route.total_length - 7.0) < 1e-9


def test_route_layers_used():
    """Route tracks the set of layers used."""
    seg = Segment(type=SegmentType.TRACK,
                  start=Coordinate(0, 0), end=Coordinate(3, 0),
                  width=0.2, layer="F.Cu", net_id="N1")
    via = Via(position=Coordinate(3, 0), diameter=0.6, drill_size=0.3,
             from_layer="F.Cu", to_layer="B.Cu", net_id="N1")
    route = Route(id="R1", net_id="N1", segments=[seg], vias=[via])
    assert route.layers_used == {"F.Cu", "B.Cu"}


def test_route_manhattan_compliant():
    """Route with only H/V segments is Manhattan compliant."""
    seg1 = Segment(type=SegmentType.TRACK,
                   start=Coordinate(0, 0), end=Coordinate(5, 0),
                   width=0.2, layer="F.Cu", net_id="N1")
    seg2 = Segment(type=SegmentType.TRACK,
                   start=Coordinate(5, 0), end=Coordinate(5, 3),
                   width=0.2, layer="F.Cu", net_id="N1")
    route = Route(id="R1", net_id="N1", segments=[seg1, seg2])
    assert route.is_manhattan_compliant() is True


def test_route_add_segment_wrong_net():
    """Adding a segment with mismatched net_id raises ValueError."""
    route = Route(id="R1", net_id="N1")
    seg = Segment(type=SegmentType.TRACK,
                  start=Coordinate(0, 0), end=Coordinate(1, 0),
                  width=0.2, layer="F.Cu", net_id="N2")
    with pytest.raises(ValueError):
        route.add_segment(seg)


def test_route_add_via_wrong_net():
    """Adding a via with mismatched net_id raises ValueError."""
    route = Route(id="R1", net_id="N1")
    via = Via(position=Coordinate(0, 0), diameter=0.6, drill_size=0.3,
             from_layer="F.Cu", to_layer="B.Cu", net_id="N2")
    with pytest.raises(ValueError):
        route.add_via(via)


def test_route_statistics():
    """Route statistics dict contains expected keys."""
    seg = Segment(type=SegmentType.TRACK,
                  start=Coordinate(0, 0), end=Coordinate(5, 0),
                  width=0.2, layer="F.Cu", net_id="N1")
    route = Route(id="R1", net_id="N1", segments=[seg])
    stats = route.get_route_statistics()
    assert 'total_length' in stats
    assert 'via_count' in stats
    assert 'is_manhattan' in stats


def test_routing_result_success():
    """RoutingResult.success_result creates a successful result."""
    route = Route(id="R1", net_id="N1")
    result = RoutingResult.success_result(route, execution_time=0.5)
    assert result.success is True
    assert result.route is route


def test_routing_result_failure():
    """RoutingResult.failure_result creates a failed result."""
    result = RoutingResult.failure_result("no path found")
    assert result.success is False
    assert result.error_message == "no path found"


def test_routing_statistics_success_rate():
    """Success rate is nets_routed / nets_attempted."""
    stats = RoutingStatistics(nets_attempted=10, nets_routed=7)
    assert abs(stats.success_rate - 0.7) < 1e-9


def test_routing_statistics_zero_attempted():
    """Success rate is 0 when no nets attempted."""
    stats = RoutingStatistics()
    assert stats.success_rate == 0.0


def test_routing_statistics_to_dict():
    """Statistics can be serialized to dict."""
    stats = RoutingStatistics(nets_attempted=5, nets_routed=3,
                              total_length=100.0, total_vias=10)
    d = stats.to_dict()
    assert d['nets_attempted'] == 5
    assert d['success_rate'] == pytest.approx(0.6)


def test_drc_constraints_defaults():
    """DRCConstraints creates a Default netclass."""
    drc = DRCConstraints()
    assert "Default" in drc.netclasses


def test_drc_constraints_validate_track_width_valid():
    """Valid track width passes validation."""
    drc = DRCConstraints()
    assert drc.validate_track_width(0.25) is True


def test_drc_constraints_validate_track_width_too_narrow():
    """Track width below global minimum fails."""
    drc = DRCConstraints(min_track_width=0.1)
    assert drc.validate_track_width(0.05) is False


def test_drc_constraints_validate_via_size_valid():
    """Valid via passes validation."""
    drc = DRCConstraints()
    assert drc.validate_via_size(0.6, 0.3) is True


def test_drc_constraints_validate_via_size_too_small():
    """Via below minimum fails."""
    drc = DRCConstraints(min_via_diameter=0.2)
    assert drc.validate_via_size(0.1, 0.05) is False


def test_drc_constraints_clearance_matrix():
    """Clearance matrix initialized with defaults."""
    drc = DRCConstraints()
    val = drc.get_clearance(ClearanceType.TRACK_TO_TRACK)
    assert val == drc.default_clearance


def test_drc_constraints_roundtrip():
    """to_dict -> from_dict roundtrip preserves values."""
    original = DRCConstraints(
        min_track_width=0.15,
        default_clearance=0.25,
        blind_via_enabled=False,
    )
    d = original.to_dict()
    restored = DRCConstraints.from_dict(d)
    assert restored.min_track_width == 0.15
    assert restored.default_clearance == 0.25
    assert restored.blind_via_enabled is False


def test_drc_via_types_allowed():
    """Via types reflect enabled flags."""
    drc = DRCConstraints(blind_via_enabled=True, buried_via_enabled=False,
                         micro_via_enabled=True)
    allowed = drc.get_via_types_allowed()
    assert "through" in allowed
    assert "blind" in allowed
    assert "buried" not in allowed
    assert "micro" in allowed


def test_netclass_track_width_valid():
    """Track width within range is valid."""
    nc = NetClass(name="Power", track_width=0.5, via_diameter=0.8, via_drill=0.4)
    assert nc.is_track_width_valid(0.5) is True


def test_netclass_track_width_invalid():
    """Track width outside max is invalid."""
    nc = NetClass(name="Power", track_width=0.5, via_diameter=0.8, via_drill=0.4)
    assert nc.is_track_width_valid(2.0) is False


def test_netclass_via_size_valid():
    """Via diameter within range is valid."""
    nc = NetClass(name="Sig", track_width=0.25, via_diameter=0.6, via_drill=0.3)
    assert nc.is_via_size_valid(0.6) is True
