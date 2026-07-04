"""Tests for DRC validation service."""
import pytest

from orthoroute.domain.models.board import Board, Component, Net, Pad, Layer, Coordinate
from orthoroute.domain.models.routing import Route, Segment, SegmentType, Via, ViaType
from orthoroute.domain.models.constraints import DRCConstraints, ClearanceType
from orthoroute.domain.services.drc_checker import DRCChecker, DRCViolation


@pytest.fixture
def checker():
    """DRC checker with default constraints."""
    return DRCChecker(DRCConstraints())


def test_clearance_validation_passes(checker):
    """Distance above clearance passes validation."""
    assert checker.constraints.validate_clearance(0.5, ClearanceType.TRACK_TO_TRACK) is True


def test_clearance_validation_fails(checker):
    """Distance below clearance fails validation."""
    assert checker.constraints.validate_clearance(0.05, ClearanceType.TRACK_TO_TRACK) is False


def test_track_width_valid(checker):
    """Valid track width passes DRC."""
    route = Route(id="R1", net_id="N1", segments=[
        Segment(type=SegmentType.TRACK, start=Coordinate(0, 0), end=Coordinate(5, 0),
                width=0.25, layer="F.Cu", net_id="N1")
    ])
    violations = checker._check_track_widths(route)
    assert len(violations) == 0


def test_track_width_too_narrow(checker):
    """Track narrower than minimum creates violation."""
    route = Route(id="R1", net_id="N1", segments=[
        Segment(type=SegmentType.TRACK, start=Coordinate(0, 0), end=Coordinate(5, 0),
                width=0.05, layer="F.Cu", net_id="N1")
    ])
    violations = checker._check_track_widths(route)
    assert len(violations) > 0
    assert violations[0].type == "track_width"


def test_via_size_valid(checker):
    """Valid via passes DRC."""
    route = Route(id="R1", net_id="N1", vias=[
        Via(position=Coordinate(0, 0), diameter=0.6, drill_size=0.3,
            from_layer="F.Cu", to_layer="B.Cu", net_id="N1")
    ])
    violations = checker._check_via_sizes(route)
    assert len(violations) == 0


def test_via_size_too_small(checker):
    """Via smaller than minimum creates violation."""
    route = Route(id="R1", net_id="N1", vias=[
        Via(position=Coordinate(0, 0), diameter=0.1, drill_size=0.05,
            from_layer="F.Cu", to_layer="B.Cu", net_id="N1")
    ])
    violations = checker._check_via_sizes(route)
    assert len(violations) > 0
    assert violations[0].type == "via_size"


def test_check_board_clean(checker, sample_board):
    """Clean sample board produces no error-level violations."""
    violations = checker.check_board(sample_board)
    errors = [v for v in violations if v.severity == 'error']
    assert len(errors) == 0


def test_check_board_empty_net(checker):
    """Board with empty net produces violation."""
    board = Board(id="b", name="test")
    net = Net(id="N1", name="empty_net", pads=[])
    board.add_net(net)
    board._build_indexes()
    violations = checker.check_board(board)
    net_violations = [v for v in violations if v.type == 'empty_net']
    assert len(net_violations) == 1


def test_check_board_single_pad_net(checker):
    """Board with single-pad net produces warning."""
    pad = Pad(id="P1", component_id="C1", net_id=None,
              position=Coordinate(0, 0), size=(0.3, 0.3))
    board = Board(id="b", name="test")
    net = Net(id="N1", name="solo", pads=[pad])
    board.add_net(net)
    board._build_indexes()
    violations = checker.check_board(board)
    single_violations = [v for v in violations if v.type == 'single_pad_net']
    assert len(single_violations) == 1


def test_check_route_clean(checker):
    """Clean route with valid widths and vias produces no violations."""
    seg = Segment(type=SegmentType.TRACK, start=Coordinate(0, 0), end=Coordinate(5, 0),
                  width=0.25, layer="F.Cu", net_id="N1")
    via = Via(position=Coordinate(5, 0), diameter=0.6, drill_size=0.3,
             from_layer="F.Cu", to_layer="B.Cu", net_id="N1")
    seg2 = Segment(type=SegmentType.TRACK, start=Coordinate(5, 0), end=Coordinate(5, 3),
                   width=0.25, layer="B.Cu", net_id="N1")
    route = Route(id="R1", net_id="N1", segments=[seg, seg2], vias=[via])
    board = Board(id="b", name="test")
    violations = checker.check_route(route, board)
    assert len(violations) == 0


def test_inter_route_same_net_no_violation(checker):
    """Routes with same net_id don't generate clearance violations."""
    seg1 = Segment(type=SegmentType.TRACK, start=Coordinate(0, 0), end=Coordinate(5, 0),
                   width=0.2, layer="F.Cu", net_id="N1")
    seg2 = Segment(type=SegmentType.TRACK, start=Coordinate(0, 0.1), end=Coordinate(5, 0.1),
                   width=0.2, layer="F.Cu", net_id="N1")
    route1 = Route(id="R1", net_id="N1", segments=[seg1])
    route2 = Route(id="R2", net_id="N1", segments=[seg2])
    violations = checker.check_routes_clearance([route1, route2])
    assert len(violations) == 0


def test_inter_route_different_net_too_close(checker):
    """Routes from different nets that are too close generate violations."""
    seg1 = Segment(type=SegmentType.TRACK, start=Coordinate(0, 0), end=Coordinate(5, 0),
                   width=0.2, layer="F.Cu", net_id="N1")
    seg2 = Segment(type=SegmentType.TRACK, start=Coordinate(0, 0.05), end=Coordinate(5, 0.05),
                   width=0.2, layer="F.Cu", net_id="N2")
    route1 = Route(id="R1", net_id="N1", segments=[seg1])
    route2 = Route(id="R2", net_id="N2", segments=[seg2])
    violations = checker.check_routes_clearance([route1, route2])
    assert len(violations) > 0


def test_drc_report_structure(checker):
    """DRC report contains expected keys."""
    violations = [
        DRCViolation(type="track_width", severity="error", message="too narrow"),
        DRCViolation(type="track_width", severity="error", message="also narrow"),
        DRCViolation(type="clearance", severity="warning", message="close"),
    ]
    report = checker.generate_drc_report(violations)
    assert report['total_violations'] == 3
    assert report['errors'] == 2
    assert report['warnings'] == 1


def test_drc_violation_str():
    """DRCViolation __str__ includes severity and message."""
    v = DRCViolation(type="test", severity="error", message="bad thing")
    s = str(v)
    assert "ERROR" in s
    assert "bad thing" in s


def test_drc_violation_with_location():
    """DRCViolation __str__ includes location when present."""
    v = DRCViolation(type="test", severity="warning", message="close",
                     location=Coordinate(1.5, 2.5))
    s = str(v)
    assert "1.500" in s
    assert "2.500" in s
