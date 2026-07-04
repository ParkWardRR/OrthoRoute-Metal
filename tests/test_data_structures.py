"""Tests for PathFinder data structures: Portal, EdgeRec, Geometry."""
import pytest

from orthoroute.algorithms.manhattan.pathfinder.data_structures import (
    Portal, EdgeRec, Geometry, canonical_edge_key,
)


def test_portal_creation():
    """Portal stores position, layer, and net."""
    p = Portal(x=1.0, y=2.0, layer=0, net="N1", pad_layer=0)
    assert p.x == 1.0
    assert p.y == 2.0
    assert p.layer == 0
    assert p.net == "N1"


def test_portal_pad_layer():
    """Portal records the pad's original layer."""
    p = Portal(x=0, y=0, layer=2, net="N1", pad_layer=0)
    assert p.pad_layer == 0


def test_edgerec_defaults():
    """EdgeRec initializes with zero usage and empty owners."""
    er = EdgeRec()
    assert er.usage == 0
    assert len(er.owners) == 0
    assert er.pres_cost == 0.0


def test_edgerec_usage_increment():
    """EdgeRec usage can be incremented."""
    er = EdgeRec()
    er.usage += 1
    assert er.usage == 1


def test_edgerec_owner_tracking():
    """EdgeRec tracks owner nets."""
    er = EdgeRec()
    er.owners.add("N1")
    er.owners.add("N2")
    assert "N1" in er.owners
    assert len(er.owners) == 2


def test_edgerec_taboo_default():
    """EdgeRec taboo_until_iter defaults to -1 (no taboo)."""
    er = EdgeRec()
    assert er.taboo_until_iter == -1


def test_edgerec_historical_cost():
    """EdgeRec historical cost starts at 0."""
    er = EdgeRec()
    assert er.historical_cost == 0.0


def test_geometry_creation():
    """Geometry initializes with empty tracks and vias lists."""
    g = Geometry()
    assert g.tracks == []
    assert g.vias == []


def test_geometry_append_track():
    """Tracks can be appended to Geometry."""
    g = Geometry()
    g.tracks.append({"start": (0, 0), "end": (1, 0)})
    assert len(g.tracks) == 1


def test_geometry_append_via():
    """Vias can be appended to Geometry."""
    g = Geometry()
    g.vias.append({"pos": (5, 5), "layers": (0, 1)})
    assert len(g.vias) == 1


def test_canonical_edge_key_order():
    """canonical_edge_key normalizes node order."""
    k1 = canonical_edge_key(0, 1, 2, 3, 4)
    k2 = canonical_edge_key(0, 3, 4, 1, 2)
    assert k1 == k2


def test_canonical_edge_key_same_nodes():
    """Same node pair gives same key regardless of direction."""
    k1 = canonical_edge_key(1, 5, 3, 2, 1)
    k2 = canonical_edge_key(1, 2, 1, 5, 3)
    assert k1 == k2


def test_canonical_edge_key_layer_matters():
    """Different layers produce different keys."""
    k1 = canonical_edge_key(0, 1, 2, 3, 4)
    k2 = canonical_edge_key(1, 1, 2, 3, 4)
    assert k1 != k2
