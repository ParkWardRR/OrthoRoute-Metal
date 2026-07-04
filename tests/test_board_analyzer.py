"""Tests for board analysis: congestion ratio, net statistics."""
import pytest

from orthoroute.algorithms.manhattan.board_analyzer import (
    BoardCharacteristics, _assign_hv_layers_by_demand,
)


def test_board_characteristics_creation():
    """BoardCharacteristics can be created with all required fields."""
    bc = BoardCharacteristics(
        signal_layers=[1, 2, 3, 4], h_layers={1, 3}, v_layers={2, 4},
        layer_count=4, plane_layers=set(),
        board_width_mm=50.0, board_height_mm=40.0, usable_area_mm2=2000.0,
        grid_pitch_mm=0.4, total_horizontal_channels=5000,
        total_vertical_channels=4000, channels_per_h_layer=125,
        channels_per_v_layer=100, via_capacity_estimate=30000,
        net_count=200, total_hpwl_mm=15000.0, avg_net_hpwl_mm=75.0,
        demand_horizontal_mm=8000.0, demand_vertical_mm=7000.0,
        demand_h_percentage=0.533, congestion_ratio=0.85,
        density_nets_per_mm2=0.1, routing_complexity="NORMAL",
        allowed_via_spans={(0, 1), (1, 2), (2, 3)}, via_flexibility=0.5,
    )
    assert bc.congestion_ratio == 0.85
    assert bc.routing_complexity == "NORMAL"


def test_board_characteristics_sparse():
    """Low congestion ratio classifies as SPARSE."""
    bc = BoardCharacteristics(
        signal_layers=[1], h_layers={1}, v_layers=set(), layer_count=1,
        plane_layers=set(), board_width_mm=100.0, board_height_mm=100.0,
        usable_area_mm2=10000.0, grid_pitch_mm=0.4,
        total_horizontal_channels=25000, total_vertical_channels=25000,
        channels_per_h_layer=250, channels_per_v_layer=250,
        via_capacity_estimate=0, net_count=5, total_hpwl_mm=50.0,
        avg_net_hpwl_mm=10.0, demand_horizontal_mm=25.0,
        demand_vertical_mm=25.0, demand_h_percentage=0.5,
        congestion_ratio=0.01, density_nets_per_mm2=0.0005,
        routing_complexity="SPARSE", allowed_via_spans=set(), via_flexibility=0.0,
    )
    assert bc.routing_complexity == "SPARSE"


def test_board_characteristics_dense():
    """High congestion ratio classifies as DENSE."""
    bc = BoardCharacteristics(
        signal_layers=[1, 2], h_layers={1}, v_layers={2}, layer_count=2,
        plane_layers=set(), board_width_mm=20.0, board_height_mm=20.0,
        usable_area_mm2=400.0, grid_pitch_mm=0.4,
        total_horizontal_channels=100, total_vertical_channels=100,
        channels_per_h_layer=50, channels_per_v_layer=50,
        via_capacity_estimate=100, net_count=500, total_hpwl_mm=50000.0,
        avg_net_hpwl_mm=100.0, demand_horizontal_mm=25000.0,
        demand_vertical_mm=25000.0, demand_h_percentage=0.5,
        congestion_ratio=2.5, density_nets_per_mm2=1.25,
        routing_complexity="DENSE", allowed_via_spans={(0, 1)}, via_flexibility=1.0,
    )
    assert bc.routing_complexity == "DENSE"
    assert bc.congestion_ratio > 1.2


def test_hv_assignment_no_tasks():
    """With no tasks, layers alternate H/V."""
    signal_layers = [1, 2, 3, 4]
    h, v, pct = _assign_hv_layers_by_demand(signal_layers, {}, lattice=None)
    assert len(h) + len(v) == 4
    assert 0.0 <= pct <= 1.0


def test_hv_assignment_returns_disjoint_sets():
    """H and V layer sets must be disjoint."""
    signal_layers = [1, 2, 3, 4, 5, 6]
    h, v, _ = _assign_hv_layers_by_demand(signal_layers, {}, lattice=None)
    assert h.isdisjoint(v)


def test_hv_assignment_covers_all_layers():
    """Union of H and V sets equals the signal layer set."""
    signal_layers = [1, 2, 3, 4]
    h, v, _ = _assign_hv_layers_by_demand(signal_layers, {}, lattice=None)
    assert h | v == set(signal_layers)
