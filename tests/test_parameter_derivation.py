"""Tests for parameter auto-derivation from board characteristics."""
import pytest

from orthoroute.algorithms.manhattan.board_analyzer import BoardCharacteristics
from orthoroute.algorithms.manhattan.parameter_derivation import (
    derive_routing_parameters, DerivedRoutingParameters, apply_derived_parameters,
)
from orthoroute.algorithms.manhattan.pathfinder.config import PathFinderConfig


def _make_board_chars(congestion_ratio=0.85, layer_count=6, net_count=100,
                      via_flexibility=0.9):
    """Helper to create BoardCharacteristics with key fields."""
    return BoardCharacteristics(
        signal_layers=list(range(1, layer_count + 1)),
        h_layers=set(range(1, layer_count // 2 + 1)),
        v_layers=set(range(layer_count // 2 + 1, layer_count + 1)),
        layer_count=layer_count, plane_layers=set(),
        board_width_mm=50.0, board_height_mm=40.0, usable_area_mm2=2000.0,
        grid_pitch_mm=0.4, total_horizontal_channels=5000,
        total_vertical_channels=4000, channels_per_h_layer=125,
        channels_per_v_layer=100, via_capacity_estimate=30000,
        net_count=net_count, total_hpwl_mm=net_count * 75.0,
        avg_net_hpwl_mm=75.0, demand_horizontal_mm=net_count * 40.0,
        demand_vertical_mm=net_count * 35.0, demand_h_percentage=0.533,
        congestion_ratio=congestion_ratio,
        density_nets_per_mm2=net_count / 2000.0,
        routing_complexity="NORMAL",
        allowed_via_spans={(i, j) for i in range(layer_count) for j in range(i+1, layer_count)},
        via_flexibility=via_flexibility,
    )


def test_derive_sparse_strategy():
    """Sparse board (rho < 0.6) gets fast-convergence strategy."""
    bc = _make_board_chars(congestion_ratio=0.3)
    params = derive_routing_parameters(bc)
    assert "SPARSE" in params.strategy


def test_derive_dense_strategy():
    """Dense board (rho >= 1.2) gets conservative strategy."""
    bc = _make_board_chars(congestion_ratio=1.5)
    params = derive_routing_parameters(bc)
    assert "DENSE" in params.strategy


def test_derive_pres_fac_mult_range():
    """Present factor multiplier is always > 1."""
    bc = _make_board_chars(congestion_ratio=0.85)
    params = derive_routing_parameters(bc)
    assert params.pres_fac_mult > 1.0


def test_derive_pres_fac_max_positive():
    """Present factor max is positive."""
    bc = _make_board_chars()
    params = derive_routing_parameters(bc)
    assert params.pres_fac_max > 0


def test_derive_hist_cost_weight_positive():
    """History cost weight is positive."""
    bc = _make_board_chars()
    params = derive_routing_parameters(bc)
    assert params.hist_cost_weight > 0


def test_derive_hotset_cap_positive():
    """Hotset cap is at least 32."""
    bc = _make_board_chars(net_count=200)
    params = derive_routing_parameters(bc)
    assert params.hotset_cap >= 32


def test_derive_via_cost_unrestricted():
    """High via flexibility gets low via cost."""
    bc = _make_board_chars(via_flexibility=0.95)
    params = derive_routing_parameters(bc)
    assert params.via_cost_base <= 0.5


def test_derive_via_cost_restricted():
    """Low via flexibility gets higher via cost."""
    bc = _make_board_chars(via_flexibility=0.5)
    params = derive_routing_parameters(bc)
    assert params.via_cost_base >= 1.0


def test_derive_stagnation_patience_positive():
    """Stagnation patience is always positive."""
    bc = _make_board_chars()
    params = derive_routing_parameters(bc)
    assert params.stagnation_patience > 0


def test_derive_max_iterations_at_least_250():
    """Max iterations is always at least 250."""
    bc = _make_board_chars()
    params = derive_routing_parameters(bc)
    assert params.max_iterations >= 250


def test_apply_derived_parameters():
    """apply_derived_parameters modifies config in-place."""
    bc = _make_board_chars(congestion_ratio=0.5)
    params = derive_routing_parameters(bc)
    config = PathFinderConfig()
    apply_derived_parameters(config, params)
    assert config.pres_fac_mult == params.pres_fac_mult
    assert config.hist_cost_weight == params.hist_cost_weight
    assert config.via_cost == params.via_cost_base


def test_derive_few_layers_hist_bonus():
    """Boards with <= 10 layers get a history bonus."""
    bc_few = _make_board_chars(layer_count=6)
    bc_many = _make_board_chars(layer_count=30)
    params_few = derive_routing_parameters(bc_few)
    params_many = derive_routing_parameters(bc_many)
    assert params_few.hist_cost_weight > params_many.hist_cost_weight


def test_derive_layer_bias_few_layers():
    """Few layers get aggressive layer bias."""
    bc = _make_board_chars(layer_count=6)
    params = derive_routing_parameters(bc)
    assert params.layer_bias_alpha == 0.20


def test_derive_empty_board():
    """Empty board (0 nets) doesn't crash derivation."""
    bc = _make_board_chars(net_count=0)
    params = derive_routing_parameters(bc)
    assert params.hotset_cap == 32
