"""Tests for PathFinder configuration."""
import pytest

from orthoroute.algorithms.manhattan.pathfinder.config import (
    PathFinderConfig, GRID_PITCH, BATCH_SIZE, MAX_ITERATIONS,
    VIA_COST, PRES_FAC_INIT, PRES_FAC_MULT, PRES_FAC_MAX,
)


def test_config_default_grid_pitch():
    """Default grid pitch is 0.4mm."""
    cfg = PathFinderConfig()
    assert cfg.grid_pitch == 0.4


def test_config_default_batch_size():
    """Default batch size matches module constant."""
    cfg = PathFinderConfig()
    assert cfg.batch_size == BATCH_SIZE


def test_config_default_max_iterations():
    """Default max iterations matches module constant."""
    cfg = PathFinderConfig()
    assert cfg.max_iters == MAX_ITERATIONS


def test_config_default_via_cost():
    """Default via cost matches module constant."""
    cfg = PathFinderConfig()
    assert cfg.via_cost == VIA_COST


def test_config_default_pres_fac_init():
    """Default initial present factor matches module constant."""
    cfg = PathFinderConfig()
    assert cfg.pres_fac_init == PRES_FAC_INIT


def test_config_override_batch_size():
    """Batch size can be overridden."""
    cfg = PathFinderConfig(batch_size=64)
    assert cfg.batch_size == 64


def test_config_override_grid_pitch():
    """Grid pitch can be overridden."""
    cfg = PathFinderConfig(grid_pitch=0.2)
    assert cfg.grid_pitch == 0.2


def test_config_override_max_iterations():
    """Max iterations can be overridden."""
    cfg = PathFinderConfig(max_iters=100)
    assert cfg.max_iters == 100


def test_config_pres_fac_mult_positive():
    """Present factor multiplier is > 1 for escalation."""
    cfg = PathFinderConfig()
    assert cfg.pres_fac_mult > 1.0


def test_config_pres_fac_max_greater_than_init():
    """Present factor max is greater than init."""
    cfg = PathFinderConfig()
    assert cfg.pres_fac_max > cfg.pres_fac_init


def test_config_via_cost_positive():
    """Via cost is positive."""
    cfg = PathFinderConfig()
    assert cfg.via_cost > 0


def test_config_default_layer_names():
    """Default layer names include F.Cu and B.Cu."""
    cfg = PathFinderConfig()
    assert "F.Cu" in cfg.layer_names
    assert "B.Cu" in cfg.layer_names


def test_config_stagnation_patience_positive():
    """Stagnation patience is positive."""
    cfg = PathFinderConfig()
    assert cfg.stagnation_patience > 0


def test_config_mode_default():
    """Default mode is 'near_far'."""
    cfg = PathFinderConfig()
    assert cfg.mode == "near_far"
