"""Tests for ORP/ORS serialization utilities."""
import pytest

from orthoroute.infrastructure.serialization import (
    derive_orp_filename, derive_ors_filename, get_solution_summary,
)


def test_derive_orp_filename():
    """ORP filename derived from board filename."""
    result = derive_orp_filename("MainController.kicad_pcb")
    assert result.endswith(".ORP")
    assert "MainController" in result


def test_derive_ors_filename():
    """ORS filename derived from ORP filename."""
    result = derive_ors_filename("MainController.ORP")
    assert result.endswith(".ORS")
    assert "MainController" in result


def test_derive_ors_from_orp_chain():
    """ORP -> ORS derivation chain works."""
    orp = derive_orp_filename("board.kicad_pcb")
    ors = derive_ors_filename(orp)
    assert ors.endswith(".ORS")


def test_solution_summary_new_format():
    """get_solution_summary handles new format data."""
    ors_data = {
        'metadata': {
            'export_timestamp': '2024-01-01', 'orthoroute_version': '1.0.0',
            'converged': True, 'total_iterations': 25, 'total_time_seconds': 10.5,
        },
        'statistics': {
            'nets_routed': 100, 'total_wirelength_mm': 5000.0,
            'total_vias': 50, 'final_overflow_cost': 0, 'total_tracks': 200,
        },
        'geometry': {'by_net': {'N1': {}, 'N2': {}}},
    }
    summary = get_solution_summary(ors_data)
    assert "Nets Routed: 100" in summary
    assert "Total Wirelength: 5000.0" in summary


def test_solution_summary_old_format():
    """get_solution_summary handles old format data."""
    ors_data = {
        'metadata': {'timestamp': '2024-01-01', 'orthoroute_version': '0.9.0'},
        'metrics': {
            'final': {
                'converged': False, 'iterations': 10, 'total_time': 5.0,
                'nets_routed': 50, 'wirelength': 2500.0, 'via_count': 20, 'overflow': 5,
            }
        },
        'nets': {
            'N1': {'traces': [1, 2, 3], 'vias': [1]},
            'N2': {'traces': [4, 5], 'vias': []},
        },
    }
    summary = get_solution_summary(ors_data)
    assert "Nets Routed: 50" in summary


def test_solution_summary_error_handling():
    """get_solution_summary handles invalid data gracefully."""
    summary = get_solution_summary({})
    assert isinstance(summary, str)
