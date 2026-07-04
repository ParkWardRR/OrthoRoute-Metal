"""Tests for layer analysis: direction detection, utilization."""
import pytest

from orthoroute.algorithms.manhattan.layer_analyzer import LayerAnalyzer


def test_layer_analyzer_creation():
    """LayerAnalyzer initializes with correct layer count."""
    la = LayerAnalyzer(layer_count=6)
    assert la.layer_count == 6


def test_layer_analyzer_layer_usage_default():
    """Layer usage defaults to empty."""
    la = LayerAnalyzer(layer_count=4)
    assert len(la.layer_usage) == 0


def test_layer_recommendation_not_needed_low_overuse():
    """Low overuse does not trigger more-layers recommendation."""
    la = LayerAnalyzer(layer_count=6)
    layer_stats = {
        i: {'total_edges': 1000, 'used_edges': 200, 'overuse': 5, 'utilization': 0.2}
        for i in range(6)
    }
    result = la._calculate_layer_recommendation(layer_stats, total_overuse=30)
    assert result['needed'] is False


def test_layer_recommendation_needed_high_overuse():
    """Severe overuse triggers more-layers recommendation."""
    la = LayerAnalyzer(layer_count=4)
    layer_stats = {
        i: {'total_edges': 1000, 'used_edges': 850, 'overuse': 100, 'utilization': 0.85}
        for i in range(4)
    }
    result = la._calculate_layer_recommendation(layer_stats, total_overuse=400)
    assert result['needed'] is True
    assert result['recommended_count'] > 4


def test_layer_recommendation_severe_congestion():
    """Even with low utilization, severe overuse triggers recommendation."""
    la = LayerAnalyzer(layer_count=4)
    layer_stats = {
        i: {'total_edges': 1000, 'used_edges': 200, 'overuse': 60, 'utilization': 0.2}
        for i in range(4)
    }
    result = la._calculate_layer_recommendation(layer_stats, total_overuse=240)
    assert result['needed'] is True


def test_layer_recommendation_reason_present():
    """Recommendation always includes a reason string."""
    la = LayerAnalyzer(layer_count=2)
    layer_stats = {
        0: {'total_edges': 100, 'used_edges': 10, 'overuse': 0, 'utilization': 0.1},
        1: {'total_edges': 100, 'used_edges': 10, 'overuse': 0, 'utilization': 0.1},
    }
    result = la._calculate_layer_recommendation(layer_stats, total_overuse=0)
    assert 'reason' in result
    assert len(result['reason']) > 0
