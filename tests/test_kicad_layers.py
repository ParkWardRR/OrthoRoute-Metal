"""Tests for layer utilities and KiCad color scheme.

Covers: norm_layer (int and string inputs, edge cases, error handling),
get_layer_stackup (size and contents), LAYER_NAME_RE regex matching,
and Qt-guarded KiCadColorScheme tests for layer/pad colors.
"""
import re
import pytest

from orthoroute.shared.utils.layers import norm_layer, get_layer_stackup, LAYER_NAME_RE

# Conditional import for Qt-dependent tests
try:
    from orthoroute.presentation.gui.kicad_colors import KiCadColorScheme
    HAS_QT = True
except ImportError:
    HAS_QT = False


# ---------------------------------------------------------------------------
# norm_layer tests
# ---------------------------------------------------------------------------


class TestNormLayer:
    """Tests for the norm_layer() function."""

    def test_norm_layer_int_0_is_fcu(self):
        """norm_layer(0) returns 'F.Cu'."""
        assert norm_layer(0) == "F.Cu"

    def test_norm_layer_int_5_is_bcu(self):
        """norm_layer(5) returns 'B.Cu'."""
        assert norm_layer(5) == "B.Cu"

    def test_norm_layer_int_1_is_in1cu(self):
        """norm_layer(1) returns 'In1.Cu'."""
        assert norm_layer(1) == "In1.Cu"

    def test_norm_layer_int_2_is_in2cu(self):
        """norm_layer(2) returns 'In2.Cu'."""
        assert norm_layer(2) == "In2.Cu"

    def test_norm_layer_int_3_is_in3cu(self):
        """norm_layer(3) returns 'In3.Cu'."""
        assert norm_layer(3) == "In3.Cu"

    def test_norm_layer_int_4_is_in4cu(self):
        """norm_layer(4) returns 'In4.Cu'."""
        assert norm_layer(4) == "In4.Cu"

    def test_norm_layer_string_passthrough(self):
        """norm_layer('F.Cu') returns 'F.Cu' unchanged."""
        assert norm_layer("F.Cu") == "F.Cu"

    def test_norm_layer_string_bcu(self):
        """norm_layer('B.Cu') returns 'B.Cu'."""
        assert norm_layer("B.Cu") == "B.Cu"

    def test_norm_layer_string_in2cu(self):
        """norm_layer('In2.Cu') returns 'In2.Cu'."""
        assert norm_layer("In2.Cu") == "In2.Cu"

    def test_norm_layer_invalid_string_raises(self):
        """norm_layer('InvalidLayer') raises ValueError."""
        with pytest.raises(ValueError, match="Unknown layer string"):
            norm_layer("InvalidLayer")

    def test_norm_layer_invalid_int_raises(self):
        """norm_layer(99) raises ValueError."""
        with pytest.raises(ValueError, match="Unknown layer index"):
            norm_layer(99)

    def test_norm_layer_negative_int_raises(self):
        """norm_layer(-1) raises ValueError."""
        with pytest.raises(ValueError, match="Unknown layer index"):
            norm_layer(-1)

    def test_norm_layer_int_6_raises(self):
        """norm_layer(6) raises ValueError (only 0-5 mapped)."""
        with pytest.raises(ValueError, match="Unknown layer index"):
            norm_layer(6)

    def test_norm_layer_in30cu_valid(self):
        """norm_layer('In30.Cu') returns 'In30.Cu' (matches regex)."""
        assert norm_layer("In30.Cu") == "In30.Cu"

    def test_norm_layer_string_with_whitespace(self):
        """norm_layer(' F.Cu ') strips whitespace and returns 'F.Cu'."""
        assert norm_layer(" F.Cu ") == "F.Cu"

    def test_norm_layer_xcu_invalid(self):
        """norm_layer('X.Cu') raises ValueError."""
        with pytest.raises(ValueError):
            norm_layer("X.Cu")

    def test_norm_layer_fcu_no_dot_invalid(self):
        """norm_layer('FCu') raises ValueError."""
        with pytest.raises(ValueError):
            norm_layer("FCu")

    def test_norm_layer_all_valid_ints(self):
        """All valid int indices (0-5) return valid layer names."""
        expected = {0: "F.Cu", 1: "In1.Cu", 2: "In2.Cu",
                    3: "In3.Cu", 4: "In4.Cu", 5: "B.Cu"}
        for idx, name in expected.items():
            assert norm_layer(idx) == name


# ---------------------------------------------------------------------------
# get_layer_stackup tests
# ---------------------------------------------------------------------------


class TestGetLayerStackup:
    """Tests for the get_layer_stackup() function."""

    def test_get_layer_stackup_contains_all(self):
        """get_layer_stackup() returns set with F.Cu, B.Cu, In1-In4.Cu."""
        stackup = get_layer_stackup()
        for name in ["F.Cu", "B.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu"]:
            assert name in stackup

    def test_get_layer_stackup_size(self):
        """len(get_layer_stackup()) == 6."""
        assert len(get_layer_stackup()) == 6

    def test_get_layer_stackup_returns_set(self):
        """get_layer_stackup() returns a set type."""
        assert isinstance(get_layer_stackup(), set)

    def test_get_layer_stackup_no_extras(self):
        """get_layer_stackup() contains exactly the 6 standard layers."""
        expected = {"F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "B.Cu"}
        assert get_layer_stackup() == expected


# ---------------------------------------------------------------------------
# LAYER_NAME_RE regex tests
# ---------------------------------------------------------------------------


class TestLayerNameRegex:
    """Tests for LAYER_NAME_RE regex pattern."""

    def test_layer_name_regex_fcu(self):
        """LAYER_NAME_RE.match('F.Cu') is not None."""
        assert LAYER_NAME_RE.match("F.Cu") is not None

    def test_layer_name_regex_bcu(self):
        """LAYER_NAME_RE.match('B.Cu') is not None."""
        assert LAYER_NAME_RE.match("B.Cu") is not None

    def test_layer_name_regex_in1cu(self):
        """LAYER_NAME_RE.match('In1.Cu') is not None."""
        assert LAYER_NAME_RE.match("In1.Cu") is not None

    def test_layer_name_regex_in15cu(self):
        """LAYER_NAME_RE.match('In15.Cu') is not None."""
        assert LAYER_NAME_RE.match("In15.Cu") is not None

    def test_layer_name_regex_in30cu(self):
        """LAYER_NAME_RE.match('In30.Cu') is not None."""
        assert LAYER_NAME_RE.match("In30.Cu") is not None

    def test_layer_name_regex_in0cu(self):
        """LAYER_NAME_RE.match('In0.Cu') is not None (\\d+ matches 0)."""
        assert LAYER_NAME_RE.match("In0.Cu") is not None

    def test_layer_name_regex_invalid_xcu(self):
        """LAYER_NAME_RE.match('X.Cu') is None."""
        assert LAYER_NAME_RE.match("X.Cu") is None

    def test_layer_name_regex_no_dot(self):
        """LAYER_NAME_RE.match('FCu') is None."""
        assert LAYER_NAME_RE.match("FCu") is None

    def test_layer_name_regex_lower_fcu(self):
        """LAYER_NAME_RE.match('f.cu') is None (case sensitive)."""
        assert LAYER_NAME_RE.match("f.cu") is None

    def test_layer_name_regex_empty_string(self):
        """LAYER_NAME_RE.match('') is None."""
        assert LAYER_NAME_RE.match("") is None

    def test_layer_name_regex_silkscreen(self):
        """LAYER_NAME_RE.match('F.SilkS') is None."""
        assert LAYER_NAME_RE.match("F.SilkS") is None

    def test_layer_name_regex_edge_cuts(self):
        """LAYER_NAME_RE.match('Edge.Cuts') is None."""
        assert LAYER_NAME_RE.match("Edge.Cuts") is None

    def test_layer_name_regex_full_match(self):
        """LAYER_NAME_RE anchors to full string – 'F.Cu extra' doesn't match."""
        assert LAYER_NAME_RE.match("F.Cu extra") is None


# ---------------------------------------------------------------------------
# KiCadColorScheme tests (Qt-dependent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_QT, reason="Qt not available")
class TestKiCadColorScheme:
    """Tests for KiCadColorScheme – guarded by Qt availability."""

    def test_color_scheme_creation(self):
        """KiCadColorScheme() doesn't crash (uses defaults if no theme file)."""
        scheme = KiCadColorScheme()
        assert scheme is not None

    def test_color_scheme_has_background(self):
        """get_color('background') returns a valid QColor."""
        scheme = KiCadColorScheme()
        color = scheme.get_color("background")
        assert color is not None
        assert color.isValid()

    def test_layer_color_fcu(self):
        """get_layer_color('F.Cu') returns a valid color."""
        scheme = KiCadColorScheme()
        color = scheme.get_layer_color("F.Cu")
        assert color is not None
        assert color.isValid()

    def test_layer_color_bcu(self):
        """get_layer_color('B.Cu') returns a valid color."""
        scheme = KiCadColorScheme()
        color = scheme.get_layer_color("B.Cu")
        assert color is not None
        assert color.isValid()

    def test_layer_color_internal(self):
        """get_layer_color('In1.Cu') returns a valid color."""
        scheme = KiCadColorScheme()
        color = scheme.get_layer_color("In1.Cu")
        assert color is not None
        assert color.isValid()

    def test_layer_color_fcu_vs_bcu_different(self):
        """F.Cu and B.Cu have different colors."""
        scheme = KiCadColorScheme()
        f_color = scheme.get_layer_color("F.Cu")
        b_color = scheme.get_layer_color("B.Cu")
        assert f_color != b_color

    def test_pad_color_through_hole(self):
        """get_pad_color('through_hole', 'F.Cu') returns a valid color."""
        scheme = KiCadColorScheme()
        color = scheme.get_pad_color("through_hole", "F.Cu")
        assert color is not None
        assert color.isValid()

    def test_pad_color_smd_front(self):
        """get_pad_color('smd', 'F.Cu') returns a valid color."""
        scheme = KiCadColorScheme()
        color = scheme.get_pad_color("smd", "F.Cu")
        assert color is not None
        assert color.isValid()

    def test_pad_color_smd_back(self):
        """get_pad_color('smd', 'B.Cu') returns back pad color."""
        scheme = KiCadColorScheme()
        color = scheme.get_pad_color("smd", "B.Cu")
        assert color is not None
        assert color.isValid()

    def test_pad_color_smd_front_vs_back_different(self):
        """SMD front and back pad colors are different."""
        scheme = KiCadColorScheme()
        front = scheme.get_pad_color("smd", "F.Cu")
        back = scheme.get_pad_color("smd", "B.Cu")
        assert front != back

    def test_get_color_unknown_returns_white(self):
        """get_color with unknown name returns white (255, 255, 255)."""
        from PyQt6.QtGui import QColor
        scheme = KiCadColorScheme()
        color = scheme.get_color("nonexistent_color_xyz")
        assert color == QColor(255, 255, 255)

    def test_color_scheme_has_ratsnest(self):
        """get_color('ratsnest') returns a valid QColor."""
        scheme = KiCadColorScheme()
        color = scheme.get_color("ratsnest")
        assert color is not None
        assert color.isValid()
