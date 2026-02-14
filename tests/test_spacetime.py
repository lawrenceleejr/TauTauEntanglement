"""Tests for spacetime interval calculations in spacetime.py."""
import numpy as np
import pytest
from spacetime import compute_spacetime_interval
from config import C_LIGHT


def _make_vtx(t_s, x_m, y_m, z_m):
    """Helper: create a vertex dict."""
    return {
        'decay_vertex_xyz': np.array([x_m, y_m, z_m]),
        'decay_vertex_t': t_s,
    }


class TestSpacetimeInterval:
    """Tests for compute_spacetime_interval."""

    def test_lightlike(self):
        """Lightlike separation: ds2 = 0."""
        d = 1e-3  # 1 mm
        t = d / C_LIGHT  # time for light to travel 1 mm
        vtx1 = _make_vtx(0, 0, 0, 0)
        vtx2 = _make_vtx(t, d, 0, 0)
        result = compute_spacetime_interval(vtx1, vtx2)
        assert abs(result['ds2_m2']) < 1e-20
        assert abs(result['v_signal_c'] - 1.0) < 1e-8

    def test_spacelike(self):
        """Spacelike separation: ds2 < 0, v_signal > c."""
        d = 1e-3
        t = d / (2 * C_LIGHT)  # half the time light would need
        vtx1 = _make_vtx(0, 0, 0, 0)
        vtx2 = _make_vtx(t, d, 0, 0)
        result = compute_spacetime_interval(vtx1, vtx2)
        assert result['ds2_m2'] < 0
        assert bool(result['is_spacelike']) is True
        assert result['v_signal_c'] > 1.0

    def test_timelike(self):
        """Timelike separation: ds2 > 0, v_signal < c."""
        d = 1e-3
        t = d / (0.5 * C_LIGHT)  # twice the time light would need
        vtx1 = _make_vtx(0, 0, 0, 0)
        vtx2 = _make_vtx(t, d, 0, 0)
        result = compute_spacetime_interval(vtx1, vtx2)
        assert result['ds2_m2'] > 0
        assert bool(result['is_spacelike']) is False
        assert result['v_signal_c'] < 1.0

    def test_same_point(self):
        """Same spacetime point: everything zero."""
        vtx = _make_vtx(1e-12, 1e-3, 2e-3, 3e-3)
        result = compute_spacetime_interval(vtx, vtx)
        assert abs(result['dr_m']) < 1e-15
        assert abs(result['ds2_m2']) < 1e-20
        assert result['v_signal_c'] == 0.0

    def test_signed_ds_sign(self):
        """signed_ds should be negative for spacelike, positive for timelike."""
        d = 1e-3
        # Spacelike
        t_sl = d / (2 * C_LIGHT)
        vtx1 = _make_vtx(0, 0, 0, 0)
        vtx2 = _make_vtx(t_sl, d, 0, 0)
        r_sl = compute_spacetime_interval(vtx1, vtx2)
        assert r_sl['signed_ds_m'] < 0

        # Timelike
        t_tl = d / (0.5 * C_LIGHT)
        vtx3 = _make_vtx(t_tl, d, 0, 0)
        r_tl = compute_spacetime_interval(vtx1, vtx3)
        assert r_tl['signed_ds_m'] > 0

    def test_mm_conversion(self):
        """signed_ds_mm should be 1000x signed_ds_m."""
        vtx1 = _make_vtx(0, 0, 0, 0)
        vtx2 = _make_vtx(1e-12, 1e-3, 0, 0)
        result = compute_spacetime_interval(vtx1, vtx2)
        assert abs(result['signed_ds_mm'] - result['signed_ds_m'] * 1e3) < 1e-10

    def test_simultaneous_events(self):
        """dt = 0, dr > 0: v_signal = inf, spacelike."""
        vtx1 = _make_vtx(0, 0, 0, 0)
        vtx2 = _make_vtx(0, 1e-3, 0, 0)
        result = compute_spacetime_interval(vtx1, vtx2)
        assert result['v_signal_c'] == np.inf
        assert bool(result['is_spacelike']) is True

    def test_ndarray_input(self):
        """Accept (t, x, y, z) array as input."""
        vtx1 = np.array([0.0, 0.0, 0.0, 0.0])
        vtx2 = np.array([1e-12, 1e-3, 0.0, 0.0])
        result = compute_spacetime_interval(vtx1, vtx2)
        assert 'dr_m' in result
        assert result['dr_m'] > 0
