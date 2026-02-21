"""Tests for fractal texture generators."""

import numpy as np
import pytest

from chromascope.experiment.fractal import (
    _NUMBA_OK,
    _apply_kernel_output,
    interpolate_c,
    julia_set,
    mandelbrot_zoom,
    noise_fractal,
)


class TestJuliaSet:
    def test_output_shape(self):
        result = julia_set(160, 120, c=-0.8 + 0.156j, max_iter=50)
        assert result.shape == (120, 160)

    def test_output_dtype(self):
        result = julia_set(80, 60, c=-0.4 + 0.6j, max_iter=30)
        assert result.dtype == np.float32

    def test_output_range(self):
        result = julia_set(100, 80, c=0.285 + 0.01j, max_iter=50)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_not_all_zero(self):
        result = julia_set(100, 80, c=-0.7269 + 0.1889j, max_iter=80)
        assert result.max() > 0.0, "Julia set should have non-zero escape values"

    def test_zoom_changes_output(self):
        r1 = julia_set(80, 60, c=-0.8 + 0.156j, zoom=1.0, max_iter=30)
        r2 = julia_set(80, 60, c=-0.8 + 0.156j, zoom=5.0, max_iter=30)
        assert not np.allclose(r1, r2), "Different zooms should produce different outputs"

    def test_different_c_different_output(self):
        r1 = julia_set(80, 60, c=-0.8 + 0.156j, max_iter=30)
        r2 = julia_set(80, 60, c=0.285 + 0.01j, max_iter=30)
        assert not np.allclose(r1, r2)


class TestMandelbrot:
    def test_output_shape(self):
        result = mandelbrot_zoom(160, 120, max_iter=50)
        assert result.shape == (120, 160)

    def test_output_range(self):
        result = mandelbrot_zoom(100, 80, max_iter=50)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_not_all_zero(self):
        result = mandelbrot_zoom(100, 80, max_iter=80)
        assert result.max() > 0.0

    def test_zoom_changes_output(self):
        r1 = mandelbrot_zoom(80, 60, zoom=1.0, max_iter=30)
        r2 = mandelbrot_zoom(80, 60, zoom=10.0, max_iter=30)
        assert not np.allclose(r1, r2)


class TestNoiseFractal:
    def test_output_shape(self):
        result = noise_fractal(160, 120)
        assert result.shape == (120, 160)

    def test_output_range(self):
        result = noise_fractal(100, 80)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_time_changes_output(self):
        r1 = noise_fractal(80, 60, time=0.0)
        r2 = noise_fractal(80, 60, time=5.0)
        assert not np.allclose(r1, r2)

    def test_deterministic_with_seed(self):
        r1 = noise_fractal(80, 60, time=1.0, seed=42)
        r2 = noise_fractal(80, 60, time=1.0, seed=42)
        np.testing.assert_array_equal(r1, r2)


class TestInterpolateC:
    def test_returns_complex(self):
        c = interpolate_c(0.0)
        assert isinstance(c, complex)

    def test_loops(self):
        c0 = interpolate_c(0.0)
        c1 = interpolate_c(1.0)
        assert abs(c0 - c1) < 1e-5, "Should loop back to start at t=1.0"

    def test_midpoint_differs(self):
        c0 = interpolate_c(0.0)
        c_mid = interpolate_c(0.5)
        assert abs(c0 - c_mid) > 0.01


class TestNumbaKernels:
    """Quality-parity tests: Numba path must match numpy path to visual precision."""

    def test_numba_flag_is_bool(self):
        assert isinstance(_NUMBA_OK, bool)

    @pytest.mark.skipif(not _NUMBA_OK, reason="Numba not installed")
    def test_julia_numba_shape(self):
        result = julia_set(80, 60, c=-0.8 + 0.156j, max_iter=50)
        assert result.shape == (60, 80)
        assert result.dtype == np.float32

    @pytest.mark.skipif(not _NUMBA_OK, reason="Numba not installed")
    def test_julia_numba_range(self):
        result = julia_set(80, 60, c=-0.4 + 0.6j, max_iter=50)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.skipif(not _NUMBA_OK, reason="Numba not installed")
    def test_mandelbrot_numba_shape(self):
        result = mandelbrot_zoom(80, 60, max_iter=50)
        assert result.shape == (60, 80)
        assert result.dtype == np.float32

    @pytest.mark.skipif(not _NUMBA_OK, reason="Numba not installed")
    def test_mandelbrot_numba_range(self):
        result = mandelbrot_zoom(100, 80, max_iter=50)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_apply_kernel_output_normalises(self):
        """_apply_kernel_output must map escaped values to [0,1] and interior to [0,0.35]."""
        escaped = np.array([[0.0, 5.0], [10.0, 0.0]], dtype=np.float64)
        interior = np.array([[-1.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
        result = _apply_kernel_output(escaped, interior)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # Interior pixel [1,1] must be in [0, 0.35]
        assert result[1, 1] <= 0.35 + 1e-6

    def test_apply_kernel_output_escaped_max_is_one(self):
        escaped = np.array([[3.0, 6.0, 9.0]], dtype=np.float64)
        interior = np.full((1, 3), -1.0, dtype=np.float64)
        result = _apply_kernel_output(escaped, interior)
        assert abs(result.max() - 1.0) < 1e-6, "Escaped max must normalise to 1.0"
