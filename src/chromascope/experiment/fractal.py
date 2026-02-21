"""
Fractal texture generators and visualizer.

Hot-path functions (julia_set, mandelbrot_zoom) are accelerated with
Numba @njit(parallel=True) when available — the same quality, ~15-40x faster
for 1080p overnight renders.  Falls back to the original NumPy implementation
transparently when Numba is not installed.

All outputs are float32 arrays in [0, 1] representing escape-time or intensity
values ready for palette mapping.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chromascope.experiment.base import BaseConfig, BaseVisualizer
from chromascope.experiment.kaleidoscope import (
    flow_field_warp,
    infinite_zoom_blend,
    polar_mirror,
    radial_warp,
)

# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------
try:
    import numba as _numba

    _NUMBA_OK: bool = True
except ImportError:  # pragma: no cover
    _numba = None  # type: ignore[assignment]
    _NUMBA_OK = False


if _NUMBA_OK:
    _LOG2 = 0.6931471805599453  # math.log(2) — scalar constant for kernel

    @_numba.njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
    def _julia_kernel(
        re_grid: np.ndarray,
        im_grid: np.ndarray,
        c_re: float,
        c_im: float,
        max_iter: int,
        escaped_out: np.ndarray,
        interior_mag: np.ndarray,
    ) -> None:
        """Parallel JIT kernel for Julia set iteration.

        Fills *escaped_out* with raw smooth-colouring values for escaped pixels
        (0.0 for interior) and *interior_mag* with |z_final| for interior pixels
        (-1.0 for escaped).  Caller normalises both arrays afterward.
        """
        h, w = re_grid.shape
        for i in _numba.prange(h):  # type: ignore[attr-defined]
            for j in range(w):
                # Cast to float64 locally for precision in the inner loop
                zr = float(re_grid[i, j])
                zi = float(im_grid[i, j])
                n = 0
                while n < max_iter and zr * zr + zi * zi <= 4.0:
                    zr, zi = zr * zr - zi * zi + c_re, 2.0 * zr * zi + c_im
                    n += 1
                # Use modulus test (not n < max_iter) to detect escape.
                # The while condition is tested BEFORE each application, so a
                # pixel that first exceeds |z|=2 on the very last application
                # has n==max_iter after the loop but |z|^2 > 4 — it escaped.
                if zr * zr + zi * zi > 4.0:
                    # Escaped — smooth (continuous) colouring
                    # Formula matches numpy path: smooth = n - log2(log2(|z|))
                    # Clamp to avoid log(log(x)) being negative near |z|=2
                    mag2 = zr * zr + zi * zi
                    if mag2 < 1.002001:  # sqrt(1.002001) ≈ 1.001
                        mag2 = 1.002001
                    log_z = math.log(math.sqrt(mag2))
                    smooth = float(n) - math.log(log_z / _LOG2) / _LOG2
                    escaped_out[i, j] = smooth
                    interior_mag[i, j] = -1.0
                else:
                    # Interior — record |z_final| for orbit-trap colouring
                    escaped_out[i, j] = 0.0
                    interior_mag[i, j] = math.sqrt(zr * zr + zi * zi)

    @_numba.njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
    def _mandelbrot_kernel(
        re_grid: np.ndarray,
        im_grid: np.ndarray,
        max_iter: int,
        escaped_out: np.ndarray,
        interior_mag: np.ndarray,
    ) -> None:
        """Parallel JIT kernel for Mandelbrot set iteration."""
        h, w = re_grid.shape
        for i in _numba.prange(h):  # type: ignore[attr-defined]
            for j in range(w):
                c_re = float(re_grid[i, j])
                c_im = float(im_grid[i, j])
                zr = 0.0
                zi = 0.0
                n = 0
                while n < max_iter and zr * zr + zi * zi <= 4.0:
                    zr, zi = zr * zr - zi * zi + c_re, 2.0 * zr * zi + c_im
                    n += 1
                if zr * zr + zi * zi > 4.0:  # escaped (handles last-iter escape)
                    mag2 = zr * zr + zi * zi
                    if mag2 < 1.002001:
                        mag2 = 1.002001
                    log_z = math.log(math.sqrt(mag2))
                    smooth = float(n) - math.log(log_z / _LOG2) / _LOG2
                    escaped_out[i, j] = smooth
                    interior_mag[i, j] = -1.0
                else:
                    escaped_out[i, j] = 0.0
                    interior_mag[i, j] = math.sqrt(zr * zr + zi * zi)


def _apply_kernel_output(
    escaped_out: np.ndarray,
    interior_mag: np.ndarray,
) -> np.ndarray:
    """Normalize kernel output arrays into a [0, 1] float32 result.

    Quality-identical to the original NumPy path:
      • escaped pixels  → escaped_out / max(escaped_out)
      • interior pixels → |z_final| / max(|z_final|) * 0.35
    """
    output = escaped_out  # reuse pre-allocated buffer

    max_val = output.max()
    if max_val > 0:
        output /= max_val

    interior_mask = interior_mag >= 0.0
    if interior_mask.any():
        iz = interior_mag[interior_mask]
        iz_max = iz.max()
        if iz_max > 0:
            output[interior_mask] = (iz / iz_max) * 0.35

    # Smooth colouring CAN produce slightly negative values (e.g. for pixels
    # that escape in 1 iteration with large |z|).  Clamp so callers always
    # receive a well-behaved [0, 1] field.
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def _warmup_fractal_jit() -> None:
    """Trigger JIT compilation once at import time on a tiny 4×4 render.

    With cache=True the compiled object is stored to disk so subsequent process
    starts skip the ~30 s compilation and load in ~0.5 s instead.
    """
    if not _NUMBA_OK:
        return
    _h, _w = 4, 4
    _re = np.linspace(-1.5, 1.5, _w, dtype=np.float64).reshape(1, _w).repeat(_h, axis=0)
    _im = np.linspace(-1.0, 1.0, _h, dtype=np.float64).reshape(_h, 1).repeat(_w, axis=1)
    _out = np.zeros((_h, _w), dtype=np.float64)
    _mag = np.full((_h, _w), -1.0, dtype=np.float64)
    _julia_kernel(_re, _im, -0.8, 0.156, 4, _out, _mag)
    _mandelbrot_kernel(_re, _im, 4, _out, _mag)


# ---------------------------------------------------------------------------
# Public texture generators
# ---------------------------------------------------------------------------

def julia_set(
    width: int,
    height: int,
    c: complex,
    center: complex = 0 + 0j,
    zoom: float = 1.0,
    max_iter: int = 256,
) -> np.ndarray:
    """Render Julia set escape-time values.

    Returns a float32 array of shape (height, width) in [0, 1].
    Uses Numba parallel JIT when available; falls back to NumPy otherwise.
    """
    aspect = width / height
    r_span = 3.0 / zoom
    i_span = r_span / aspect

    # float64 grids — ensures complex128 z arithmetic in the NumPy fallback,
    # matching the float64 Numba kernel for parity.
    re = np.linspace(
        center.real - r_span / 2,
        center.real + r_span / 2,
        width,
        dtype=np.float64,
    )
    im = np.linspace(
        center.imag - i_span / 2,
        center.imag + i_span / 2,
        height,
        dtype=np.float64,
    )
    re_grid, im_grid = np.meshgrid(re, im)

    if _NUMBA_OK:
        escaped_out = np.zeros((height, width), dtype=np.float64)
        interior_mag = np.full((height, width), -1.0, dtype=np.float64)
        _julia_kernel(
            re_grid, im_grid,
            float(c.real), float(c.imag),
            int(max_iter),
            escaped_out, interior_mag,
        )
        return _apply_kernel_output(escaped_out, interior_mag)

    # --- NumPy fallback (float64 arithmetic, same precision as Numba kernel) ---
    z = re_grid + 1j * im_grid   # complex128 since re_grid is float64
    output = np.zeros((height, width), dtype=np.float64)
    mask = np.ones((height, width), dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + c
        escaped = mask & (np.abs(z) > 2.0)
        if np.any(escaped):
            abs_z = np.abs(z[escaped])
            smooth_val = i + 1 - np.log2(np.log2(np.maximum(abs_z, 1.001)))
            output[escaped] = smooth_val
        mask &= ~escaped

    max_val = output.max()
    if max_val > 0:
        output /= max_val

    if np.any(mask):
        interior_z = np.abs(z[mask])
        iz_max = interior_z.max()
        if iz_max > 0:
            output[mask] = (interior_z / iz_max) * 0.35

    return np.clip(output, 0.0, 1.0).astype(np.float32)


def mandelbrot_zoom(
    width: int,
    height: int,
    center: complex = -0.75 + 0.1j,
    zoom: float = 1.0,
    max_iter: int = 256,
) -> np.ndarray:
    """Render Mandelbrot set at a given zoom and center.

    Returns a float32 array of shape (height, width) in [0, 1].
    Uses Numba parallel JIT when available; falls back to NumPy otherwise.
    """
    aspect = width / height
    r_span = 3.5 / zoom
    i_span = r_span / aspect

    re = np.linspace(
        center.real - r_span / 2,
        center.real + r_span / 2,
        width,
        dtype=np.float64,
    )
    im = np.linspace(
        center.imag - i_span / 2,
        center.imag + i_span / 2,
        height,
        dtype=np.float64,
    )
    re_grid, im_grid = np.meshgrid(re, im)

    if _NUMBA_OK:
        escaped_out = np.zeros((height, width), dtype=np.float64)
        interior_mag = np.full((height, width), -1.0, dtype=np.float64)
        _mandelbrot_kernel(
            re_grid, im_grid,
            int(max_iter),
            escaped_out, interior_mag,
        )
        return _apply_kernel_output(escaped_out, interior_mag)

    # --- NumPy fallback (float64 arithmetic, same precision as Numba kernel) ---
    c = re_grid + 1j * im_grid   # complex128 since re_grid is float64
    z = np.zeros_like(c)
    output = np.zeros((height, width), dtype=np.float64)
    mask = np.ones((height, width), dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + c[mask]
        escaped = mask & (np.abs(z) > 2.0)
        if np.any(escaped):
            abs_z = np.abs(z[escaped])
            smooth_val = i + 1 - np.log2(np.log2(np.maximum(abs_z, 1.001)))
            output[escaped] = smooth_val
        mask &= ~escaped

    max_val = output.max()
    if max_val > 0:
        output /= max_val

    if np.any(mask):
        interior_z = np.abs(z[mask])
        iz_max = interior_z.max()
        if iz_max > 0:
            output[mask] = (interior_z / iz_max) * 0.35

    return np.clip(output, 0.0, 1.0).astype(np.float32)


def noise_fractal(
    width: int,
    height: int,
    time: float = 0.0,
    octaves: int = 4,
    scale: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Multi-octave sine-based fractal noise field."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, scale, width, dtype=np.float32)
    y = np.linspace(0, scale, height, dtype=np.float32)
    xg, yg = np.meshgrid(x, y)

    output = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0

    for octave in range(octaves):
        freq = 2.0 ** octave
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        angle = rng.uniform(0, np.pi)

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xr = xg * cos_a - yg * sin_a
        yr = xg * sin_a + yg * cos_a

        layer = np.sin(xr * freq * 2 * np.pi + phase_x + time * (octave + 1) * 0.5)
        layer += np.sin(yr * freq * 2 * np.pi + phase_y + time * (octave + 1) * 0.3)
        layer *= 0.5

        output += layer * amplitude
        amplitude *= 0.5

    output = (output - output.min()) / (output.max() - output.min() + 1e-8)
    return output


JULIA_C_PATH = [
    -0.7269 + 0.1889j,
    -0.8 + 0.156j,
    -0.4 + 0.6j,
    0.285 + 0.01j,
    0.285 + 0.0j,
    -0.70176 - 0.3842j,
    -0.835 - 0.2321j,
    -0.1 + 0.651j,
    0.0 + 0.8j,
    -0.7269 + 0.1889j,
]


def interpolate_c(t: float) -> complex:
    """Interpolate along the curated Julia c-value path."""
    t = t % 1.0
    n = len(JULIA_C_PATH) - 1
    segment = t * n
    idx = int(segment)
    frac = segment - idx
    idx = min(idx, n - 1)

    c0 = JULIA_C_PATH[idx]
    c1 = JULIA_C_PATH[idx + 1]
    frac = frac * frac * (3 - 2 * frac)

    return complex(
        c0.real + (c1.real - c0.real) * frac,
        c0.imag + (c1.imag - c0.imag) * frac,
    )


@dataclass
class FractalConfig(BaseConfig):
    """Configuration for the fractal kaleidoscope renderer."""
    num_segments: int = 8
    fractal_mode: str = "blend"
    base_zoom_speed: float = 1.0
    zoom_beat_punch: float = 1.08
    base_rotation_speed: float = 1.0
    base_max_iter: int = 200
    max_max_iter: int = 400
    feedback_alpha: float = 0.20
    base_zoom_factor: float = 1.015
    warp_amplitude: float = 0.03
    warp_frequency: float = 4.0
    mandelbrot_center: complex = -0.7435669 + 0.1314023j


class FractalKaleidoscopeRenderer(BaseVisualizer):
    """
    Renders audio-reactive fractal kaleidoscope frames.
    Modernized for the OPEN UP architecture.
    """

    def __init__(
        self,
        config: FractalConfig | None = None,
        seed: int | None = None,
        center_pos: Tuple[float, float] | None = None,
    ):
        super().__init__(config or FractalConfig(), seed, center_pos)
        self.cfg: FractalConfig = self.cfg

        # State
        self.accumulated_rotation = 0.0
        self.julia_t = 0.0
        self._drift_phase = 0.0
        self._last_good_c = complex(-0.7269, 0.1889)

        # Feedback buffer for infinite zoom
        self.feedback_field: np.ndarray | None = None

    def _probe_c(self, c: complex, zoom: float, probe_iter: int) -> bool:
        probe = julia_set(32, 24, c=c, center=complex(0, 0), zoom=zoom, max_iter=probe_iter)
        boundary_frac = float((probe > 0.4).mean())
        return 0.10 < boundary_frac < 0.85

    def _pick_best_c(self, julia_c: complex, zoom: float, probe_iter: int) -> complex:
        if self._probe_c(julia_c, zoom, probe_iter):
            self._last_good_c = julia_c
            return julia_c
        if self._probe_c(self._last_good_c, zoom, probe_iter):
            return self._last_good_c
        return JULIA_C_PATH[0]

    def update(self, frame_data: Dict[str, Any]):
        """Advance the fractal simulation."""
        dt = 1.0 / self.cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)

        # Rotation
        rotation_delta = (
            0.01 * self.cfg.base_rotation_speed *
            (1.0 + self._smooth_harmonic * 2.0 + self._smooth_brilliance * 3.0)
        )
        self.accumulated_rotation += rotation_delta

        # Julia c drift
        c_speed = 0.0003 * (1.0 + self._smooth_harmonic)
        self.julia_t += c_speed

        # Lissajous drift
        self._drift_phase += dt * 0.3

    def get_raw_field(self) -> np.ndarray:
        """Returns the raw float32 escape-time field."""
        cfg = self.cfg
        dt = 1.0 / cfg.fps

        # Max iterations
        max_iter = int(
            cfg.base_max_iter +
            (self._smooth_energy + self._smooth_flux * 0.5) * (cfg.max_max_iter - cfg.base_max_iter)
        )

        # Zoom
        breath = 0.4 * math.sin(self.time * 0.4)
        fractal_zoom = 1.0 + self._smooth_low * 0.5 + self._smooth_sub_bass * 0.8 + breath
        fractal_zoom = max(0.6, min(fractal_zoom, 2.5))

        # Drift
        drift_x = math.sin(self._drift_phase * 1.3) * 0.25 / max(fractal_zoom, 1)
        drift_y = math.cos(self._drift_phase * 0.9) * 0.18 / max(fractal_zoom, 1)

        julia_c = interpolate_c(self.julia_t)
        use_mandelbrot = cfg.fractal_mode == "mandelbrot"

        if not use_mandelbrot:
            probe_iter = min(max_iter, 100)
            effective_c = self._pick_best_c(julia_c, fractal_zoom, probe_iter)
        else:
            effective_c = julia_c

        # Generate core texture
        if use_mandelbrot:
            texture = mandelbrot_zoom(
                cfg.width, cfg.height,
                center=cfg.mandelbrot_center + complex(drift_x * 0.01, drift_y * 0.01),
                zoom=fractal_zoom * 0.5,
                max_iter=max_iter,
            )
        else:
            texture = julia_set(
                cfg.width, cfg.height,
                c=effective_c,
                center=complex(drift_x, drift_y),
                zoom=fractal_zoom,
                max_iter=max_iter,
            )

        # Organic noise
        if self._smooth_energy > 0.3 or self._smooth_flatness > 0.2:
            noise = noise_fractal(
                cfg.width, cfg.height,
                time=self.time,
                octaves=4,
                scale=2.0 + self._smooth_harmonic * 2 + self._smooth_flatness * 4,
                seed=42,
            )
            noise_blend = 0.03 + self._smooth_energy * 0.04 + self._smooth_flatness * 0.1
            texture = texture * (1 - noise_blend) + noise * noise_blend

        # Vapor Warp 2.0 — Perlin-style flow-field displacement
        warp_amp = cfg.warp_amplitude * (0.5 + self._smooth_low * 1.5 + self._smooth_flux * 1.0)
        if warp_amp > 0.005:
            texture = flow_field_warp(
                texture,
                amplitude=warp_amp,
                scale=cfg.warp_frequency,
                time=self.time * 2,
                octaves=3,
            )

        # Kaleidoscope mirror
        seg_mod = int(self._smooth_high * 4 + self._smooth_flux * 2)
        num_seg = max(4, cfg.num_segments + seg_mod - 2)

        texture = polar_mirror(
            texture,
            num_segments=num_seg,
            rotation=self.accumulated_rotation,
        )

        # Infinite zoom feedback (field-level)
        if self.feedback_field is not None:
            zoom_f = cfg.base_zoom_factor * (1.0 + self._smooth_energy * 0.01 + self._smooth_sub_bass * 0.02)
            alpha = cfg.feedback_alpha * (1.0 - self._smooth_percussive * 0.6)
            texture = texture * (1 - alpha) + self.feedback_field * alpha

        self.feedback_field = texture.copy()

        return texture


# ---------------------------------------------------------------------------
# Trigger JIT compilation at import time (no-op if Numba not installed)
# ---------------------------------------------------------------------------
_warmup_fractal_jit()
