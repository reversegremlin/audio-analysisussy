"""
Fractal texture generators and visualizer.

Vectorized with numpy — no per-pixel Python loops.
All outputs are float32 arrays in [0, 1] representing escape-time
or intensity values ready for palette mapping.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chromascope.experiment.base import BaseConfig, BaseVisualizer
from chromascope.experiment.kaleidoscope import (
    infinite_zoom_blend,
    polar_mirror,
    radial_warp,
)


def julia_set(
    width: int,
    height: int,
    c: complex,
    center: complex = 0 + 0j,
    zoom: float = 1.0,
    max_iter: int = 256,
) -> np.ndarray:
    """Render Julia set escape-time values."""
    aspect = width / height
    r_span = 3.0 / zoom
    i_span = r_span / aspect

    re = np.linspace(
        center.real - r_span / 2,
        center.real + r_span / 2,
        width,
        dtype=np.float32,
    )
    im = np.linspace(
        center.imag - i_span / 2,
        center.imag + i_span / 2,
        height,
        dtype=np.float32,
    )

    re_grid, im_grid = np.meshgrid(re, im)
    z = re_grid + 1j * im_grid

    output = np.zeros((height, width), dtype=np.float32)
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

    return output


def mandelbrot_zoom(
    width: int,
    height: int,
    center: complex = -0.75 + 0.1j,
    zoom: float = 1.0,
    max_iter: int = 256,
) -> np.ndarray:
    """Render Mandelbrot set at a given zoom and center."""
    aspect = width / height
    r_span = 3.5 / zoom
    i_span = r_span / aspect

    re = np.linspace(
        center.real - r_span / 2,
        center.real + r_span / 2,
        width,
        dtype=np.float32,
    )
    im = np.linspace(
        center.imag - i_span / 2,
        center.imag + i_span / 2,
        height,
        dtype=np.float32,
    )

    re_grid, im_grid = np.meshgrid(re, im)
    c = re_grid + 1j * im_grid
    z = np.zeros_like(c)

    output = np.zeros((height, width), dtype=np.float32)
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

    return output


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
        center_pos: Tuple[float, float] | None = None
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
                seed=42 # Fixed seed for consistency, or use self.rng
            )
            noise_blend = 0.03 + self._smooth_energy * 0.04 + self._smooth_flatness * 0.1
            texture = texture * (1 - noise_blend) + noise * noise_blend

        # Radial warp
        warp_amp = cfg.warp_amplitude * (0.5 + self._smooth_low * 1.5 + self._smooth_flux * 1.0)
        if warp_amp > 0.005:
            texture = radial_warp(
                texture,
                amplitude=warp_amp,
                frequency=cfg.warp_frequency,
                time=self.time * 2,
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
            # We need a float-based infinite zoom blend or just use the image-based one later.
            # To keep it "Energy First", we should probably do it here.
            # For now, I'll skip field-level feedback or implement a simple version.
            texture = texture * (1 - alpha) + self.feedback_field * alpha
            
        self.feedback_field = texture.copy()

        return texture
