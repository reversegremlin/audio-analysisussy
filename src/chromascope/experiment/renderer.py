"""
Frame orchestrator for the fractal kaleidoscope renderer.

Reads manifest frame data, drives fractal generation, kaleidoscope
mirroring, color grading, and post-processing. Yields frames as
a generator for memory-efficient piping to the encoder.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from chromascope.experiment.colorgrade import (
    add_glow,
    apply_palette,
    chromatic_aberration,
    tone_map_soft,
    vignette,
)
from chromascope.experiment.fractal import (
    interpolate_c,
    julia_set,
    mandelbrot_zoom,
    noise_fractal,
)
from chromascope.experiment.kaleidoscope import (
    infinite_zoom_blend,
    polar_mirror,
    radial_warp,
)


@dataclass
class RenderConfig:
    """Configuration for the fractal kaleidoscope renderer."""

    width: int = 1920
    height: int = 1080
    fps: int = 60
    num_segments: int = 8
    fractal_mode: str = "blend"  # "julia", "mandelbrot", or "blend"

    # Zoom
    base_zoom_speed: float = 1.0
    zoom_beat_punch: float = 1.08

    # Rotation
    base_rotation_speed: float = 1.0

    # Fractal detail
    base_max_iter: int = 200
    max_max_iter: int = 400

    # Post-processing
    glow_enabled: bool = True
    glow_intensity: float = 0.35
    glow_radius: int = 15
    aberration_enabled: bool = True
    aberration_offset: int = 3
    vignette_strength: float = 0.3

    # Feedback / infinite zoom
    feedback_alpha: float = 0.20
    base_zoom_factor: float = 1.015

    # Warp
    warp_amplitude: float = 0.03
    warp_frequency: float = 4.0

    # Color
    base_saturation: float = 0.88
    contrast: float = 1.4

    # Mandelbrot zoom target — a visually interesting spiral region
    mandelbrot_center: complex = field(
        default_factory=lambda: -0.7435669 + 0.1314023j
    )


class FractalKaleidoscopeRenderer:
    """
    Renders audio-reactive fractal kaleidoscope frames.

    Driven by manifest data from the AudioPipeline.
    """

    def __init__(self, config: RenderConfig | None = None):
        self.cfg = config or RenderConfig()

        # State
        self.feedback_buffer: np.ndarray | None = None
        self.accumulated_rotation = 0.0
        self.julia_t = 0.0  # parameter along the c-value path
        self.time = 0.0

        # Smoothed audio values
        self._smooth_percussive = 0.0
        self._smooth_harmonic = 0.3
        self._smooth_brightness = 0.5
        self._smooth_energy = 0.3
        self._smooth_low = 0.3
        self._smooth_high = 0.3

        # Lissajous drift state
        self._drift_phase = 0.0

    # Default fallback c — always produces rich boundary detail
    _DEFAULT_GOOD_C = complex(-0.7269, 0.1889)

    def _lerp(self, current: float, target: float, factor: float) -> float:
        return current + (target - current) * factor

    @staticmethod
    def _probe_c(c: complex, zoom: float, probe_iter: int) -> bool:
        """Return True if *c* produces a detailed Julia set at *zoom*."""
        probe = julia_set(32, 24, c=c, center=complex(0, 0),
                          zoom=zoom, max_iter=probe_iter)
        boundary_frac = float((probe > 0.4).mean())
        return 0.10 < boundary_frac < 0.85

    def _pick_best_c(
        self, julia_c: complex, zoom: float, probe_iter: int,
    ) -> complex:
        """Choose the best c-value, falling back if the primary is flat."""
        # 1. Try the current interpolated c
        if self._probe_c(julia_c, zoom, probe_iter):
            self._last_good_c = julia_c
            return julia_c

        # 2. Try the cached last-good c (may be stale at new zoom)
        cached = getattr(self, '_last_good_c', None)
        if cached is not None and self._probe_c(cached, zoom, probe_iter):
            return cached

        # 3. Guaranteed-good default
        return self._DEFAULT_GOOD_C

    def _smooth_audio(self, frame_data: dict[str, Any]):
        """Update smoothed audio values from frame data."""
        is_beat = frame_data.get("is_beat", False)
        # Fast attack on beats, but not so fast that it feels jerky.
        # 0.3 reaches ~87% of target in 6 frames (100ms) — punchy
        # but fluid.  Non-beat frames use 0.12 for gentle drift.
        fast = 0.3 if is_beat else 0.12
        slow = 0.08

        self._smooth_percussive = self._lerp(
            self._smooth_percussive,
            frame_data.get("percussive_impact", 0.0),
            fast,
        )
        self._smooth_harmonic = self._lerp(
            self._smooth_harmonic,
            frame_data.get("harmonic_energy", 0.3),
            slow,
        )
        self._smooth_brightness = self._lerp(
            self._smooth_brightness,
            frame_data.get("spectral_brightness", 0.5),
            slow,
        )
        self._smooth_energy = self._lerp(
            self._smooth_energy,
            frame_data.get("global_energy", 0.3),
            slow,
        )
        self._smooth_low = self._lerp(
            self._smooth_low,
            frame_data.get("low_energy", 0.3),
            slow,
        )
        self._smooth_high = self._lerp(
            self._smooth_high,
            frame_data.get("high_energy", 0.3),
            slow,
        )

    def render_frame(
        self,
        frame_data: dict[str, Any],
        frame_index: int,
    ) -> np.ndarray:
        """
        Render a single frame.

        Args:
            frame_data: Frame dict from manifest.
            frame_index: Frame index.

        Returns:
            (H, W, 3) uint8 RGB numpy array.
        """
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        self.time += dt

        # Smooth audio
        self._smooth_audio(frame_data)
        is_beat = frame_data.get("is_beat", False)
        is_onset = frame_data.get("is_onset", False)

        # --- Update state ---

        # Rotation
        rotation_delta = (
            0.01
            * cfg.base_rotation_speed
            * (1.0 + self._smooth_harmonic * 2.0)
        )
        self.accumulated_rotation += rotation_delta

        # Julia c-parameter drift
        c_speed = 0.0003 * (1.0 + self._smooth_harmonic)
        self.julia_t += c_speed
        julia_c = interpolate_c(self.julia_t)

        # Max iterations from brightness
        max_iter = int(
            cfg.base_max_iter
            + self._smooth_brightness * (cfg.max_max_iter - cfg.base_max_iter)
        )

        # Hue from chroma
        hue_base = frame_data.get("pitch_hue", 0.0)
        if is_onset:
            hue_base = (hue_base + 0.1) % 1.0

        # --- Generate fractal texture ---

        # Fractal zoom breathes with the music rather than monotonically
        # increasing.  Bass energy pushes the zoom inward (revealing
        # finer boundary structure), while a slow sine oscillation keeps
        # it moving so the Julia set boundary stays visible.  The
        # feedback buffer provides the infinite-tunnel effect separately.
        breath = 0.4 * math.sin(self.time * 0.4)
        fractal_zoom = 1.0 + self._smooth_low * 0.5 + breath
        fractal_zoom = max(0.6, min(fractal_zoom, 1.8))

        # Lissajous drift — generous amplitude keeps the viewport
        # exploring the Julia set boundary rather than sitting in a
        # dark interior pocket.
        self._drift_phase += dt * 0.3
        drift_x = math.sin(self._drift_phase * 1.3) * 0.25 / max(fractal_zoom, 1)
        drift_y = math.cos(self._drift_phase * 0.9) * 0.18 / max(fractal_zoom, 1)

        # Cheap low-res probe to detect flat Julia sets BEFORE
        # spending time on the full-res render.  When the current
        # c-value produces a featureless set, fall back to a
        # known-good c.  Two-tier fallback: first try the last
        # cached good c (re-probed at the CURRENT zoom), then
        # the always-reliable default.
        use_mandelbrot = cfg.fractal_mode == "mandelbrot"
        effective_c = julia_c

        if not use_mandelbrot:
            probe_iter = min(max_iter, 100)
            effective_c = self._pick_best_c(
                julia_c, fractal_zoom, probe_iter,
            )

        if use_mandelbrot:
            texture = mandelbrot_zoom(
                cfg.width,
                cfg.height,
                center=cfg.mandelbrot_center + complex(drift_x * 0.01, drift_y * 0.01),
                zoom=fractal_zoom * 0.5,
                max_iter=max_iter,
            )
        else:
            texture = julia_set(
                cfg.width,
                cfg.height,
                c=effective_c,
                center=complex(drift_x, drift_y),
                zoom=fractal_zoom,
                max_iter=max_iter,
            )

        # Light organic noise — very subtle, just to add living texture.
        # Heavy noise injection destroys fractal boundary detail after
        # the kaleidoscope mirror spreads it, so we keep this minimal.
        if self._smooth_energy > 0.3:
            noise = noise_fractal(
                cfg.width,
                cfg.height,
                time=self.time,
                octaves=4,
                scale=2.0 + self._smooth_harmonic * 2,
            )
            noise_blend = 0.03 + self._smooth_energy * 0.04
            texture = texture * (1 - noise_blend) + noise * noise_blend

        # --- Radial warp (breathing) ---

        warp_amp = cfg.warp_amplitude * (0.5 + self._smooth_low * 1.5)
        if warp_amp > 0.005:
            texture = radial_warp(
                texture,
                amplitude=warp_amp,
                frequency=cfg.warp_frequency,
                time=self.time * 2,
            )

        # --- Kaleidoscope mirror ---

        # Segment count: base ± smoothed high energy modulation.
        # Using the smoothed value prevents frame-to-frame jitter
        # that makes the kaleidoscope pattern jump.
        seg_mod = int(self._smooth_high * 4)
        num_seg = max(4, cfg.num_segments + seg_mod - 2)

        texture = polar_mirror(
            texture,
            num_segments=num_seg,
            rotation=self.accumulated_rotation,
        )

        # --- Color grading ---

        frame_rgb = apply_palette(
            texture,
            hue_base=hue_base,
            time=self.time,
            saturation=cfg.base_saturation * (0.8 + self._smooth_harmonic * 0.4),
            contrast=cfg.contrast,
        )

        # --- Infinite zoom blend ---

        zoom_factor = cfg.base_zoom_factor * (1.0 + self._smooth_energy * 0.01)
        # Reduce feedback on heavy percussion so fresh fractal detail dominates
        feedback_alpha = cfg.feedback_alpha * (1.0 - self._smooth_percussive * 0.6)

        frame_rgb = infinite_zoom_blend(
            frame_rgb,
            self.feedback_buffer,
            zoom_factor=zoom_factor,
            feedback_alpha=feedback_alpha,
        )

        # --- Post-processing ---

        if cfg.glow_enabled:
            glow_int = cfg.glow_intensity * (1.0 + self._smooth_percussive * 0.3)
            glow_int = min(glow_int, 0.50)
            frame_rgb = add_glow(frame_rgb, intensity=glow_int, radius=cfg.glow_radius)

        if cfg.aberration_enabled:
            ab_offset = int(
                cfg.aberration_offset * (1.0 + self._smooth_percussive * 1.5)
            )
            frame_rgb = chromatic_aberration(frame_rgb, offset=ab_offset)

        if cfg.vignette_strength > 0:
            # Vignette pulses with percussion — beats contract the
            # spotlight, drawing the eye inward to the fractal detail.
            vign_str = cfg.vignette_strength * (
                1.0 + self._smooth_percussive * 0.6
            )
            frame_rgb = vignette(frame_rgb, strength=vign_str)

        # Tone-map before storing in feedback buffer to break brightness accumulation
        frame_rgb = tone_map_soft(frame_rgb)

        # Update feedback buffer
        self.feedback_buffer = frame_rgb.copy()

        return frame_rgb

    def render_manifest(
        self,
        manifest: dict[str, Any],
        progress_callback: callable = None,
    ) -> Iterator[np.ndarray]:
        """
        Render all frames from a manifest as a generator.

        Args:
            manifest: Complete manifest dict with "frames" list.
            progress_callback: Optional callback(current, total).

        Yields:
            (H, W, 3) uint8 RGB arrays, one per frame.
        """
        frames = manifest.get("frames", [])
        total = len(frames)

        # Reset state
        self.feedback_buffer = None
        self.accumulated_rotation = 0.0
        self.julia_t = 0.0
        self.time = 0.0
        self._drift_phase = 0.0
        self._smooth_percussive = 0.0
        self._smooth_harmonic = 0.3
        self._smooth_brightness = 0.5
        self._smooth_energy = 0.3
        self._smooth_low = 0.3
        self._smooth_high = 0.3

        for i, frame_data in enumerate(frames):
            frame = self.render_frame(frame_data, i)
            yield frame

            if progress_callback:
                progress_callback(i + 1, total)
