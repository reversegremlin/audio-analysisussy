"""
Kaleidoscope visualization renderer.

Maps audio features to geometric transformations:
- Percussive Impact → Scale & Thickness (pulse)
- Harmonic Energy → Rotation & Orbit speed
- Spectral Centroid → Polygon complexity (sides)
- Chroma → Hue/Color
- Spectral Flatness → Particle jitter/noise
- 7-Band EQ → Layered background dynamics
"""

import colorsys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from chromascope.visualizers.styles import get_kaleidoscope_style


@dataclass
class KaleidoscopeConfig:
    """Configuration for the kaleidoscope renderer."""

    width: int = 1920
    height: int = 1080
    fps: int = 60
    num_mirrors: int = 8  # Radial symmetry count
    trail_alpha: int = 40  # Frame persistence (0-255)
    base_radius: float = 150.0  # Base shape size
    max_scale: float = 1.8  # Maximum pulse scale
    base_thickness: int = 3  # Base line thickness
    max_thickness: int = 12  # Maximum line thickness
    orbit_radius: float = 200.0  # Base orbit distance
    rotation_speed: float = 2.0  # Base rotation multiplier
    min_sides: int = 3  # Triangle
    max_sides: int = 12  # Dodecagon (approaches circle)
    background_color: tuple[int, int, int] = (5, 5, 15)
    background_color2: tuple[int, int, int] = (26, 10, 46)  # Secondary gradient color
    dynamic_background: bool = True  # Enable dynamic background effects
    bg_reactivity: float = 0.7  # Background reactivity to audio (0-1)
    bg_particles: bool = True  # Enable particle effects
    bg_pulse: bool = True  # Enable beat pulse rings
    style: str = "geometric"  # Visualization style (geometric, glass, flower, spiral, circuit, fibonacci, fractal, dmt, sacred, mycelial, fluid, orrery, quark)


class KaleidoscopeRenderer:
    """
    Renders kaleidoscopic visuals driven by audio manifest data.

    Creates a geometric visualization where shapes pulse, rotate, and
    change complexity based on the character of the music.
    """

    # Map chroma indices to hue values (0-1 range)
    # C=red, progressing through spectrum
    CHROMA_TO_HUE = {
        0: 0.0,    # C - Red
        1: 0.083,  # C# - Orange-red
        2: 0.167,  # D - Orange
        3: 0.25,   # D# - Yellow
        4: 0.333,  # E - Yellow-green
        5: 0.417,  # F - Green
        6: 0.5,    # F# - Cyan
        7: 0.583,  # G - Light blue
        8: 0.667,  # G# - Blue
        9: 0.75,   # A - Purple
        10: 0.833, # A# - Magenta
        11: 0.917, # B - Pink
    }

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, config: KaleidoscopeConfig | None = None):
        """
        Initialize the renderer.

        Args:
            config: Rendering configuration. Uses defaults if None.
        """
        self.config = config or KaleidoscopeConfig()

        # Apply shared style presets as defaults — only fill in values
        # that the caller didn't explicitly set (i.e. still at class default).
        style_overrides = get_kaleidoscope_style(self.config.style)
        if style_overrides:
            defaults = KaleidoscopeConfig()
            for key, value in style_overrides.items():
                if hasattr(self.config, key) and getattr(self.config, key) == getattr(defaults, key):
                    setattr(self.config, key, value)
        self.accumulated_rotation = 0.0
        self.surface: pygame.Surface | None = None

        # Dynamic background state
        self.gradient_angle = 0.0
        self.pulse_intensity = 0.0
        self.particles = self._init_particles()

        # Smoothed values for fluid animation
        self.smoothed_percussive = 0.0
        self.smoothed_harmonic = 0.3
        self.smoothed_brightness = 0.5
        self.smoothed_flatness = 0.0
        self.smoothed_flux = 0.0
        self.smoothed_sharpness = 0.0
        self.smoothed_sub_bass = 0.0
        self.smoothed_brilliance = 0.0

    def _init_particles(self) -> list[dict]:
        """Initialize background particles."""
        import random
        particles = []
        for _ in range(80):
            particles.append({
                'x': random.random() * self.config.width,
                'y': random.random() * self.config.height,
                'size': random.random() * 2 + 0.5,
                'speed': random.random() * 0.5 + 0.1,
                'angle': random.random() * math.pi * 2,
                'brightness': random.random() * 0.5 + 0.3,
                'pulse': random.random() * math.pi * 2,
                'jitter': 0.0
            })
        return particles

    def _note_to_hue(self, note: str) -> float:
        """Convert note name to hue value."""
        try:
            idx = self.NOTE_NAMES.index(note)
            return self.CHROMA_TO_HUE[idx]
        except ValueError:
            return 0.5  # Default cyan

    def _hue_to_rgb(self, hue: float, saturation: float = 0.8, value: float = 0.9) -> tuple[int, int, int]:
        """Convert HSV to RGB color tuple."""
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _compute_polygon_points(
        self,
        center: tuple[float, float],
        radius: float,
        num_sides: int,
        rotation: float,
    ) -> list[tuple[float, float]]:
        """Compute vertices of a regular polygon."""
        points = []
        for i in range(num_sides):
            angle = rotation + (2 * math.pi * i / num_sides)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        return points

    def _draw_polygon(
        self,
        surface: pygame.Surface,
        center: tuple[float, float],
        radius: float,
        num_sides: int,
        rotation: float,
        color: tuple[int, int, int],
        thickness: int,
    ):
        """Draw a single polygon."""
        points = self._compute_polygon_points(center, radius, num_sides, rotation)
        if len(points) >= 3:
            pygame.draw.polygon(surface, color, points, thickness)

    def _lerp(self, current: float, target: float, factor: float) -> float:
        """Linear interpolation helper."""
        return current + (target - current) * factor

    def _render_dynamic_background(
        self,
        surface: pygame.Surface,
        frame_data: dict[str, Any],
    ):
        """Render dynamic, music-reactive background."""
        cfg = self.config
        width, height = cfg.width, cfg.height
        reactivity = cfg.bg_reactivity

        # Update gradient angle, influenced by brilliance
        self.gradient_angle += 0.002 * (1 + self.smoothed_harmonic + self.smoothed_brilliance * 2)

        # Update pulse intensity
        is_beat = frame_data.get("is_beat", False)
        if is_beat:
            self.pulse_intensity = self._lerp(self.pulse_intensity, 1.0, 0.8)
        else:
            self.pulse_intensity = self._lerp(self.pulse_intensity, 0.0, 0.05)

        # Calculate gradient blend
        blend_phase = math.sin(self.gradient_angle * 2) * 0.5 + 0.5
        energy_blend = self.smoothed_harmonic * reactivity
        # Sub-bass influences the depth of the color blend
        blend = blend_phase * (0.3 - self.smoothed_sub_bass * 0.1) + energy_blend * 0.4

        # Interpolate background colors
        c1 = cfg.background_color
        c2 = cfg.background_color2
        mid_color = (
            int(c1[0] + (c2[0] - c1[0]) * blend),
            int(c1[1] + (c2[1] - c1[1]) * blend),
            int(c1[2] + (c2[2] - c1[2]) * blend),
        )

        # Add brightness boost on beats, influenced by spectral flux
        flux_boost = self.smoothed_flux * 20 * reactivity
        brightness_boost = int(self.pulse_intensity * 30 * reactivity + flux_boost)
        boosted_color = (
            min(255, max(0, mid_color[0] + brightness_boost)),
            min(255, max(0, mid_color[1] + brightness_boost)),
            min(255, max(0, mid_color[2] + brightness_boost)),
        )

        # Create gradient effect using concentric circles
        # Jitter the center based on zero crossing rate
        zcr_jitter = self.smoothed_flatness * 50 * reactivity
        import random
        center_x = width // 2 + int(math.sin(self.gradient_angle * 3) * width * 0.1 * reactivity) + int((random.random() - 0.5) * zcr_jitter)
        center_y = height // 2 + int(math.cos(self.gradient_angle * 2) * height * 0.1 * reactivity) + int((random.random() - 0.5) * zcr_jitter)

        # Draw radial gradient approximation
        # Sub-bass expands the background radius
        max_radius = int(max(width, height) * (0.9 + self.smoothed_sub_bass * 0.2) + self.pulse_intensity * 200 * reactivity)
        steps = 20

        for i in range(steps, 0, -1):
            ratio = i / steps
            radius = int(max_radius * ratio)
            # Interpolate from boosted center to dark edge
            step_color = (
                int(boosted_color[0] * ratio + c1[0] * (1 - ratio)),
                int(boosted_color[1] * ratio + c1[1] * (1 - ratio)),
                int(boosted_color[2] * ratio + c1[2] * (1 - ratio)),
            )
            pygame.draw.circle(surface, step_color, (center_x, center_y), radius)

        # Render particles
        if cfg.bg_particles:
            self._render_particles(surface)

        # Render pulse rings
        if cfg.bg_pulse and self.pulse_intensity > 0.1:
            self._render_pulse_rings(surface)

    def _render_particles(self, surface: pygame.Surface):
        """Render floating particle effects."""
        cfg = self.config
        reactivity = cfg.bg_reactivity
        energy_boost = 1 + self.smoothed_harmonic * 2 * reactivity
        
        # Spectral flatness adds jitter to particle movement
        jitter_amt = self.smoothed_flatness * 5 * reactivity
        import random

        for particle in self.particles:
            # Update position
            particle['jitter'] = self._lerp(particle['jitter'], (random.random() - 0.5) * jitter_amt, 0.2)
            particle['x'] += math.cos(particle['angle']) * particle['speed'] * energy_boost + particle['jitter']
            particle['y'] += math.sin(particle['angle']) * particle['speed'] * energy_boost + particle['jitter']

            # Wrap around edges
            if particle['x'] < 0:
                particle['x'] = cfg.width
            if particle['x'] > cfg.width:
                particle['x'] = 0
            if particle['y'] < 0:
                particle['y'] = cfg.height
            if particle['y'] > cfg.height:
                particle['y'] = 0

            # Pulse brightness
            particle['pulse'] += 0.05
            pulse_brightness = math.sin(particle['pulse']) * 0.3 + 0.7
            beat_brightness = 1 + self.pulse_intensity * 0.5

            # Calculate alpha and size
            # Sharpness increases particle alpha
            alpha = particle['brightness'] * pulse_brightness * beat_brightness * reactivity * (1 + self.smoothed_sharpness * 0.5)
            size = particle['size'] * (1 + self.smoothed_percussive * 0.5)

            # Draw particle
            color_val = int(255 * min(1, alpha))
            pygame.draw.circle(
                surface,
                (color_val, color_val, color_val),
                (int(particle['x']), int(particle['y'])),
                max(1, int(size))
            )

            # Add glow on high energy
            if self.smoothed_percussive > 0.5:
                glow_alpha = (self.smoothed_percussive - 0.5) * alpha * 0.3
                glow_val = int(255 * min(1, glow_alpha))
                if glow_val > 10:
                    pygame.draw.circle(
                        surface,
                        (glow_val, glow_val, glow_val),
                        (int(particle['x']), int(particle['y'])),
                        max(1, int(size * 3))
                    )

    def _render_pulse_rings(self, surface: pygame.Surface):
        """Render expanding pulse rings on beats."""
        cfg = self.config
        center = (cfg.width // 2, cfg.height // 2)
        max_radius = int(max(cfg.width, cfg.height) * 0.6)

        # Get accent color
        accent = (245, 158, 11)  # Default amber

        for i in range(3):
            phase = (1 - self.pulse_intensity + i * 0.2) % 1
            radius = int(phase * max_radius)
            # Sharpness makes rings thinner but brighter
            alpha = self.pulse_intensity * 0.15 * (1 - phase) * (1 + self.smoothed_sharpness)

            if alpha > 0.01 and radius > 0:
                ring_color = (
                    int(min(255, accent[0] * alpha * 2)),
                    int(min(255, accent[1] * alpha * 2)),
                    int(min(255, accent[2] * alpha * 2))
                )
                line_width = max(1, int(2 - self.smoothed_sharpness))
                pygame.draw.circle(surface, ring_color, center, radius, line_width)

    def _draw_kaleidoscope(
        self,
        surface: pygame.Surface,
        frame_data: dict[str, Any],
        center: tuple[float, float],
    ):
        """Draw the full kaleidoscope pattern for a single frame."""
        cfg = self.config

        # Extract values from frame (using smoothed versions from render_frame)
        percussive = self.smoothed_percussive
        harmonic = self.smoothed_harmonic
        brightness = self.smoothed_brightness
        sharpness = self.smoothed_sharpness
        
        dominant_chroma = frame_data.get("dominant_chroma", "C")
        is_beat = frame_data.get("is_beat", False)

        # Map audio to visual properties
        # Scale: base + percussive boost + sharp spikes
        scale = 1.0 + (percussive * (cfg.max_scale - 1.0)) + (sharpness * 0.2)
        if is_beat:
            scale *= 1.1  # Extra pop on beats

        radius = cfg.base_radius * scale

        # Thickness: driven by percussive impact, but sharpness thins the lines
        thickness = int(cfg.base_thickness + percussive * (cfg.max_thickness - cfg.base_thickness))
        thickness = max(1, int(thickness * (1 - sharpness * 0.5)))

        # Rotation: accumulate based on harmonic energy, brilliance adds speed spikes
        rotation_delta = (harmonic + self.smoothed_brilliance) * cfg.rotation_speed * (math.pi / 30)
        self.accumulated_rotation += rotation_delta

        # Polygon sides: brightness controls complexity, flux adds temporary complexity spikes
        side_boost = int(self.smoothed_flux * 3)
        num_sides = int(cfg.min_sides + brightness * (cfg.max_sides - cfg.min_sides)) + side_boost
        num_sides = max(cfg.min_sides, min(cfg.max_sides + 5, num_sides))

        # Orbit distance: modulated by harmonic and sub-bass
        orbit = cfg.orbit_radius * (0.5 + harmonic * 0.5 + self.smoothed_sub_bass * 0.3)

        # Color from chroma
        hue = self._note_to_hue(dominant_chroma)
        # Brilliance increases color saturation
        base_color = self._hue_to_rgb(hue, 0.7 + self.smoothed_brilliance * 0.3, 0.95)

        # Secondary color (complementary)
        secondary_hue = (hue + 0.5) % 1.0
        secondary_color = self._hue_to_rgb(secondary_hue, 0.6, 0.8)

        # Draw radial mirrors
        for i in range(cfg.num_mirrors):
            mirror_angle = (2 * math.pi * i / cfg.num_mirrors) + self.accumulated_rotation * 0.3

            # Calculate orbit position
            orbit_x = center[0] + orbit * math.cos(mirror_angle)
            orbit_y = center[1] + orbit * math.sin(mirror_angle)

            # Draw outer polygon
            self._draw_polygon(
                surface,
                (orbit_x, orbit_y),
                radius * 0.8,
                num_sides,
                self.accumulated_rotation + mirror_angle,
                base_color,
                thickness,
            )

            # Draw inner polygon (counter-rotating)
            # Sharpness makes the inner shape more distinct
            inner_sides = max(3, num_sides - 2)
            self._draw_polygon(
                surface,
                (orbit_x, orbit_y),
                radius * (0.35 + sharpness * 0.1),
                inner_sides,
                -self.accumulated_rotation * 1.5 + mirror_angle,
                secondary_color,
                max(1, thickness // 2),
            )

        # Central shape
        self._draw_polygon(
            surface,
            center,
            radius * 0.6,
            num_sides,
            self.accumulated_rotation * 0.5,
            self._hue_to_rgb(hue, 0.9, 1.0),
            thickness + 2,
        )

    def render_frame(
        self,
        frame_data: dict[str, Any],
        previous_surface: pygame.Surface | None = None,
    ) -> pygame.Surface:
        """
        Render a single frame.

        Args:
            frame_data: Frame data from manifest.
            previous_surface: Previous frame for trail effect.

        Returns:
            Rendered pygame Surface.
        """
        cfg = self.config

        # Extract values from frame (v1.1 schema)
        percussive = frame_data.get("percussive_impact", 0.1)
        harmonic = frame_data.get("harmonic_energy", 0.3)
        brightness = frame_data.get("spectral_brightness", 0.5)
        
        # New features from v1.1
        flatness = frame_data.get("spectral_flatness", 0.0)
        flux = frame_data.get("spectral_flux", 0.0)
        sharpness = frame_data.get("sharpness", 0.0)
        sub_bass = frame_data.get("sub_bass", 0.0)
        brilliance = frame_data.get("brilliance", 0.0)
        
        is_beat = frame_data.get("is_beat", False)

        # Smoothing factors
        smooth_fast = 0.2
        smooth_med = 0.1
        smooth_slow = 0.05

        self.smoothed_percussive = self._lerp(
            self.smoothed_percussive, percussive,
            0.5 if is_beat else smooth_fast
        )
        self.smoothed_harmonic = self._lerp(
            self.smoothed_harmonic, harmonic, smooth_med
        )
        self.smoothed_brightness = self._lerp(
            self.smoothed_brightness, brightness, smooth_slow
        )
        self.smoothed_flatness = self._lerp(
            self.smoothed_flatness, flatness, smooth_med
        )
        self.smoothed_flux = self._lerp(
            self.smoothed_flux, flux, smooth_fast
        )
        self.smoothed_sharpness = self._lerp(
            self.smoothed_sharpness, sharpness, smooth_med
        )
        self.smoothed_sub_bass = self._lerp(
            self.smoothed_sub_bass, sub_bass, smooth_med
        )
        self.smoothed_brilliance = self._lerp(
            self.smoothed_brilliance, brilliance, smooth_fast
        )

        # Create or reuse surface
        surface = pygame.Surface((cfg.width, cfg.height))

        # Trail effect: blend with previous frame
        if previous_surface is not None and cfg.trail_alpha > 0:
            # Darken previous frame
            fade = pygame.Surface((cfg.width, cfg.height))
            fade.fill(cfg.background_color)
            fade_alpha = int((100 - cfg.trail_alpha) / 100 * 80) + 5
            fade.set_alpha(fade_alpha)
            previous_surface.blit(fade, (0, 0))
            surface.blit(previous_surface, (0, 0))
        else:
            surface.fill(cfg.background_color)

        # Render dynamic background if enabled
        if cfg.dynamic_background:
            self._render_dynamic_background(surface, frame_data)

        # Draw kaleidoscope
        center = (cfg.width / 2, cfg.height / 2)
        self._draw_kaleidoscope(surface, frame_data, center)

        return surface

    def render_manifest(
        self,
        manifest: dict[str, Any],
        progress_callback: callable = None,
    ) -> list[pygame.Surface]:
        """
        Render all frames from a manifest.

        Args:
            manifest: Complete manifest dictionary.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            List of rendered pygame Surfaces.
        """
        frames = manifest.get("frames", [])
        surfaces = []
        previous = None

        self.accumulated_rotation = 0.0  # Reset rotation

        for i, frame_data in enumerate(frames):
            surface = self.render_frame(frame_data, previous)
            surfaces.append(surface.copy())
            previous = surface

            if progress_callback:
                progress_callback(i + 1, len(frames))

        return surfaces

    def surface_to_array(self, surface: pygame.Surface) -> np.ndarray:
        """Convert pygame surface to numpy array for video encoding."""
        # pygame uses (width, height) but numpy expects (height, width)
        arr = pygame.surfarray.array3d(surface)
        # Transpose from (width, height, 3) to (height, width, 3)
        arr = np.transpose(arr, (1, 0, 2))
        return arr
