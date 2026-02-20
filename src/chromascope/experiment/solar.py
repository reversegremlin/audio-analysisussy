"""
Solar visualizer.
Modernized for the OPEN UP architecture.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import noise
from PIL import Image, ImageDraw, ImageFilter

from chromascope.experiment.base import BaseConfig, BaseVisualizer


@dataclass
class SolarConfig(BaseConfig):
    """Configuration for the Solar visualizer."""
    pan_speed_x: float = 0.5
    pan_speed_y: float = 0.3
    zoom_speed: float = 0.2
    palette_type: str = "solar"


class SolarRenderer(BaseVisualizer):
    """
    Renders dynamic solar plasma and flares.
    Driven by BaseVisualizer for unified state and audio reactivity.
    """

    def __init__(
        self, 
        config: SolarConfig | None = None, 
        seed: int | None = None, 
        center_pos: Tuple[float, float] | None = None
    ):
        super().__init__(config or SolarConfig(), seed, center_pos)
        self.cfg: SolarConfig = self.cfg  # Type hint
        
        # State
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_zoom = 1.0
        self.sun_radius = min(self.cfg.width, self.cfg.height) // 4
        
        # Localized hot spots
        self.hot_spot_center = (0, 0)
        self.hot_spot_intensity = 0.0
        self.hot_spot_decay_rate = 0.85
        
        # Solar flares
        self.flares: List[Dict[str, Any]] = []
        
        # Random offsets for noise (decouple randomness)
        self.noise_offsets = [self.rng.uniform(0, 1000) for _ in range(8)]

    def _update_camera(self, frame_data: Dict[str, Any]):
        """Updates camera position and zoom."""
        energy = self._smooth_energy
        high = self._smooth_high
        flat = self._smooth_flatness
        
        # Base pan
        self.camera_x = math.sin(self.time * self.cfg.pan_speed_x) * 100
        self.camera_y = math.cos(self.time * self.cfg.pan_speed_y) * 100
        
        # Shake
        shake = (energy * 50) + (high * 100) + (flat * 50)
        self.camera_x += (self.rng.random() - 0.5) * shake
        self.camera_y += (self.rng.random() - 0.5) * shake
        
        # Zoom
        self.camera_zoom = 1.0 + math.sin(self.time * self.cfg.zoom_speed) * 0.5
        self.camera_zoom += (energy * 0.5) + (high * 1.0)

    def _generate_noise_layer(self, scale: float, octaves: int, persistence: float, 
                             lacunarity: float, offset_idx: int, angle: float = 0.0) -> np.ndarray:
        """Generates a Perlin noise layer."""
        h, w = self.cfg.height, self.cfg.width
        layer = np.zeros((h, w), dtype=np.float32)
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        cx, cy = w / 2, h / 2
        
        # Optimization: We could vectorize this with np.fromfunction or similar, 
        # but the original used a loop. Let's try a more vectorized approach for speed.
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        tx = x - cx
        ty = y - cy
        
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        
        off_x = self.camera_x + self.noise_offsets[offset_idx]
        off_y = self.camera_y + self.noise_offsets[offset_idx+1]
        
        nx = ((rx + cx + off_x) * scale / w) * self.camera_zoom
        ny = ((ry + cy + off_y) * scale / h) * self.camera_zoom
        
        # noise.pnoise3 is not vectorized, so we still need a way to fill this efficiently
        # or use a different noise library. For now, keep the loop but maybe optimized.
        # Actually, let's use a simpler vectorized noise if possible, but pnoise3 is what was there.
        
        # For a 1920x1080 frame, this loop is VERY slow in Python.
        # The original code had this loop. It must have been slow.
        # I'll use a small optimization by pre-calculating coordinates.
        
        # To make it faster, I'll only sample a grid and interpolate?
        # No, let's stick to the original logic but try to be faster if I can.
        
        for i in range(h):
            for j in range(w):
                layer[i, j] = noise.pnoise3(
                    ny[i, j], nx[i, j], self.time,
                    octaves=octaves, persistence=persistence, lacunarity=lacunarity
                )
        return layer

    def _generate_hot_spot(self) -> np.ndarray:
        """Generates a localized radial gradient."""
        h, w = self.cfg.height, self.cfg.width
        hot_spot_img = np.zeros((h, w), dtype=np.float32)
        if self.hot_spot_intensity > 0.01:
            cx, cy = self.hot_spot_center
            y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
            distance = np.sqrt(x*x + y*y)
            max_dist = min(w, h) / 8
            gradient = 1.0 - np.clip(distance / max_dist, 0, 1)
            hot_spot_img = (gradient * self.hot_spot_intensity).astype(np.float32)
            self.hot_spot_intensity *= self.hot_spot_decay_rate
        return hot_spot_img

    def update(self, frame_data: Dict[str, Any]):
        """Advance the solar simulation."""
        self.time += frame_data.get("global_energy", 0.1) * 0.1
        self._smooth_audio(frame_data)
        self._update_camera(frame_data)
        
        # Sun pulse
        base_radius = min(self.cfg.width, self.cfg.height) // 4
        radius_pulse = (self._smooth_low * 50) + (self._smooth_sub_bass * 100) + (self._smooth_percussive * 120)
        self.sun_radius = int(base_radius * self.camera_zoom + radius_pulse)
        
        # Hot spots
        if frame_data.get("is_beat", False) or self._smooth_flux > 0.8:
            self.hot_spot_center = (
                self.rng.integers(self.cfg.width // 4, 3 * self.cfg.width // 4),
                self.rng.integers(self.cfg.height // 4, 3 * self.cfg.height // 4)
            )
            self.hot_spot_intensity = min(1.5, 1.0 + self._smooth_flux)
            
        # Flare Spawning
        if self._smooth_percussive > 0.7 or self._smooth_flux > 0.8:
            self._spawn_flare()

    def _spawn_flare(self):
        angle_start = self.rng.uniform(0, 2 * math.pi)
        cx, cy = self.center_pos
        fx_start = int(cx + self.sun_radius * math.cos(angle_start))
        fy_start = int(cy + self.sun_radius * math.sin(angle_start))
        
        angle_offset = self.rng.uniform(-math.pi / 4, math.pi / 4)
        angle_end = angle_start + angle_offset
        flare_length = self.sun_radius * (1.0 + (self._smooth_percussive + self._smooth_sharpness) * 1.5)
        fx_end = int(cx + flare_length * math.cos(angle_end))
        fy_end = int(cy + flare_length * math.sin(angle_end))

        mid_x, mid_y = (fx_start + fx_end) / 2, (fy_start + fy_end) / 2
        perp_angle = angle_start + math.pi / 2
        arc_height = self.sun_radius * (0.2 + (self._smooth_high + self._smooth_flux) * 0.5)
        fcx = int(mid_x + arc_height * math.cos(perp_angle + self.rng.uniform(-0.5, 0.5)))
        fcy = int(mid_y + arc_height * math.sin(perp_angle + self.rng.uniform(-0.5, 0.5)))

        self.flares.append({
            "start_point": (fx_start, fy_start),
            "control_point": (fcx, fcy),
            "end_point": (fx_end, fy_end),
            "max_width": max(1, int((self._smooth_percussive + self._smooth_brilliance) * self.sun_radius * 0.15)),
            "current_intensity": min(1.0, 0.5 + self._smooth_percussive * 0.5 + self._smooth_flux * 0.5),
            "decay_rate": 0.85 + (1 - 0.85) * self._smooth_energy * 0.5
        })

    def get_raw_field(self) -> np.ndarray:
        """Returns the raw float32 energy field (sun + flares)."""
        # 1. Sun Texture
        swirl_angle = self.time * 0.01 + (self._smooth_energy + self._smooth_brilliance) * math.pi * 0.5
        
        # Layer 1: Large turbulence
        layer1 = self._generate_noise_layer(0.5 + self._smooth_energy * 2.5, 6, 0.5, 2.0, 0, swirl_angle)
        
        # Layer 2: Plasma flow
        flow_speed = 1.0 + self._smooth_harmonic * 2.0
        layer2 = self._generate_noise_layer(0.2 + self._smooth_harmonic * 2.0, 8, 0.6, 2.5, 2, swirl_angle * 1.5)
        
        combined = (layer1 * 0.5) + (layer2 * 0.5)
        combined = np.clip(combined + self._generate_hot_spot(), 0, 2.0)
        
        # Normalize sun texture
        sun_field = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        
        # 2. Masking
        y, x = np.mgrid[0:self.cfg.height, 0:self.cfg.width].astype(np.float32)
        cx, cy = self.center_pos
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = 1.0 - np.clip((dist - self.sun_radius) / 20.0, 0, 1) # Soft edge
        sun_field *= mask
        
        # 3. Flares
        flare_field = self._generate_flare_field()
        
        return np.clip(sun_field + flare_field, 0, 1)

    def _generate_flare_field(self) -> np.ndarray:
        """Draws all active flares into a single float field."""
        h, w = self.cfg.height, self.cfg.width
        field = Image.new("F", (w, h), 0.0)
        draw = ImageDraw.Draw(field)
        
        new_flares = []
        for flare in self.flares:
            # Simplified drawing for flares in energy field
            # In a real app, we'd use Bezier points
            pts = self._get_bezier_points(flare)
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i+1]
                width = max(1, int(flare["max_width"] * (1 - i/len(pts))))
                intensity = flare["current_intensity"] * (1 - i/len(pts))
                draw.line([p1, p2], fill=float(intensity), width=width)
            
            # Decay
            flare["current_intensity"] *= flare["decay_rate"]
            flare["max_width"] *= 0.95
            if flare["current_intensity"] > 0.01 and flare["max_width"] > 1:
                new_flares.append(flare)
        
        self.flares = new_flares
        
        # Blur the flare field for glow/softness
        field_np = np.array(field)
        if np.any(field_np > 0):
            # PIL blur is faster than scipy for this
            field_img = Image.fromarray(field_np, mode="F")
            field_img = field_img.filter(ImageFilter.GaussianBlur(radius=5))
            field_np = np.array(field_img)
            
        return field_np

    def _get_bezier_points(self, flare: Dict[str, Any], num_pts: int = 20) -> List[Tuple[float, float]]:
        p0 = flare["start_point"]
        p1 = flare["control_point"]
        p2 = flare["end_point"]
        pts = []
        for i in range(num_pts + 1):
            t = i / num_pts
            x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
            y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
            pts.append((x, y))
        return pts
