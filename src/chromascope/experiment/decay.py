"""
Audio-reactive cloud chamber decay renderer.

Translates music into a field of radioactive trails:
- Alpha: short, thick, bright
- Beta: long, thin, fast
- Gamma: sparse flashes/speck events
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, List

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from scipy.ndimage import gaussian_filter

from chromascope.experiment.colorgrade import (
    add_glow,
    tone_map_soft,
    vignette,
)


@dataclass
class Particle:
    """Represents a single decay event trail."""
    x: float
    y: float
    vx: float
    vy: float
    life: float  # 1.0 to 0.0
    decay_rate: float
    thickness: float
    intensity: float
    type: str  # "alpha", "beta", "gamma"
    last_x: float = 0.0
    last_y: float = 0.0


@dataclass
class DecayConfig:
    """Configuration for the decay renderer."""
    width: int = 1920
    height: int = 1080
    fps: int = 60
    
    # Decay-specific
    base_cpm: int = 6000  # Counts per minute baseline
    trail_persistence: float = 0.92
    diffusion: float = 0.05
    ionization_gain: float = 1.0
    style: str = "uranium"  # "lab", "uranium", "noir", "neon"
    
    # Post-processing
    glow_enabled: bool = True
    vignette_strength: float = 0.3
    
    # Performance
    max_particles: int = 2000


class DecayRenderer:
    """
    Renders audio-reactive decay trails.
    """

    def __init__(self, config: DecayConfig | None = None):
        self.cfg = config or DecayConfig()
        
        # State
        self.particles: List[Particle] = []
        self.time = 0.0
        
        # Buffer (H, W) float32
        self.trail_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        
        # Smoothed audio
        self._smooth_energy = 0.1
        self._smooth_percussive = 0.0
        self._smooth_low = 0.1
        self._smooth_high = 0.1
        self._smooth_flux = 0.0
        self._smooth_sharpness = 0.0
        self._smooth_centroid = 0.5

        # Drift state
        self.drift_angle = 0.0

    def _lerp(self, current: float, target: float, factor: float) -> float:
        return current + (target - current) * factor

    def _smooth_audio(self, frame_data: dict[str, Any]):
        """Update smoothed audio values."""
        is_beat = frame_data.get("is_beat", False)
        fast = 0.3 if is_beat else 0.15
        slow = 0.08

        self._smooth_energy = self._lerp(
            self._smooth_energy, frame_data.get("global_energy", 0.1), slow
        )
        self._smooth_percussive = self._lerp(
            self._smooth_percussive, frame_data.get("percussive_impact", 0.0), fast
        )
        self._smooth_low = self._lerp(
            self._smooth_low, frame_data.get("low_energy", 0.1), slow
        )
        self._smooth_high = self._lerp(
            self._smooth_high, frame_data.get("high_energy", 0.1), slow
        )
        self._smooth_flux = self._lerp(
            self._smooth_flux, frame_data.get("spectral_flux", 0.0), fast
        )
        self._smooth_sharpness = self._lerp(
            self._smooth_sharpness, frame_data.get("sharpness", 0.0), slow
        )
        self._smooth_centroid = self._lerp(
            self._smooth_centroid, frame_data.get("spectral_centroid", 0.5), slow
        )

    def spawn_particle(self, p_type: str, intensity_mult: float = 1.0):
        """Spawn a new decay particle."""
        if len(self.particles) >= self.cfg.max_particles:
            return

        x = random.uniform(0, self.cfg.width)
        y = random.uniform(0, self.cfg.height)
        
        # Base angle + drift
        angle = random.uniform(0, 2 * math.pi) + self.drift_angle
        
        if p_type == "alpha":
            speed = random.uniform(1.0, 3.0)
            life = 1.0
            decay_rate = random.uniform(0.04, 0.08)
            thickness = random.uniform(4.0, 8.0)
            intensity = random.uniform(0.7, 1.0) * intensity_mult
        elif p_type == "beta":
            # Speed influenced by centroid
            speed = random.uniform(15.0, 40.0) * (0.8 + self._smooth_centroid * 0.4)
            life = 1.0
            decay_rate = random.uniform(0.01, 0.03)
            thickness = random.uniform(1.0, 2.5)
            intensity = random.uniform(0.4, 0.7) * intensity_mult
        else:  # gamma
            speed = 0.0
            life = 1.0
            decay_rate = 0.2
            thickness = random.uniform(2.0, 5.0)
            intensity = random.uniform(0.8, 1.0) * intensity_mult

        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        
        p = Particle(
            x=x, y=y, last_x=x, last_y=y,
            vx=vx, vy=vy, 
            life=life, decay_rate=decay_rate, 
            thickness=thickness, intensity=intensity,
            type=p_type
        )
        self.particles.append(p)

    def update_particles(self, dt: float):
        """Move particles and update their life."""
        new_particles = []
        for p in self.particles:
            p.life -= p.decay_rate
            if p.life > 0:
                p.last_x, p.last_y = p.x, p.y
                p.x += p.vx
                p.y += p.vy
                
                # Add some wobble
                if p.type != "gamma":
                    # Wobble scale influenced by sharpness
                    wobble = 0.5 + self._smooth_sharpness * 2.0
                    p.vx += random.uniform(-wobble, wobble)
                    p.vy += random.uniform(-wobble, wobble)
                    # Drag
                    p.vx *= 0.98
                    p.vy *= 0.98
                
                new_particles.append(p)
        self.particles = new_particles

    def _apply_styles(self, buffer: np.ndarray) -> np.ndarray:
        """Apply color palettes based on style."""
        style = self.cfg.style
        # buffer is (H, W) float32 [0.0, 1.0]
        
        if style == "lab":
            # Monochrome
            rgb = np.stack([buffer, buffer, buffer], axis=-1)
        elif style == "uranium":
            # Green-white luminescence
            r = buffer * 0.4
            g = buffer * 1.0
            b = buffer * 0.3
            rgb = np.stack([r, g, b], axis=-1)
        elif style == "noir":
            # High contrast B&W
            b2 = np.power(buffer, 1.5)
            rgb = np.stack([b2, b2, b2], axis=-1)
        elif style == "neon":
            # Cyan/Magenta spectral glow
            r = buffer * (0.8 + 0.2 * math.sin(self.time * 1.2))
            g = buffer * (0.2 + 0.8 * math.cos(self.time * 0.8))
            b = buffer * 0.9
            rgb = np.stack([r, g, b], axis=-1)
        else:
            rgb = np.stack([buffer, buffer, buffer], axis=-1)
            
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def render_frame(
        self,
        frame_data: dict[str, Any],
        frame_index: int,
    ) -> np.ndarray:
        """Render a single frame of decay trails."""
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)
        
        # Update drift angle based on centroid
        self.drift_angle += (self._smooth_centroid - 0.5) * dt * 2.0

        # 1. Spawning logic
        cpm = cfg.base_cpm * (0.5 + self._smooth_energy * 2.0)
        spawn_prob = (cpm / 60.0) * dt
        num_spawns = np.random.poisson(spawn_prob)
        
        if frame_data.get("is_beat", False):
            num_spawns += int(25 * self._smooth_percussive * cfg.ionization_gain)

        total_energy = self._smooth_low + self._smooth_high + 1e-6
        alpha_prob = self._smooth_low / total_energy
        
        for _ in range(num_spawns):
            r = random.random()
            if r < alpha_prob * 0.5:
                self.spawn_particle("alpha")
            elif r < 0.92:
                self.spawn_particle("beta")
            else:
                self.spawn_particle("gamma")

        # 2. Update and draw to trail buffer
        self.trail_buffer *= cfg.trail_persistence
        
        # Add diffusion
        if cfg.diffusion > 0:
            # Use scipy gaussian_filter for float32 support
            self.trail_buffer = gaussian_filter(self.trail_buffer, sigma=cfg.diffusion * 5)

        # Draw current particles
        self.update_particles(dt)
        
        # Use PIL Draw for lines - ensure we use mode "F" and properly handle it
        deposit_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        draw = ImageDraw.Draw(deposit_img)
        
        for p in self.particles:
            color = float(p.intensity * p.life)
            if p.type == "gamma":
                draw.ellipse([p.x - p.thickness, p.y - p.thickness, 
                              p.x + p.thickness, p.y + p.thickness], 
                             fill=color)
            else:
                draw.line([(p.last_x, p.last_y), (p.x, p.y)], 
                          fill=color, width=int(p.thickness))
        
        # Add deposits to trail buffer
        self.trail_buffer = np.maximum(self.trail_buffer, np.array(deposit_img))

        # 4. Color and Post-processing
        rgb = self._apply_styles(self.trail_buffer)
        
        if cfg.glow_enabled:
            # Dynamic glow intensity based on flux
            g_int = 0.3 + self._smooth_flux * 0.4
            rgb = add_glow(rgb, intensity=g_int, radius=12)
            
        if cfg.vignette_strength > 0:
            rgb = vignette(rgb, strength=cfg.vignette_strength)
            
        return tone_map_soft(rgb)

    def render_manifest(
        self,
        manifest: dict[str, Any],
        progress_callback: callable = None,
    ) -> Iterator[np.ndarray]:
        """Render all frames from a manifest."""
        frames = manifest.get("frames", [])
        total = len(frames)
        
        # Reset state
        self.particles = []
        self.trail_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.time = 0.0
        self.drift_angle = 0.0

        for i, frame_data in enumerate(frames):
            frame = self.render_frame(frame_data, i)
            yield frame

            if progress_callback:
                progress_callback(i + 1, total)
