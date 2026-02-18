"""
Audio-reactive cloud chamber decay renderer.

Translates music into a field of radioactive trails:
- Alpha: short, thick, bright
- Beta: long, thin, fast
- Gamma: sparse flashes/speck events
- Central Ore: Pulsating radioactive core spawning decay events.
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
    base_cpm: int = 8000  # Counts per minute baseline (Increased)
    trail_persistence: float = 0.94
    diffusion: float = 0.06
    ionization_gain: float = 1.2
    style: str = "uranium"  # "lab", "uranium", "noir", "neon"
    
    # Post-processing
    glow_enabled: bool = True
    vignette_strength: float = 0.4
    
    # Performance
    max_particles: int = 4000


class DecayRenderer:
    """
    Renders audio-reactive decay trails originating from a central ore.
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
        self._smooth_sub_bass = 0.0
        self._smooth_brilliance = 0.0

        # Drift & Ore state
        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0

    def _lerp(self, current: float, target: float, factor: float) -> float:
        return current + (target - current) * factor

    def _smooth_audio(self, frame_data: dict[str, Any]):
        """Update smoothed audio values."""
        is_beat = frame_data.get("is_beat", False)
        fast = 0.4 if is_beat else 0.2
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
        self._smooth_sub_bass = self._lerp(
            self._smooth_sub_bass, frame_data.get("sub_bass", 0.0), fast
        )
        self._smooth_brilliance = self._lerp(
            self._smooth_brilliance, frame_data.get("brilliance", 0.0), fast
        )

    def spawn_particle(self, p_type: str, intensity_mult: float = 1.0):
        """Spawn a new decay particle originating from the central ore."""
        if len(self.particles) >= self.cfg.max_particles:
            return

        center_x = self.cfg.width / 2
        center_y = self.cfg.height / 2
        
        # Radius depends on ore scale
        base_radius = 50.0 * self.ore_scale
        spawn_angle = random.uniform(0, 2 * math.pi)
        
        # Spawn on the surface of the ore
        r = base_radius * random.uniform(0.7, 1.1)
        x = center_x + math.cos(spawn_angle) * r
        y = center_y + math.sin(spawn_angle) * r
        
        # Travel mostly outwards
        angle = spawn_angle + random.uniform(-0.4, 0.4) + self.drift_angle
        
        if p_type == "alpha":
            # Short, thick, intense
            speed = random.uniform(3.0, 8.0) * (1.0 + self._smooth_low * 2.0)
            life = 1.0
            decay_rate = random.uniform(0.04, 0.08)
            thickness = random.uniform(8.0, 16.0)
            intensity = random.uniform(0.8, 1.4) * intensity_mult
        elif p_type == "beta":
            # Fast, thin, long
            speed = random.uniform(25.0, 70.0) * (0.8 + self._smooth_centroid * 1.0)
            life = 1.0
            decay_rate = random.uniform(0.01, 0.025)
            thickness = random.uniform(1.5, 4.5)
            intensity = random.uniform(0.5, 1.0) * intensity_mult
        else:  # gamma
            # Sudden bursts/points
            speed = random.uniform(0, 15.0)
            life = 1.0
            decay_rate = 0.2
            thickness = random.uniform(4.0, 10.0)
            intensity = random.uniform(1.0, 1.5) * intensity_mult

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
        # Field turbulence from flux and sharpness
        turbulence = (self._smooth_flux * 8.0 + self._smooth_sharpness * 4.0)
        
        for p in self.particles:
            p.life -= p.decay_rate
            if p.life > 0:
                p.last_x, p.last_y = p.x, p.y
                p.x += p.vx
                p.y += p.vy
                
                # Add wobble/field effects
                if p.type != "gamma":
                    w = turbulence * p.life
                    p.vx += random.uniform(-w, w)
                    p.vy += random.uniform(-w, w)
                    # Gentle drag
                    p.vx *= 0.985
                    p.vy *= 0.985
                
                new_particles.append(p)
        self.particles = new_particles

    def _draw_ore(self, draw: ImageDraw.Draw, center: tuple[float, float], scale: float):
        """Draw the central radioactive ore core."""
        cx, cy = center
        # Jagged crystalline ore shape (Uraninite)
        num_points = 16
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points) + self.ore_rotation
            # Radius influenced by brilliance and energy
            r_jitter = random.uniform(-15, 15) * self._smooth_energy
            r = (50.0 + r_jitter) * scale
            points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
        
        # Draw the core body
        core_int = 0.7 + self._smooth_sub_bass * 0.5
        draw.polygon(points, fill=core_int)
        
        # Inner hotspots/veins
        for _ in range(5):
            hx = cx + random.uniform(-25, 25) * scale
            hy = cy + random.uniform(-25, 25) * scale
            hr = random.uniform(5, 20) * scale * (0.5 + self._smooth_brilliance)
            draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=1.0)

    def _apply_styles(self, buffer: np.ndarray) -> np.ndarray:
        """Apply color palettes based on style."""
        style = self.cfg.style
        # buffer is (H, W) float32 [0.0, 1.0]
        
        if style == "lab":
            # Monochrome
            rgb = np.stack([buffer, buffer, buffer], axis=-1)
        elif style == "uranium":
            # Green-white luminescence with hot core
            r = buffer * 0.25
            g = buffer * 1.0
            b = buffer * 0.15
            # Heat mask for the core
            y, x = np.ogrid[:self.cfg.height, :self.cfg.width]
            dist = np.sqrt((x - self.cfg.width/2)**2 + (y - self.cfg.height/2)**2)
            heat = np.exp(-dist / (60.0 * self.ore_scale))
            r = np.maximum(r, buffer * heat * 0.9)
            rgb = np.stack([r, g, b], axis=-1)
        elif style == "noir":
            # High contrast B&W
            b2 = np.power(buffer, 1.4)
            rgb = np.stack([b2, b2, b2], axis=-1)
        elif style == "neon":
            # Spectral shifting glow
            r = buffer * (0.7 + 0.5 * math.sin(self.time * 2.0))
            g = buffer * (0.1 + 0.4 * math.cos(self.time * 1.3))
            b = buffer * (0.8 + self._smooth_sub_bass * 0.4)
            rgb = np.stack([r, g, b], axis=-1)
        else:
            rgb = np.stack([buffer, buffer, buffer], axis=-1)
            
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def render_frame(
        self,
        frame_data: dict[str, Any],
        frame_index: int,
    ) -> np.ndarray:
        """Render a single frame of intense decay trails."""
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)
        
        # Update state
        self.ore_rotation += (0.04 + self._smooth_brilliance * 0.6)
        # Sub-bass drives the ore's massive heart-beat pulse
        self.ore_scale = self._lerp(self.ore_scale, 1.0 + self._smooth_sub_bass * 1.5, 0.45)
        self.drift_angle += (self._smooth_centroid - 0.5) * dt * 5.0

        # 1. Spawning logic - HIGH INTENSITY
        cpm = cfg.base_cpm * (1.0 + self._smooth_energy * 5.0 + self._smooth_flux * 8.0)
        spawn_prob = (cpm / 60.0) * dt
        num_spawns = np.random.poisson(spawn_prob)
        
        # Violent beat bursts
        if frame_data.get("is_beat", False):
            num_spawns += int(80 * self._smooth_percussive * cfg.ionization_gain)
        
        # Spectral eruptions
        if self._smooth_flux > 0.5:
            num_spawns += int(40 * self._smooth_flux)

        total_energy = self._smooth_low + self._smooth_high + 1e-6
        alpha_prob = self._smooth_low / total_energy
        
        for _ in range(num_spawns):
            r = random.random()
            if r < alpha_prob * 0.6:
                self.spawn_particle("alpha")
            elif r < 0.94:
                self.spawn_particle("beta")
            else:
                self.spawn_particle("gamma")

        # 2. Update and draw to trail buffer
        # Lower persistence on high flux for "electric" feel
        persistence = cfg.trail_persistence * (1.0 - self._smooth_flux * 0.04)
        self.trail_buffer *= persistence
        
        # Dynamic diffusion
        if cfg.diffusion > 0:
            sigma = cfg.diffusion * 6 * (0.6 + self._smooth_energy * 0.8)
            self.trail_buffer = gaussian_filter(self.trail_buffer, sigma=sigma)

        # 3. Draw current state
        self.update_particles(dt)
        
        deposit_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        draw = ImageDraw.Draw(deposit_img)
        
        # Draw central Ore
        self._draw_ore(draw, (cfg.width/2, cfg.height/2), self.ore_scale)
        
        for p in self.particles:
            # Flux increases brightness and thickness of current trails
            flux_mult = (1.0 + self._smooth_flux * 0.7)
            color = float(p.intensity * p.life * flux_mult)
            thickness = int(p.thickness * (0.8 + p.life * 0.2))
            
            if p.type == "gamma":
                draw.ellipse([p.x - thickness, p.y - thickness, 
                              p.x + thickness, p.y + thickness], 
                             fill=color)
            else:
                draw.line([(p.last_x, p.last_y), (p.x, p.y)], 
                          fill=color, width=thickness)
        
        # Add deposits to trail buffer
        self.trail_buffer = np.maximum(self.trail_buffer, np.array(deposit_img))

        # 4. Color and Post-processing
        rgb = self._apply_styles(self.trail_buffer)
        
        if cfg.glow_enabled:
            # Massive bloom on flux and sub-bass
            g_int = 0.45 + self._smooth_flux * 0.7 + self._smooth_sub_bass * 0.4
            g_int = min(g_int, 0.98)
            rgb = add_glow(rgb, intensity=g_int, radius=18)
            
        if cfg.vignette_strength > 0:
            v_str = cfg.vignette_strength * (1.2 + self._smooth_sub_bass * 1.5)
            rgb = vignette(rgb, strength=v_str)
            
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
        self.ore_rotation = 0.0
        self.ore_scale = 1.0

        for i, frame_data in enumerate(frames):
            frame = self.render_frame(frame_data, i)
            yield frame

            if progress_callback:
                progress_callback(i + 1, total)
