"""
Audio-reactive cloud chamber decay renderer.
Modernized for the OPEN UP architecture.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from chromascope.experiment.base import BaseConfig, BaseVisualizer


@dataclass
class Particle:
    """Represents a single decay event trail."""
    x: float
    y: float
    vx: float
    vy: float
    life: float
    decay_rate: float
    thickness: float
    intensity: float
    type: str
    last_x: float = 0.0
    last_y: float = 0.0
    drag: float = 0.95
    generation: int = 0


@dataclass
class DecayConfig(BaseConfig):
    """Configuration for the decay renderer."""
    base_cpm: int = 12000 
    trail_persistence: float = 0.95
    vapor_persistence: float = 0.98 
    base_diffusion: float = 0.08
    ionization_gain: float = 1.2
    max_particles: int = 6000
    palette_type: str = "jewel" # Can use jewel or custom


class DecayRenderer(BaseVisualizer):
    """
    Renders organic, smokey decay trails.
    """

    def __init__(
        self, 
        config: DecayConfig | None = None, 
        seed: int | None = None, 
        center_pos: Tuple[float, float] | None = None
    ):
        super().__init__(config or DecayConfig(), seed, center_pos)
        self.cfg: DecayConfig = self.cfg
        
        # State
        self.particles: List[Particle] = []
        self.track_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.vapor_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        
        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0

    def spawn_particle(self, p_type: str, x: float = None, y: float = None, 
                       vx: float = None, vy: float = None, gen: int = 0):
        if len(self.particles) >= self.cfg.max_particles: return
        
        if x is None:
            cx, cy = self.center_pos
            base_radius = 50.0 * self.ore_scale
            spawn_angle = self.rng.uniform(0, 2 * math.pi)
            r = base_radius * self.rng.uniform(0.7, 1.1)
            x = cx + math.cos(spawn_angle) * r
            y = cy + math.sin(spawn_angle) * r
            angle = spawn_angle + self.rng.uniform(-0.4, 0.4) + self.drift_angle
        else:
            angle = math.atan2(vy, vx) + self.rng.uniform(-0.3, 0.3)

        kick = 1.0 + self._smooth_percussive * 1.5
        if p_type == "alpha":
            speed = self.rng.uniform(5, 15) * kick
            decay = self.rng.uniform(0.02, 0.05)
            thick = self.rng.uniform(8, 16)
            drag = 0.88
        elif p_type == "beta":
            speed = self.rng.uniform(30, 60) * kick
            decay = self.rng.uniform(0.005, 0.015)
            thick = self.rng.uniform(2, 4)
            drag = 0.98
        else:
            speed = self.rng.uniform(0, 10)
            decay = 0.1
            thick = self.rng.uniform(4, 10)
            drag = 0.95

        p = Particle(
            x=x, y=y, last_x=x, last_y=y, 
            vx=math.cos(angle)*speed, vy=math.sin(angle)*speed, 
            life=1.0, decay_rate=decay, thickness=thick, 
            intensity=self.rng.uniform(0.7, 1.5), 
            type=p_type, drag=drag, generation=gen
        )
        self.particles.append(p)

    def update(self, frame_data: Dict[str, Any]):
        """Advance the decay simulation."""
        dt = 1.0 / self.cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)
        
        self.ore_rotation += (0.04 + self._smooth_harmonic * 0.3)
        self.ore_scale = self._lerp(self.ore_scale, 1.0 + self._smooth_sub_bass * 1.2, 0.4)
        self.drift_angle += (self._smooth_centroid - 0.5) * dt * 6.0
        self.view_zoom = self._lerp(self.view_zoom, 1.0 + self._smooth_sub_bass * 0.3, 0.1)

        # Spawning
        cpm = self.cfg.base_cpm * (1.0 + self._smooth_energy * 4.0 + self._smooth_flux * 3.0)
        num_spawns = self.rng.poisson((cpm / 60.0) * dt)
        if frame_data.get("is_beat", False): 
            num_spawns += int(60 * self._smooth_percussive * self.cfg.ionization_gain)

        alpha_prob = self._smooth_low / (self._smooth_low + self._smooth_high + 1e-6)
        for _ in range(num_spawns):
            r = self.rng.random()
            if r < alpha_prob * 0.5: self.spawn_particle("alpha")
            elif r < 0.92: self.spawn_particle("beta")
            else: self.spawn_particle("gamma")

        # Update existing
        new_particles = []
        h_wobble = self._smooth_harmonic * 4.0
        for p in self.particles:
            p.life -= p.decay_rate
            if p.life > 0:
                p.last_x, p.last_y = p.x, p.y
                p.x += p.vx
                p.y += p.vy
                p.vx *= p.drag
                p.vy *= p.drag
                p.vx += self.rng.uniform(-h_wobble, h_wobble)
                p.vy += self.rng.uniform(-h_wobble, h_wobble)
                
                # Branching
                if p.generation < 1 and self.rng.random() < (0.01 * self._smooth_flux):
                    self.spawn_particle(p.type, p.x, p.y, p.vx, p.vy, p.generation + 1)
                new_particles.append(p)
        self.particles = new_particles

    def get_raw_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (track_buffer, vapor_buffer)."""
        cfg = self.cfg
        
        # Persistence and diffusion
        self.track_buffer *= cfg.trail_persistence
        self.vapor_buffer *= cfg.vapor_persistence
        self.vapor_buffer = gaussian_filter(
            self.vapor_buffer, 
            sigma=cfg.base_diffusion * 6 * (0.5 + self._smooth_harmonic)
        )

        # Draw to buffers
        track_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        vapor_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        draw_t = ImageDraw.Draw(track_img)
        draw_v = ImageDraw.Draw(vapor_img)
        
        self._draw_ore(draw_t, self.center_pos, self.ore_scale)
        
        v_cx, v_cy = cfg.width / 2, cfg.height / 2
        for p in self.particles:
            color = float(p.intensity * p.life)
            thickness = int(p.thickness * (0.7 + p.life * 0.3))
            
            def transform(px, py): 
                return (v_cx + (px - v_cx) * self.view_zoom, 
                        v_cy + (py - v_cy) * self.view_zoom)
            
            tx, ty = transform(p.x, p.y)
            tlx, tly = transform(p.last_x, p.last_y)
            
            if p.type == "gamma":
                draw_t.ellipse([tx - thickness, ty - thickness, tx + thickness, ty + thickness], fill=color)
            else:
                draw_t.line([(tlx, tly), (tx, ty)], fill=color, width=max(1, thickness // 2))
                draw_v.line([(tlx, tly), (tx, ty)], fill=color * 0.6, width=thickness)

        self.track_buffer = np.maximum(self.track_buffer, np.array(track_img))
        self.vapor_buffer = np.maximum(self.vapor_buffer, np.array(vapor_img))
        
        return self.track_buffer, self.vapor_buffer

    def _draw_ore(self, draw: ImageDraw.Draw, center: Tuple[float, float], scale: float):
        cx, cy = center
        num_pts = 16
        pts = []
        for i in range(num_pts):
            angle = i * (2 * math.pi / num_pts) + self.ore_rotation
            r = (50.0 + self.rng.uniform(-10, 10) * self._smooth_energy) * scale
            pts.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
        
        draw.polygon(pts, fill=0.6 + self._smooth_harmonic * 0.4)
        for _ in range(3):
            hr = self.rng.uniform(5, 15) * scale
            hx = cx + self.rng.uniform(-10, 10) * scale
            hy = cy + self.rng.uniform(-10, 10) * scale
            draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=1.0)
