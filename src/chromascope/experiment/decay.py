"""
Audio-reactive cloud chamber decay renderer.

Translates music into a field of radioactive trails:
- Alpha: short, thick, bright
- Beta: long, thin, fast
- Gamma: sparse flashes/speck events
- Central Ore: Pulsating radioactive core spawning decay events.

Enhanced features:
- Slow Motion: Temporal smoothing and slower particle dynamics.
- Secondary Ionization: Trails split and branch.
- Vapor Distortion: Shader-like heat haze displacement.
- Dynamic Zoom: Reactive viewport scaling.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, List

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates

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
    generation: int = 0


@dataclass
class DecayConfig:
    """Configuration for the decay renderer."""
    width: int = 1920
    height: int = 1080
    fps: int = 60
    
    # Decay-specific
    base_cpm: int = 10000 
    trail_persistence: float = 0.96
    diffusion: float = 0.08
    ionization_gain: float = 1.5
    style: str = "uranium"  # "lab", "uranium", "noir", "neon"
    
    # Post-processing
    glow_enabled: bool = True
    vignette_strength: float = 0.5
    distortion_strength: float = 0.2
    
    # Performance
    max_particles: int = 5000


class DecayRenderer:
    """
    Renders high-intensity, chaotic decay trails with shader-like effects.
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

        # Dynamics state
        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0
        
        # Distortion map (pre-calculated or dynamic)
        self._distortion_offsets = self._init_distortion_map()

    def _init_distortion_map(self):
        """Initialize a static or base noise map for distortion."""
        h, w = self.cfg.height, self.cfg.width
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        return x, y

    def _lerp(self, current: float, target: float, factor: float) -> float:
        return current + (target - current) * factor

    def _smooth_audio(self, frame_data: dict[str, Any]):
        """Update smoothed audio values."""
        is_beat = frame_data.get("is_beat", False)
        # Slower smoothing for "slow-motion" feel in some channels
        fast = 0.4 if is_beat else 0.2
        slow = 0.06

        self._smooth_energy = self._lerp(self._smooth_energy, frame_data.get("global_energy", 0.1), slow)
        self._smooth_percussive = self._lerp(self._smooth_percussive, frame_data.get("percussive_impact", 0.0), fast)
        self._smooth_low = self._lerp(self._smooth_low, frame_data.get("low_energy", 0.1), slow)
        self._smooth_high = self._lerp(self._smooth_high, frame_data.get("high_energy", 0.1), slow)
        self._smooth_flux = self._lerp(self._smooth_flux, frame_data.get("spectral_flux", 0.0), fast)
        self._smooth_sharpness = self._lerp(self._smooth_sharpness, frame_data.get("sharpness", 0.0), slow)
        self._smooth_centroid = self._lerp(self._smooth_centroid, frame_data.get("spectral_centroid", 0.5), slow)
        self._smooth_sub_bass = self._lerp(self._smooth_sub_bass, frame_data.get("sub_bass", 0.0), fast)
        self._smooth_brilliance = self._lerp(self._smooth_brilliance, frame_data.get("brilliance", 0.0), fast)

    def spawn_particle(self, p_type: str, x: float = None, y: float = None, 
                       vx: float = None, vy: float = None, gen: int = 0):
        """Spawn a particle, optionally at a specific location for branching."""
        if len(self.particles) >= self.cfg.max_particles:
            return

        if x is None or y is None:
            # Spawn from Ore surface
            center_x, center_y = self.cfg.width / 2, self.cfg.height / 2
            base_radius = 60.0 * self.ore_scale
            spawn_angle = random.uniform(0, 2 * math.pi)
            r = base_radius * random.uniform(0.6, 1.2)
            x = center_x + math.cos(spawn_angle) * r
            y = center_y + math.sin(spawn_angle) * r
            angle = spawn_angle + random.uniform(-0.5, 0.5) + self.drift_angle
        else:
            # Branching spawn - use provided velocity or jitter
            angle = math.atan2(vy, vx) + random.uniform(-0.8, 0.8)

        # "Slow motion" velocity scaling
        speed_scale = 0.4 + self._smooth_energy * 0.6
        
        if p_type == "alpha":
            speed = random.uniform(1.0, 4.0) * (1.0 + self._smooth_low) * speed_scale
            life = 1.0
            decay_rate = random.uniform(0.015, 0.04) # Slower decay
            thickness = random.uniform(10.0, 20.0)
            intensity = random.uniform(0.8, 1.5)
        elif p_type == "beta":
            speed = random.uniform(10.0, 30.0) * (0.5 + self._smooth_centroid) * speed_scale
            life = 1.0
            decay_rate = random.uniform(0.005, 0.015) # Long trails
            thickness = random.uniform(2.0, 5.0)
            intensity = random.uniform(0.6, 1.2)
        else: # gamma
            speed = random.uniform(0, 5.0) * speed_scale
            life = 1.0
            decay_rate = 0.1
            thickness = random.uniform(5.0, 15.0)
            intensity = random.uniform(1.0, 2.0)

        nvx = math.cos(angle) * speed
        nvy = math.sin(angle) * speed
        
        p = Particle(
            x=x, y=y, last_x=x, last_y=y,
            vx=nvx, vy=nvy, 
            life=life, decay_rate=decay_rate, 
            thickness=thickness, intensity=intensity,
            type=p_type, generation=gen
        )
        self.particles.append(p)

    def update_particles(self, dt: float):
        """Move particles, update life, and handle branching."""
        new_particles = []
        # Chaotic field from flux
        field_power = self._smooth_flux * 10.0
        
        for p in self.particles:
            p.life -= p.decay_rate
            if p.life > 0:
                p.last_x, p.last_y = p.x, p.y
                p.x += p.vx
                p.y += p.vy
                
                # Magnetic/Chaos deviation
                drift = (math.sin(self.time * 2 + p.x * 0.01) * field_power)
                p.vx += drift * 0.1
                p.vy += math.cos(self.time * 1.5 + p.y * 0.01) * field_power * 0.1
                
                # Secondary Ionization (Branching)
                # High energy and flux trigger splits
                if p.generation < 2 and random.random() < (0.02 * self._smooth_energy * self._smooth_flux):
                    self.spawn_particle(p.type, p.x, p.y, p.vx, p.vy, p.generation + 1)
                
                # Drag
                p.vx *= 0.99
                p.vy *= 0.99
                new_particles.append(p)
        self.particles = new_particles

    def _draw_ore(self, draw: ImageDraw.Draw, center: tuple[float, float], scale: float):
        """Draw the chaotic central ore core."""
        cx, cy = center
        # More points for a "shattered" look
        num_points = 24
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points) + self.ore_rotation
            # Chaos radius
            r_noise = random.uniform(-20, 20) * (self._smooth_flux + self._smooth_brilliance)
            r = (70.0 + r_noise) * scale
            points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
        
        draw.polygon(points, fill=0.9 + self._smooth_sub_bass * 0.3)
        
        # Internal crackling
        for _ in range(8):
            hx = cx + random.uniform(-40, 40) * scale
            hy = cy + random.uniform(-40, 40) * scale
            hr = random.uniform(2, 25) * scale * (0.8 + self._smooth_brilliance)
            draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=1.0)

    def _apply_vapor_distortion(self, buffer: np.ndarray) -> np.ndarray:
        """Apply a 'shader-like' heat haze distortion."""
        h, w = buffer.shape
        strength = self.cfg.distortion_strength * (1.0 + self._smooth_flux * 3.0)
        
        if strength <= 0.01:
            return buffer
            
        # Create displacement field
        t = self.time * 3.0
        # Low frequency "wobble"
        dx = np.sin(t + np.linspace(0, 10, w)) * strength * 10
        dy = np.cos(t * 1.2 + np.linspace(0, 10, h)) * strength * 10
        
        # Using map_coordinates for high-quality displacement
        # This is a bit slow on CPU but looks very "shader-y"
        grid_x, grid_y = self._distortion_offsets
        # Add dynamic wobble to grid
        # We'll just shift the coordinates
        distorted_coords = np.array([
            grid_y + dy[:, None],
            grid_x + dx[None, :]
        ])
        
        distorted = map_coordinates(buffer, distorted_coords, order=1, mode='reflect')
        return distorted

    def _apply_styles(self, buffer: np.ndarray) -> np.ndarray:
        """Apply color palettes with spectral shifts."""
        # buffer is (H, W) float32
        style = self.cfg.style
        
        if style == "uranium":
            r = buffer * 0.2
            g = buffer * 1.0
            b = buffer * (0.1 + self._smooth_brilliance * 0.5)
            # Add plasma-heat core
            y, x = np.ogrid[:self.cfg.height, :self.cfg.width]
            dist = np.sqrt((x - self.cfg.width/2)**2 + (y - self.cfg.height/2)**2)
            glow = np.exp(-dist / (80.0 * self.ore_scale * self.view_zoom))
            r = np.maximum(r, buffer * glow * 1.2)
            rgb = np.stack([r, g, b], axis=-1)
        elif style == "neon":
            # Fast spectral cycling on peaks
            shift = self.time * 1.0 + self._smooth_flux * 5.0
            r = buffer * (0.5 + 0.5 * math.sin(shift))
            g = buffer * (0.5 + 0.5 * math.cos(shift * 0.8))
            b = buffer * (0.8 + self._smooth_sub_bass)
            rgb = np.stack([r, g, b], axis=-1)
        elif style == "noir":
            # Extreme contrast
            b2 = np.power(buffer, 1.6)
            rgb = np.stack([b2, b2, b2], axis=-1)
        else: # lab
            rgb = np.stack([buffer, buffer, buffer], axis=-1)
            
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def render_frame(
        self,
        frame_data: dict[str, Any],
        frame_index: int,
    ) -> np.ndarray:
        """Render a single high-chaos frame."""
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)
        
        # Dynamic Viewport State
        self.ore_rotation += (0.05 + self._smooth_flux * 0.8)
        self.ore_scale = self._lerp(self.ore_scale, 1.0 + self._smooth_sub_bass * 2.0, 0.4)
        self.drift_angle += (self._smooth_centroid - 0.5) * dt * 8.0
        
        # Zoom reactive to sub-bass and global energy
        target_zoom = 1.0 + self._smooth_sub_bass * 0.3 + math.sin(self.time * 0.5) * 0.1
        self.view_zoom = self._lerp(self.view_zoom, target_zoom, 0.1)

        # 1. Spawning - CRANKED
        cpm = cfg.base_cpm * (1.0 + self._smooth_energy * 6.0 + self._smooth_flux * 10.0)
        spawn_prob = (cpm / 60.0) * dt
        num_spawns = np.random.poisson(spawn_prob)
        
        if frame_data.get("is_beat", False):
            num_spawns += int(100 * self._smooth_percussive * cfg.ionization_gain)

        total_energy = self._smooth_low + self._smooth_high + 1e-6
        alpha_prob = self._smooth_low / total_energy
        
        for _ in range(num_spawns):
            r = random.random()
            if r < alpha_prob * 0.6:
                self.spawn_particle("alpha")
            elif r < 0.93:
                self.spawn_particle("beta")
            else:
                self.spawn_particle("gamma")

        # 2. Update buffer with Persistence
        p_fade = cfg.trail_persistence * (1.0 - self._smooth_flux * 0.05)
        self.trail_buffer *= p_fade
        
        # Diffusion pulses
        if cfg.diffusion > 0:
            sigma = cfg.diffusion * 8 * (0.5 + self._smooth_energy)
            self.trail_buffer = gaussian_filter(self.trail_buffer, sigma=sigma)

        # 3. Draw
        self.update_particles(dt)
        
        # Scaling coordinates for Zoom
        # We can simulate zoom by drawing everything shifted/scaled
        # But for simplicity, we'll draw to a temp image and then transform
        deposit_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        draw = ImageDraw.Draw(deposit_img)
        
        center_x, center_y = cfg.width / 2, cfg.height / 2
        
        # Draw Ore
        self._draw_ore(draw, (center_x, center_y), self.ore_scale)
        
        # Draw Particles
        for p in self.particles:
            # Scale intensity and thickness by life and flux
            f_boost = (1.0 + self._smooth_flux * 1.5)
            color = float(p.intensity * p.life * f_boost)
            thickness = int(p.thickness * (0.7 + p.life * 0.3) * f_boost)
            
            # Simple zoom implementation: transform points
            def transform(px, py):
                return (
                    center_x + (px - center_x) * self.view_zoom,
                    center_y + (py - center_y) * self.view_zoom
                )

            tx, ty = transform(p.x, p.y)
            tlx, tly = transform(p.last_x, p.last_y)

            if p.type == "gamma":
                draw.ellipse([tx - thickness, ty - thickness, 
                              tx + thickness, ty + thickness], 
                             fill=color)
            else:
                draw.line([(tlx, tly), (tx, ty)], fill=color, width=thickness)
        
        # Accumulate
        current_frame_buffer = np.array(deposit_img)
        
        # 4. Vapor Distortion "Shader"
        current_frame_buffer = self._apply_vapor_distortion(current_frame_buffer)
        
        self.trail_buffer = np.maximum(self.trail_buffer, current_frame_buffer)

        # 5. Grading and Post
        rgb = self._apply_styles(self.trail_buffer)
        
        if cfg.glow_enabled:
            # Massive bloom
            g_int = 0.5 + self._smooth_flux * 0.8 + self._smooth_sub_bass * 0.5
            rgb = add_glow(rgb, intensity=min(g_int, 0.99), radius=22)
            
        if cfg.vignette_strength > 0:
            v_str = cfg.vignette_strength * (1.5 + self._smooth_sub_bass * 2.0)
            rgb = vignette(rgb, strength=v_str)
            
        return tone_map_soft(rgb)

    def render_manifest(
        self,
        manifest: dict[str, Any],
        progress_callback: callable = None,
    ) -> Iterator[np.ndarray]:
        """Render manifest with full chaos."""
        frames = manifest.get("frames", [])
        total = len(frames)
        
        # Reset State
        self.particles = []
        self.trail_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.time = 0.0
        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0

        for i, frame_data in enumerate(frames):
            frame = self.render_frame(frame_data, i)
            yield frame

            if progress_callback:
                progress_callback(i + 1, total)
