"""
Audio-reactive cloud chamber decay renderer.

Translates music into a field of radioactive trails:
- Alpha: short, thick, bright (starts fast, slows down rapidly)
- Beta: long, thin, fast (sustained velocity, subtle drag)
- Gamma: sparse flashes/speck events
- Central Ore: Pulsating radioactive core spawning decay events.

Aesthetic focus:
- Drag Dynamics: Particles decelerate as they lose energy in the vapor.
- Smokey Trails: Dual-buffer system for sharp tracks and diffuse lingering vapor.
- Harmonic Sync: Diffusion and trail "wobble" tied to musical harmonics.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Tuple

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
    """Represents a single decay event trail with drag dynamics."""
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
    drag: float = 0.95
    generation: int = 0


@dataclass
class DecayConfig:
    """Configuration for the 'smokey' decay renderer."""
    width: int = 1920
    height: int = 1080
    fps: int = 60
    
    # Decay-specific
    base_cpm: int = 12000 
    trail_persistence: float = 0.95
    vapor_persistence: float = 0.98 # Vapor lingers longer
    base_diffusion: float = 0.08
    ionization_gain: float = 1.2
    style: str = "uranium"
    
    # Post-processing
    glow_enabled: bool = True
    vignette_strength: float = 0.4
    distortion_strength: float = 0.15
    
    # Performance
    max_particles: int = 6000


class DecayRenderer:
    """
    Renders organic, smokey decay trails with drag and harmonic reactivity.
    """

    def __init__(self, config: DecayConfig | None = None, seed: int | None = None):
        self.cfg = config or DecayConfig()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # State
        self.particles: List[Particle] = []
        self.time = 0.0
        
        # Buffers: track (sharp) and vapor (diffuse)
        self.track_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.vapor_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        
        # Smoothed audio
        self._smooth_energy = 0.1
        self._smooth_percussive = 0.0
        self._smooth_harmonic = 0.2
        self._smooth_low = 0.1
        self._smooth_high = 0.1
        self._smooth_flux = 0.0
        self._smooth_sharpness = 0.0
        self._smooth_centroid = 0.5
        self._smooth_sub_bass = 0.0
        self._smooth_brilliance = 0.0

        # Dynamics
        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0
        
        self._distortion_offsets = np.mgrid[0:self.cfg.height, 0:self.cfg.width].astype(np.float32)

    def _lerp(self, current: float, target: float, factor: float) -> float:
        return current + (target - current) * factor

    def _smooth_audio(self, frame_data: dict[str, Any]):
        """Update smoothed audio values."""
        is_beat = frame_data.get("is_beat", False)
        fast = 0.35 if is_beat else 0.18
        slow = 0.06

        self._smooth_energy = self._lerp(self._smooth_energy, frame_data.get("global_energy", 0.1), slow)
        self._smooth_percussive = self._lerp(self._smooth_percussive, frame_data.get("percussive_impact", 0.0), fast)
        self._smooth_harmonic = self._lerp(self._smooth_harmonic, frame_data.get("harmonic_energy", 0.2), slow)
        self._smooth_low = self._lerp(self._smooth_low, frame_data.get("low_energy", 0.1), slow)
        self._smooth_high = self._lerp(self._smooth_high, frame_data.get("high_energy", 0.1), slow)
        self._smooth_flux = self._lerp(self._smooth_flux, frame_data.get("spectral_flux", 0.0), fast)
        self._smooth_sharpness = self._lerp(self._smooth_sharpness, frame_data.get("sharpness", 0.0), slow)
        self._smooth_centroid = self._lerp(self._smooth_centroid, frame_data.get("spectral_centroid", 0.5), slow)
        self._smooth_sub_bass = self._lerp(self._smooth_sub_bass, frame_data.get("sub_bass", 0.0), fast)
        self._smooth_brilliance = self._lerp(self._smooth_brilliance, frame_data.get("brilliance", 0.0), fast)

    def spawn_particle(self, p_type: str, x: float = None, y: float = None, 
                       vx: float = None, vy: float = None, gen: int = 0):
        if len(self.particles) >= self.cfg.max_particles:
            return

        if x is None:
            center_x, center_y = self.cfg.width / 2, self.cfg.height / 2
            base_radius = 50.0 * self.ore_scale
            spawn_angle = random.uniform(0, 2 * math.pi)
            r = base_radius * random.uniform(0.7, 1.1)
            x = center_x + math.cos(spawn_angle) * r
            y = center_y + math.sin(spawn_angle) * r
            angle = spawn_angle + random.uniform(-0.4, 0.4) + self.drift_angle
        else:
            angle = math.atan2(vy, vx) + random.uniform(-0.3, 0.3)

        # Initial velocity "kick" on beats
        kick = 1.0 + self._smooth_percussive * 1.5
        
        if p_type == "alpha":
            speed = random.uniform(5.0, 15.0) * kick
            life = 1.0
            decay_rate = random.uniform(0.02, 0.05)
            thickness = random.uniform(8.0, 16.0)
            intensity = random.uniform(1.0, 1.5)
            drag = 0.88 # Heavy drag for Alpha
        elif p_type == "beta":
            speed = random.uniform(30.0, 60.0) * kick
            life = 1.0
            decay_rate = random.uniform(0.005, 0.015)
            thickness = random.uniform(2.0, 4.0)
            intensity = random.uniform(0.7, 1.1)
            drag = 0.98 # Light drag for Beta
        else: # gamma
            speed = random.uniform(0, 10.0)
            life = 1.0
            decay_rate = 0.1
            thickness = random.uniform(4.0, 10.0)
            intensity = random.uniform(1.0, 2.0)
            drag = 0.95

        nvx = math.cos(angle) * speed
        nvy = math.sin(angle) * speed
        
        p = Particle(
            x=x, y=y, last_x=x, last_y=y,
            vx=nvx, vy=nvy, 
            life=life, decay_rate=decay_rate, 
            thickness=thickness, intensity=intensity,
            type=p_type, drag=drag, generation=gen
        )
        self.particles.append(p)

    def update_particles(self, dt: float):
        new_particles = []
        # Harmonic resonance wobble
        h_wobble = self._smooth_harmonic * 4.0
        
        for p in self.particles:
            p.life -= p.decay_rate
            if p.life > 0:
                p.last_x, p.last_y = p.x, p.y
                p.x += p.vx
                p.y += p.vy
                
                # Apply drag (fast-then-slow)
                p.vx *= p.drag
                p.vy *= p.drag
                
                # Harmonic wobble
                p.vx += random.uniform(-h_wobble, h_wobble)
                p.vy += random.uniform(-h_wobble, h_wobble)
                
                # Secondary branching
                if p.generation < 1 and random.random() < (0.01 * self._smooth_flux):
                    self.spawn_particle(p.type, p.x, p.y, p.vx, p.vy, p.generation + 1)

                new_particles.append(p)
        self.particles = new_particles

    def _draw_ore(self, draw: ImageDraw.Draw, center: tuple[float, float], scale: float):
        cx, cy = center
        num_points = 16
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points) + self.ore_rotation
            r = (50.0 + random.uniform(-10, 10) * self._smooth_energy) * scale
            points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
        
        # Ore glows with harmonics
        fill = 0.6 + self._smooth_harmonic * 0.4
        draw.polygon(points, fill=fill)
        # Core
        for _ in range(3):
            hr = random.uniform(5, 15) * scale
            draw.ellipse([cx-hr, cy-hr, cx+hr, cy+hr], fill=1.0)

    def _apply_vapor_distortion(self, buffer: np.ndarray) -> np.ndarray:
        strength = self.cfg.distortion_strength * (1.0 + self._smooth_flux * 2.0)
        if strength <= 0.01: return buffer
        h, w = buffer.shape
        t = self.time * 2.5
        dx = np.sin(t + np.linspace(0, 12, w)) * strength * 8
        dy = np.cos(t * 1.1 + np.linspace(0, 12, h)) * strength * 8
        y, x = self._distortion_offsets
        coords = np.array([y + dy[:, None], x + dx[None, :]])
        return map_coordinates(buffer, coords, order=1, mode='reflect')

    def get_raw_buffers(self, frame_data: dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Update and return raw (track, vapor) buffers before styling."""
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)
        
        # Dynamics
        self.ore_rotation += (0.04 + self._smooth_harmonic * 0.3)
        self.ore_scale = self._lerp(self.ore_scale, 1.0 + self._smooth_sub_bass * 1.2, 0.4)
        self.drift_angle += (self._smooth_centroid - 0.5) * dt * 6.0
        self.view_zoom = self._lerp(self.view_zoom, 1.0 + self._smooth_sub_bass * 0.3, 0.1)

        # Spawning
        cpm = cfg.base_cpm * (1.0 + self._smooth_energy * 4.0 + self._smooth_flux * 3.0)
        spawn_prob = (cpm / 60.0) * dt
        num_spawns = np.random.poisson(spawn_prob)
        if frame_data.get("is_beat", False):
            num_spawns += int(60 * self._smooth_percussive * cfg.ionization_gain)

        alpha_prob = self._smooth_low / (self._smooth_low + self._smooth_high + 1e-6)
        for _ in range(num_spawns):
            r = random.random()
            if r < alpha_prob * 0.5: self.spawn_particle("alpha")
            elif r < 0.92: self.spawn_particle("beta")
            else: self.spawn_particle("gamma")

        # Persistence
        self.track_buffer *= cfg.trail_persistence
        self.vapor_buffer *= cfg.vapor_persistence
        
        # Diffusion
        sigma = cfg.base_diffusion * 6 * (0.5 + self._smooth_harmonic)
        self.vapor_buffer = gaussian_filter(self.vapor_buffer, sigma=sigma)

        # Draw
        self.update_particles(dt)
        track_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        vapor_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        draw_t = ImageDraw.Draw(track_img)
        draw_v = ImageDraw.Draw(vapor_img)
        cx, cy = cfg.width / 2, cfg.height / 2
        self._draw_ore(draw_t, (cx, cy), self.ore_scale)
        
        for p in self.particles:
            color = float(p.intensity * p.life)
            thickness = int(p.thickness * (0.7 + p.life * 0.3))
            def transform(px, py):
                return (cx + (px - cx) * self.view_zoom, cy + (py - cy) * self.view_zoom)
            tx, ty = transform(p.x, p.y)
            tlx, tly = transform(p.last_x, p.last_y)
            if p.type == "gamma":
                draw_t.ellipse([tx - thickness, ty - thickness, tx + thickness, ty + thickness], fill=color)
            else:
                draw_t.line([(tlx, tly), (tx, ty)], fill=color, width=max(1, thickness // 2))
                draw_v.line([(tlx, tly), (tx, ty)], fill=color * 0.6, width=thickness)

        current_t = self._apply_vapor_distortion(np.array(track_img))
        current_v = self._apply_vapor_distortion(np.array(vapor_img))
        self.track_buffer = np.maximum(self.track_buffer, current_t)
        self.vapor_buffer = np.maximum(self.vapor_buffer, current_v)
        
        return self.track_buffer, self.vapor_buffer

    def _apply_styles(self, track: np.ndarray, vapor: np.ndarray) -> np.ndarray:
        style = self.cfg.style
        h, w = self.cfg.height, self.cfg.width
        cx, cy = w/2, h/2
        
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        core_radius = 50.0 * self.ore_scale
        dist_norm = dist / core_radius
        shift_mask = np.clip((dist_norm - 1.2) / 1.5, 0, 1)
        
        hue_base = (self.time * 0.05 + self._smooth_centroid * 0.5) % 1.0
        hue_tip = (hue_base + 0.33 + self._smooth_harmonic * 0.2) % 1.0

        if style in ["uranium", "neon"]:
            sat_in = 0.8 + self._smooth_energy * 0.2
            sat_out = 0.6 + self._smooth_brilliance * 0.4
            val_in = np.clip(track * 1.5 + vapor * 0.8, 0, 1)
            val_out = np.clip(track * 1.0 + vapor * 1.2, 0, 1)
            rgb_in = self._hsv_to_rgb(hue_base, sat_in, val_in)
            rgb_out = self._hsv_to_rgb(hue_tip, sat_out, val_out)
            rgb = rgb_in * (1 - shift_mask[:,:,None]) + rgb_out * shift_mask[:,:,None]
            heat_mask = np.exp(-dist / (60.0 * self.ore_scale))
            heat_hue = (hue_base - 0.1) % 1.0
            rgb_heat = self._hsv_to_rgb(heat_hue, 1.0, heat_mask * (0.8 + self._smooth_sub_bass))
            rgb = np.maximum(rgb, rgb_heat)
        elif style == "noir":
            val = np.clip(track * 1.3 + vapor * 0.7, 0, 1)
            val = np.power(val, 1.5)
            rgb = self._hsv_to_rgb(hue_base, 0.2, val)
        else: # lab
            val = np.clip(track * 1.2 + vapor * 0.6, 0, 1)
            rgb = np.stack([val, val, val], axis=-1)
            
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def _hsv_to_rgb(self, h: float, s: float, v: np.ndarray) -> np.ndarray:
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        sector = int(h * 6) % 6
        if sector == 0: r, g, b = c, x, 0
        elif sector == 1: r, g, b = x, c, 0
        elif sector == 2: r, g, b = 0, c, x
        elif sector == 3: r, g, b = 0, x, c
        elif sector == 4: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        return np.stack([r + m, g + m, b + m], axis=-1)

    def render_frame(self, frame_data: dict[str, Any], frame_index: int) -> np.ndarray:
        track, vapor = self.get_raw_buffers(frame_data)
        rgb = self._apply_styles(track, vapor)
        cfg = self.cfg
        if cfg.glow_enabled:
            g_int = 0.35 + self._smooth_flux * 0.5 + self._smooth_harmonic * 0.3
            rgb = add_glow(rgb, intensity=min(g_int, 0.9), radius=18)
        if cfg.vignette_strength > 0:
            v_str = cfg.vignette_strength * (1.0 + self._smooth_sub_bass * 1.5)
            rgb = vignette(rgb, strength=v_str)
        return tone_map_soft(rgb)

    def render_manifest(self, manifest: dict[str, Any], progress_callback: callable = None) -> Iterator[np.ndarray]:
        frames = manifest.get("frames", [])
        total = len(frames)
        self.particles = []
        self.track_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.vapor_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.time = 0.0
        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0
        for i, frame_data in enumerate(frames):
            yield self.render_frame(frame_data, i)
            if progress_callback: progress_callback(i + 1, total)


class MirrorRenderer:
    """
    Manages two independent DecayRenderer instances and composites them
    using spatial splits and interference logic.
    """
    def __init__(self, config: DecayConfig, split_mode: str = "vertical", 
                 interference_mode: str = "resonance"):
        self.cfg = config
        self.split_mode = split_mode # vertical, horizontal, diagonal, circular
        self.int_mode = interference_mode # resonance, constructive, destructive, difference
        
        # Instance A and B (different seeds for independence)
        self.instance_a = DecayRenderer(config, seed=42)
        self.instance_b = DecayRenderer(config, seed=1337)
        
        h, w = config.height, config.width
        self.y, self.x = np.ogrid[:h, :w]

    def _generate_masks(self, frame_data: dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = self.cfg.height, self.cfg.width
        cx, cy = w/2, h/2
        
        # Pulse the split line slightly to music
        pulse = math.sin(self.instance_a.time * 2) * 20 * self.instance_a._smooth_energy
        
        if self.split_mode == "vertical":
            dist_field = self.x - (cx + pulse)
            mask_a = np.clip(0.5 - dist_field / 100.0, 0, 1)
            mask_b = np.clip(0.5 + dist_field / 100.0, 0, 1)
        elif self.split_mode == "horizontal":
            dist_field = self.y - (cy + pulse)
            mask_a = np.clip(0.5 - dist_field / 100.0, 0, 1)
            mask_b = np.clip(0.5 + dist_field / 100.0, 0, 1)
        elif self.split_mode == "diagonal":
            dist_field = (self.x - cx) - (self.y - cy) + pulse
            mask_a = np.clip(0.5 - dist_field / 100.0, 0, 1)
            mask_b = np.clip(0.5 + dist_field / 100.0, 0, 1)
        else: # circular
            r = np.sqrt((self.x - cx)**2 + (self.y - cy)**2)
            # Split at 1.5x current ore size
            boundary = 150.0 * self.instance_a.ore_scale + pulse
            mask_a = np.clip(0.5 - (r - boundary) / 100.0, 0, 1)
            mask_b = np.clip(0.5 + (r - boundary) / 100.0, 0, 1)
            
        return mask_a, mask_b

    def render_frame(self, frame_data: dict[str, Any], frame_index: int) -> np.ndarray:
        # 1. Get raw buffers from both independent instances
        t_a, v_a = self.instance_a.get_raw_buffers(frame_data)
        t_b, v_b = self.instance_b.get_raw_buffers(frame_data)
        
        # 2. Generate spatial masks for A and B
        mask_a, mask_b = self._generate_masks(frame_data)
        
        # 3. Interference logic
        # Resonance (Multiplicative): Shows where both are high (The "Sweet Spot")
        # Constructive (Additive): Combined energy
        # Destructive (Difference): Energy where they vary
        
        def compute_interference(a, b):
            if self.int_mode == "resonance":
                return (a * b) * 4.0 # Boost resonance
            elif self.int_mode == "constructive":
                return (a + b) * 0.5
            elif self.int_mode == "destructive":
                return np.abs(a - b)
            else: # "sweet_spot" hybrid
                return np.maximum(a * mask_a, b * mask_b) + (a * b * 2.0)

        # Composite track and vapor buffers with interference in the overlap
        # overlap zone is where mask_a * mask_b > 0
        overlap = mask_a * mask_b
        
        track_final = (t_a * mask_a * (1-overlap)) + (t_b * mask_b * (1-overlap)) + compute_interference(t_a, t_b) * overlap
        vapor_final = (v_a * mask_a * (1-overlap)) + (v_b * mask_b * (1-overlap)) + compute_interference(v_a, v_b) * overlap
        
        # 4. Style the final composite buffer using instance_a's color state (synchronized)
        rgb = self.instance_a._apply_styles(track_final, vapor_final)
        
        # 5. Final post-processing
        cfg = self.cfg
        if cfg.glow_enabled:
            g_int = 0.35 + self.instance_a._smooth_flux * 0.5 + self.instance_a._smooth_harmonic * 0.3
            rgb = add_glow(rgb, intensity=min(g_int, 0.9), radius=18)
        if cfg.vignette_strength > 0:
            v_str = cfg.vignette_strength * (1.0 + self.instance_a._smooth_sub_bass * 1.5)
            rgb = vignette(rgb, strength=v_str)
            
        return tone_map_soft(rgb)

    def render_manifest(self, manifest: dict[str, Any], progress_callback: callable = None) -> Iterator[np.ndarray]:
        frames = manifest.get("frames", [])
        total = len(frames)
        # Reset instances
        self.instance_a = DecayRenderer(self.cfg, seed=42)
        self.instance_b = DecayRenderer(self.cfg, seed=1337)
        for i, frame_data in enumerate(frames):
            yield self.render_frame(frame_data, i)
            if progress_callback: progress_callback(i + 1, total)
