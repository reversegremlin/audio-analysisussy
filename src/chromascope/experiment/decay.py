"""
Audio-reactive cloud chamber decay renderer.

Translates music into a field of radioactive trails:
- Alpha: short, thick, bright
- Beta: long, thin, fast
- Gamma: sparse flashes/speck events
- Central Ore: Pulsating radioactive core.

Dynamic Sliding Mirror Architecture:
- Symmetrical Opposites: Two halves move in exact opposite directions.
- Centered Intersection: Collision and interference ALWAYS happen around the center.
- Axis-Locked Motion: Slides are strictly horizontal, vertical, or diagonal.
- Audio-Reactive Phasing: Music energy drives the speed and "rush" of the overlap.
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
    vapor_persistence: float = 0.98 
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
    Truly independent via local random state.
    """

    def __init__(self, config: DecayConfig | None = None, seed: int | None = None, 
                 center_pos: Tuple[float, float] | None = None):
        self.cfg = config or DecayConfig()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.center_pos = center_pos or (self.cfg.width / 2, self.cfg.height / 2)
        
        # State
        self.particles: List[Particle] = []
        self.time = 0.0
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

        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0
        self._distortion_offsets = np.mgrid[0:self.cfg.height, 0:self.cfg.width].astype(np.float32)

    def _lerp(self, current: float, target: float, factor: float) -> float:
        return current + (target - current) * factor

    def _smooth_audio(self, frame_data: dict[str, Any]):
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
            speed, decay, thick, drag = self.rng.uniform(5, 15)*kick, self.rng.uniform(0.02, 0.05), self.rng.uniform(8, 16), 0.88
        elif p_type == "beta":
            speed, decay, thick, drag = self.rng.uniform(30, 60)*kick, self.rng.uniform(0.005, 0.015), self.rng.uniform(2, 4), 0.98
        else:
            speed, decay, thick, drag = self.rng.uniform(0, 10), 0.1, self.rng.uniform(4, 10), 0.95

        p = Particle(x=x, y=y, last_x=x, last_y=y, vx=math.cos(angle)*speed, vy=math.sin(angle)*speed, 
                     life=1.0, decay_rate=decay, thickness=thick, intensity=self.rng.uniform(0.7, 1.5), 
                     type=p_type, drag=drag, generation=gen)
        self.particles.append(p)

    def update_particles(self, dt: float):
        new_particles = []
        h_wobble = self._smooth_harmonic * 4.0
        for p in self.particles:
            p.life -= p.decay_rate
            if p.life > 0:
                p.last_x, p.last_y = p.x, p.y
                p.x += p.vx; p.y += p.vy
                p.vx *= p.drag; p.vy *= p.drag
                p.vx += self.rng.uniform(-h_wobble, h_wobble)
                p.vy += self.rng.uniform(-h_wobble, h_wobble)
                if p.generation < 1 and self.rng.random() < (0.01 * self._smooth_flux):
                    self.spawn_particle(p.type, p.x, p.y, p.vx, p.vy, p.generation + 1)
                new_particles.append(p)
        self.particles = new_particles

    def _draw_ore(self, draw: ImageDraw.Draw, center: tuple[float, float], scale: float):
        cx, cy = center
        num_pts = 16; pts = []
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
        cfg = self.cfg; dt = 1.0 / cfg.fps
        self.time += dt; self._smooth_audio(frame_data)
        self.ore_rotation += (0.04 + self._smooth_harmonic * 0.3)
        self.ore_scale = self._lerp(self.ore_scale, 1.0 + self._smooth_sub_bass * 1.2, 0.4)
        self.drift_angle += (self._smooth_centroid - 0.5) * dt * 6.0
        self.view_zoom = self._lerp(self.view_zoom, 1.0 + self._smooth_sub_bass * 0.3, 0.1)

        cpm = cfg.base_cpm * (1.0 + self._smooth_energy * 4.0 + self._smooth_flux * 3.0)
        num_spawns = self.np_rng.poisson((cpm / 60.0) * dt)
        if frame_data.get("is_beat", False): num_spawns += int(60 * self._smooth_percussive * cfg.ionization_gain)

        alpha_prob = self._smooth_low / (self._smooth_low + self._smooth_high + 1e-6)
        for _ in range(num_spawns):
            r = self.rng.random()
            if r < alpha_prob * 0.5: self.spawn_particle("alpha")
            elif r < 0.92: self.spawn_particle("beta")
            else: self.spawn_particle("gamma")

        self.track_buffer *= cfg.trail_persistence; self.vapor_buffer *= cfg.vapor_persistence
        self.vapor_buffer = gaussian_filter(self.vapor_buffer, sigma=cfg.base_diffusion * 6 * (0.5 + self._smooth_harmonic))

        self.update_particles(dt)
        track_img = Image.new("F", (cfg.width, cfg.height), 0.0); vapor_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        draw_t = ImageDraw.Draw(track_img); draw_v = ImageDraw.Draw(vapor_img)
        self._draw_ore(draw_t, self.center_pos, self.ore_scale)
        
        v_cx, v_cy = cfg.width / 2, cfg.height / 2
        for p in self.particles:
            color = float(p.intensity * p.life); thickness = int(p.thickness * (0.7 + p.life * 0.3))
            def transform(px, py): return (v_cx + (px - v_cx) * self.view_zoom, v_cy + (py - v_cy) * self.view_zoom)
            tx, ty = transform(p.x, p.y); tlx, tly = transform(p.last_x, p.last_y)
            if p.type == "gamma": draw_t.ellipse([tx - thickness, ty - thickness, tx + thickness, ty + thickness], fill=color)
            else:
                draw_t.line([(tlx, tly), (tx, ty)], fill=color, width=max(1, thickness // 2))
                draw_v.line([(tlx, tly), (tx, ty)], fill=color * 0.6, width=thickness)

        self.track_buffer = np.maximum(self.track_buffer, np.array(track_img))
        self.vapor_buffer = np.maximum(self.vapor_buffer, np.array(vapor_img))
        return self.track_buffer, self.vapor_buffer

    def _hsv_to_rgb(self, h: float, s: float, v: np.ndarray) -> np.ndarray:
        c = v * s; x = c * (1 - abs((h * 6) % 2 - 1)); m = v - c
        sector = int(h * 6) % 6
        if sector == 0: r, g, b = c, x, 0
        elif sector == 1: r, g, b = x, c, 0
        elif sector == 2: r, g, b = 0, c, x
        elif sector == 3: r, g, b = 0, x, c
        elif sector == 4: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        return np.stack([r + m, g + m, b + m], axis=-1)

    def _apply_styles(self, track: np.ndarray, vapor: np.ndarray, 
                      dist_field: np.ndarray | None = None, 
                      int_mask: np.ndarray | None = None) -> np.ndarray:
        style = self.cfg.style; h, w = self.cfg.height, self.cfg.width; cx, cy = w/2, h/2
        if dist_field is None:
            y, x = np.ogrid[:h, :w]
            dist_field = np.sqrt((x - cx)**2 + (y - cy)**2)
        shift_mask = np.clip((dist_field / (50.0 * self.ore_scale) - 1.2) / 1.5, 0, 1)
        hue_base = (self.time * 0.05 + self._smooth_centroid * 0.5) % 1.0
        hue_tip = (hue_base + 0.33 + self._smooth_harmonic * 0.2) % 1.0
        if style in ["uranium", "neon"]:
            rgb_in = self._hsv_to_rgb(hue_base, 0.8 + self._smooth_energy * 0.2, np.clip(track * 1.5 + vapor * 0.8, 0, 1))
            rgb_out = self._hsv_to_rgb(hue_tip, 0.6 + self._smooth_brilliance * 0.4, np.clip(track * 1.0 + vapor * 1.2, 0, 1))
            rgb = rgb_in * (1 - shift_mask[:,:,None]) + rgb_out * shift_mask[:,:,None]
            
            # Annihilation color (matter-antimatter)
            if int_mask is not None:
                # Pulse the interference intensity with percussive impact
                p_kick = self._smooth_percussive * 3.0
                hue_int = (hue_base + 0.5 + self._smooth_harmonic * 0.1) % 1.0
                rgb_int = self._hsv_to_rgb(hue_int, 0.9, np.clip(track * (2.0 + p_kick) + vapor * 1.5, 0, 1))
                
                # Add white-hot core to interference, scaled by percussive impact
                white_hot = np.clip(track * (3.0 + p_kick * 2.0), 0, 1)[:,:,None]
                rgb_int = np.maximum(rgb_int, white_hot)
                
                # Blend in the interference color
                m = int_mask[:,:,None]
                rgb = rgb * (1 - m) + rgb_int * m

            heat_mask = np.exp(-dist_field / (60.0 * self.ore_scale))
            rgb_heat = self._hsv_to_rgb((hue_base - 0.1) % 1.0, 1.0, heat_mask * (0.8 + self._smooth_sub_bass))
            rgb = np.maximum(rgb, rgb_heat)
        elif style == "noir":
            rgb = self._hsv_to_rgb(hue_base, 0.2, np.power(np.clip(track * 1.3 + vapor * 0.7, 0, 1), 1.5))
        else:
            v = np.clip(track * 1.2 + vapor * 0.6, 0, 1)
            rgb = np.stack([v, v, v], axis=-1)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def render_frame(self, frame_data: dict[str, Any], frame_index: int) -> np.ndarray:
        track, vapor = self.get_raw_buffers(frame_data)
        rgb = self._apply_styles(track, vapor)
        if self.cfg.glow_enabled: rgb = add_glow(rgb, intensity=min(0.35 + self._smooth_flux * 0.5, 0.9), radius=18)
        if self.cfg.vignette_strength > 0: rgb = vignette(rgb, strength=self.cfg.vignette_strength * (1.0 + self._smooth_sub_bass * 1.5))
        return tone_map_soft(rgb)

    def render_manifest(self, manifest: dict[str, Any], progress_callback: callable = None) -> Iterator[np.ndarray]:
        for i, frame_data in enumerate(manifest.get("frames", [])):
            yield self.render_frame(frame_data, i)
            if progress_callback: progress_callback(i + 1, len(manifest["frames"]))


class MirrorRenderer:
    """
    Dynamic Symmetrical Sliding Compositor. 
    Halves move in strictly opposite directions along a specific axis.
    Intersection ALWAYS centered.
    """
    MIRROR_MODES = ["vertical", "horizontal", "diagonal", "circular"]
    INT_MODES = ["resonance", "constructive", "destructive", "sweet_spot"]

    def __init__(self, config: DecayConfig, split_mode: str = "vertical", 
                 interference_mode: str = "resonance"):
        self.cfg = config
        self.requested_split = split_mode
        self.requested_int = interference_mode
        self.curr_split_idx = 0 if split_mode == "cycle" else self.MIRROR_MODES.index(split_mode)
        self.next_split_idx = self.curr_split_idx
        self.curr_int_idx = 0 if interference_mode == "cycle" else self.INT_MODES.index(interference_mode)
        self.next_int_idx = self.curr_int_idx
        self.transition_alpha = 0.0
        self.change_potential = 0.0
        
        # Symmetrical instances (identical settings, different seeds)
        self.instance_a = DecayRenderer(config, seed=42)
        self.instance_b = DecayRenderer(config, seed=1337)
        
        self.phase = 0.0
        h, w = config.height, config.width
        self.yg, self.xg = np.mgrid[0:h, 0:w].astype(np.float32)

    def _get_identity_mask(self, mode: str) -> np.ndarray:
        h, w = self.cfg.height, self.cfg.width; cx, cy = w/2, h/2; grad = 50.0 # Sharper split
        if mode == "vertical": return np.clip(0.5 - (self.xg - cx) / grad, 0, 1)
        elif mode == "horizontal": return np.clip(0.5 - (self.yg - cy) / grad, 0, 1)
        elif mode == "diagonal": return np.clip(0.5 - ((self.xg - cx) - (self.yg - cy)) / grad, 0, 1)
        else: r = np.sqrt((self.xg - cx)**2 + (self.yg - cy)**2); return np.clip(0.5 - (r - 200.0) / grad, 0, 1)

    def _smooth_shift(self, buffer: np.ndarray, dy: float, dx: float) -> np.ndarray:
        coords = np.array([self.yg - dy, self.xg - dx])
        return map_coordinates(buffer, coords, order=1, mode='wrap')

    def render_frame(self, frame_data: dict[str, Any], frame_index: int) -> np.ndarray:
        dt = 1.0 / self.cfg.fps; energy = frame_data.get("global_energy", 0.1)
        is_beat = frame_data.get("is_beat", False); percussive = frame_data.get("percussive_impact", 0.0)
        sub_bass = frame_data.get("sub_bass", 0.0)
        
        # 1. Update State - Synchronized Motion & Reversals
        # Beat-locked reversals
        if is_beat and sub_bass > 0.6: self.phase_dir = -getattr(self, 'phase_dir', 1.0)
        else: self.phase_dir = getattr(self, 'phase_dir', 1.0)
        
        # Audio-reactive phase increment with beat "shiver"
        p_inc = (0.12 + energy * 0.35) * self.phase_dir
        if is_beat: p_inc *= (1.5 + percussive * 2.0)
        self.phase += dt * p_inc
        
        # 2. Beat-Locked Mode Cycling
        if self.requested_split == "cycle" or self.requested_int == "cycle":
            self.change_potential += energy * dt * 2.0
            # Wait for both threshold AND a beat to trigger change
            if self.change_potential > 0.85 and is_beat and self.transition_alpha <= 0:
                self.change_potential = 0
                if self.requested_split == "cycle": self.next_split_idx = (self.curr_split_idx + 1) % len(self.MIRROR_MODES)
                if self.requested_int == "cycle": self.next_int_idx = (self.curr_int_idx + 1) % len(self.INT_MODES)
            
            if self.next_split_idx != self.curr_split_idx or self.next_int_idx != self.curr_int_idx:
                self.transition_alpha += dt * 0.7
                if self.transition_alpha >= 1.0:
                    self.curr_split_idx = self.next_split_idx; self.curr_int_idx = self.next_int_idx; self.transition_alpha = 0.0

        # 3. Determine strictly symmetrical panning axis
        mode = self.MIRROR_MODES[self.curr_split_idx]
        amp_x, amp_y = self.cfg.width * 0.45, self.cfg.height * 0.45
        
        if mode == "vertical": axis_x, axis_y = 1.0, 0.0
        elif mode == "horizontal": axis_x, axis_y = 0.0, 1.0
        elif mode == "diagonal": axis_x, axis_y = 1.0, 1.0
        else: axis_x, axis_y = 1.0, -1.0 # Circular mode slides along opposite diagonal
        
        # strictly opposite oscillation around center
        osc = math.sin(self.phase)
        off_a_x, off_a_y = axis_x * osc * amp_x, axis_y * osc * amp_y
        off_b_x, off_b_y = -off_a_x, -off_a_y

        # 4. Process Instances
        t_a, v_a = self.instance_a.get_raw_buffers(frame_data)
        t_b, v_b = self.instance_b.get_raw_buffers(frame_data)
        
        mask_src_a_curr = self._get_identity_mask(self.MIRROR_MODES[self.curr_split_idx])
        mask_src_a_next = self._get_identity_mask(self.MIRROR_MODES[self.next_split_idx])
        mask_src_a = mask_src_a_curr * (1 - self.transition_alpha) + mask_src_a_next * self.transition_alpha
        mask_src_b = 1.0 - mask_src_a
        
        # Shift buffers
        t_a_s = self._smooth_shift(t_a * mask_src_a, off_a_y, off_a_x)
        v_a_s = self._smooth_shift(v_a * mask_src_a, off_a_y, off_a_x)
        mask_a_s = self._smooth_shift(mask_src_a, off_a_y, off_a_x)
        
        t_b_s = self._smooth_shift(t_b * mask_src_b, off_b_y, off_b_x)
        v_b_s = self._smooth_shift(v_b * mask_src_b, off_b_y, off_b_x)
        mask_b_s = self._smooth_shift(mask_src_b, off_b_y, off_b_x)
        
        # Shift Distance Fields
        cx, cy = self.cfg.width/2, self.cfg.height/2
        dist_base = np.sqrt((self.xg - cx)**2 + (self.yg - cy)**2)
        dist_final = np.minimum(self._smooth_shift(dist_base, off_a_y, off_a_x),
                               self._smooth_shift(dist_base, off_b_y, off_b_x))
        
        overlap = np.clip(mask_a_s * mask_b_s * 4.0, 0, 1)
        
        def compute_int(a, b, mode, p_boost):
            # Violent beat-reactive gain
            gain = 1.0 + p_boost * 3.5
            if mode == "resonance": return (a * b) * 12.0 * gain
            elif mode == "constructive": return (a + b) * 0.8 * gain
            elif mode == "destructive": return np.abs(a - b) * 5.0 * gain
            else: return (np.maximum(a, b) + (a * b * 10.0)) * gain

        # Calculate interference with percussive boost
        track_int = compute_int(t_a_s, t_b_s, self.INT_MODES[self.curr_int_idx], percussive) * (1-self.transition_alpha) + \
                    compute_int(t_a_s, t_b_s, self.INT_MODES[self.next_int_idx], percussive) * self.transition_alpha
        vapor_int = compute_int(v_a_s, v_b_s, self.INT_MODES[self.curr_int_idx], percussive) * (1-self.transition_alpha) + \
                    compute_int(v_a_s, v_b_s, self.INT_MODES[self.next_int_idx], percussive) * self.transition_alpha

        track_final = (t_a_s * (1-overlap)) + (t_b_s * (1-overlap)) + (track_int * overlap)
        vapor_final = (v_a_s * (1-overlap)) + (v_b_s * (1-overlap)) + (vapor_int * overlap)
        
        rgb = self.instance_a._apply_styles(track_final, vapor_final, dist_field=dist_final, int_mask=overlap)
        if self.cfg.glow_enabled: rgb = add_glow(rgb, intensity=min(0.35 + self.instance_a._smooth_flux * 0.5, 0.9), radius=18)
        if self.cfg.vignette_strength > 0: rgb = vignette(rgb, strength=self.cfg.vignette_strength * (1.0 + self.instance_a._smooth_sub_bass * 1.5))
        return tone_map_soft(rgb)

    def render_manifest(self, manifest: dict[str, Any], progress_callback: callable = None) -> Iterator[np.ndarray]:
        self.instance_a = DecayRenderer(self.cfg, seed=42); self.instance_b = DecayRenderer(self.cfg, seed=1337)
        self.phase = 0.0
        for i, frame_data in enumerate(manifest.get("frames", [])):
            yield self.render_frame(frame_data, i)
            if progress_callback: progress_callback(i + 1, len(manifest["frames"]))
