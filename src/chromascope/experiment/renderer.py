"""
Universal Master Compositor and Orchestrator.
Handles mirroring, interference, and heterogeneous visualizer blending.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from scipy.ndimage import map_coordinates

from chromascope.experiment.base import BaseConfig, BaseVisualizer, VisualPolisher


class UniversalMirrorCompositor:
    """
    Handles symmetrical mirroring and interference for any BaseVisualizer.
    Can also blend different visualizer types.
    """

    MIRROR_MODES = ["off", "vertical", "horizontal", "diagonal", "circular"]
    INT_MODES = ["resonance", "constructive", "destructive", "sweet_spot"]

    def __init__(
        self, 
        viz_class_a: Type[BaseVisualizer], 
        config: BaseConfig,
        viz_class_b: Optional[Type[BaseVisualizer]] = None,
        seed: int = 42
    ):
        self.cfg = config
        self.viz_class_a = viz_class_a
        self.viz_class_b = viz_class_b or viz_class_a
        
        # Scaling logic (MultiProfile)
        self.render_w, self.render_h = config.width, config.height
        self.sim_w, self.sim_h = config.get_profile_dims()
        
        # Adjust config for instances to use sim resolution
        # We'll pass a copy to avoid side effects
        from dataclasses import replace
        sim_cfg = replace(config, width=self.sim_w, height=self.sim_h)
        
        # Symmetrical instances
        self.instance_a = self.viz_class_a(sim_cfg, seed=seed)
        self.instance_b = self.viz_class_b(sim_cfg, seed=seed + 1337)
        
        self.phase = 0.0
        self.phase_dir = 1.0
        self.time = 0.0
        
        # State for cycling
        self.curr_mirror_idx = self.MIRROR_MODES.index(
            config.mirror_mode if config.mirror_mode in self.MIRROR_MODES else "vertical"
        )
        self.next_mirror_idx = self.curr_mirror_idx
        self.curr_int_idx = self.INT_MODES.index(
            config.interference_mode if config.interference_mode in self.INT_MODES else "resonance"
        )
        self.next_int_idx = self.curr_int_idx
        self.transition_alpha = 0.0
        self.change_potential = 0.0
        
        # Grid for shifts (sim resolution)
        self.yg, self.xg = np.mgrid[0:self.sim_h, 0:self.sim_w].astype(np.float32)

    def _get_identity_mask(self, mode: str) -> np.ndarray:
        h, w = self.sim_h, self.sim_w
        cx, cy = w / 2, h / 2
        grad = 50.0 * (w / self.render_w) # Adjusted gradient for sim resolution
        
        if mode == "vertical":
            return np.clip(0.5 - (self.xg - cx) / grad, 0, 1)
        elif mode == "horizontal":
            return np.clip(0.5 - (self.yg - cy) / grad, 0, 1)
        elif mode == "diagonal":
            return np.clip(0.5 - ((self.xg - cx) - (self.yg - cy)) / grad, 0, 1)
        elif mode == "circular":
            r = np.sqrt((self.xg - cx)**2 + (self.yg - cy)**2)
            return np.clip(0.5 - (r - min(h, w) // 4) / grad, 0, 1)
        return np.ones((h, w), dtype=np.float32)


    def _smooth_shift(self, buffer: np.ndarray, dy: float, dx: float) -> np.ndarray:
        # Use wrap mode for infinite feel
        coords = np.array([self.yg - dy, self.xg - dx])
        return map_coordinates(buffer, coords, order=1, mode='wrap')

    def render_frame(self, frame_data: Dict[str, Any], frame_index: int) -> np.ndarray:
        """Composites a single frame."""
        dt = 1.0 / self.cfg.fps
        self.time += dt
        energy = frame_data.get("global_energy", 0.1)
        is_beat = frame_data.get("is_beat", False)
        percussive = frame_data.get("percussive_impact", 0.0)
        sub_bass = frame_data.get("sub_bass", 0.0)
        
        # 1. Update Phase
        if is_beat and sub_bass > 0.7:
            self.phase_dir = -self.phase_dir
            
        p_inc = (0.02 + energy * 0.08) * self.phase_dir
        if is_beat:
            p_inc *= (1.1 + percussive * 1.2)
        self.phase += dt * p_inc
        
        # 2. Cycle logic
        if self.cfg.mirror_mode == "cycle" or self.cfg.interference_mode == "cycle":
            self.change_potential += energy * dt * 2.0
            if self.change_potential > 0.85 and is_beat and self.transition_alpha <= 0:
                self.change_potential = 0
                if self.cfg.mirror_mode == "cycle":
                    self.next_mirror_idx = (self.curr_mirror_idx + 1) % len(self.MIRROR_MODES)
                    if self.MIRROR_MODES[self.next_mirror_idx] == "off": # Skip off in cycle
                         self.next_mirror_idx = (self.next_mirror_idx + 1) % len(self.MIRROR_MODES)
                if self.cfg.interference_mode == "cycle":
                    self.next_int_idx = (self.curr_int_idx + 1) % len(self.INT_MODES)
            
            if self.next_mirror_idx != self.curr_mirror_idx or self.next_int_idx != self.curr_int_idx:
                self.transition_alpha += dt * 0.6
                if self.transition_alpha >= 1.0:
                    self.curr_mirror_idx = self.next_mirror_idx
                    self.curr_int_idx = self.next_int_idx
                    self.transition_alpha = 0.0

        # 3. Off mode optimization
        mode_str = self.MIRROR_MODES[self.curr_mirror_idx]
        if mode_str == "off" and self.transition_alpha <= 0:
            return self.instance_a.render_frame(frame_data, frame_index)

        # 4. Axis-Locked Symmetrical Clashing
        amp_x, amp_y = self.cfg.width * 0.4, self.cfg.height * 0.4
        
        if mode_str == "vertical": axis_x, axis_y = 1.0, 0.0
        elif mode_str == "horizontal": axis_x, axis_y = 0.0, 1.0
        elif mode_str == "diagonal": axis_x, axis_y = 1.0, 1.0
        else: axis_x, axis_y = 0.0, 0.0 # circular/off
        
        osc = math.sin(self.phase)
        off_a_x, off_a_y = axis_x * osc * amp_x, axis_y * osc * amp_y
        off_b_x, off_b_y = -off_a_x, -off_a_y

        # 5. Process Instances
        self.instance_a.update(frame_data)
        self.instance_b.update(frame_data)
        
        field_a = self.instance_a.get_raw_field()
        field_b = self.instance_b.get_raw_field()
        
        # Merge multi-fields if necessary (Decay style)
        def normalize_field(f):
            if isinstance(f, tuple):
                return np.clip(f[0] * 1.5 + f[1] * 0.8, 0, 1)
            return f
            
        field_a = normalize_field(field_a)
        field_b = normalize_field(field_b)
        
        # Masks
        mask_src_a_curr = self._get_identity_mask(self.MIRROR_MODES[self.curr_mirror_idx])
        mask_src_a_next = self._get_identity_mask(self.MIRROR_MODES[self.next_mirror_idx])
        mask_src_a = mask_src_a_curr * (1 - self.transition_alpha) + mask_src_a_next * self.transition_alpha
        mask_src_b = 1.0 - mask_src_a
        
        # Shift
        field_a_s = self._smooth_shift(field_a * mask_src_a, off_a_y, off_a_x)
        mask_a_s = self._smooth_shift(mask_src_a, off_a_y, off_a_x)
        
        field_b_s = self._smooth_shift(field_b * mask_src_b, off_b_y, off_b_x)
        mask_b_s = self._smooth_shift(mask_src_b, off_b_y, off_b_x)
        
        overlap = np.clip(mask_a_s * mask_b_s * 4.0, 0, 1)
        
        # 6. Interference
        def compute_int(a, b, mode, p_boost):
            gain = 1.0 + p_boost * 3.5
            if mode == "resonance": return (a * b) * 5.0 * gain
            elif mode == "constructive": return (a + b) * 0.8 * gain
            elif mode == "destructive": return np.abs(a - b) * 5.0 * gain
            else: return (np.maximum(a, b) + (a * b * 10.0)) * gain


        int_mode_curr = self.INT_MODES[self.curr_int_idx]
        int_mode_next = self.INT_MODES[self.next_int_idx]
        
        field_int = compute_int(field_a_s, field_b_s, int_mode_curr, percussive) * (1-self.transition_alpha) + \
                    compute_int(field_a_s, field_b_s, int_mode_next, percussive) * self.transition_alpha

        field_final = (field_a_s * (1-overlap)) + (field_b_s * (1-overlap)) + (field_int * overlap)
        field_final = np.clip(field_final, 0, 1)
        
        # 7. Polish
        polisher = VisualPolisher(self.cfg)
        return polisher.apply(field_final, frame_data, self.time, self.instance_a._smooth_audio_dict())
