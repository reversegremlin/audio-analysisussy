"""
Base classes and universal styling for the Chromascope engine.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from chromascope.experiment.colorgrade import (
    add_glow,
    apply_palette,
    chromatic_aberration,
    tone_map_soft,
    vignette,
)


@dataclass
class BaseConfig:
    """Universal configuration for all visualizers."""
    width: int = 1920
    height: int = 1080
    fps: int = 60
    
    # Common post-processing
    glow_enabled: bool = True
    glow_intensity: float = 0.35
    glow_radius: int = 15
    aberration_enabled: bool = True
    aberration_offset: int = 3
    vignette_strength: float = 0.3
    
    # Palette
    palette_type: str = "jewel"  # "jewel", "solar", "nebula"
    
    # Mirror/Interference defaults
    mirror_mode: str = "off"  # "off", "vertical", "horizontal", "diagonal", "circular", "cycle"
    interference_mode: str = "resonance"  # "resonance", "constructive", "destructive", "sweet_spot", "cycle"

    # Performance
    low_res_mirror: bool = True # Scaling down internal simulations during mirror

    def get_profile_dims(self) -> Tuple[int, int]:
        """Returns (width, height) adjusted for performance."""
        if self.mirror_mode != "off" and self.low_res_mirror:
            # Scale down to 75% for mirrored renders to keep FPS up
            return (int(self.width * 0.75), int(self.height * 0.75))
        return (self.width, self.height)



class BaseVisualizer(abc.ABC):
    """
    Abstract Base Class for all Chromascope visualizers.
    Standardizes state management and audio-reactive advanced simulations.
    """

    def __init__(
        self, 
        config: Optional[BaseConfig] = None, 
        seed: Optional[int] = None, 
        center_pos: Optional[Tuple[float, float]] = None
    ):
        self.cfg = config or BaseConfig()
        self.rng = np.random.default_rng(seed)
        self.center_pos = center_pos or (self.cfg.width / 2, self.cfg.height / 2)
        
        # Internal state
        self.time = 0.0
        
        # Smoothed audio values (standardized across all renderers)
        self._smooth_energy = 0.1
        self._smooth_percussive = 0.0
        self._smooth_harmonic = 0.2
        self._smooth_low = 0.1
        self._smooth_high = 0.1
        self._smooth_flux = 0.0
        self._smooth_flatness = 0.0
        self._smooth_sharpness = 0.0
        self._smooth_sub_bass = 0.0
        self._smooth_brilliance = 0.0
        self._smooth_centroid = 0.5

    def _lerp(self, current: float, target: float, factor: float) -> float:
        return current + (target - current) * factor

    def _smooth_audio(self, frame_data: Dict[str, Any]):
        """Standard audio smoothing logic."""
        is_beat = frame_data.get("is_beat", False)
        fast = 0.35 if is_beat else 0.15
        med = 0.12
        slow = 0.06

        self._smooth_energy = self._lerp(self._smooth_energy, frame_data.get("global_energy", 0.1), slow)
        self._smooth_percussive = self._lerp(self._smooth_percussive, frame_data.get("percussive_impact", 0.0), fast)
        self._smooth_harmonic = self._lerp(self._smooth_harmonic, frame_data.get("harmonic_energy", 0.2), slow)
        self._smooth_low = self._lerp(self._smooth_low, frame_data.get("low_energy", 0.1), slow)
        self._smooth_high = self._lerp(self._smooth_high, frame_data.get("high_energy", 0.1), slow)
        
        self._smooth_flux = self._lerp(self._smooth_flux, frame_data.get("spectral_flux", 0.0), fast)
        self._smooth_flatness = self._lerp(self._smooth_flatness, frame_data.get("spectral_flatness", 0.0), med)
        self._smooth_sharpness = self._lerp(self._smooth_sharpness, frame_data.get("sharpness", 0.0), med)
        self._smooth_sub_bass = self._lerp(self._smooth_sub_bass, frame_data.get("sub_bass", 0.0), fast)
        self._smooth_brilliance = self._lerp(self._smooth_brilliance, frame_data.get("brilliance", 0.0), fast)
        self._smooth_centroid = self._lerp(self._smooth_centroid, frame_data.get("spectral_centroid", 0.5), slow)

    @abc.abstractmethod
    def update(self, frame_data: Dict[str, Any]):
        """Advance the simulation state."""
        pass

    @abc.abstractmethod
    def get_raw_field(self) -> np.ndarray:
        """
        Returns the raw float32 energy field (0.0 - 1.0).
        Some visualizers may return a tuple of fields.
        """
        pass

    def render_frame(self, frame_data: Dict[str, Any], frame_index: int) -> np.ndarray:
        """Default implementation that combines update, field generation, and polishing."""
        self.update(frame_data)
        field = self.get_raw_field()
        
        polisher = VisualPolisher(self.cfg)
        return polisher.apply(field, frame_data, self.time, self._smooth_audio_dict())


    def _smooth_audio_dict(self) -> Dict[str, float]:
        """Returns a dict of current smoothed audio values."""
        return {
            "energy": self._smooth_energy,
            "percussive": self._smooth_percussive,
            "harmonic": self._smooth_harmonic,
            "low": self._smooth_low,
            "high": self._smooth_high,
            "flux": self._smooth_flux,
            "flatness": self._smooth_flatness,
            "sharpness": self._smooth_sharpness,
            "sub_bass": self._smooth_sub_bass,
            "brilliance": self._smooth_brilliance,
            "centroid": self._smooth_centroid,
        }


class VisualPolisher:
    """
    Universal Styling Engine.
    Turns raw energy fields into finished, audio-reactive frames.
    """

    def __init__(self, config: BaseConfig):
        self.cfg = config

    def apply(
        self, 
        field: Any, 
        frame_data: Dict[str, Any], 
        time: float, 
        smoothed: Dict[str, float]
    ) -> np.ndarray:
        """
        Applies palettes and post-processing.
        """
        # 1. Field Normalization & Merging
        if isinstance(field, tuple):
            energy_map = np.clip(sum(f for f in field), 0, 1)
        else:
            energy_map = field

        # Upscale if needed (low_res_mirror)
        h, w = energy_map.shape
        if w != self.cfg.width or h != self.cfg.height:
            from PIL import Image
            img = Image.fromarray(energy_map, mode="F")
            img = img.resize((self.cfg.width, self.cfg.height), Image.BILINEAR)
            energy_map = np.array(img)

        # 2. Palette Mapping
        hue_base = frame_data.get("pitch_hue", 0.0)
        hue_base = (hue_base + smoothed["centroid"] * 0.2) % 1.0
        
        sat = 0.85 * (0.8 + smoothed["harmonic"] * 0.2 + smoothed["brilliance"] * 0.5)
        sat = np.clip(sat, 0.5, 1.0)

        if self.cfg.palette_type == "solar":
            # Fire/Solar colormap logic
            frame_rgb = self._apply_solar_palette(energy_map, smoothed)
        else:
            # Default Jewel-tone logic
            frame_rgb = apply_palette(
                energy_map,
                hue_base=hue_base,
                time=time,
                saturation=sat,
                contrast=1.5,
            )

        # 3. Post-Processing
        if self.cfg.glow_enabled:
            glow_int = self.cfg.glow_intensity * (1.0 + smoothed["percussive"] * 0.3 + smoothed["flux"] * 0.4)
            glow_int = min(glow_int, 0.8)
            frame_rgb = add_glow(frame_rgb, intensity=glow_int, radius=self.cfg.glow_radius)

        if self.cfg.aberration_enabled:
            ab_offset = int(
                self.cfg.aberration_offset * (1.0 + smoothed["percussive"] * 1.5 + smoothed["sharpness"] * 3.0)
            )
            frame_rgb = chromatic_aberration(frame_rgb, offset=ab_offset)

        if self.cfg.vignette_strength > 0:
            vign_str = self.cfg.vignette_strength * (1.0 + smoothed["low"] * 0.5 + smoothed["sub_bass"] * 1.0)
            frame_rgb = vignette(frame_rgb, strength=vign_str)

        return tone_map_soft(frame_rgb)

    def _apply_solar_palette(self, energy: np.ndarray, smoothed: Dict[str, float]) -> np.ndarray:
        """Custom Solar/Fire mapping."""
        # energy is 0-1
        # 0.0 -> black, 0.5 -> red, 0.8 -> yellow, 1.0 -> white
        h, w = energy.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        
        # Red channel
        rgb[:,:,0] = np.clip(energy * 2.0, 0, 1)
        # Green channel (starts later)
        rgb[:,:,1] = np.clip((energy - 0.4) * 2.0, 0, 1)
        # Blue channel (starts even later)
        rgb[:,:,2] = np.clip((energy - 0.8) * 5.0, 0, 1)
        
        # Add some audio-reactive color shift
        shift = smoothed["brilliance"] * 0.2
        rgb[:,:,1] = np.clip(rgb[:,:,1] + shift * energy, 0, 1)
        
        return (rgb * 255).astype(np.uint8)

