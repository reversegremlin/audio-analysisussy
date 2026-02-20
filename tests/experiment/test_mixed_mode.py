"""Tests for Mixed Mode rendering and configuration."""

import numpy as np
import pytest
from chromascope.experiment.cli import MixedConfig
from chromascope.experiment.solar import SolarRenderer
from chromascope.experiment.decay import DecayRenderer
from chromascope.experiment.renderer import UniversalMirrorCompositor

def _mock_frame_data(i: int = 0) -> dict:
    """Build a minimal frame data for testing."""
    return {
        "frame_index": i,
        "time": i / 30.0,
        "is_beat": i % 8 == 0,
        "percussive_impact": 0.5,
        "harmonic_energy": 0.5,
        "global_energy": 0.5,
        "low_energy": 0.5,
        "high_energy": 0.5,
        "spectral_flux": 0.5,
        "spectral_flatness": 0.5,
        "sharpness": 0.5,
        "brilliance": 0.5,
        "sub_bass": 0.5,
        "spectral_centroid": 0.5,
        "pitch_hue": 0.5,
    }

def test_mixed_config_attributes():
    """Verify MixedConfig has all required attributes from its parents."""
    config = MixedConfig()
    
    # BaseConfig attributes
    assert hasattr(config, "width")
    assert hasattr(config, "mirror_mode")
    
    # SolarConfig attributes
    assert hasattr(config, "pan_speed_x")
    assert hasattr(config, "zoom_speed")
    
    # DecayConfig attributes
    assert hasattr(config, "base_cpm")
    assert hasattr(config, "max_particles")
    
    # FractalConfig attributes
    assert hasattr(config, "num_segments")
    assert hasattr(config, "fractal_mode")

def test_mixed_mode_rendering():
    """Verify that UniversalMirrorCompositor can render with Solar + Decay using MixedConfig."""
    config = MixedConfig(
        width=160,
        height=120,
        fps=30,
        mirror_mode="vertical",
        interference_mode="resonance"
    )
    
    renderer = UniversalMirrorCompositor(
        SolarRenderer, 
        config, 
        viz_class_b=DecayRenderer
    )
    
    frame_data = _mock_frame_data(0)
    frame = renderer.render_frame(frame_data, 0)
    
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8
    # Should not be all black
    assert np.any(frame > 0)

def test_mixed_mode_cycling():
    """Verify that mixed mode works even when cycling mirror/interference."""
    config = MixedConfig(
        width=80,
        height=60,
        fps=30,
        mirror_mode="cycle",
        interference_mode="cycle"
    )
    
    renderer = UniversalMirrorCompositor(
        SolarRenderer, 
        config, 
        viz_class_b=DecayRenderer
    )
    
    # Render a few frames to trigger potential transitions or at least exercise the logic
    for i in range(5):
        frame_data = _mock_frame_data(i)
        # Force a beat and high energy to trigger cycling logic
        frame_data["is_beat"] = True
        frame_data["sub_bass"] = 1.0
        frame_data["global_energy"] = 1.0
        
        frame = renderer.render_frame(frame_data, i)
        assert frame.shape == (60, 80, 3)
