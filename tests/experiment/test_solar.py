"""
Tests for the solar renderer.
"""
import numpy as np

from chromascope.experiment.solar import SolarRenderer, SolarConfig

def test_solar_renderer_runs():
    """
    Test that the solar renderer runs for a few frames without errors.
    """
    config = SolarConfig(width=100, height=100, fps=60, pan_speed_x=0.1, pan_speed_y=0.05, zoom_speed=0.05)
    renderer = SolarRenderer(config)
    
    manifest = {
        "frames": [
            {"global_energy": 0.5, "low_energy": 0.5, "harmonic_energy": 0.1, "percussive_impact": 0.0, "high_energy": 0.0, "is_beat": False, "is_onset": False, "spectral_brightness": 0.0},
            {"global_energy": 0.6, "low_energy": 0.4, "harmonic_energy": 0.2, "percussive_impact": 0.1, "high_energy": 0.1, "is_beat": True, "is_onset": False, "spectral_brightness": 0.1},
            {"global_energy": 0.7, "low_energy": 0.3, "harmonic_energy": 0.3, "percussive_impact": 0.2, "high_energy": 0.2, "is_beat": False, "is_onset": True, "spectral_brightness": 0.2},
        ]
    }
    frames = list(renderer.render_manifest(manifest))
    
    assert len(frames) == 3
    for frame in frames:
        assert frame.shape == (100, 100, 3)
        # Check that the frame is not all black
        assert np.sum(frame) > 0
