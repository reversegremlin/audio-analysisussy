import numpy as np
import pytest
from chromascope.experiment.decay import DecayRenderer, DecayConfig

def test_decay_renderer_init():
    config = DecayConfig(width=640, height=480, fps=30)
    renderer = DecayRenderer(config)
    assert renderer.cfg.width == 640
    assert renderer.trail_buffer.shape == (480, 640)
    assert len(renderer.particles) == 0

def test_decay_renderer_spawn():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    renderer.spawn_particle("alpha")
    assert len(renderer.particles) == 1
    assert renderer.particles[0].type == "alpha"
    
    renderer.spawn_particle("beta")
    assert len(renderer.particles) == 2
    assert renderer.particles[1].type == "beta"

def test_render_frame():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    frame_data = {
        "global_energy": 0.5,
        "percussive_impact": 0.8,
        "low_energy": 0.6,
        "high_energy": 0.2,
        "is_beat": True,
        "spectral_flux": 0.5
    }
    frame = renderer.render_frame(frame_data, 0)
    assert frame.shape == (100, 100, 3)
    assert frame.dtype == np.uint8
    # Particles should have been spawned
    assert len(renderer.particles) > 0

def test_render_manifest():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    manifest = {
        "frames": [
            {"global_energy": 0.1, "is_beat": False},
            {"global_energy": 0.9, "is_beat": True}
        ]
    }
    frames = list(renderer.render_manifest(manifest))
    assert len(frames) == 2
    assert frames[0].shape == (100, 100, 3)

def test_styles():
    config = DecayConfig(width=10, height=10, style="uranium")
    renderer = DecayRenderer(config)
    buffer = np.ones((10, 10), dtype=np.float32)
    rgb = renderer._apply_styles(buffer)
    assert rgb.shape == (10, 10, 3)
    # Uranium should be greenish: G should be max, R and B lower
    assert rgb[0, 0, 1] == 255
    assert rgb[0, 0, 0] < 255
    assert rgb[0, 0, 2] < 255

    renderer.cfg.style = "lab"
    rgb_lab = renderer._apply_styles(buffer)
    assert np.all(rgb_lab[0, 0, 0] == rgb_lab[0, 0, 1] == rgb_lab[0, 0, 2])
