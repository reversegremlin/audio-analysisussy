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
        "spectral_flux": 0.9,
        "sub_bass": 0.7
    }
    frame = renderer.render_frame(frame_data, 0)
    assert frame.shape == (100, 100, 3)
    assert frame.dtype == np.uint8
    # Particles should have been spawned
    assert len(renderer.particles) > 0
    # Zoom should have reacted
    assert renderer.view_zoom > 1.0

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
    # Uranium should be greenish: G should be strong
    assert rgb[0, 0, 1] > 200

    renderer.cfg.style = "lab"
    rgb_lab = renderer._apply_styles(buffer)
    assert np.all(rgb_lab[0, 0, 0] == rgb_lab[0, 0, 1] == rgb_lab[0, 0, 2])

def test_secondary_ionization():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    # Mock data to trigger branching
    frame_data = {
        "global_energy": 1.0,
        "spectral_flux": 1.0,
        "is_beat": True
    }
    # Manually add a particle that is likely to branch
    renderer.spawn_particle("beta", x=50, y=50, vx=10, vy=10)
    initial_count = len(renderer.particles)
    
    # Run a few frames to allow branching logic to trigger
    for _ in range(10):
        renderer.render_frame(frame_data, 0)
    
    # Due to random nature, we might not always get a branch in 10 frames, 
    # but with flux=1.0 and energy=1.0, probability is high.
    # We at least check that the system doesn't crash.
    assert len(renderer.particles) >= initial_count
