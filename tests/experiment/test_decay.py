import numpy as np
import pytest
from chromascope.experiment.decay import DecayRenderer, DecayConfig, MirrorRenderer

def test_decay_renderer_init():
    config = DecayConfig(width=640, height=480, fps=30)
    renderer = DecayRenderer(config)
    assert renderer.cfg.width == 640
    assert renderer.track_buffer.shape == (480, 640)
    assert renderer.vapor_buffer.shape == (480, 640)
    assert len(renderer.particles) == 0

def test_decay_renderer_spawn():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    renderer.spawn_particle("alpha")
    assert len(renderer.particles) == 1
    assert renderer.particles[0].type == "alpha"
    # Alpha should have drag
    assert renderer.particles[0].drag < 1.0
    
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
        "spectral_flux": 0.5,
        "harmonic_energy": 0.4,
        "sub_bass": 0.3
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
    track = np.ones((10, 10), dtype=np.float32)
    vapor = np.ones((10, 10), dtype=np.float32)
    rgb = renderer._apply_styles(track, vapor)
    assert rgb.shape == (10, 10, 3)
    # Uranium should be greenish: G should be strong
    assert rgb[0, 0, 1] > 200

    renderer.cfg.style = "lab"
    rgb_lab = renderer._apply_styles(track, vapor)
    assert np.all(rgb_lab[0, 0, 0] == rgb_lab[0, 0, 1] == rgb_lab[0, 0, 2])

def test_particle_drag():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    # Alpha particle with high initial velocity
    renderer.spawn_particle("alpha", x=50, y=50, vx=10, vy=0)
    initial_vx = renderer.particles[0].vx
    
    # Update
    renderer.update_particles(1.0/60.0)
    
    # vx should have decreased due to drag
    assert renderer.particles[0].vx < initial_vx

def test_mirror_renderer():
    config = DecayConfig(width=100, height=100)
    renderer = MirrorRenderer(config, split_mode="vertical", interference_mode="resonance")
    
    frame_data = {
        "global_energy": 0.5,
        "percussive_impact": 0.8,
        "is_beat": True,
        "spectral_flux": 0.5,
        "harmonic_energy": 0.4,
        "sub_bass": 0.3
    }
    
    frame = renderer.render_frame(frame_data, 0)
    assert frame.shape == (100, 100, 3)
    # Both instances should have particles
    assert len(renderer.instance_a.particles) > 0
    assert len(renderer.instance_b.particles) > 0
