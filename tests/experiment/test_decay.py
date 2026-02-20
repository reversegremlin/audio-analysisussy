"""Tests for the DecayRenderer (post OPEN UP refactor + Numba SoA)."""

import numpy as np
import pytest

from chromascope.experiment.decay import DecayRenderer, DecayConfig
from chromascope.experiment.renderer import UniversalMirrorCompositor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_data(**overrides):
    base = {
        "global_energy": 0.5,
        "percussive_impact": 0.8,
        "low_energy": 0.6,
        "high_energy": 0.2,
        "is_beat": True,
        "spectral_flux": 0.5,
        "harmonic_energy": 0.4,
        "sub_bass": 0.3,
        "spectral_centroid": 0.5,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_decay_renderer_init():
    config = DecayConfig(width=640, height=480, fps=30)
    renderer = DecayRenderer(config)
    assert renderer.cfg.width == 640
    assert renderer.track_buffer.shape == (480, 640)
    assert renderer.vapor_buffer.shape == (480, 640)
    assert renderer.particle_count == 0


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------

def test_decay_renderer_spawn():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    renderer.spawn_particle("alpha")
    assert renderer.particle_count == 1
    p = renderer.particles[0]
    assert p.type == "alpha"
    assert p.drag < 1.0

    renderer.spawn_particle("beta")
    assert renderer.particle_count == 2
    assert renderer.particles[1].type == "beta"


def test_spawn_all_types():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    for t in ("alpha", "beta", "gamma"):
        renderer.spawn_particle(t)
    types = {p.type for p in renderer.particles}
    assert types == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def test_render_frame():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    frame = renderer.render_frame(_frame_data(), 0)
    assert frame.shape == (100, 100, 3)
    assert frame.dtype == np.uint8
    assert renderer.particle_count > 0


def test_render_manifest():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    manifest = {
        "frames": [
            _frame_data(global_energy=0.1, is_beat=False),
            _frame_data(global_energy=0.9, is_beat=True),
        ]
    }
    frames = list(renderer.render_manifest(manifest))
    assert len(frames) == 2
    assert frames[0].shape == (100, 100, 3)


def test_render_manifest_progress_callback():
    renderer = DecayRenderer(DecayConfig(width=50, height=50))
    manifest = {"frames": [_frame_data() for _ in range(3)]}
    log = []
    list(renderer.render_manifest(manifest, progress_callback=lambda c, t: log.append((c, t))))
    assert log == [(1, 3), (2, 3), (3, 3)]


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def test_particle_drag():
    """Alpha particle velocity should decrease due to drag after an update."""
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    renderer.spawn_particle("alpha", x=50.0, y=50.0, vx=10.0, vy=0.0)

    # Snapshot initial velocity
    initial_vx = renderer.particles[0].vx

    # One physics step (no audio)
    renderer.update(_frame_data(global_energy=0.0, percussive_impact=0.0,
                                harmonic_energy=0.0, spectral_flux=0.0,
                                is_beat=False))

    # Particle may still be alive; check velocity reduced
    alive = renderer.particles
    if alive:
        assert alive[0].vx < initial_vx, (
            f"Drag did not reduce vx: initial={initial_vx}, new={alive[0].vx}"
        )


# ---------------------------------------------------------------------------
# UniversalMirrorCompositor (was MirrorRenderer)
# ---------------------------------------------------------------------------

def test_mirror_compositor_vertical():
    config = DecayConfig(
        width=100, height=100,
        mirror_mode="vertical",
        interference_mode="resonance",
    )
    compositor = UniversalMirrorCompositor(DecayRenderer, config)

    frame = compositor.render_frame(_frame_data(), 0)
    assert frame.shape == (100, 100, 3)
    # Both instances should have been seeded and started generating particles
    assert compositor.instance_a is not None
    assert compositor.instance_b is not None


def test_mirror_compositor_cycle():
    """Cycle mode should trigger a transition after accumulating potential."""
    config = DecayConfig(
        width=100, height=100,
        mirror_mode="cycle",
        interference_mode="cycle",
    )
    compositor = UniversalMirrorCompositor(DecayRenderer, config)

    # energy=1.0, dt=1/60 → potential += 2/60 ≈ 0.033/frame
    # need ~26 frames to reach 0.85; also requires is_beat=True
    for i in range(50):
        compositor.render_frame(
            _frame_data(global_energy=1.0, is_beat=True), i
        )

    # Should be in (or past) a transition
    assert compositor.transition_alpha > 0 or compositor.curr_mirror_idx != 0
