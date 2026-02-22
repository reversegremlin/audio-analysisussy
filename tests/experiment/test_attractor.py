"""Tests for AttractorRenderer and AttractorConfig."""

import numpy as np
import pytest

from chromascope.experiment.attractor import (
    AttractorConfig,
    AttractorRenderer,
    _rk4_lorenz_numpy,
    _rk4_rossler_numpy,
    _splat_glow_numpy,
    rk4_lorenz,
    rk4_rossler,
    splat_glow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(
    energy: float = 0.5,
    percussive: float = 0.3,
    low: float = 0.4,
    high: float = 0.2,
    sub_bass: float = 0.2,
    harmonic: float = 0.3,
    brilliance: float = 0.1,
    flux: float = 0.2,
    flatness: float = 0.4,
    centroid: float = 0.5,
    is_beat: bool = False,
) -> dict:
    return {
        "global_energy": energy,
        "percussive_impact": percussive,
        "low_energy": low,
        "high_energy": high,
        "sub_bass": sub_bass,
        "harmonic_energy": harmonic,
        "brilliance": brilliance,
        "spectral_flux": flux,
        "spectral_flatness": flatness,
        "spectral_centroid": centroid,
        "is_beat": is_beat,
        "pitch_hue": 0.0,
        "sharpness": 0.1,
    }


def _renderer(w: int = 64, h: int = 64, seed: int = 42, **kwargs) -> AttractorRenderer:
    cfg = AttractorConfig(width=w, height=h, fps=30, num_particles=50, **kwargs)
    return AttractorRenderer(cfg, seed=seed)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestAttractorConfigDefaults:
    def test_default_blend_mode(self):
        cfg = AttractorConfig()
        assert cfg.blend_mode == "dual"

    def test_default_palette(self):
        cfg = AttractorConfig()
        assert cfg.attractor_palette == "neon_aurora"

    def test_default_num_particles(self):
        cfg = AttractorConfig()
        assert cfg.num_particles == 3000

    def test_inherits_base(self):
        cfg = AttractorConfig(width=1280, height=720)
        assert cfg.width == 1280
        assert cfg.fps == 60

    def test_custom_values(self):
        cfg = AttractorConfig(
            num_particles=500,
            lorenz_sigma=12.0,
            trail_decay=0.90,
            blend_mode="morph",
            attractor_palette="void_fire",
        )
        assert cfg.num_particles == 500
        assert cfg.lorenz_sigma == 12.0
        assert cfg.trail_decay == 0.90
        assert cfg.blend_mode == "morph"
        assert cfg.attractor_palette == "void_fire"


# ---------------------------------------------------------------------------
# Renderer initialisation
# ---------------------------------------------------------------------------

class TestAttractorRendererInit:
    def test_lorenz_pts_shape(self):
        r = _renderer()
        assert r._lorenz_pts.shape == (r.cfg.num_particles, 3)

    def test_rossler_pts_shape(self):
        r = _renderer()
        assert r._rossler_pts.shape == (r.cfg.num_particles, 3)

    def test_lorenz_pts_dtype(self):
        r = _renderer()
        assert r._lorenz_pts.dtype == np.float64

    def test_rossler_pts_dtype(self):
        r = _renderer()
        assert r._rossler_pts.dtype == np.float64

    def test_accum_shape(self):
        r = _renderer(w=64, h=48)
        assert r._accum.shape == (48, 64, 3)

    def test_accum_dtype(self):
        r = _renderer()
        assert r._accum.dtype == np.float32

    def test_accum_starts_zero(self):
        r = _renderer()
        assert r._accum.sum() == 0.0

    def test_lorenz_pts_finite_after_warmup(self):
        r = _renderer()
        assert np.all(np.isfinite(r._lorenz_pts))

    def test_rossler_pts_finite_after_warmup(self):
        r = _renderer()
        assert np.all(np.isfinite(r._rossler_pts))

    def test_normalization_computed(self):
        r = _renderer()
        center_l, scale_l = r._lorenz_norm
        center_r, scale_r = r._rossler_norm
        assert center_l.shape == (3,)
        assert scale_l > 0
        assert center_r.shape == (3,)
        assert scale_r > 0


# ---------------------------------------------------------------------------
# Single frame rendering
# ---------------------------------------------------------------------------

class TestRenderFrame:
    def test_output_shape(self):
        r = _renderer(w=64, h=64)
        frame = r.render_frame(_frame(), 0)
        assert frame.shape == (64, 64, 3)

    def test_output_dtype(self):
        r = _renderer()
        frame = r.render_frame(_frame(), 0)
        assert frame.dtype == np.uint8

    def test_output_range(self):
        r = _renderer()
        frame = r.render_frame(_frame(), 0)
        assert frame.min() >= 0
        assert frame.max() <= 255

    def test_not_all_black_after_frames(self):
        r = _renderer()
        for i in range(5):
            frame = r.render_frame(_frame(), i)
        assert frame.max() > 0


# ---------------------------------------------------------------------------
# Simulation correctness
# ---------------------------------------------------------------------------

class TestSimulationCorrectness:
    def test_lorenz_no_nan_after_frame(self):
        r = _renderer()
        r.update(_frame())
        assert np.all(np.isfinite(r._lorenz_pts))

    def test_rossler_no_nan_after_frame(self):
        r = _renderer()
        r.update(_frame())
        assert np.all(np.isfinite(r._rossler_pts))

    def test_accum_evolves(self):
        r = _renderer()
        r.render_frame(_frame(), 0)
        accum_1 = r._accum.copy()
        r.render_frame(_frame(), 1)
        accum_2 = r._accum.copy()
        # Two frames should produce different accumulation states
        assert not np.allclose(accum_1, accum_2)

    def test_trails_persist_in_accum(self):
        r = _renderer()
        for i in range(3):
            r.render_frame(_frame(), i)
        assert r._accum.sum() > 0

    def test_multiple_frames_stable(self):
        r = _renderer()
        for i in range(10):
            frame = r.render_frame(_frame(), i)
        assert np.all(np.isfinite(frame.astype(np.float32)))


# ---------------------------------------------------------------------------
# Blend modes
# ---------------------------------------------------------------------------

class TestBlendModes:
    def test_lorenz_only_mode(self):
        r = _renderer(blend_mode="lorenz")
        frame = r.render_frame(_frame(), 0)
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8

    def test_rossler_only_mode(self):
        r = _renderer(blend_mode="rossler")
        frame = r.render_frame(_frame(), 0)
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8

    def test_dual_mode_produces_output(self):
        r = _renderer(blend_mode="dual")
        for i in range(3):
            frame = r.render_frame(_frame(), i)
        assert frame.max() > 0

    def test_morph_mode_produces_output(self):
        r = _renderer(blend_mode="morph")
        for i in range(3):
            frame = r.render_frame(_frame(flatness=0.5), i)
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# Audio reactivity
# ---------------------------------------------------------------------------

class TestAudioReactivity:
    def test_beat_increases_brightness(self):
        r_no_beat = _renderer(seed=1)
        r_beat = _renderer(seed=1)

        # Same initial state, render one frame each
        frame_no_beat = r_no_beat.render_frame(_frame(is_beat=False), 0)
        frame_beat = r_beat.render_frame(_frame(is_beat=True), 0)

        # Beat frame should be brighter (higher mean)
        assert frame_beat.astype(float).mean() >= frame_no_beat.astype(float).mean()

    def test_sub_bass_modulates_sigma(self):
        r = _renderer()
        # High sub_bass should produce higher sigma
        sigma_low, *_ = r._modulate_params()
        r._smooth_sub_bass = 0.9
        sigma_high, *_ = r._modulate_params()
        assert sigma_high > sigma_low

    def test_harmonic_modulates_rossler_a(self):
        r = _renderer()
        r._smooth_harmonic = 0.1
        _, _, _, a_low, _, _ = r._modulate_params()
        r._smooth_harmonic = 0.9
        _, _, _, a_high, _, _ = r._modulate_params()
        assert a_high > a_low

    def test_params_stay_in_safe_range(self):
        r = _renderer()
        # Even with extreme audio values, params should be clamped
        r._smooth_sub_bass = 1.0
        r._smooth_percussive = 1.0
        r._smooth_harmonic = 1.0
        sigma, rho, beta, a, b, c = r._modulate_params()
        assert 2.0 <= sigma <= 25.0
        assert 10.0 <= rho <= 45.0
        assert 0.05 <= a <= 0.45


# ---------------------------------------------------------------------------
# Kernel tests (direct)
# ---------------------------------------------------------------------------

class TestRK4LorenzKernel:
    def test_no_nan(self):
        pts = np.ones((20, 3), dtype=np.float64)
        rk4_lorenz(pts, 10.0, 28.0, 2.6667, 0.01, 10)
        assert np.all(np.isfinite(pts))

    def test_numpy_fallback_no_nan(self):
        pts = np.ones((20, 3), dtype=np.float64)
        _rk4_lorenz_numpy(pts, 10.0, 28.0, 2.6667, 0.01, 10)
        assert np.all(np.isfinite(pts))

    def test_particles_diverge_from_same_start(self):
        # Two particles with different starting positions should diverge
        # Lorenz Lyapunov exponent ~0.9; after 2000 steps × dt=0.01 (20 time
        # units), a 1e-6 perturbation grows by ~e^18 ≈ 6.5e7 — easily detectable.
        pts = np.ones((5, 3), dtype=np.float64)
        pts[0] += 1e-6  # tiny perturbation on one particle
        rk4_lorenz(pts, 10.0, 28.0, 2.6667, 0.01, 2000)
        # After many steps they must differ by more than the initial perturbation
        assert np.abs(pts[0] - pts[1]).max() > 1e-4


class TestRK4RosslerKernel:
    def test_no_nan(self):
        pts = np.ones((20, 3), dtype=np.float64)
        pts[:, 2] = 3.0
        rk4_rossler(pts, 0.2, 0.2, 5.7, 0.01, 10)
        assert np.all(np.isfinite(pts))

    def test_numpy_fallback_no_nan(self):
        pts = np.ones((20, 3), dtype=np.float64)
        pts[:, 2] = 3.0
        _rk4_rossler_numpy(pts, 0.2, 0.2, 5.7, 0.01, 10)
        assert np.all(np.isfinite(pts))


class TestSplatKernel:
    def test_in_bounds_particle_writes_to_accum(self):
        accum = np.zeros((64, 64, 3), dtype=np.float32)
        x = np.array([32.0], dtype=np.float64)
        y = np.array([32.0], dtype=np.float64)
        r = np.array([1.0], dtype=np.float32)
        g = np.array([0.5], dtype=np.float32)
        b = np.array([0.25], dtype=np.float32)
        w = np.array([1.0], dtype=np.float32)
        splat_glow(x, y, r, g, b, w, accum, 64, 64)
        assert np.any(accum > 0)

    def test_out_of_bounds_particle_no_crash(self):
        accum = np.zeros((64, 64, 3), dtype=np.float32)
        # Particles entirely outside the frame
        x = np.array([-10.0, 100.0, 32.0], dtype=np.float64)
        y = np.array([32.0, 32.0, -10.0], dtype=np.float64)
        r = np.ones(3, dtype=np.float32)
        g = np.ones(3, dtype=np.float32)
        b = np.ones(3, dtype=np.float32)
        w = np.ones(3, dtype=np.float32)
        splat_glow(x, y, r, g, b, w, accum, 64, 64)  # should not raise
        assert np.all(np.isfinite(accum))

    def test_numpy_splat_bounds(self):
        accum = np.zeros((64, 64, 3), dtype=np.float32)
        x = np.array([-5.0, 32.0, 70.0], dtype=np.float64)
        y = np.array([32.0, 32.0, 32.0], dtype=np.float64)
        r = np.ones(3, dtype=np.float32)
        g = np.ones(3, dtype=np.float32)
        b = np.ones(3, dtype=np.float32)
        w = np.ones(3, dtype=np.float32)
        _splat_glow_numpy(x, y, r, g, b, w, accum, 64, 64)
        assert np.any(accum > 0)
        assert np.all(np.isfinite(accum))

    def test_bilinear_sums_to_one(self):
        """Bilinear weights for a single particle should sum to particle weight."""
        accum = np.zeros((10, 10, 3), dtype=np.float32)
        x = np.array([4.5], dtype=np.float64)  # exactly between 4 and 5
        y = np.array([4.5], dtype=np.float64)
        r = np.array([1.0], dtype=np.float32)
        g = np.array([0.0], dtype=np.float32)
        b = np.array([0.0], dtype=np.float32)
        w = np.array([1.0], dtype=np.float32)
        _splat_glow_numpy(x, y, r, g, b, w, accum, 10, 10)
        # All red contributions should sum to 1.0 (bilinear weights sum to 1)
        np.testing.assert_allclose(accum[:, :, 0].sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Manifest / batch rendering
# ---------------------------------------------------------------------------

class TestManifestRender:
    def test_manifest_render_yields_frames(self):
        r = _renderer(w=32, h=32)
        frames_data = [_frame() for _ in range(3)]
        manifest = {"frames": frames_data}
        rendered = list(r.render_manifest(manifest))
        assert len(rendered) == 3
        for f in rendered:
            assert f.shape == (32, 32, 3)
            assert f.dtype == np.uint8

    def test_progress_callback_called(self):
        r = _renderer(w=32, h=32)
        manifest = {"frames": [_frame() for _ in range(4)]}
        calls = []
        list(r.render_manifest(manifest, progress_callback=lambda c, t: calls.append((c, t))))
        assert len(calls) == 4
        assert calls[-1] == (4, 4)

    def test_empty_manifest(self):
        r = _renderer()
        rendered = list(r.render_manifest({"frames": []}))
        assert rendered == []
