"""Tests for FractalKaleidoscopeRenderer (post OPEN UP refactor)."""

import numpy as np
import pytest

from chromascope.experiment.fractal import FractalKaleidoscopeRenderer, FractalConfig


def _mock_manifest(n_frames: int = 10, fps: int = 30) -> dict:
    """Build a minimal manifest for testing."""
    frames = []
    for i in range(n_frames):
        frames.append({
            "frame_index": i,
            "time": i / fps,
            "is_beat": i % 8 == 0,
            "is_onset": i % 4 == 0,
            "percussive_impact": 0.3 + 0.4 * (i % 8 == 0),
            "harmonic_energy": 0.5,
            "global_energy": 0.4,
            "low_energy": 0.3,
            "mid_energy": 0.4,
            "high_energy": 0.5,
            "spectral_brightness": 0.5,
            "dominant_chroma": "G",
            "pitch_hue": 0.583,
            "texture": 0.5,
            "spectral_flux": 0.3,
            "spectral_flatness": 0.2,
            "sub_bass": 0.2,
            "brilliance": 0.3,
            "sharpness": 0.2,
            "spectral_centroid": 0.5,
        })
    return {
        "metadata": {"bpm": 120, "duration": n_frames / fps, "fps": fps},
        "frames": frames,
    }


class TestFractalKaleidoscopeRenderer:
    def test_single_frame(self):
        config = FractalConfig(
            width=160, height=120, fps=30,
            glow_enabled=False, aberration_enabled=False,
        )
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=1)
        frame = renderer.render_frame(manifest["frames"][0], 0)
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8

    def test_multiple_frames_different(self):
        config = FractalConfig(
            width=80, height=60, fps=30,
            glow_enabled=False, aberration_enabled=False,
        )
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=3)

        frames = list(renderer.render_manifest(manifest))
        assert len(frames) == 3
        assert not np.array_equal(frames[0], frames[1])

    def test_generator_yields(self):
        config = FractalConfig(
            width=80, height=60, fps=30,
            glow_enabled=False, aberration_enabled=False,
        )
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=5)

        count = 0
        for frame in renderer.render_manifest(manifest):
            assert frame.shape == (60, 80, 3)
            count += 1
        assert count == 5

    def test_julia_mode(self):
        config = FractalConfig(
            width=80, height=60, fps=30,
            fractal_mode="julia",
            glow_enabled=False, aberration_enabled=False,
        )
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=2)
        frames = list(renderer.render_manifest(manifest))
        assert len(frames) == 2

    def test_mandelbrot_mode(self):
        config = FractalConfig(
            width=80, height=60, fps=30,
            fractal_mode="mandelbrot",
            glow_enabled=False, aberration_enabled=False,
        )
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=2)
        frames = list(renderer.render_manifest(manifest))
        assert len(frames) == 2

    def test_progress_callback(self):
        config = FractalConfig(
            width=80, height=60, fps=30,
            glow_enabled=False, aberration_enabled=False,
        )
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=3)

        progress = []
        for _ in renderer.render_manifest(
            manifest,
            progress_callback=lambda c, t: progress.append((c, t)),
        ):
            pass
        assert len(progress) == 3
        assert progress[-1] == (3, 3)

    def test_sustained_bass_no_washout(self):
        """Regression: sustained heavy bass should not wash out to white."""
        config = FractalConfig(
            width=80, height=60, fps=30,
            glow_enabled=True, aberration_enabled=False,
        )
        renderer = FractalKaleidoscopeRenderer(config)

        n_frames = 120
        frames_data = []
        for i in range(n_frames):
            frames_data.append({
                "frame_index": i, "time": i / 30,
                "is_beat": i % 4 == 0,
                "is_onset": i % 2 == 0,
                "percussive_impact": 0.9,
                "harmonic_energy": 0.7,
                "global_energy": 0.8,
                "low_energy": 0.9,
                "mid_energy": 0.5,
                "high_energy": 0.6,
                "spectral_brightness": 0.7,
                "dominant_chroma": "C",
                "pitch_hue": 0.0,
                "texture": 0.5,
                "spectral_flux": 0.4,
                "spectral_flatness": 0.2,
                "sub_bass": 0.5,
                "brilliance": 0.4,
                "sharpness": 0.3,
                "spectral_centroid": 0.5,
            })
        manifest = {
            "metadata": {"bpm": 140, "duration": n_frames / 30, "fps": 30},
            "frames": frames_data,
        }

        rendered = list(renderer.render_manifest(manifest))
        last_frame = rendered[-1]
        mean_brightness = last_frame.mean()

        assert mean_brightness < 200, (
            f"Brightness washout detected: mean={mean_brightness:.1f}"
        )
        assert last_frame.std() > 10, (
            f"Detail lost: std={last_frame.std():.1f}"
        )

    def test_sustained_beats_preserve_detail(self):
        """After many beat-heavy frames, fractal should still have texture."""
        config = FractalConfig(
            width=80, height=60, fps=30,
            glow_enabled=False, aberration_enabled=False,
            vignette_strength=0.0,
        )
        renderer = FractalKaleidoscopeRenderer(config)

        for i in range(200):
            frame_data = {
                "frame_index": i, "time": i / 30,
                "is_beat": True,
                "is_onset": True,
                "percussive_impact": 1.0,
                "harmonic_energy": 0.5,
                "global_energy": 0.9,
                "low_energy": 0.8,
                "mid_energy": 0.5,
                "high_energy": 0.5,
                "spectral_brightness": 0.5,
                "dominant_chroma": "C",
                "pitch_hue": 0.0,
                "texture": 0.5,
                "spectral_flux": 0.4,
                "spectral_flatness": 0.2,
                "sub_bass": 0.5,
                "brilliance": 0.3,
                "sharpness": 0.2,
                "spectral_centroid": 0.5,
            }
            frame = renderer.render_frame(frame_data, i)

        assert frame.std() > 8, (
            f"Fractal detail lost under sustained beats: std={frame.std():.1f}"
        )
