"""Tests for the KaleidoscopeRenderer module."""

import numpy as np
import pytest

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYGAME_AVAILABLE, reason="pygame not installed")


class TestKaleidoscopeRenderer:
    """Tests for kaleidoscope visualization."""

    @pytest.fixture
    def renderer(self):
        """Create a renderer instance."""
        pygame.init()
        from chromascope.visualizers.kaleidoscope import (
            KaleidoscopeConfig,
            KaleidoscopeRenderer,
        )
        config = KaleidoscopeConfig(width=640, height=480, fps=30)
        return KaleidoscopeRenderer(config)

    @pytest.fixture
    def sample_frame(self):
        """Sample frame data for testing."""
        return {
            "frame_index": 0,
            "time": 0.0,
            "is_beat": True,
            "percussive_impact": 0.8,
            "harmonic_energy": 0.5,
            "spectral_brightness": 0.6,
            "dominant_chroma": "G",
        }

    def test_renderer_initialization(self, renderer):
        """Renderer should initialize with correct config."""
        assert renderer.config.width == 640
        assert renderer.config.height == 480
        assert renderer.config.fps == 30
        assert renderer.accumulated_rotation == 0.0

    def test_particle_initialization(self, renderer):
        """Particles should be initialized."""
        assert len(renderer.particles) == 80
        for particle in renderer.particles:
            assert 'x' in particle
            assert 'y' in particle
            assert 'size' in particle
            assert 'brightness' in particle

    def test_render_frame_returns_surface(self, renderer, sample_frame):
        """render_frame should return a pygame Surface."""
        surface = renderer.render_frame(sample_frame)
        assert isinstance(surface, pygame.Surface)
        assert surface.get_width() == 640
        assert surface.get_height() == 480

    def test_render_frame_with_trail(self, renderer, sample_frame):
        """render_frame should work with previous frame for trails."""
        first_surface = renderer.render_frame(sample_frame)
        second_surface = renderer.render_frame(sample_frame, first_surface)
        assert isinstance(second_surface, pygame.Surface)

    def test_accumulated_rotation_increases(self, renderer, sample_frame):
        """Rotation should accumulate over frames."""
        initial_rotation = renderer.accumulated_rotation
        renderer.render_frame(sample_frame)
        # Rotation increases based on harmonic energy
        assert renderer.accumulated_rotation != initial_rotation

    def test_smoothed_values_update(self, renderer, sample_frame):
        """Smoothed values should update towards frame data."""
        initial_percussive = renderer.smoothed_percussive
        renderer.render_frame(sample_frame)
        # Should move towards the frame's percussive_impact (0.8)
        assert renderer.smoothed_percussive > initial_percussive

    def test_pulse_intensity_on_beat(self, renderer, sample_frame):
        """Pulse intensity should increase on beats."""
        # Frame with beat
        renderer.render_frame(sample_frame)
        beat_pulse = renderer.pulse_intensity

        # Frame without beat
        no_beat_frame = sample_frame.copy()
        no_beat_frame["is_beat"] = False
        for _ in range(10):  # Multiple frames to decay
            renderer.render_frame(no_beat_frame)

        assert renderer.pulse_intensity < beat_pulse

    def test_dynamic_background_disabled(self, renderer, sample_frame):
        """Should work with dynamic background disabled."""
        renderer.config.dynamic_background = False
        surface = renderer.render_frame(sample_frame)
        assert isinstance(surface, pygame.Surface)

    def test_particles_disabled(self, renderer, sample_frame):
        """Should work with particles disabled."""
        renderer.config.bg_particles = False
        surface = renderer.render_frame(sample_frame)
        assert isinstance(surface, pygame.Surface)

    def test_pulse_disabled(self, renderer, sample_frame):
        """Should work with pulse rings disabled."""
        renderer.config.bg_pulse = False
        surface = renderer.render_frame(sample_frame)
        assert isinstance(surface, pygame.Surface)

    def test_surface_to_array(self, renderer, sample_frame):
        """surface_to_array should convert to correct numpy shape."""
        surface = renderer.render_frame(sample_frame)
        arr = renderer.surface_to_array(surface)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (480, 640, 3)  # (height, width, RGB)

    def test_chroma_to_hue_mapping(self, renderer):
        """Chroma names should map to hue values."""
        assert renderer.CHROMA_TO_HUE[0] == 0.0  # C = Red
        assert renderer.CHROMA_TO_HUE[7] == 0.583  # G = Light blue

    def test_different_chroma_colors(self, renderer):
        """Different chroma values should produce different frames."""
        frame_c = {"is_beat": False, "percussive_impact": 0.5,
                   "harmonic_energy": 0.5, "spectral_brightness": 0.5,
                   "dominant_chroma": "C"}
        frame_g = {"is_beat": False, "percussive_impact": 0.5,
                   "harmonic_energy": 0.5, "spectral_brightness": 0.5,
                   "dominant_chroma": "G"}

        surface_c = renderer.render_frame(frame_c)
        # Reset state
        renderer.accumulated_rotation = 0
        renderer.smoothed_percussive = 0
        renderer.smoothed_harmonic = 0.3
        surface_g = renderer.render_frame(frame_g)

        # Surfaces should be different due to different hue
        arr_c = renderer.surface_to_array(surface_c)
        arr_g = renderer.surface_to_array(surface_g)
        assert not np.array_equal(arr_c, arr_g)

    def test_render_manifest(self, renderer):
        """render_manifest should process all frames."""
        manifest = {
            "metadata": {"fps": 30, "n_frames": 5},
            "frames": [
                {"frame_index": i, "time": i/30, "is_beat": i % 2 == 0,
                 "percussive_impact": 0.5, "harmonic_energy": 0.5,
                 "spectral_brightness": 0.5, "dominant_chroma": "C"}
                for i in range(5)
            ]
        }

        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))

        surfaces = renderer.render_manifest(manifest, progress_callback)

        assert len(surfaces) == 5
        assert len(progress_calls) == 5
        assert progress_calls[-1] == (5, 5)

    def test_config_background_colors(self):
        """Config should support two background colors."""
        from chromascope.visualizers.kaleidoscope import KaleidoscopeConfig

        config = KaleidoscopeConfig(
            background_color=(10, 10, 20),
            background_color2=(30, 15, 50)
        )
        assert config.background_color == (10, 10, 20)
        assert config.background_color2 == (30, 15, 50)

    def test_config_reactivity(self):
        """Config should support reactivity setting."""
        from chromascope.visualizers.kaleidoscope import KaleidoscopeConfig

        config = KaleidoscopeConfig(bg_reactivity=0.5)
        assert config.bg_reactivity == 0.5
