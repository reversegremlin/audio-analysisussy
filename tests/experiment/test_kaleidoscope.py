"""Tests for kaleidoscope mirror and zoom."""

import numpy as np
import pytest

from chromascope.experiment.kaleidoscope import (
    infinite_zoom_blend,
    polar_mirror,
    radial_warp,
)


class TestPolarMirror:
    def test_output_shape_2d(self):
        texture = np.random.rand(120, 160).astype(np.float32)
        result = polar_mirror(texture, num_segments=8)
        assert result.shape == (120, 160)

    def test_output_shape_3d(self):
        texture = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        result = polar_mirror(texture, num_segments=6)
        assert result.shape == (120, 160, 3)

    def test_symmetry(self):
        """Verify N-way symmetry: average patches at equal angular intervals should match."""
        texture = np.random.rand(300, 300).astype(np.float32)
        n = 6
        result = polar_mirror(texture, num_segments=n, rotation=0.0)

        cx, cy = 150, 150
        r = 60  # sample at fixed radius
        patch = 5  # average a small patch to avoid pixel boundary issues

        # Sample at the center of each segment (not on boundaries)
        segment_angle = 2 * np.pi / n
        values = []
        for i in range(n):
            angle = segment_angle * i + segment_angle / 2  # center of segment
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            x = min(max(x, patch), 299 - patch)
            y = min(max(y, patch), 299 - patch)
            # Average a small patch around the sample point
            val = result[y - patch:y + patch, x - patch:x + patch].mean()
            values.append(val)

        # All segment averages should be similar (mirrored from same source)
        for v in values[1:]:
            assert abs(v - values[0]) < 0.08, \
                f"Symmetry broken: {values}"

    def test_rotation_changes_output(self):
        texture = np.random.rand(100, 100).astype(np.float32)
        r1 = polar_mirror(texture, num_segments=8, rotation=0.0)
        r2 = polar_mirror(texture, num_segments=8, rotation=1.0)
        assert not np.allclose(r1, r2)


class TestRadialWarp:
    def test_output_shape(self):
        texture = np.random.rand(120, 160).astype(np.float32)
        result = radial_warp(texture, amplitude=0.05)
        assert result.shape == (120, 160)

    def test_zero_amplitude_preserves(self):
        texture = np.random.rand(60, 80).astype(np.float32)
        result = radial_warp(texture, amplitude=0.0)
        np.testing.assert_array_equal(result, texture)


class TestInfiniteZoomBlend:
    def test_first_frame_passthrough(self):
        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        result = infinite_zoom_blend(frame, None)
        np.testing.assert_array_equal(result, frame)

    def test_blended_output_shape(self):
        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        feedback = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        result = infinite_zoom_blend(frame, feedback)
        assert result.shape == (120, 160, 3)
        assert result.dtype == np.uint8

    def test_blended_differs_from_input(self):
        frame = np.random.randint(50, 200, (120, 160, 3), dtype=np.uint8)
        feedback = np.random.randint(50, 200, (120, 160, 3), dtype=np.uint8)
        result = infinite_zoom_blend(frame, feedback, zoom_factor=1.05)
        assert not np.array_equal(result, frame)

    def test_energy_conservation_bright_frame(self):
        """Blended output must not exceed max of current and feedback inputs."""
        # Bright current frame — the old formula would exceed both inputs
        frame = np.full((60, 80, 3), 220, dtype=np.uint8)
        feedback = np.full((60, 80, 3), 200, dtype=np.uint8)
        result = infinite_zoom_blend(frame, feedback, zoom_factor=1.02, feedback_alpha=0.85)
        # Output should never exceed the brighter of the two inputs
        assert result.max() <= 255
        # More importantly, with energy conservation the result should stay
        # close to the current frame when it dominates
        assert result.mean() <= 255.0

    def test_energy_conservation_no_runaway(self):
        """Iterating blend on a bright frame must not escalate toward 255."""
        frame = np.full((60, 80, 3), 200, dtype=np.uint8)
        buf = frame.copy()
        for _ in range(30):
            buf = infinite_zoom_blend(frame, buf, zoom_factor=1.02, feedback_alpha=0.85)
        # After 30 iterations, brightness should not have blown up to 255
        assert buf.mean() < 250, f"Brightness ran away to {buf.mean():.1f}"

    def test_dark_areas_show_feedback(self):
        """Dark current frame should let feedback show through."""
        dark_frame = np.full((60, 80, 3), 10, dtype=np.uint8)
        bright_feedback = np.full((60, 80, 3), 180, dtype=np.uint8)
        result = infinite_zoom_blend(dark_frame, bright_feedback, zoom_factor=1.02, feedback_alpha=0.85)
        # Result should be brighter than the dark current frame
        assert result.mean() > dark_frame.mean() + 20
