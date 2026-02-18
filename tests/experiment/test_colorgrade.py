"""Tests for color grading and post-processing."""

import numpy as np
import pytest

from chromascope.experiment.colorgrade import (
    add_glow,
    apply_palette,
    chromatic_aberration,
    tone_map_soft,
    vignette,
)


class TestApplyPalette:
    def test_output_shape(self):
        vals = np.random.rand(120, 160).astype(np.float32)
        result = apply_palette(vals)
        assert result.shape == (120, 160, 3)

    def test_output_dtype(self):
        vals = np.random.rand(60, 80).astype(np.float32)
        result = apply_palette(vals)
        assert result.dtype == np.uint8

    def test_output_range(self):
        vals = np.random.rand(60, 80).astype(np.float32)
        result = apply_palette(vals)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_hue_affects_color(self):
        vals = np.full((60, 80), 0.5, dtype=np.float32)
        r1 = apply_palette(vals, hue_base=0.0)
        r2 = apply_palette(vals, hue_base=0.5)
        assert not np.array_equal(r1, r2)

    def test_low_values_produce_dark_colors(self):
        """Low escape values should produce dark but not necessarily black output."""
        vals = np.full((60, 80), 0.1, dtype=np.float32)
        result = apply_palette(vals, saturation=0.85, contrast=1.4)
        # Low values should be dim (dark jewel tones, not white)
        assert result.mean() < 80, (
            f"Low values too bright: mean={result.mean():.1f}"
        )


class TestAddGlow:
    def test_output_shape(self):
        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        result = add_glow(frame, intensity=0.3)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_zero_intensity_passthrough(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        result = add_glow(frame, intensity=0.0)
        np.testing.assert_array_equal(result, frame)

    def test_glow_brightens(self):
        frame = np.full((60, 80, 3), 128, dtype=np.uint8)
        result = add_glow(frame, intensity=0.5, radius=10)
        assert result.mean() >= frame.mean()


class TestChromaticAberration:
    def test_output_shape(self):
        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        result = chromatic_aberration(frame, offset=3)
        assert result.shape == frame.shape

    def test_zero_offset_passthrough(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        result = chromatic_aberration(frame, offset=0)
        np.testing.assert_array_equal(result, frame)

    def test_channels_shifted(self):
        frame = np.random.randint(50, 200, (60, 80, 3), dtype=np.uint8)
        result = chromatic_aberration(frame, offset=5)
        # Green channel should be unchanged
        np.testing.assert_array_equal(result[:, :, 1], frame[:, :, 1])
        # R and B channels should differ
        assert not np.array_equal(result[:, :, 0], frame[:, :, 0])


class TestVignette:
    def test_output_shape(self):
        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        result = vignette(frame, strength=0.4)
        assert result.shape == frame.shape

    def test_center_brighter_than_edges(self):
        frame = np.full((200, 200, 3), 200, dtype=np.uint8)
        result = vignette(frame, strength=0.5)
        center_val = result[100, 100].mean()
        corner_val = result[0, 0].mean()
        assert center_val > corner_val

    def test_zero_strength_passthrough(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        result = vignette(frame, strength=0.0)
        np.testing.assert_array_equal(result, frame)


class TestToneMapSoft:
    def test_output_shape_and_dtype(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        result = tone_map_soft(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_below_shoulder_unchanged(self):
        """Pixels below shoulder threshold should pass through untouched."""
        frame = np.full((60, 80, 3), 100, dtype=np.uint8)  # well below 0.85*255=216
        result = tone_map_soft(frame, shoulder=0.85)
        np.testing.assert_array_equal(result, frame)

    def test_above_shoulder_compressed(self):
        """Pixels above shoulder should be compressed downward."""
        frame = np.full((60, 80, 3), 250, dtype=np.uint8)
        result = tone_map_soft(frame, shoulder=0.85)
        # Should be less than original 250
        assert result.mean() < 250
        # But still relatively bright (not crushed)
        assert result.mean() > 220

    def test_pure_white_compressed(self):
        """Pure 255 should be compressed below 255."""
        frame = np.full((60, 80, 3), 255, dtype=np.uint8)
        result = tone_map_soft(frame, shoulder=0.85)
        assert result.max() < 255

    def test_monotonicity(self):
        """Brighter input should still produce brighter output."""
        low = np.full((10, 10, 3), 220, dtype=np.uint8)
        high = np.full((10, 10, 3), 250, dtype=np.uint8)
        r_low = tone_map_soft(low, shoulder=0.85)
        r_high = tone_map_soft(high, shoulder=0.85)
        assert r_high.mean() > r_low.mean()

    def test_shoulder_zero_compresses_everything(self):
        """With shoulder=0, all non-zero pixels are compressed."""
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = tone_map_soft(frame, shoulder=0.0)
        # Should be less than original since everything is above shoulder
        assert result.mean() < 128

    def test_default_shoulder_catches_highlights(self):
        """Default shoulder (0.78) should compress pixels above ~199."""
        frame = np.full((10, 10, 3), 220, dtype=np.uint8)
        result = tone_map_soft(frame)  # default shoulder=0.78
        # 220 is above 0.78*255=198.9, so it should be compressed
        assert result.mean() < 220
