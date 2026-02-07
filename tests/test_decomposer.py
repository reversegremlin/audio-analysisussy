"""Tests for the AudioDecomposer module."""

import numpy as np
import pytest

from chromascope.core.decomposer import AudioDecomposer, DecomposedAudio


class TestAudioDecomposer:
    """Tests for HPSS decomposition."""

    def test_separate_preserves_length(self, mixed_signal, sample_rate):
        """Separated signals should have same length as original."""
        y, sr = mixed_signal
        decomposer = AudioDecomposer()
        result = decomposer.separate(y, sr)

        assert len(result.harmonic) == len(y)
        assert len(result.percussive) == len(y)
        assert len(result.original) == len(y)

    def test_separate_returns_decomposed_audio(self, mixed_signal, sample_rate):
        """Result should be a DecomposedAudio dataclass."""
        y, sr = mixed_signal
        decomposer = AudioDecomposer()
        result = decomposer.separate(y, sr)

        assert isinstance(result, DecomposedAudio)
        assert result.sample_rate == sr
        assert result.duration > 0

    def test_harmonic_contains_tonal_content(self, pure_sine, sample_rate):
        """Pure sine wave should be mostly in harmonic component."""
        y, sr = pure_sine
        decomposer = AudioDecomposer()
        result = decomposer.separate(y, sr)

        # Harmonic component should have most of the energy
        harmonic_energy = np.sum(result.harmonic ** 2)
        percussive_energy = np.sum(result.percussive ** 2)

        assert harmonic_energy > percussive_energy * 5

    def test_percussive_contains_transients(self, click_track, sample_rate):
        """Click track should have significant percussive energy."""
        y, sr = click_track
        decomposer = AudioDecomposer()
        result = decomposer.separate(y, sr)

        # Percussive component should capture the clicks
        percussive_energy = np.sum(result.percussive ** 2)
        assert percussive_energy > 0

    def test_n_samples_property(self, mixed_signal, sample_rate):
        """n_samples property should return correct length."""
        y, sr = mixed_signal
        decomposer = AudioDecomposer()
        result = decomposer.separate(y, sr)

        assert result.n_samples == len(y)

    def test_custom_margin(self, mixed_signal, sample_rate):
        """Custom HPSS margin should affect separation."""
        y, sr = mixed_signal

        # Default margin
        decomposer_default = AudioDecomposer(margin=(1.0, 1.0))
        result_default = decomposer_default.separate(y, sr)

        # Aggressive margin
        decomposer_aggressive = AudioDecomposer(margin=(3.0, 3.0))
        result_aggressive = decomposer_aggressive.separate(y, sr)

        # Results should differ
        diff = np.abs(result_default.harmonic - result_aggressive.harmonic).sum()
        assert diff > 0

    def test_decompose_file(self, temp_audio_file):
        """decompose_file should load and separate in one step."""
        decomposer = AudioDecomposer()
        result = decomposer.decompose_file(temp_audio_file)

        assert isinstance(result, DecomposedAudio)
        assert result.n_samples > 0
        assert result.duration > 0
