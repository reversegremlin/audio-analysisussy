"""Tests for the FeatureAnalyzer module."""

import numpy as np
import pytest

from chromascope.core.analyzer import (
    ExtractedFeatures,
    FeatureAnalyzer,
    FrequencyBands,
)
from chromascope.core.decomposer import AudioDecomposer


class TestFeatureAnalyzer:
    """Tests for feature extraction."""

    @pytest.fixture
    def decomposed_mixed(self, mixed_signal):
        """Get decomposed mixed signal for testing."""
        y, sr = mixed_signal
        decomposer = AudioDecomposer()
        return decomposer.separate(y, sr)

    @pytest.fixture
    def decomposed_sine(self, pure_sine):
        """Get decomposed sine wave for testing."""
        y, sr = pure_sine
        decomposer = AudioDecomposer()
        return decomposer.separate(y, sr)

    @pytest.fixture
    def decomposed_clicks(self, click_track):
        """Get decomposed click track for testing."""
        y, sr = click_track
        decomposer = AudioDecomposer()
        return decomposer.separate(y, sr)

    def test_analyze_returns_extracted_features(self, decomposed_mixed):
        """analyze() should return ExtractedFeatures."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_mixed)

        assert isinstance(result, ExtractedFeatures)

    def test_hop_length_for_60fps(self, sample_rate):
        """hop_length should give ~60 frames per second."""
        analyzer = FeatureAnalyzer(target_fps=60)
        hop = analyzer.compute_hop_length(sample_rate)

        # At 22050 Hz and 60 fps, hop should be ~367
        expected = sample_rate // 60
        assert hop == expected

    def test_beat_detection(self, decomposed_clicks):
        """Should detect beats in click track."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_clicks)

        # Should detect some beats
        assert len(result.temporal.beat_frames) > 0
        assert len(result.temporal.beat_times) > 0

    def test_bpm_in_reasonable_range(self, decomposed_mixed):
        """BPM should be in a reasonable range."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_mixed)

        # BPM should be between 30 and 300
        assert 30 <= result.temporal.bpm <= 300

    def test_onset_detection(self, decomposed_clicks):
        """Should detect onsets in percussive signal."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_clicks)

        # Should detect some onsets
        assert len(result.temporal.onset_frames) > 0

    def test_rms_energy(self, decomposed_mixed):
        """RMS energy should be computed for all components."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_mixed)

        assert len(result.energy.rms) > 0
        assert len(result.energy.rms_harmonic) > 0
        assert len(result.energy.rms_percussive) > 0

    def test_frequency_bands(self, decomposed_mixed):
        """Frequency bands should be extracted."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_mixed)

        bands = result.energy.frequency_bands
        assert isinstance(bands, FrequencyBands)
        assert len(bands.low) > 0
        assert len(bands.mid) > 0
        assert len(bands.high) > 0

    def test_chroma_features(self, decomposed_sine):
        """Chroma should detect the dominant pitch."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_sine)

        # Chroma should be 12 x n_frames
        assert result.tonality.chroma.shape[0] == 12
        assert result.tonality.chroma.shape[1] == result.n_frames

    def test_spectral_centroid(self, decomposed_mixed):
        """Spectral centroid should be computed."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_mixed)

        assert len(result.tonality.spectral_centroid) == result.n_frames
        assert np.all(result.tonality.spectral_centroid >= 0)

    def test_dominant_chroma(self, decomposed_sine):
        """Should identify dominant chroma per frame."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_sine)

        indices = result.tonality.dominant_chroma_indices
        assert len(indices) == result.n_frames
        assert np.all(indices >= 0)
        assert np.all(indices < 12)

    def test_chroma_index_to_name(self):
        """Should convert chroma indices to note names."""
        assert FeatureAnalyzer.chroma_index_to_name(0) == "C"
        assert FeatureAnalyzer.chroma_index_to_name(1) == "C#"
        assert FeatureAnalyzer.chroma_index_to_name(9) == "A"
        assert FeatureAnalyzer.chroma_index_to_name(11) == "B"

    def test_frame_times_populated(self, decomposed_mixed):
        """Frame times should be populated."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_mixed)

        assert len(result.frame_times) == result.n_frames
        assert result.frame_times[0] >= 0
        # Times should be increasing
        assert np.all(np.diff(result.frame_times) > 0)
