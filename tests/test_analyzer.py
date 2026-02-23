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

    def test_default_target_fps_if_none(self, sample_rate):
        """FeatureAnalyzer should handle target_fps=None by using default 60."""
        analyzer = FeatureAnalyzer(target_fps=None)
        assert analyzer.target_fps == 60
        hop = analyzer.compute_hop_length(sample_rate)
        assert hop == sample_rate // 60

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
        assert len(result.energy.spectral_flux) > 0

    def test_frequency_bands(self, decomposed_mixed):
        """Frequency bands should be extracted."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_mixed)

        bands = result.energy.frequency_bands
        assert isinstance(bands, FrequencyBands)
        
        # New 7-band system
        assert len(bands.sub_bass) > 0
        assert len(bands.bass) > 0
        assert len(bands.low_mid) > 0
        assert len(bands.mid) > 0
        assert len(bands.high_mid) > 0
        assert len(bands.presence) > 0
        assert len(bands.brilliance) > 0
        
        # Legacy bands
        assert len(bands.low) > 0
        assert len(bands.mid_aggregate) > 0
        assert len(bands.high) > 0

    def test_chroma_features(self, decomposed_sine):
        """Chroma should detect the dominant pitch."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_sine)

        # Chroma should be 12 x n_frames
        assert result.tonality.chroma.shape[0] == 12
        assert result.tonality.chroma.shape[1] == result.n_frames

    def test_spectral_features(self, decomposed_mixed):
        """Spectral characteristics should be computed."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_mixed)

        assert len(result.tonality.spectral_centroid) == result.n_frames
        assert len(result.tonality.spectral_flatness) == result.n_frames
        assert len(result.tonality.spectral_rolloff) == result.n_frames
        assert len(result.tonality.zero_crossing_rate) == result.n_frames
        
        assert np.all(result.tonality.spectral_centroid >= 0)
        assert np.all(result.tonality.spectral_flatness >= 0)
        assert np.all(result.tonality.spectral_rolloff >= 0)
        assert np.all(result.tonality.zero_crossing_rate >= 0)

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

    def test_frame_time_spacing_matches_fps(self, decomposed_mixed):
        """Average frame spacing in seconds should match target FPS."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_mixed)

        diffs = np.diff(result.frame_times)
        mean_step = np.mean(diffs)
        target_step = 1.0 / analyzer.target_fps

        # Allow small numerical tolerance, but enforce tight alignment.
        assert np.isclose(mean_step, target_step, rtol=0.05)

    def test_tempo_curve_follows_click_track(self, decomposed_clicks):
        """
        Tempo curve should reflect local tempo and agree with global BPM
        for a simple click track.
        """
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_clicks)

        tempo_curve = result.temporal.tempo_curve_bpm

        # Should produce a non-empty, positive tempo curve
        assert tempo_curve.size > 0
        assert np.all(tempo_curve > 0)

        # Median of tempo curve should be close to global BPM
        median_tempo = np.median(tempo_curve)
        assert np.isclose(median_tempo, result.temporal.bpm, atol=10.0)

    def test_mfcc_timbre_features_present(self, decomposed_mixed):
        """MFCC-based timbre features should be extracted alongside tonality."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_mixed)

        mfcc = result.tonality.mfcc

        # Expect a compact MFCC representation aligned with frame count
        assert mfcc.ndim == 2
        assert mfcc.shape[0] == 13  # default n_mfcc
        assert mfcc.shape[1] == result.n_frames
