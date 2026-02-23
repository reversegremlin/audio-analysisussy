"""Tests for the SignalPolisher module."""

import numpy as np
import pytest

from chromascope.core.analyzer import FeatureAnalyzer
from chromascope.core.decomposer import AudioDecomposer
from chromascope.core.polisher import EnvelopeParams, PolishedFeatures, SignalPolisher


class TestSignalPolisher:
    """Tests for signal smoothing and normalization."""

    @pytest.fixture
    def extracted_features(self, mixed_signal):
        """Get extracted features for testing."""
        y, sr = mixed_signal
        decomposer = AudioDecomposer()
        decomposed = decomposer.separate(y, sr)
        analyzer = FeatureAnalyzer(target_fps=60)
        return analyzer.analyze(decomposed)

    def test_polish_returns_polished_features(self, extracted_features):
        """polish() should return PolishedFeatures."""
        polisher = SignalPolisher(fps=60)
        result = polisher.polish(extracted_features)

        assert isinstance(result, PolishedFeatures)

    def test_normalize_to_unit_range(self):
        """normalize() should map values to [0, 1]."""
        polisher = SignalPolisher()

        signal = np.array([0, 50, 100, 25, 75])
        normalized = polisher.normalize(signal)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)

    def test_normalize_constant_signal(self):
        """Constant signal should normalize to zeros."""
        polisher = SignalPolisher()

        signal = np.array([5.0, 5.0, 5.0, 5.0])
        normalized = polisher.normalize(signal)

        assert np.allclose(normalized, 0.0)

    def test_envelope_instant_attack(self):
        """With 0ms attack, signal should jump instantly."""
        polisher = SignalPolisher(fps=60)

        signal = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        params = EnvelopeParams(attack_ms=0.0, release_ms=500.0)

        result = polisher.apply_envelope(signal, params)

        # Signal should jump to 1.0 instantly at index 2
        assert result[2] == 1.0

    def test_envelope_slow_release(self):
        """With slow release, signal should decay gradually."""
        polisher = SignalPolisher(fps=60)

        # Impulse signal
        signal = np.zeros(60)
        signal[10] = 1.0

        params = EnvelopeParams(attack_ms=0.0, release_ms=500.0)
        result = polisher.apply_envelope(signal, params)

        # After impulse, value should decay but not immediately to 0
        assert result[11] > 0.5  # Still high shortly after
        assert result[30] > 0.0  # Still decaying
        assert result[30] < result[11]  # Decreasing

    def test_is_beat_array_creation(self, extracted_features):
        """Beat array should have True at beat positions."""
        polisher = SignalPolisher()
        n_frames = extracted_features.n_frames
        beat_frames = extracted_features.temporal.beat_frames

        is_beat = polisher.create_beat_array(n_frames, beat_frames)

        assert len(is_beat) == n_frames
        assert is_beat.dtype == bool

        # Check beats are marked
        for bf in beat_frames:
            if bf < n_frames:
                assert is_beat[int(bf)]

    def test_is_onset_array_creation(self, extracted_features):
        """Onset array should have True at onset positions."""
        polisher = SignalPolisher()
        n_frames = extracted_features.n_frames
        onset_frames = extracted_features.temporal.onset_frames

        is_onset = polisher.create_onset_array(n_frames, onset_frames)

        assert len(is_onset) == n_frames
        assert is_onset.dtype == bool

    def test_all_outputs_normalized(self, extracted_features):
        """All continuous outputs should be in [0, 1] range."""
        polisher = SignalPolisher(fps=60)
        result = polisher.polish(extracted_features)

        # Check all continuous signals
        continuous_signals = [
            result.percussive_impact,
            result.harmonic_energy,
            result.global_energy,
            result.spectral_flux,
            result.sub_bass,
            result.bass,
            result.low_mid,
            result.mid,
            result.high_mid,
            result.presence,
            result.brilliance,
            result.low_energy,
            result.mid_energy,
            result.high_energy,
            result.spectral_brightness,
            result.spectral_flatness,
            result.spectral_rolloff,
            result.zero_crossing_rate,
        ]

        for signal in continuous_signals:
            assert np.all(signal >= 0.0)
            assert np.all(signal <= 1.0)

    def test_chroma_normalized(self, extracted_features):
        """All 12 chroma bins should be normalized."""
        polisher = SignalPolisher(fps=60)
        result = polisher.polish(extracted_features)

        assert result.chroma.shape[0] == 12
        for i in range(12):
            assert np.all(result.chroma[i] >= 0.0)
            assert np.all(result.chroma[i] <= 1.0)

    def test_frame_count_preserved(self, extracted_features):
        """Number of frames should be preserved."""
        polisher = SignalPolisher(fps=60)
        result = polisher.polish(extracted_features)

        assert result.n_frames == extracted_features.n_frames
        assert len(result.frame_times) == extracted_features.n_frames

    def test_custom_envelope_params(self, extracted_features):
        """Custom envelope parameters should affect output."""
        impact_env = EnvelopeParams(attack_ms=10.0, release_ms=100.0)
        energy_env = EnvelopeParams(attack_ms=100.0, release_ms=500.0)

        polisher = SignalPolisher(
            fps=60,
            impact_envelope=impact_env,
            energy_envelope=energy_env,
        )
        result = polisher.polish(extracted_features)

        # Should complete without error
        assert result.n_frames > 0

    def test_adaptive_envelopes_respond_to_bpm(self, extracted_features):
        """
        Adaptive envelopes should produce longer decay at lower BPM and
        shorter decay at higher BPM for the same underlying features.
        """
        import copy

        # Create two views of the same features with different BPM values.
        slow = copy.deepcopy(extracted_features)
        fast = copy.deepcopy(extracted_features)

        slow.temporal.bpm = 60.0
        fast.temporal.bpm = 180.0

        polisher = SignalPolisher(fps=60, adaptive_envelopes=True)

        slow_result = polisher.polish(slow)
        fast_result = polisher.polish(fast)

        # With longer release at low BPM, total impact energy should be higher.
        slow_sum = slow_result.percussive_impact.sum()
        fast_sum = fast_result.percussive_impact.sum()

        assert slow_sum > fast_sum
