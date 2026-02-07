"""
Signal smoothing and normalization module.

Applies aesthetic processing to raw audio features to prevent
visual flickering and ensure smooth, organic visuals.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from chromascope.core.analyzer import ExtractedFeatures


@dataclass
class EnvelopeParams:
    """Attack/Release envelope parameters in milliseconds."""

    attack_ms: float = 0.0  # Instant attack
    release_ms: float = 500.0  # 500ms decay


@dataclass
class PolishedFeatures:
    """Smoothed and normalized features ready for visualization."""

    # Per-frame boolean triggers
    is_beat: np.ndarray  # Shape: (n_frames,)
    is_onset: np.ndarray  # Shape: (n_frames,)

    # Smoothed continuous signals [0.0, 1.0]
    percussive_impact: np.ndarray
    harmonic_energy: np.ndarray
    global_energy: np.ndarray

    # Frequency bands [0.0, 1.0]
    low_energy: np.ndarray
    mid_energy: np.ndarray
    high_energy: np.ndarray

    # Tonality [0.0, 1.0]
    spectral_brightness: np.ndarray
    chroma: np.ndarray  # Shape: (12, n_frames), normalized

    # Dominant note per frame
    dominant_chroma_indices: np.ndarray

    # Metadata
    n_frames: int
    fps: int
    frame_times: np.ndarray


class SignalPolisher:
    """
    Applies aesthetic smoothing to raw audio features.

    Implements attack/release envelopes and normalization to create
    visually pleasing, flicker-free signals.
    """

    def __init__(
        self,
        fps: int = 60,
        impact_envelope: EnvelopeParams | None = None,
        energy_envelope: EnvelopeParams | None = None,
    ):
        """
        Initialize the polisher.

        Args:
            fps: Target frames per second.
            impact_envelope: Envelope for percussive signals (default: 0ms attack, 200ms release).
            energy_envelope: Envelope for continuous energy signals (default: 50ms attack, 300ms release).
        """
        self.fps = fps
        self.impact_envelope = impact_envelope or EnvelopeParams(
            attack_ms=0.0,
            release_ms=200.0,
        )
        self.energy_envelope = energy_envelope or EnvelopeParams(
            attack_ms=50.0,
            release_ms=300.0,
        )

    def _ms_to_frames(self, ms: float) -> int:
        """Convert milliseconds to number of frames at current FPS."""
        return max(1, int((ms / 1000.0) * self.fps))

    def normalize(self, signal: np.ndarray, floor: float = 0.001) -> np.ndarray:
        """
        Normalize signal to [0.0, 1.0] range.

        Args:
            signal: Input signal.
            floor: Minimum value to prevent division by zero.

        Returns:
            Normalized signal in [0.0, 1.0].
        """
        min_val = np.min(signal)
        max_val = np.max(signal)
        range_val = max_val - min_val

        if range_val < floor:
            return np.zeros_like(signal)

        normalized = (signal - min_val) / range_val
        return np.clip(normalized, 0.0, 1.0)

    def apply_envelope(
        self,
        signal: np.ndarray,
        params: EnvelopeParams,
    ) -> np.ndarray:
        """
        Apply attack/release envelope to a signal.

        Creates smooth "glow" effects where values jump up quickly
        but fade down slowly.

        Args:
            signal: Input signal (should be normalized first).
            params: Envelope attack/release parameters.

        Returns:
            Envelope-smoothed signal.
        """
        attack_frames = self._ms_to_frames(params.attack_ms)
        release_frames = self._ms_to_frames(params.release_ms)

        output = np.zeros_like(signal)
        current = 0.0

        for i, target in enumerate(signal):
            if target > current:
                # Attack phase - rise towards target
                if attack_frames <= 1:
                    current = target
                else:
                    attack_rate = 1.0 / attack_frames
                    current = current + (target - current) * attack_rate
            else:
                # Release phase - decay towards target
                if release_frames <= 1:
                    current = target
                else:
                    release_rate = 1.0 / release_frames
                    current = current - (current - target) * release_rate

            output[i] = current

        return np.clip(output, 0.0, 1.0)

    def create_beat_array(
        self,
        n_frames: int,
        beat_frames: np.ndarray,
    ) -> np.ndarray:
        """
        Create boolean beat trigger array aligned to frame indices.

        Args:
            n_frames: Total number of output frames.
            beat_frames: Frame indices where beats occur.

        Returns:
            Boolean array with True at beat positions.
        """
        is_beat = np.zeros(n_frames, dtype=bool)
        valid_beats = beat_frames[beat_frames < n_frames]
        is_beat[valid_beats.astype(int)] = True
        return is_beat

    def create_onset_array(
        self,
        n_frames: int,
        onset_frames: np.ndarray,
    ) -> np.ndarray:
        """
        Create boolean onset trigger array.

        Args:
            n_frames: Total number of output frames.
            onset_frames: Frame indices where onsets occur.

        Returns:
            Boolean array with True at onset positions.
        """
        is_onset = np.zeros(n_frames, dtype=bool)
        valid_onsets = onset_frames[onset_frames < n_frames]
        is_onset[valid_onsets.astype(int)] = True
        return is_onset

    def smooth_spectral_centroid(
        self,
        centroid: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """
        Normalize spectral centroid to [0.0, 1.0] as "brightness".

        Maps typical speech/music range (100Hz - 8000Hz) to 0-1.
        """
        # Typical range for music
        min_hz = 100.0
        max_hz = 8000.0

        brightness = (centroid - min_hz) / (max_hz - min_hz)
        brightness = np.clip(brightness, 0.0, 1.0)

        # Apply light smoothing
        return self.apply_envelope(brightness, self.energy_envelope)

    def polish(self, features: ExtractedFeatures) -> PolishedFeatures:
        """
        Apply full aesthetic processing to extracted features.

        Args:
            features: Raw features from FeatureAnalyzer.

        Returns:
            PolishedFeatures ready for visualization.
        """
        n_frames = features.n_frames

        # Boolean triggers
        is_beat = self.create_beat_array(n_frames, features.temporal.beat_frames)
        is_onset = self.create_onset_array(n_frames, features.temporal.onset_frames)

        # Energy signals with envelope smoothing
        percussive_impact = self.apply_envelope(
            self.normalize(features.energy.rms_percussive),
            self.impact_envelope,
        )

        harmonic_energy = self.apply_envelope(
            self.normalize(features.energy.rms_harmonic),
            self.energy_envelope,
        )

        global_energy = self.apply_envelope(
            self.normalize(features.energy.rms),
            self.energy_envelope,
        )

        # Frequency bands
        low_energy = self.apply_envelope(
            self.normalize(features.energy.frequency_bands.low),
            self.energy_envelope,
        )
        mid_energy = self.apply_envelope(
            self.normalize(features.energy.frequency_bands.mid),
            self.energy_envelope,
        )
        high_energy = self.apply_envelope(
            self.normalize(features.energy.frequency_bands.high),
            self.energy_envelope,
        )

        # Spectral brightness
        spectral_brightness = self.smooth_spectral_centroid(
            features.tonality.spectral_centroid,
            features.sample_rate,
        )

        # Normalize chroma (each bin independently)
        chroma_normalized = np.zeros_like(features.tonality.chroma)
        for i in range(12):
            chroma_normalized[i] = self.normalize(features.tonality.chroma[i])

        return PolishedFeatures(
            is_beat=is_beat,
            is_onset=is_onset,
            percussive_impact=percussive_impact,
            harmonic_energy=harmonic_energy,
            global_energy=global_energy,
            low_energy=low_energy,
            mid_energy=mid_energy,
            high_energy=high_energy,
            spectral_brightness=spectral_brightness,
            chroma=chroma_normalized,
            dominant_chroma_indices=features.tonality.dominant_chroma_indices,
            n_frames=n_frames,
            fps=self.fps,
            frame_times=features.frame_times,
        )
