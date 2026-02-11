"""
Feature extraction module for audio analysis.

Extracts visual drivers: beats, onsets, RMS energy,
frequency bands, chroma features, and spectral centroid.
"""

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy import signal as scipy_signal

from chromascope.core.decomposer import DecomposedAudio


@dataclass
class FrequencyBands:
    """Energy levels for frequency sub-bands."""

    low: np.ndarray  # 0-200Hz
    mid: np.ndarray  # 200Hz-4kHz
    high: np.ndarray  # 4kHz+


@dataclass
class TemporalFeatures:
    """Beat and onset timing features."""

    bpm: float
    beat_frames: np.ndarray
    beat_times: np.ndarray
    onset_frames: np.ndarray
    onset_times: np.ndarray
    # Local tempo estimates derived from beat spacing
    tempo_curve_bpm: np.ndarray | None = None
    tempo_curve_times: np.ndarray | None = None


@dataclass
class EnergyFeatures:
    """Energy-related features."""

    rms: np.ndarray
    rms_harmonic: np.ndarray
    rms_percussive: np.ndarray
    frequency_bands: FrequencyBands


@dataclass
class TonalityFeatures:
    """Pitch and timbre features."""

    chroma: np.ndarray  # Shape: (12, n_frames)
    spectral_centroid: np.ndarray
    dominant_chroma_indices: np.ndarray
    # Compact timbre representation (MFCCs)
    mfcc: np.ndarray | None = None


@dataclass
class ExtractedFeatures:
    """Complete feature set from audio analysis."""

    temporal: TemporalFeatures
    energy: EnergyFeatures
    tonality: TonalityFeatures
    n_frames: int
    hop_length: int
    sample_rate: int
    frame_times: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if len(self.frame_times) == 0:
            self.frame_times = librosa.frames_to_time(
                np.arange(self.n_frames),
                sr=self.sample_rate,
                hop_length=self.hop_length,
            )


class FeatureAnalyzer:
    """
    Extracts visual driver features from decomposed audio.

    Features are aligned to a consistent frame rate determined by hop_length.
    """

    CHROMA_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(
        self,
        target_fps: int = 60,
        n_fft: int = 2048,
    ):
        """
        Initialize the analyzer.

        Args:
            target_fps: Target frames per second for output alignment.
            n_fft: FFT window size.
        """
        self.target_fps = target_fps
        self.n_fft = n_fft

    def compute_hop_length(self, sr: int) -> int:
        """
        Calculate hop_length to achieve target FPS.

        Args:
            sr: Sample rate.

        Returns:
            Hop length in samples.
        """
        return int(sr / self.target_fps)

    def extract_temporal(
        self,
        decomposed: DecomposedAudio,
        hop_length: int,
    ) -> TemporalFeatures:
        """
        Extract beat and onset features.

        Beat tracking uses the full signal while onset detection
        is enhanced by focusing on the percussive component.
        """
        y = decomposed.original
        sr = decomposed.sample_rate

        # Global tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )
        # Handle both scalar tempo and array tempo (librosa version differences)
        if isinstance(tempo, np.ndarray):
            bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            bpm = float(tempo)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

        # Onset detection - percussive component gives cleaner transients
        onset_env = librosa.onset.onset_strength(
            y=decomposed.percussive,
            sr=sr,
            hop_length=hop_length,
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

        # Derive a simple tempo curve from beat-to-beat intervals.
        if len(beat_times) >= 2:
            intervals = np.diff(beat_times)
            # Guard against extremely small or zero intervals.
            intervals = np.clip(intervals, 1e-3, None)
            tempo_curve_bpm = 60.0 / intervals
            tempo_curve_times = beat_times[:-1] + intervals / 2.0
        else:
            tempo_curve_bpm = np.array([], dtype=float)
            tempo_curve_times = np.array([], dtype=float)

        return TemporalFeatures(
            bpm=bpm,
            beat_frames=beat_frames,
            beat_times=beat_times,
            onset_frames=onset_frames,
            onset_times=onset_times,
            tempo_curve_bpm=tempo_curve_bpm,
            tempo_curve_times=tempo_curve_times,
        )

    def extract_energy(
        self,
        decomposed: DecomposedAudio,
        hop_length: int,
    ) -> EnergyFeatures:
        """
        Extract RMS energy and frequency band levels.
        """
        sr = decomposed.sample_rate

        # Global and component RMS
        rms = librosa.feature.rms(
            y=decomposed.original,
            hop_length=hop_length,
        )[0]

        rms_harmonic = librosa.feature.rms(
            y=decomposed.harmonic,
            hop_length=hop_length,
        )[0]

        rms_percussive = librosa.feature.rms(
            y=decomposed.percussive,
            hop_length=hop_length,
        )[0]

        # Frequency band separation using bandpass filtering
        frequency_bands = self._extract_frequency_bands(
            decomposed.original,
            sr,
            hop_length,
        )

        return EnergyFeatures(
            rms=rms,
            rms_harmonic=rms_harmonic,
            rms_percussive=rms_percussive,
            frequency_bands=frequency_bands,
        )

    def _extract_frequency_bands(
        self,
        y: np.ndarray,
        sr: int,
        hop_length: int,
    ) -> FrequencyBands:
        """
        Extract energy in Low (0-200Hz), Mid (200Hz-4kHz), High (4kHz+) bands.
        """
        nyquist = sr / 2

        # Design bandpass filters
        # Low: 20Hz - 200Hz
        low_band = self._bandpass_rms(y, 20, 200, sr, hop_length)

        # Mid: 200Hz - 4000Hz
        mid_band = self._bandpass_rms(y, 200, 4000, sr, hop_length)

        # High: 4000Hz - Nyquist
        high_cutoff = min(16000, nyquist - 100)
        high_band = self._bandpass_rms(y, 4000, high_cutoff, sr, hop_length)

        return FrequencyBands(low=low_band, mid=mid_band, high=high_band)

    def _bandpass_rms(
        self,
        y: np.ndarray,
        low_freq: float,
        high_freq: float,
        sr: int,
        hop_length: int,
    ) -> np.ndarray:
        """Apply bandpass filter and compute RMS."""
        nyquist = sr / 2

        # Normalize frequencies
        low_norm = low_freq / nyquist
        high_norm = min(high_freq / nyquist, 0.99)

        # Design Butterworth bandpass filter
        sos = scipy_signal.butter(
            4,
            [low_norm, high_norm],
            btype="band",
            output="sos",
        )

        # Apply filter
        filtered = scipy_signal.sosfilt(sos, y)

        # Compute RMS
        rms = librosa.feature.rms(y=filtered, hop_length=hop_length)[0]

        return rms

    def extract_tonality(
        self,
        decomposed: DecomposedAudio,
        hop_length: int,
    ) -> TonalityFeatures:
        """
        Extract chroma features and spectral centroid.

        Chroma is extracted from harmonic component for cleaner pitch content.
        """
        sr = decomposed.sample_rate

        # Chroma from harmonic component (cleaner pitch representation)
        chroma = librosa.feature.chroma_stft(
            y=decomposed.harmonic,
            sr=sr,
            hop_length=hop_length,
            n_fft=self.n_fft,
        )

        # Spectral centroid from original (overall brightness)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=decomposed.original,
            sr=sr,
            hop_length=hop_length,
        )[0]

        # Dominant chroma per frame
        dominant_chroma_indices = np.argmax(chroma, axis=0)

        # MFCC-based timbre representation from the original signal
        mfcc = librosa.feature.mfcc(
            y=decomposed.original,
            sr=sr,
            hop_length=hop_length,
            n_mfcc=13,
        )

        return TonalityFeatures(
            chroma=chroma,
            spectral_centroid=spectral_centroid,
            dominant_chroma_indices=dominant_chroma_indices,
            mfcc=mfcc,
        )

    def analyze(self, decomposed: DecomposedAudio) -> ExtractedFeatures:
        """
        Perform complete feature extraction on decomposed audio.

        Args:
            decomposed: Audio separated into harmonic/percussive components.

        Returns:
            ExtractedFeatures with all visual drivers.
        """
        sr = decomposed.sample_rate
        hop_length = self.compute_hop_length(sr)

        temporal = self.extract_temporal(decomposed, hop_length)
        energy = self.extract_energy(decomposed, hop_length)
        tonality = self.extract_tonality(decomposed, hop_length)

        # Calculate number of frames
        n_frames = len(energy.rms)

        return ExtractedFeatures(
            temporal=temporal,
            energy=energy,
            tonality=tonality,
            n_frames=n_frames,
            hop_length=hop_length,
            sample_rate=sr,
        )

    @classmethod
    def chroma_index_to_name(cls, index: int) -> str:
        """Convert chroma index (0-11) to note name."""
        return cls.CHROMA_NAMES[index % 12]
