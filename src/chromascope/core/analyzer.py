"""
Feature extraction module for audio analysis.

Extracts visual drivers: beats, onsets, RMS energy,
frequency bands, chroma features, spectral characteristics, and tonality.

Phase 1 (C1–C7) and Phase 2 W4 audio intelligence features are implemented
here.  All new fields on existing dataclasses default to None so downstream
consumers are unaffected when they are not populated.
"""

from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np
from scipy import signal as scipy_signal

from chromascope.core.decomposer import DecomposedAudio


# ---------------------------------------------------------------------------
# New Phase-1 dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StructuralFeatures:
    """Musical structure segmentation (C2)."""

    section_labels: np.ndarray       # per-frame section index (int)
    section_novelty: np.ndarray      # per-frame novelty score [0,1]
    section_boundaries: np.ndarray   # section start times in seconds
    n_sections: int


@dataclass
class KeyFeatures:
    """Musical key detection result (C4)."""

    root_index: int             # 0–11  (C, C#, D … B)
    root_name: str              # e.g. "F#"
    mode: str                   # "major" | "minor"
    confidence: float           # [0,1]
    key_stability: np.ndarray   # per-frame cosine similarity with key profile


@dataclass
class HarmonicFeatures:
    """Phase 2 W3 stub — chord recognition.

    Populated only if the optional ``autochord`` package is installed.
    """

    chord_labels: list  # [(start_sec, end_sec, label), ...]


# ---------------------------------------------------------------------------
# Existing dataclasses (extended)
# ---------------------------------------------------------------------------

@dataclass
class FrequencyBands:
    """Energy levels for frequency sub-bands."""

    sub_bass: np.ndarray    # 20-60Hz
    bass: np.ndarray        # 60-250Hz
    low_mid: np.ndarray     # 250-500Hz
    mid: np.ndarray         # 500-2000Hz
    high_mid: np.ndarray    # 2000-4000Hz
    presence: np.ndarray    # 4000-6000Hz
    brilliance: np.ndarray  # 6000-20000Hz

    # Legacy bands (for compatibility if needed, or aggregate)
    low: np.ndarray         # 0-200Hz
    mid_aggregate: np.ndarray  # 200Hz-4kHz
    high: np.ndarray        # 4kHz+

    # C6: CQT-derived sub-bands (None unless FeatureAnalyzer.use_cqt=True)
    sub_bass_cqt: Optional[np.ndarray] = None
    bass_cqt: Optional[np.ndarray] = None


@dataclass
class TemporalFeatures:
    """Beat and onset timing features."""

    bpm: float
    beat_frames: np.ndarray
    beat_times: np.ndarray
    onset_frames: np.ndarray
    onset_times: np.ndarray
    # Local tempo estimates derived from beat spacing
    tempo_curve_bpm: Optional[np.ndarray] = None
    tempo_curve_times: Optional[np.ndarray] = None
    # C5: downbeat positions
    downbeat_frames: Optional[np.ndarray] = None
    downbeat_times: Optional[np.ndarray] = None
    # W4: per-onset classification
    onset_types: Optional[list] = None        # e.g. ["percussive", "harmonic", …]
    onset_sharpness: Optional[np.ndarray] = None


@dataclass
class EnergyFeatures:
    """Energy-related features."""

    rms: np.ndarray
    rms_harmonic: np.ndarray
    rms_percussive: np.ndarray
    spectral_flux: np.ndarray
    frequency_bands: FrequencyBands


@dataclass
class TonalityFeatures:
    """Pitch and timbre features."""

    chroma: np.ndarray  # Shape: (12, n_frames)
    spectral_centroid: np.ndarray
    spectral_flatness: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    dominant_chroma_indices: np.ndarray
    # C1: compact timbre representation
    mfcc: Optional[np.ndarray] = None             # (13, n_frames)
    mfcc_delta: Optional[np.ndarray] = None       # (13, n_frames)
    mfcc_delta2: Optional[np.ndarray] = None      # (13, n_frames)
    # C3: fundamental frequency (pyin)
    f0_hz: Optional[np.ndarray] = None            # (n_frames,) NaN where unvoiced
    f0_voiced: Optional[np.ndarray] = None        # (n_frames,) bool
    f0_probs: Optional[np.ndarray] = None         # (n_frames,) voiced probability
    # C7: additional spectral descriptors
    spectral_bandwidth: Optional[np.ndarray] = None   # (n_frames,)
    spectral_contrast: Optional[np.ndarray] = None    # (7, n_frames)


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
    # Phase-1 high-level features
    structure: Optional[StructuralFeatures] = None
    key: Optional[KeyFeatures] = None
    harmony: Optional[HarmonicFeatures] = None   # Phase 2 W3 stub

    def __post_init__(self):
        if len(self.frame_times) == 0:
            self.frame_times = librosa.frames_to_time(
                np.arange(self.n_frames),
                sr=self.sample_rate,
                hop_length=self.hop_length,
            )


# ---------------------------------------------------------------------------
# Feature analyzer
# ---------------------------------------------------------------------------

class FeatureAnalyzer:
    """
    Extracts visual driver features from decomposed audio.

    Features are aligned to a consistent frame rate determined by hop_length.
    """

    CHROMA_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Krumhansl-Schmuckler key profiles (major / natural minor)
    MAJOR_PROFILE = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    MINOR_PROFILE = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    def __init__(
        self,
        target_fps: int = 60,
        n_fft: int = 2048,
        use_cqt: bool = False,
        use_neural_beats: bool = False,
    ):
        """
        Initialize the analyzer.

        Args:
            target_fps: Target frames per second for output alignment.
            n_fft: FFT window size.
            use_cqt: If True, compute CQT-based sub-bass/bass bands (C6).
            use_neural_beats: Reserved for Phase 2 W2 — madmom beat tracker.
        """
        self.target_fps = target_fps or 60
        self.n_fft = n_fft
        self.use_cqt = use_cqt
        self.use_neural_beats = use_neural_beats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def compute_hop_length(self, sr: int) -> int:
        """
        Calculate hop_length to achieve target FPS.

        Args:
            sr: Sample rate.

        Returns:
            Hop length in samples.
        """
        return int(sr / self.target_fps)

    @staticmethod
    def _trim_or_pad_1d(arr: np.ndarray, n: int, pad_value: float = 0.0) -> np.ndarray:
        """Trim or zero/nan-pad a 1-D array to exactly *n* elements."""
        if len(arr) >= n:
            return arr[:n]
        pad = np.full(n - len(arr), pad_value, dtype=arr.dtype)
        return np.concatenate([arr, pad])

    @staticmethod
    def _trim_or_pad_2d(arr: np.ndarray, n: int) -> np.ndarray:
        """Trim or zero-pad the last axis of a 2-D array to *n* columns."""
        if arr.shape[1] >= n:
            return arr[:, :n]
        pad = np.zeros((arr.shape[0], n - arr.shape[1]), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=1)

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Energy features
    # ------------------------------------------------------------------

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

        # Spectral flux (onset strength envelope)
        spectral_flux = librosa.onset.onset_strength(
            y=decomposed.original,
            sr=sr,
            hop_length=hop_length,
        )

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
            spectral_flux=spectral_flux,
            frequency_bands=frequency_bands,
        )

    def _extract_frequency_bands(
        self,
        y: np.ndarray,
        sr: int,
        hop_length: int,
    ) -> FrequencyBands:
        """
        Extract energy in multiple frequency sub-bands.
        """
        nyquist = sr / 2

        # 7-band subdivision
        sub_bass = self._bandpass_rms(y, 20, 60, sr, hop_length)
        bass = self._bandpass_rms(y, 60, 250, sr, hop_length)
        low_mid = self._bandpass_rms(y, 250, 500, sr, hop_length)
        mid = self._bandpass_rms(y, 500, 2000, sr, hop_length)
        high_mid = self._bandpass_rms(y, 2000, 4000, sr, hop_length)
        presence = self._bandpass_rms(y, 4000, 6000, sr, hop_length)
        brilliance = self._bandpass_rms(y, 6000, min(20000, nyquist - 100), sr, hop_length)

        # Legacy/Aggregate bands
        low = self._bandpass_rms(y, 20, 200, sr, hop_length)
        mid_agg = self._bandpass_rms(y, 200, 4000, sr, hop_length)
        high = self._bandpass_rms(y, 4000, min(16000, nyquist - 100), sr, hop_length)

        # C6: optional CQT-derived sub-bass / bass
        sub_bass_cqt = None
        bass_cqt = None
        if self.use_cqt:
            sub_bass_cqt, bass_cqt = self._extract_cqt_bands(y, sr, hop_length)

        return FrequencyBands(
            sub_bass=sub_bass,
            bass=bass,
            low_mid=low_mid,
            mid=mid,
            high_mid=high_mid,
            presence=presence,
            brilliance=brilliance,
            low=low,
            mid_aggregate=mid_agg,
            high=high,
            sub_bass_cqt=sub_bass_cqt,
            bass_cqt=bass_cqt,
        )

    def _extract_cqt_bands(
        self,
        y: np.ndarray,
        sr: int,
        hop_length: int,
    ) -> tuple:
        """
        Compute CQT-derived sub-bass and bass band energies.

        Returns:
            Tuple of (sub_bass_cqt, bass_cqt) arrays, or (None, None) on failure.
        """
        try:
            fmin = float(librosa.note_to_hz("C1"))  # ~32.7 Hz
            cqt = librosa.cqt(
                y=y,
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=36,         # 3 octaves: C1–C4
                bins_per_octave=12,
            )
            cqt_mag = np.abs(cqt)
            # Sub-bass: C1–C2 (bins 0–11, ~32.7–65.4 Hz)
            sub_bass_cqt = cqt_mag[:12].mean(axis=0)
            # Bass: C2–C4 (bins 12–35, ~65.4–261.6 Hz)
            bass_cqt = cqt_mag[12:36].mean(axis=0)
            return sub_bass_cqt, bass_cqt
        except Exception:
            return None, None

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

        # Ensure low < high
        if low_norm >= high_norm:
            return np.zeros(int(len(y) / hop_length) + 1)

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

    # ------------------------------------------------------------------
    # Tonality features
    # ------------------------------------------------------------------

    def extract_tonality(
        self,
        decomposed: DecomposedAudio,
        hop_length: int,
    ) -> TonalityFeatures:
        """
        Extract chroma features and spectral characteristics.

        Chroma is extracted from harmonic component for cleaner pitch content.
        C1 MFCC deltas, C3 pyin pitch, and C7 bandwidth/contrast are added here.
        """
        y = decomposed.original
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
            y=y,
            sr=sr,
            hop_length=hop_length,
        )[0]

        # Spectral flatness (noisiness vs tonality)
        spectral_flatness = librosa.feature.spectral_flatness(
            y=y,
            hop_length=hop_length,
        )[0]

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )[0]

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=y,
            hop_length=hop_length,
        )[0]

        # Dominant chroma per frame
        dominant_chroma_indices = np.argmax(chroma, axis=0)

        # C1: MFCC-based timbre representation + deltas
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            hop_length=hop_length,
            n_mfcc=13,
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # C3: pyin fundamental frequency estimation
        # Use a 4× coarser hop to keep peak memory manageable on long files
        # (pyin allocates full (frame_length × n_frames) matrices; at 60fps
        # hop=368 that is ~500 MB+ on a 5-minute track).
        # 15fps resolution is ample for pitch tracking; we interpolate back.
        n_ref = len(spectral_centroid)  # reference length for alignment
        pyin_hop = hop_length * 4
        try:
            f0, f0_voiced, f0_probs = librosa.pyin(
                y=decomposed.harmonic,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr,
                hop_length=pyin_hop,
            )
            # Resize coarse arrays back to n_ref via nearest-neighbour
            def _resize_1d(arr, n, nan_fill=False):
                if len(arr) == n:
                    return arr
                idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
                return arr[idx]

            f0 = _resize_1d(f0, n_ref)
            f0_voiced = _resize_1d(f0_voiced.astype(np.float32), n_ref).astype(bool)
            f0_probs = _resize_1d(f0_probs, n_ref)
        except Exception:
            f0 = np.full(n_ref, np.nan)
            f0_voiced = np.zeros(n_ref, dtype=bool)
            f0_probs = np.zeros(n_ref)

        # C7: spectral bandwidth and contrast
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )[0]
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )  # shape (7, n_frames)

        return TonalityFeatures(
            chroma=chroma,
            spectral_centroid=spectral_centroid,
            spectral_flatness=spectral_flatness,
            spectral_rolloff=spectral_rolloff,
            zero_crossing_rate=zcr,
            dominant_chroma_indices=dominant_chroma_indices,
            mfcc=mfcc,
            mfcc_delta=mfcc_delta,
            mfcc_delta2=mfcc_delta2,
            f0_hz=f0,
            f0_voiced=f0_voiced,
            f0_probs=f0_probs,
            spectral_bandwidth=spectral_bandwidth,
            spectral_contrast=spectral_contrast,
        )

    # ------------------------------------------------------------------
    # Structure segmentation (C2)
    # ------------------------------------------------------------------

    def extract_structure(
        self,
        decomposed: DecomposedAudio,
        hop_length: int,
        n_frames: int,
    ) -> StructuralFeatures:
        """
        Segment the track into musically homogeneous sections.

        Uses librosa's agglomerative clustering on chroma features to find
        section boundaries, and onset strength as a per-frame novelty proxy.

        Args:
            decomposed: Separated audio components.
            hop_length: Hop length in samples.
            n_frames: Reference frame count for alignment.

        Returns:
            StructuralFeatures with section labels, novelty, and boundaries.
        """
        sr = decomposed.sample_rate

        # Chroma from harmonic (stable representation for segmentation).
        # chroma_stft is used instead of chroma_cqt: both give comparable
        # quality for structural segmentation, but chroma_cqt computes a full
        # CQT which can allocate hundreds of MB on long tracks.
        chroma = librosa.feature.chroma_stft(
            y=decomposed.harmonic,
            sr=sr,
            hop_length=hop_length,
        )
        # Align chroma to n_frames
        chroma = self._trim_or_pad_2d(chroma, n_frames)

        # Per-frame novelty (onset strength on the full signal)
        novelty_raw = librosa.onset.onset_strength(
            y=decomposed.original,
            sr=sr,
            hop_length=hop_length,
        )
        novelty_raw = self._trim_or_pad_1d(novelty_raw, n_frames)
        max_nov = novelty_raw.max()
        section_novelty = novelty_raw / (max_nov + 1e-8)

        # Number of segments: heuristic based on duration
        duration_sec = decomposed.duration
        k = max(2, min(8, int(duration_sec / 8) + 2))
        k = min(k, n_frames)

        try:
            segment_ids = librosa.segment.agglomerative(chroma, k=k)
            # Trim/pad to n_frames
            if len(segment_ids) > n_frames:
                segment_ids = segment_ids[:n_frames]
            elif len(segment_ids) < n_frames:
                segment_ids = np.pad(
                    segment_ids, (0, n_frames - len(segment_ids)), mode="edge"
                )
            segment_ids = segment_ids.astype(int)
        except Exception:
            segment_ids = np.zeros(n_frames, dtype=int)

        # Section boundaries: frames where the label changes
        boundary_frames = np.concatenate(
            [[0], np.where(np.diff(segment_ids) != 0)[0] + 1]
        )
        boundary_times = librosa.frames_to_time(
            boundary_frames, sr=sr, hop_length=hop_length
        )
        n_sections = int(len(boundary_frames))

        return StructuralFeatures(
            section_labels=segment_ids,
            section_novelty=section_novelty,
            section_boundaries=boundary_times,
            n_sections=n_sections,
        )

    # ------------------------------------------------------------------
    # Key detection (C4) — Krumhansl-Schmuckler
    # ------------------------------------------------------------------

    def detect_key(
        self,
        chroma_mean: np.ndarray,
        chroma: Optional[np.ndarray] = None,
    ) -> KeyFeatures:
        """
        Detect the musical key using the Krumhansl-Schmuckler algorithm.

        Correlates the mean chroma vector against all 24 major/minor key
        profiles (12 roots × 2 modes) and returns the best match.

        Args:
            chroma_mean: Mean chroma across the track, shape (12,).
            chroma: Full chroma matrix (12, n_frames) for per-frame stability.
                    If None, stability is a constant equal to the confidence.

        Returns:
            KeyFeatures with root, mode, confidence, and per-frame stability.
        """
        best_corr = -np.inf
        best_root = 0
        best_mode = "major"

        all_correlations = np.zeros(24)

        for i in range(12):
            prof_maj = np.roll(self.MAJOR_PROFILE, i)
            corr_maj = float(np.corrcoef(chroma_mean, prof_maj)[0, 1])
            if np.isnan(corr_maj):
                corr_maj = 0.0
            all_correlations[i] = corr_maj

            prof_min = np.roll(self.MINOR_PROFILE, i)
            corr_min = float(np.corrcoef(chroma_mean, prof_min)[0, 1])
            if np.isnan(corr_min):
                corr_min = 0.0
            all_correlations[12 + i] = corr_min

            if corr_maj > best_corr:
                best_corr = corr_maj
                best_root = i
                best_mode = "major"
            if corr_min > best_corr:
                best_corr = corr_min
                best_root = i
                best_mode = "minor"

        # Confidence: relative position of best score in observed range
        c_min = float(all_correlations.min())
        c_max = float(all_correlations.max())
        if c_max > c_min:
            confidence = float(
                np.clip((best_corr - c_min) / (c_max - c_min), 0.0, 1.0)
            )
        else:
            confidence = 0.5

        # Per-frame key stability: cosine similarity between frame chroma
        # and the detected key profile.
        if chroma is not None and chroma.shape[1] > 0:
            profile = (
                np.roll(self.MAJOR_PROFILE, best_root)
                if best_mode == "major"
                else np.roll(self.MINOR_PROFILE, best_root)
            )
            profile_norm = profile / (np.linalg.norm(profile) + 1e-8)
            # (12, n_frames) → (n_frames,)
            frame_norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
            chroma_unit = chroma / frame_norms
            key_stability = np.clip(profile_norm @ chroma_unit, 0.0, 1.0)
        else:
            key_stability = np.array([confidence])

        return KeyFeatures(
            root_index=best_root,
            root_name=self.CHROMA_NAMES[best_root],
            mode=best_mode,
            confidence=confidence,
            key_stability=key_stability,
        )

    # ------------------------------------------------------------------
    # Onset shape classification (W4)
    # ------------------------------------------------------------------

    def classify_onset_shape(
        self,
        onset_frame: int,
        rms: np.ndarray,
        hop_length: int,
        sr: int,
    ) -> tuple:
        """
        Classify an onset by its amplitude envelope shape.

        Looks at the RMS energy in a window around the onset frame and uses
        rise/fall ratios to distinguish transient, percussive, and harmonic
        attack shapes.

        Args:
            onset_frame: Frame index of the onset.
            rms: RMS energy array (percussive component recommended).
            hop_length: Hop length in samples.
            sr: Sample rate.

        Returns:
            Tuple of (label, sharpness) where label is one of
            "transient", "percussive", "harmonic" and sharpness ∈ [0, 1].
        """
        n_frames = len(rms)
        # Window sizes in frames
        lookbehind = max(1, int(0.03 * sr / hop_length))  # ~30 ms
        lookahead = max(1, int(0.05 * sr / hop_length))   # ~50 ms

        before = int(max(0, onset_frame - lookbehind))
        after = int(min(n_frames - 1, onset_frame + lookahead))

        rms_before = float(rms[before])
        rms_at = float(rms[onset_frame]) if onset_frame < n_frames else 0.0
        rms_after = float(rms[after])

        rise = rms_at / (rms_before + 1e-8)
        fall = rms_at / (rms_after + 1e-8)

        sharpness = float(np.clip((rise - 1.0) / (rise + 1.0), 0.0, 1.0))

        if rise > 5.0 and fall > 2.0:
            label = "transient"
        elif rise > 2.0:
            label = "percussive"
        else:
            label = "harmonic"

        return label, sharpness

    # ------------------------------------------------------------------
    # Private helpers used inside analyze()
    # ------------------------------------------------------------------

    def _detect_downbeats(
        self,
        beat_frames: np.ndarray,
        beat_times: np.ndarray,
    ) -> tuple:
        """
        Estimate downbeat positions using a simple 4/4 heuristic.

        Every 4th detected beat is treated as a downbeat.
        """
        if len(beat_frames) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        idx = np.arange(0, len(beat_frames), 4)
        return beat_frames[idx], beat_times[idx]

    def _classify_onsets(
        self,
        onset_frames: np.ndarray,
        rms_percussive: np.ndarray,
        hop_length: int,
        sr: int,
    ) -> tuple:
        """
        Classify all detected onsets and compute per-onset sharpness.

        Returns:
            (onset_types list[str], onset_sharpness np.ndarray)
        """
        labels = []
        sharpness_vals = []
        for of in onset_frames:
            label, sharp = self.classify_onset_shape(
                int(of), rms_percussive, hop_length, sr
            )
            labels.append(label)
            sharpness_vals.append(sharp)
        return labels, np.array(sharpness_vals, dtype=float)

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def analyze(self, decomposed: DecomposedAudio) -> ExtractedFeatures:
        """
        Perform complete feature extraction on decomposed audio.

        Runs all Phase-1 (C1–C7) and W4 extractors in addition to the
        existing temporal/energy/tonality pipeline.

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

        # Reference frame count (from the most stable source)
        n_frames = len(energy.rms)

        # W4: classify onset shapes (needs both temporal + energy)
        onset_types, onset_sharpness = self._classify_onsets(
            temporal.onset_frames,
            energy.rms_percussive,
            hop_length,
            sr,
        )
        temporal.onset_types = onset_types
        temporal.onset_sharpness = onset_sharpness

        # C5: downbeat heuristic (needs beats)
        db_frames, db_times = self._detect_downbeats(
            temporal.beat_frames, temporal.beat_times
        )
        temporal.downbeat_frames = db_frames
        temporal.downbeat_times = db_times

        # C2: structure segmentation
        try:
            structure = self.extract_structure(decomposed, hop_length, n_frames)
        except Exception:
            structure = None

        # C4: key detection from mean chroma
        try:
            chroma_mean = np.mean(tonality.chroma, axis=1)
            key = self.detect_key(chroma_mean, tonality.chroma)
        except Exception:
            key = None

        return ExtractedFeatures(
            temporal=temporal,
            energy=energy,
            tonality=tonality,
            n_frames=n_frames,
            hop_length=hop_length,
            sample_rate=sr,
            structure=structure,
            key=key,
        )

    @classmethod
    def chroma_index_to_name(cls, index: int) -> str:
        """Convert chroma index (0-11) to note name."""
        return cls.CHROMA_NAMES[index % 12]
