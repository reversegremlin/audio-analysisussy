"""
Signal smoothing and normalization module.

Applies aesthetic processing to raw audio features to prevent
visual flickering and ensure smooth, organic visuals.

Phase-1 (C1–C7) and W4 polished fields are added here with None defaults
so old consumers are unaffected when the features are not populated.
"""

from dataclasses import dataclass
from typing import Optional

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
    spectral_flux: np.ndarray

    # Frequency bands [0.0, 1.0]
    sub_bass: np.ndarray
    bass: np.ndarray
    low_mid: np.ndarray
    mid: np.ndarray
    high_mid: np.ndarray
    presence: np.ndarray
    brilliance: np.ndarray

    # Legacy bands (optional, but keeping for compatibility)
    low_energy: np.ndarray
    mid_energy: np.ndarray
    high_energy: np.ndarray

    # Tonality/Texture [0.0, 1.0]
    spectral_brightness: np.ndarray
    spectral_flatness: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    chroma: np.ndarray  # Shape: (12, n_frames), normalized

    # Dominant note per frame
    dominant_chroma_indices: np.ndarray

    # Metadata
    n_frames: int
    fps: int
    frame_times: np.ndarray

    # ----------------------------------------------------------------
    # C1 — Timbre (MFCC)
    # ----------------------------------------------------------------
    mfcc: Optional[np.ndarray] = None          # (13, n_frames)
    mfcc_delta: Optional[np.ndarray] = None    # (13, n_frames)
    mfcc_delta2: Optional[np.ndarray] = None   # (13, n_frames)
    timbre_velocity: Optional[np.ndarray] = None  # (n_frames,)  L2-norm of delta

    # ----------------------------------------------------------------
    # C2 — Structure
    # ----------------------------------------------------------------
    section_index: Optional[np.ndarray] = None    # (n_frames,) int
    section_novelty: Optional[np.ndarray] = None  # (n_frames,) [0,1]
    section_progress: Optional[np.ndarray] = None # (n_frames,) [0,1]
    section_change: Optional[np.ndarray] = None   # (n_frames,) bool
    n_sections: Optional[int] = None
    section_boundary_times: Optional[np.ndarray] = None

    # ----------------------------------------------------------------
    # C3 — Pitch / F0
    # ----------------------------------------------------------------
    f0_hz: Optional[np.ndarray] = None           # (n_frames,) NaN = unvoiced
    f0_confidence: Optional[np.ndarray] = None   # (n_frames,) [0,1]
    f0_voiced: Optional[np.ndarray] = None       # (n_frames,) bool
    pitch_velocity: Optional[np.ndarray] = None  # (n_frames,) [0,1]
    pitch_register: Optional[np.ndarray] = None  # (n_frames,) [0,1]

    # ----------------------------------------------------------------
    # C4 — Key
    # ----------------------------------------------------------------
    key_stability: Optional[np.ndarray] = None   # (n_frames,) [0,1]
    key_root_index: Optional[int] = None         # scalar 0–11
    key_mode: Optional[str] = None               # "major" | "minor"
    key_confidence: Optional[float] = None       # scalar [0,1]

    # ----------------------------------------------------------------
    # C5 — Downbeats / rhythm grid
    # ----------------------------------------------------------------
    is_downbeat: Optional[np.ndarray] = None     # (n_frames,) bool
    beat_position: Optional[np.ndarray] = None   # (n_frames,) [0,1]
    bar_index: Optional[np.ndarray] = None       # (n_frames,) int
    bar_progress: Optional[np.ndarray] = None    # (n_frames,) [0,1]

    # ----------------------------------------------------------------
    # C6 — CQT sub-bass / bass  (None when use_cqt=False)
    # ----------------------------------------------------------------
    sub_bass_cqt: Optional[np.ndarray] = None    # (n_frames,) [0,1]
    bass_cqt: Optional[np.ndarray] = None        # (n_frames,) [0,1]

    # ----------------------------------------------------------------
    # C7 — Extended spectral descriptors
    # ----------------------------------------------------------------
    spectral_bandwidth: Optional[np.ndarray] = None  # (n_frames,) [0,1]
    spectral_contrast: Optional[np.ndarray] = None   # (7, n_frames) [0,1]

    # ----------------------------------------------------------------
    # W4 — Onset shape
    # ----------------------------------------------------------------
    onset_type: Optional[np.ndarray] = None      # (n_frames,) object dtype str|None
    onset_sharpness: Optional[np.ndarray] = None # (n_frames,) [0,1]


class SignalPolisher:
    """
    Applies aesthetic smoothing to raw audio features.

    Implements attack/release envelopes and normalization to create
    visually pleasing, flicker-free signals.
    """

    def __init__(
        self,
        fps: int = 60,
        impact_envelope: Optional[EnvelopeParams] = None,
        energy_envelope: Optional[EnvelopeParams] = None,
        adaptive_envelopes: bool = False,
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
        # When enabled, envelope timings are adaptively scaled based on BPM.
        self.adaptive_envelopes = adaptive_envelopes

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

        Maps typical music range (100Hz - 10000Hz) to 0-1.
        """
        # Typical range for music
        min_hz = 100.0
        max_hz = 10000.0

        brightness = (centroid - min_hz) / (max_hz - min_hz)
        brightness = np.clip(brightness, 0.0, 1.0)

        # Apply light smoothing
        return self.apply_envelope(brightness, self.energy_envelope)

    # ------------------------------------------------------------------
    # Phase-1 helper: safe array fetch (handles None + length mismatch)
    # ------------------------------------------------------------------

    def _safe_get(
        self,
        arr: Optional[np.ndarray],
        n_frames: int,
        fill: float = 0.0,
    ) -> np.ndarray:
        """Return *arr* trimmed/padded to *n_frames*, or a zero array if None."""
        if arr is None:
            return np.full(n_frames, fill)
        if len(arr) >= n_frames:
            return arr[:n_frames]
        pad = np.full(n_frames - len(arr), fill, dtype=arr.dtype)
        return np.concatenate([arr, pad])

    # ------------------------------------------------------------------
    # Main polishing entry point
    # ------------------------------------------------------------------

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

        # Optionally adapt envelope timings based on detected BPM.
        impact_env = self.impact_envelope
        energy_env = self.energy_envelope
        if self.adaptive_envelopes:
            bpm = getattr(features.temporal, "bpm", 120.0) or 120.0
            # Scale release inversely with tempo: slower songs -> longer glow.
            scale = 120.0 / max(bpm, 1.0)
            scale = float(np.clip(scale, 0.5, 2.0))

            impact_env = EnvelopeParams(
                attack_ms=impact_env.attack_ms,
                release_ms=impact_env.release_ms * scale,
            )
            energy_env = EnvelopeParams(
                attack_ms=energy_env.attack_ms,
                release_ms=energy_env.release_ms * scale,
            )

        # Energy signals with envelope smoothing
        percussive_impact = self.apply_envelope(
            self.normalize(features.energy.rms_percussive),
            impact_env,
        )

        harmonic_energy = self.apply_envelope(
            self.normalize(features.energy.rms_harmonic),
            energy_env,
        )

        global_energy = self.apply_envelope(
            self.normalize(features.energy.rms),
            energy_env,
        )

        spectral_flux = self.apply_envelope(
            self.normalize(features.energy.spectral_flux),
            impact_env,
        )

        # 7-band frequency energy
        fb = features.energy.frequency_bands
        sub_bass = self.apply_envelope(self.normalize(fb.sub_bass), energy_env)
        bass = self.apply_envelope(self.normalize(fb.bass), energy_env)
        low_mid = self.apply_envelope(self.normalize(fb.low_mid), energy_env)
        mid = self.apply_envelope(self.normalize(fb.mid), energy_env)
        high_mid = self.apply_envelope(self.normalize(fb.high_mid), energy_env)
        presence = self.apply_envelope(self.normalize(fb.presence), energy_env)
        brilliance = self.apply_envelope(self.normalize(fb.brilliance), energy_env)

        # Legacy bands
        low_energy = self.apply_envelope(self.normalize(fb.low), energy_env)
        mid_energy = self.apply_envelope(self.normalize(fb.mid_aggregate), energy_env)
        high_energy = self.apply_envelope(self.normalize(fb.high), energy_env)

        # Tonality/Texture
        spectral_brightness = self.smooth_spectral_centroid(
            features.tonality.spectral_centroid,
            features.sample_rate,
        )

        spectral_flatness = self.apply_envelope(
            self.normalize(features.tonality.spectral_flatness),
            energy_env,
        )

        spectral_rolloff = self.apply_envelope(
            self.normalize(features.tonality.spectral_rolloff),
            energy_env,
        )

        zero_crossing_rate = self.apply_envelope(
            self.normalize(features.tonality.zero_crossing_rate),
            energy_env,
        )

        # Normalize chroma (each bin independently)
        chroma_normalized = np.zeros_like(features.tonality.chroma)
        for i in range(12):
            chroma_normalized[i] = self.normalize(features.tonality.chroma[i])

        # ----------------------------------------------------------------
        # C1 — MFCC timbre
        # ----------------------------------------------------------------
        mfcc_raw = features.tonality.mfcc
        mfcc_delta_raw = features.tonality.mfcc_delta
        mfcc_delta2_raw = features.tonality.mfcc_delta2

        if mfcc_raw is not None:
            # Normalize each coefficient independently
            mfcc_pol = np.zeros_like(mfcc_raw)
            for i in range(mfcc_raw.shape[0]):
                mfcc_pol[i] = self.normalize(mfcc_raw[i])
        else:
            mfcc_pol = None

        if mfcc_delta_raw is not None:
            mfcc_delta_pol = np.zeros_like(mfcc_delta_raw)
            for i in range(mfcc_delta_raw.shape[0]):
                mfcc_delta_pol[i] = self.normalize(mfcc_delta_raw[i])
        else:
            mfcc_delta_pol = None

        if mfcc_delta2_raw is not None:
            mfcc_delta2_pol = np.zeros_like(mfcc_delta2_raw)
            for i in range(mfcc_delta2_raw.shape[0]):
                mfcc_delta2_pol[i] = self.normalize(mfcc_delta2_raw[i])
        else:
            mfcc_delta2_pol = None

        # timbre_velocity: L2-norm of mfcc_delta per frame, normalized
        if mfcc_delta_raw is not None:
            timbre_vel_raw = np.linalg.norm(mfcc_delta_raw, axis=0)
            timbre_velocity = self.normalize(timbre_vel_raw)
        else:
            timbre_velocity = None

        # ----------------------------------------------------------------
        # C2 — Structure
        # ----------------------------------------------------------------
        struct = features.structure
        if struct is not None:
            section_labels = self._safe_get(
                struct.section_labels.astype(float), n_frames
            ).astype(int)
            section_novelty = self.apply_envelope(
                self._safe_get(struct.section_novelty, n_frames),
                energy_env,
            )
            # Per-frame progress within each section
            section_progress = np.zeros(n_frames)
            current_start = 0
            current_lbl = section_labels[0] if n_frames > 0 else 0
            for i in range(1, n_frames + 1):
                if i == n_frames or section_labels[i] != current_lbl:
                    length = i - current_start
                    section_progress[current_start:i] = np.linspace(
                        0.0, 1.0, length, endpoint=False
                    )
                    if i < n_frames:
                        current_start = i
                        current_lbl = section_labels[i]
            section_change = np.concatenate(
                [[False], np.diff(section_labels) != 0]
            )
            n_sections = struct.n_sections
            section_boundary_times = struct.section_boundaries
        else:
            section_labels = np.zeros(n_frames, dtype=int)
            section_novelty = np.zeros(n_frames)
            section_progress = np.linspace(0.0, 1.0, n_frames)
            section_change = np.zeros(n_frames, dtype=bool)
            n_sections = 1
            section_boundary_times = np.array([0.0])

        # ----------------------------------------------------------------
        # C3 — Pitch / F0
        # ----------------------------------------------------------------
        f0_raw = features.tonality.f0_hz
        if f0_raw is not None:
            f0_hz_pol = self._safe_get(f0_raw, n_frames, fill=np.nan)
            # voiced flag
            f0_voiced_raw = features.tonality.f0_voiced
            f0_voiced_pol = (
                self._safe_get(f0_voiced_raw.astype(float), n_frames).astype(bool)
                if f0_voiced_raw is not None
                else (np.isfinite(f0_hz_pol) & (f0_hz_pol > 0))
            )
            # confidence from voiced probabilities
            f0_probs_raw = features.tonality.f0_probs
            if f0_probs_raw is not None:
                f0_confidence = self.apply_envelope(
                    self._safe_get(f0_probs_raw, n_frames),
                    energy_env,
                )
            else:
                f0_confidence = f0_voiced_pol.astype(float)
            # pitch register: log-scale map to [0,1] over C2–C7
            import librosa as _lb
            log_min = np.log(float(_lb.note_to_hz("C2")))
            log_max = np.log(float(_lb.note_to_hz("C7")))
            pitch_register = np.zeros(n_frames)
            voiced_mask = f0_voiced_pol & np.isfinite(f0_hz_pol) & (f0_hz_pol > 0)
            if np.any(voiced_mask):
                pitch_register[voiced_mask] = np.clip(
                    (np.log(f0_hz_pol[voiced_mask]) - log_min) / (log_max - log_min),
                    0.0, 1.0,
                )
            # pitch velocity: frame-to-frame change
            f0_clean = np.where(np.isfinite(f0_hz_pol), f0_hz_pol, 0.0)
            pv_raw = np.zeros(n_frames)
            pv_raw[1:] = np.abs(np.diff(f0_clean))
            pitch_velocity = self.normalize(pv_raw)
        else:
            f0_hz_pol = None
            f0_voiced_pol = None
            f0_confidence = None
            pitch_velocity = None
            pitch_register = None

        # ----------------------------------------------------------------
        # C4 — Key
        # ----------------------------------------------------------------
        key_feat = features.key
        if key_feat is not None:
            key_stability = self.apply_envelope(
                self._safe_get(key_feat.key_stability, n_frames),
                energy_env,
            )
            key_root_index = key_feat.root_index
            key_mode = key_feat.mode
            key_confidence = key_feat.confidence
        else:
            key_stability = None
            key_root_index = None
            key_mode = None
            key_confidence = None

        # ----------------------------------------------------------------
        # C5 — Downbeats / rhythm grid
        # ----------------------------------------------------------------
        beat_frames = features.temporal.beat_frames
        downbeat_frames = getattr(features.temporal, "downbeat_frames", None)

        # is_downbeat boolean array
        is_downbeat = np.zeros(n_frames, dtype=bool)
        if downbeat_frames is not None and len(downbeat_frames) > 0:
            valid_db = downbeat_frames[downbeat_frames < n_frames].astype(int)
            is_downbeat[valid_db] = True

        # beat_position: position within current beat [0,1]
        beat_position = np.zeros(n_frames)
        if len(beat_frames) >= 2:
            beats = np.append(beat_frames, n_frames).astype(int)
            for b1, b2 in zip(beats[:-1], beats[1:]):
                b1 = min(b1, n_frames)
                b2 = min(b2, n_frames)
                length = b2 - b1
                if length > 0:
                    beat_position[b1:b2] = np.linspace(0.0, 1.0, length, endpoint=False)

        # bar_index and bar_progress (every 4 beats = 1 bar)
        bar_index = np.zeros(n_frames, dtype=int)
        bar_progress = np.zeros(n_frames)
        bars_per_group = 4
        if len(beat_frames) >= 2:
            beats_ext = np.append(beat_frames, n_frames).astype(int)
            for bi, (b1, b2) in enumerate(zip(beats_ext[:-1], beats_ext[1:])):
                b1 = min(b1, n_frames)
                b2 = min(b2, n_frames)
                length = b2 - b1
                if length > 0:
                    bar_idx = bi // bars_per_group
                    beat_in_bar = (bi % bars_per_group) / bars_per_group
                    bar_index[b1:b2] = bar_idx
                    within = np.linspace(0.0, 1.0, length, endpoint=False) / bars_per_group
                    bar_progress[b1:b2] = beat_in_bar + within
        bar_progress = np.clip(bar_progress, 0.0, 1.0)

        # ----------------------------------------------------------------
        # C6 — CQT sub-bass / bass
        # ----------------------------------------------------------------
        if fb.sub_bass_cqt is not None:
            sub_bass_cqt_pol = self.apply_envelope(
                self.normalize(self._safe_get(fb.sub_bass_cqt, n_frames)),
                energy_env,
            )
        else:
            sub_bass_cqt_pol = None

        if fb.bass_cqt is not None:
            bass_cqt_pol = self.apply_envelope(
                self.normalize(self._safe_get(fb.bass_cqt, n_frames)),
                energy_env,
            )
        else:
            bass_cqt_pol = None

        # ----------------------------------------------------------------
        # C7 — Extended spectral descriptors
        # ----------------------------------------------------------------
        bw_raw = features.tonality.spectral_bandwidth
        if bw_raw is not None:
            spectral_bandwidth_pol = self.apply_envelope(
                self.normalize(self._safe_get(bw_raw, n_frames)),
                energy_env,
            )
        else:
            spectral_bandwidth_pol = None

        sc_raw = features.tonality.spectral_contrast
        if sc_raw is not None:
            # Align columns then normalize each of the 7 bands
            sc_aligned = (
                sc_raw[:, :n_frames]
                if sc_raw.shape[1] >= n_frames
                else np.pad(sc_raw, ((0, 0), (0, n_frames - sc_raw.shape[1])))
            )
            spectral_contrast_pol = np.zeros_like(sc_aligned)
            for b in range(sc_aligned.shape[0]):
                spectral_contrast_pol[b] = self.normalize(sc_aligned[b])
        else:
            spectral_contrast_pol = None

        # ----------------------------------------------------------------
        # W4 — Onset shape (per-frame)
        # ----------------------------------------------------------------
        onset_type_arr = np.full(n_frames, None, dtype=object)
        onset_sharpness_arr = np.zeros(n_frames)

        ot = getattr(features.temporal, "onset_types", None)
        os_ = getattr(features.temporal, "onset_sharpness", None)
        onset_frames_arr = features.temporal.onset_frames

        if ot is not None and os_ is not None:
            for k_i, of in enumerate(onset_frames_arr):
                fi = int(of)
                if fi < n_frames:
                    onset_type_arr[fi] = ot[k_i] if k_i < len(ot) else None
                    onset_sharpness_arr[fi] = os_[k_i] if k_i < len(os_) else 0.0

        return PolishedFeatures(
            is_beat=is_beat,
            is_onset=is_onset,
            percussive_impact=percussive_impact,
            harmonic_energy=harmonic_energy,
            global_energy=global_energy,
            spectral_flux=spectral_flux,
            sub_bass=sub_bass,
            bass=bass,
            low_mid=low_mid,
            mid=mid,
            high_mid=high_mid,
            presence=presence,
            brilliance=brilliance,
            low_energy=low_energy,
            mid_energy=mid_energy,
            high_energy=high_energy,
            spectral_brightness=spectral_brightness,
            spectral_flatness=spectral_flatness,
            spectral_rolloff=spectral_rolloff,
            zero_crossing_rate=zero_crossing_rate,
            chroma=chroma_normalized,
            dominant_chroma_indices=features.tonality.dominant_chroma_indices,
            n_frames=n_frames,
            fps=self.fps,
            frame_times=features.frame_times,
            # C1
            mfcc=mfcc_pol,
            mfcc_delta=mfcc_delta_pol,
            mfcc_delta2=mfcc_delta2_pol,
            timbre_velocity=timbre_velocity,
            # C2
            section_index=section_labels,
            section_novelty=section_novelty,
            section_progress=section_progress,
            section_change=section_change,
            n_sections=n_sections,
            section_boundary_times=section_boundary_times,
            # C3
            f0_hz=f0_hz_pol,
            f0_confidence=f0_confidence,
            f0_voiced=f0_voiced_pol,
            pitch_velocity=pitch_velocity,
            pitch_register=pitch_register,
            # C4
            key_stability=key_stability,
            key_root_index=key_root_index,
            key_mode=key_mode,
            key_confidence=key_confidence,
            # C5
            is_downbeat=is_downbeat,
            beat_position=beat_position,
            bar_index=bar_index,
            bar_progress=bar_progress,
            # C6
            sub_bass_cqt=sub_bass_cqt_pol,
            bass_cqt=bass_cqt_pol,
            # C7
            spectral_bandwidth=spectral_bandwidth_pol,
            spectral_contrast=spectral_contrast_pol,
            # W4
            onset_type=onset_type_arr,
            onset_sharpness=onset_sharpness_arr,
        )
