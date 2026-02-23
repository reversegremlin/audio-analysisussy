"""
Manifest serialization module.

Exports polished audio features to JSON format aligned to target FPS
for use in rendering engines and visualization systems.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from chromascope.core.analyzer import FeatureAnalyzer
from chromascope.core.polisher import PolishedFeatures


@dataclass
class ManifestMetadata:
    """Metadata header for the visual driver manifest."""

    bpm: float
    duration: float
    fps: int
    n_frames: int
    # Engine or exporter version (kept for backward compatibility)
    version: str = "2.0"
    # Explicit schema version for the manifest payload
    schema_version: str = "2.0"


class ManifestExporter:
    """
    Exports polished features to JSON manifest format.

    The manifest follows the schema defined in the architecture doc,
    with each frame containing all visual driver values.
    Schema version 2.0 adds Phase-1 audio intelligence fields.
    """

    def __init__(self, precision: int = 4):
        """
        Initialize the exporter.

        Args:
            precision: Decimal places for floating point values.
        """
        self.precision = precision

    def _round(self, value: float) -> float:
        """Round to configured precision."""
        return round(float(value), self.precision)

    def _safe_float(self, polished: PolishedFeatures, field: str, index: int) -> Optional[float]:
        """Safely get a per-frame float from a polished field, returning None if absent."""
        arr = getattr(polished, field, None)
        if arr is None:
            return None
        if index >= len(arr):
            return None
        val = arr[index]
        if val is None:
            return None
        try:
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return None
            return self._round(f)
        except (TypeError, ValueError):
            return None

    def _build_frame(
        self,
        index: int,
        polished: PolishedFeatures,
    ) -> dict[str, Any]:
        """
        Build a single frame's data dictionary.

        Args:
            index: Frame index.
            polished: Source polished features.

        Returns:
            Dictionary with all frame data.
        """
        chroma_idx = int(polished.dominant_chroma_indices[index])
        dominant_chroma = FeatureAnalyzer.chroma_index_to_name(chroma_idx)

        # Base feature fields derived directly from polished features
        frame: dict[str, Any] = {
            "frame_index": index,
            "time": self._round(polished.frame_times[index]),
            "is_beat": bool(polished.is_beat[index]),
            "is_onset": bool(polished.is_onset[index]),
            "percussive_impact": self._round(polished.percussive_impact[index]),
            "harmonic_energy": self._round(polished.harmonic_energy[index]),
            "global_energy": self._round(polished.global_energy[index]),
            "spectral_flux": self._round(polished.spectral_flux[index]),

            # 7-band frequency energy
            "sub_bass": self._round(polished.sub_bass[index]),
            "bass": self._round(polished.bass[index]),
            "low_mid": self._round(polished.low_mid[index]),
            "mid": self._round(polished.mid[index]),
            "high_mid": self._round(polished.high_mid[index]),
            "presence": self._round(polished.presence[index]),
            "brilliance": self._round(polished.brilliance[index]),

            # Legacy bands (kept for backward compatibility)
            "low_energy": self._round(polished.low_energy[index]),
            "mid_energy": self._round(polished.mid_energy[index]),
            "high_energy": self._round(polished.high_energy[index]),

            # Tonality/Texture
            "spectral_brightness": self._round(polished.spectral_brightness[index]),
            "spectral_flatness": self._round(polished.spectral_flatness[index]),
            "spectral_rolloff": self._round(polished.spectral_rolloff[index]),
            "zero_crossing_rate": self._round(polished.zero_crossing_rate[index]),

            "dominant_chroma": dominant_chroma,
            "chroma_values": {
                FeatureAnalyzer.CHROMA_NAMES[i]: self._round(polished.chroma[i, index])
                for i in range(12)
            },
        }

        # ------------------------------------------------------------------
        # C1 — Timbre
        # ------------------------------------------------------------------
        frame["timbre_velocity"] = self._safe_float(polished, "timbre_velocity", index)

        # ------------------------------------------------------------------
        # C2 — Structure
        # ------------------------------------------------------------------
        si = getattr(polished, "section_index", None)
        frame["section_index"] = int(si[index]) if si is not None else None
        frame["section_novelty"] = self._safe_float(polished, "section_novelty", index)
        frame["section_progress"] = self._safe_float(polished, "section_progress", index)
        sc = getattr(polished, "section_change", None)
        frame["section_change"] = bool(sc[index]) if sc is not None else None

        # ------------------------------------------------------------------
        # C3 — Pitch / F0
        # ------------------------------------------------------------------
        frame["f0_hz"] = self._safe_float(polished, "f0_hz", index)
        frame["f0_confidence"] = self._safe_float(polished, "f0_confidence", index)
        fv = getattr(polished, "f0_voiced", None)
        frame["f0_voiced"] = bool(fv[index]) if fv is not None else None
        frame["pitch_velocity"] = self._safe_float(polished, "pitch_velocity", index)
        frame["pitch_register"] = self._safe_float(polished, "pitch_register", index)

        # ------------------------------------------------------------------
        # C4 — Key stability (per-frame)
        # ------------------------------------------------------------------
        frame["key_stability"] = self._safe_float(polished, "key_stability", index)

        # ------------------------------------------------------------------
        # C5 — Downbeats / rhythm grid
        # ------------------------------------------------------------------
        idb = getattr(polished, "is_downbeat", None)
        frame["is_downbeat"] = bool(idb[index]) if idb is not None else None
        frame["beat_position"] = self._safe_float(polished, "beat_position", index)
        bi = getattr(polished, "bar_index", None)
        frame["bar_index"] = int(bi[index]) if bi is not None else None
        frame["bar_progress"] = self._safe_float(polished, "bar_progress", index)

        # ------------------------------------------------------------------
        # C6 — CQT bands (None when use_cqt=False)
        # ------------------------------------------------------------------
        frame["sub_bass_cqt"] = self._safe_float(polished, "sub_bass_cqt", index)
        frame["bass_cqt"] = self._safe_float(polished, "bass_cqt", index)

        # ------------------------------------------------------------------
        # C7 — Extended spectral
        # ------------------------------------------------------------------
        frame["spectral_bandwidth"] = self._safe_float(polished, "spectral_bandwidth", index)

        # ------------------------------------------------------------------
        # W4 — Onset shape
        # ------------------------------------------------------------------
        ot = getattr(polished, "onset_type", None)
        frame["onset_type"] = str(ot[index]) if (ot is not None and ot[index] is not None) else None
        frame["onset_sharpness"] = self._safe_float(polished, "onset_sharpness", index)

        # Derived visual primitives
        primitives = self._compute_primitives(frame, polished, index)
        frame.update(primitives)

        return frame

    def _compute_primitives(
        self,
        frame: dict[str, Any],
        polished: PolishedFeatures,
        index: int,
    ) -> dict[str, float]:
        """
        Compute high-level visual primitives from a frame's raw fields.

        This provides a small, stable set of semantic controls that renderers
        can rely on, even as lower-level features evolve.
        """
        # Core primitives map 1:1 to key polished signals
        impact = frame["percussive_impact"]
        fluidity = frame["harmonic_energy"]
        brightness = frame["spectral_brightness"]

        # Map dominant chroma onto a [0.0, 1.0] hue-like scale
        dominant = frame.get("dominant_chroma", "C")
        try:
            chroma_index = FeatureAnalyzer.CHROMA_NAMES.index(dominant)
        except ValueError:
            chroma_index = 0
        # Use 0-1 scale over the 12 chroma bins
        pitch_hue = chroma_index / (len(FeatureAnalyzer.CHROMA_NAMES) - 1)

        # Texture: richer aggregation of noisiness and high-frequency content
        flatness = frame["spectral_flatness"]
        zcr = frame["zero_crossing_rate"]
        presence = frame["presence"]
        brilliance = frame["brilliance"]
        texture = max(0.0, min(1.0, (flatness + zcr + presence + brilliance) / 4.0))

        # Sharpness: focus on spectral rolloff and flux
        flux = frame["spectral_flux"]
        rolloff = frame["spectral_rolloff"]
        sharpness = max(0.0, min(1.0, (flux + rolloff) / 2.0))

        # Phase-1 primitives
        tv = self._safe_float(polished, "timbre_velocity", index)
        timbre_velocity = float(tv) if tv is not None else 0.0

        bw = self._safe_float(polished, "spectral_bandwidth", index)
        bandwidth_norm = float(bw) if bw is not None else 0.0

        return {
            "impact": impact,
            "fluidity": fluidity,
            "brightness": brightness,
            "pitch_hue": pitch_hue,
            "texture": texture,
            "sharpness": sharpness,
            "timbre_velocity": timbre_velocity,
            "bandwidth_norm": bandwidth_norm,
        }

    def build_manifest(
        self,
        polished: PolishedFeatures,
        bpm: float,
        duration: float,
    ) -> dict[str, Any]:
        """
        Build the complete manifest dictionary.

        Args:
            polished: Polished features from SignalPolisher.
            bpm: Detected BPM from analysis.
            duration: Audio duration in seconds.

        Returns:
            Complete manifest dictionary ready for serialization.
        """
        metadata = ManifestMetadata(
            bpm=self._round(bpm),
            duration=self._round(duration),
            fps=polished.fps,
            n_frames=polished.n_frames,
        )

        frames = [
            self._build_frame(i, polished)
            for i in range(polished.n_frames)
        ]

        manifest: dict[str, Any] = {
            "metadata": {
                "bpm": metadata.bpm,
                "duration": metadata.duration,
                "fps": metadata.fps,
                "n_frames": metadata.n_frames,
                "version": metadata.version,
                "schema_version": metadata.schema_version,
            },
            "frames": frames,
        }

        # ------------------------------------------------------------------
        # C2 — Structure block
        # ------------------------------------------------------------------
        n_sec = getattr(polished, "n_sections", None)
        sec_boundaries = getattr(polished, "section_boundary_times", None)
        if n_sec is not None and sec_boundaries is not None:
            boundaries_list = [self._round(float(t)) for t in sec_boundaries]
            # Compute per-section durations from boundary times + total duration
            if len(boundaries_list) >= 2:
                sec_durations = [
                    self._round(b2 - b1)
                    for b1, b2 in zip(boundaries_list[:-1], boundaries_list[1:])
                ]
                # Last section runs to end
                sec_durations.append(self._round(duration - boundaries_list[-1]))
            elif len(boundaries_list) == 1:
                sec_durations = [self._round(duration)]
            else:
                sec_durations = []

            manifest["structure"] = {
                "n_sections": n_sec,
                "section_boundaries": boundaries_list,
                "section_durations": sec_durations,
            }

        # ------------------------------------------------------------------
        # C4 — Key block
        # ------------------------------------------------------------------
        kr = getattr(polished, "key_root_index", None)
        km = getattr(polished, "key_mode", None)
        kc = getattr(polished, "key_confidence", None)
        if kr is not None and km is not None:
            CHROMA_NAMES = FeatureAnalyzer.CHROMA_NAMES
            root_name = CHROMA_NAMES[kr % 12]
            # Relative major: for major key it's the same root; for minor +3
            if km == "minor":
                rel_major_idx = (kr + 3) % 12
            else:
                rel_major_idx = kr
            manifest["key"] = {
                "root": root_name,
                "root_index": int(kr),
                "mode": km,
                "confidence": self._round(float(kc)) if kc is not None else None,
                "relative_major": CHROMA_NAMES[rel_major_idx],
            }

        return manifest

    def export_json(
        self,
        polished: PolishedFeatures,
        bpm: float,
        duration: float,
        output_path: Union[str, Path],
        indent: int = 2,
    ) -> Path:
        """
        Export manifest to JSON file.

        Args:
            polished: Polished features.
            bpm: Detected BPM.
            duration: Audio duration.
            output_path: Path for output JSON file.
            indent: JSON indentation level.

        Returns:
            Path to written file.
        """
        manifest = self.build_manifest(polished, bpm, duration)
        output_path = Path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=indent)

        return output_path

    def export_numpy(
        self,
        polished: PolishedFeatures,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Export features as NumPy .npz archive for faster loading.

        Args:
            polished: Polished features.
            output_path: Path for output .npz file.

        Returns:
            Path to written file.
        """
        output_path = Path(output_path)

        arrays: dict[str, Any] = dict(
            is_beat=polished.is_beat,
            is_onset=polished.is_onset,
            percussive_impact=polished.percussive_impact,
            harmonic_energy=polished.harmonic_energy,
            global_energy=polished.global_energy,
            spectral_flux=polished.spectral_flux,
            sub_bass=polished.sub_bass,
            bass=polished.bass,
            low_mid=polished.low_mid,
            mid=polished.mid,
            high_mid=polished.high_mid,
            presence=polished.presence,
            brilliance=polished.brilliance,
            low_energy=polished.low_energy,
            mid_energy=polished.mid_energy,
            high_energy=polished.high_energy,
            spectral_brightness=polished.spectral_brightness,
            spectral_flatness=polished.spectral_flatness,
            spectral_rolloff=polished.spectral_rolloff,
            zero_crossing_rate=polished.zero_crossing_rate,
            chroma=polished.chroma,
            dominant_chroma_indices=polished.dominant_chroma_indices,
            frame_times=polished.frame_times,
            fps=np.array([polished.fps]),
            n_frames=np.array([polished.n_frames]),
        )

        # Phase-1 arrays (include when available)
        for field in (
            # C1
            "mfcc", "mfcc_delta", "mfcc_delta2", "timbre_velocity",
            # C2
            "section_index", "section_novelty", "section_progress",
            "section_boundary_times",
            # C3
            "f0_hz", "f0_confidence", "pitch_velocity", "pitch_register",
            # C4
            "key_stability",
            # C5
            "beat_position", "bar_index", "bar_progress",
            # C6
            "sub_bass_cqt", "bass_cqt",
            # C7
            "spectral_bandwidth", "spectral_contrast",
            # W4
            "onset_sharpness",
        ):
            val = getattr(polished, field, None)
            if val is not None:
                arrays[field] = val

        np.savez_compressed(output_path, **arrays)

        return output_path

    def to_dict(
        self,
        polished: PolishedFeatures,
        bpm: float,
        duration: float,
    ) -> dict[str, Any]:
        """
        Return manifest as dictionary (for in-memory use).

        Args:
            polished: Polished features.
            bpm: Detected BPM.
            duration: Audio duration.

        Returns:
            Manifest dictionary.
        """
        return self.build_manifest(polished, bpm, duration)
