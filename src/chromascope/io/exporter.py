"""
Manifest serialization module.

Exports polished audio features to JSON format aligned to target FPS
for use in rendering engines and visualization systems.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

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
    version: str = "1.0"


class ManifestExporter:
    """
    Exports polished features to JSON manifest format.

    The manifest follows the schema defined in the architecture doc,
    with each frame containing all visual driver values.
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

        return {
            "frame_index": index,
            "time": self._round(polished.frame_times[index]),
            "is_beat": bool(polished.is_beat[index]),
            "is_onset": bool(polished.is_onset[index]),
            "percussive_impact": self._round(polished.percussive_impact[index]),
            "harmonic_energy": self._round(polished.harmonic_energy[index]),
            "global_energy": self._round(polished.global_energy[index]),
            "low_energy": self._round(polished.low_energy[index]),
            "mid_energy": self._round(polished.mid_energy[index]),
            "high_energy": self._round(polished.high_energy[index]),
            "spectral_brightness": self._round(polished.spectral_brightness[index]),
            "dominant_chroma": dominant_chroma,
            "chroma_values": {
                FeatureAnalyzer.CHROMA_NAMES[i]: self._round(polished.chroma[i, index])
                for i in range(12)
            },
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

        return {
            "metadata": {
                "bpm": metadata.bpm,
                "duration": metadata.duration,
                "fps": metadata.fps,
                "n_frames": metadata.n_frames,
                "version": metadata.version,
            },
            "frames": frames,
        }

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

        np.savez_compressed(
            output_path,
            is_beat=polished.is_beat,
            is_onset=polished.is_onset,
            percussive_impact=polished.percussive_impact,
            harmonic_energy=polished.harmonic_energy,
            global_energy=polished.global_energy,
            low_energy=polished.low_energy,
            mid_energy=polished.mid_energy,
            high_energy=polished.high_energy,
            spectral_brightness=polished.spectral_brightness,
            chroma=polished.chroma,
            dominant_chroma_indices=polished.dominant_chroma_indices,
            frame_times=polished.frame_times,
            fps=polished.fps,
            n_frames=polished.n_frames,
        )

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
