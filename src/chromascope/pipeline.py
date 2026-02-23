"""
Main audio analysis pipeline.

Orchestrates the complete flow from audio file to visual driver manifest.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Union

from chromascope.core.analyzer import ExtractedFeatures, FeatureAnalyzer
from chromascope.core.decomposer import AudioDecomposer, DecomposedAudio
from chromascope.core.polisher import (
    EnvelopeParams,
    PolishedFeatures,
    SignalPolisher,
)
from chromascope.io.exporter import ManifestExporter


class AudioPipeline:
    """
    Complete audio-to-manifest processing pipeline.

    Combines decomposition, feature extraction, signal polishing,
    and export into a single unified interface.
    """

    # Version of the analysis logic/schema.
    # Increment this whenever the feature extraction or polishing logic changes
    # to ensure that cached manifests are invalidated and re-generated.
    ANALYSIS_VERSION = "2.0"

    def __init__(
        self,
        target_fps: int = 60,
        sample_rate: int = 22050,
        hpss_margin: tuple[float, float] = (1.0, 1.0),
        impact_envelope: EnvelopeParams | None = None,
        energy_envelope: EnvelopeParams | None = None,
        use_cqt: bool = False,
        use_neural_beats: bool = False,
        use_demucs: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            target_fps: Target frames per second for output alignment.
            sample_rate: Audio sample rate (22050 is efficient for analysis).
            hpss_margin: Harmonic-Percussive separation aggressiveness.
            impact_envelope: Envelope for percussive signals.
            energy_envelope: Envelope for continuous energy signals.
            use_cqt: Enable CQT-derived sub-bass/bass bands (C6).
            use_neural_beats: Reserved — Phase 2 W2 madmom beat tracker.
            use_demucs: Reserved — Phase 2 W1 Demucs source separation.
        """
        self.target_fps = target_fps or 60
        self.sample_rate = sample_rate
        self.use_demucs = use_demucs

        self.decomposer = AudioDecomposer(margin=hpss_margin)
        self.analyzer = FeatureAnalyzer(
            target_fps=target_fps,
            use_cqt=use_cqt,
            use_neural_beats=use_neural_beats,
        )
        self.polisher = SignalPolisher(
            fps=target_fps,
            impact_envelope=impact_envelope,
            energy_envelope=energy_envelope,
        )
        self.exporter = ManifestExporter()

    def _get_cache_dir(self) -> Path:
        """Return the directory for caching manifests."""
        # Use ~/.cache/chromascope/manifests
        cache_dir = Path.home() / ".cache" / "chromascope" / "manifests"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of the file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_config_hash(self) -> str:
        """Calculate hash of the pipeline configuration."""
        # Extract envelope params safely
        impact = self.polisher.impact_envelope
        energy = self.polisher.energy_envelope
        
        config = {
            "version": self.ANALYSIS_VERSION,
            "fps": self.target_fps,
            "sr": self.sample_rate,
            "margin": self.decomposer.margin,
            "impact": (impact.attack_ms, impact.release_ms) if impact else None,
            "energy": (energy.attack_ms, energy.release_ms) if energy else None,
        }
        return hashlib.md5(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()

    def _get_cache_path(self, audio_path: Path) -> Path:
        """Get the cache file path for a given audio file."""
        file_hash = self._calculate_file_hash(audio_path)
        config_hash = self._get_config_hash()
        
        # Filename includes file content hash and config hash
        filename = f"manifest_{file_hash}_{config_hash}.json"
        return self._get_cache_dir() / filename

    def clear_cache(self):
        """Clear the manifest cache."""
        cache_dir = self._get_cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

    def decompose(self, audio_path: Union[str, Path]) -> DecomposedAudio:
        """
        Phase A: Load and decompose audio into harmonic/percussive.

        Args:
            audio_path: Path to audio file.

        Returns:
            DecomposedAudio with separated components.
        """
        return self.decomposer.decompose_file(audio_path, sr=self.sample_rate)

    def analyze(self, decomposed: DecomposedAudio) -> ExtractedFeatures:
        """
        Phase B: Extract visual driver features.

        Args:
            decomposed: Separated audio components.

        Returns:
            ExtractedFeatures with all raw features.
        """
        return self.analyzer.analyze(decomposed)

    def polish(self, features: ExtractedFeatures) -> PolishedFeatures:
        """
        Phase C: Apply smoothing and normalization.

        Args:
            features: Raw extracted features.

        Returns:
            PolishedFeatures ready for visualization.
        """
        return self.polisher.polish(features)

    def export(
        self,
        polished: PolishedFeatures,
        bpm: float,
        duration: float,
        output_path: Union[str, Path],
        format: str = "json",
    ) -> Path:
        """
        Phase D: Export to manifest file.

        Args:
            polished: Polished features.
            bpm: Detected BPM.
            duration: Audio duration.
            output_path: Output file path.
            format: "json" or "numpy".

        Returns:
            Path to written file.
        """
        if format == "numpy":
            return self.exporter.export_numpy(polished, output_path)
        return self.exporter.export_json(polished, bpm, duration, output_path)

    def process(
        self,
        audio_path: Union[str, Path],
        output_path: Union[str, Path] | None = None,
        format: str = "json",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Run the complete pipeline from audio file to manifest.

        Args:
            audio_path: Path to input audio file.
            output_path: Path for output manifest. If None, only returns dict.
            format: Output format ("json" or "numpy").
            use_cache: Whether to use cached manifest if available.

        Returns:
            Dictionary containing manifest data and processing info.
        """
        audio_path = Path(audio_path)

        # Caching logic
        if use_cache:
            try:
                cache_path = self._get_cache_path(audio_path)
                if cache_path.exists():
                    with open(cache_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)

                    # Reconstruct result dict
                    metadata = manifest.get("metadata", {})
                    print(f"Loaded analysis from cache: {cache_path}")

                    result = {
                        "manifest": manifest,
                        "bpm": metadata.get("bpm", 0.0),
                        "duration": metadata.get("duration", 0.0),
                        "n_frames": metadata.get("n_frames", 0),
                        "fps": self.target_fps,
                    }

                    if output_path:
                        # If output path requested, write the cached manifest to it
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(manifest, f, indent=2)
                        result["output_path"] = str(output_path)

                    return result
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-analyzing.")
                # Fall through to normal processing

        # Phase A: Decompose
        decomposed = self.decompose(audio_path)

        # Phase B: Analyze
        features = self.analyze(decomposed)

        # Phase C: Polish
        polished = self.polish(features)

        # Build manifest dict
        manifest = self.exporter.to_dict(
            polished,
            bpm=features.temporal.bpm,
            duration=decomposed.duration,
        )

        # Save to cache if enabled
        if use_cache:
            try:
                cache_path = self._get_cache_path(audio_path)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
            except Exception as e:
                print(f"Failed to save cache: {e}")

        # Phase D: Export if path provided
        result = {
            "manifest": manifest,
            "bpm": features.temporal.bpm,
            "duration": decomposed.duration,
            "n_frames": polished.n_frames,
            "fps": self.target_fps,
        }

        if output_path:
            written_path = self.export(
                polished,
                features.temporal.bpm,
                decomposed.duration,
                output_path,
                format,
            )
            result["output_path"] = str(written_path)

        return result

    def process_to_manifest(
        self,
        audio_path: Union[str, Path],
    ) -> dict[str, Any]:
        """
        Process audio and return manifest dictionary directly.

        Convenience method for in-memory usage without file output.

        Args:
            audio_path: Path to input audio file.

        Returns:
            Manifest dictionary.
        """
        result = self.process(audio_path)
        return result["manifest"]
