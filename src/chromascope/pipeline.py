"""
Main audio analysis pipeline.

Orchestrates the complete flow from audio file to visual driver manifest.
"""

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

    def __init__(
        self,
        target_fps: int = 60,
        sample_rate: int = 22050,
        hpss_margin: tuple[float, float] = (1.0, 1.0),
        impact_envelope: EnvelopeParams | None = None,
        energy_envelope: EnvelopeParams | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            target_fps: Target frames per second for output alignment.
            sample_rate: Audio sample rate (22050 is efficient for analysis).
            hpss_margin: Harmonic-Percussive separation aggressiveness.
            impact_envelope: Envelope for percussive signals.
            energy_envelope: Envelope for continuous energy signals.
        """
        self.target_fps = target_fps
        self.sample_rate = sample_rate

        self.decomposer = AudioDecomposer(margin=hpss_margin)
        self.analyzer = FeatureAnalyzer(target_fps=target_fps)
        self.polisher = SignalPolisher(
            fps=target_fps,
            impact_envelope=impact_envelope,
            energy_envelope=energy_envelope,
        )
        self.exporter = ManifestExporter()

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
    ) -> dict[str, Any]:
        """
        Run the complete pipeline from audio file to manifest.

        Args:
            audio_path: Path to input audio file.
            output_path: Path for output manifest. If None, only returns dict.
            format: Output format ("json" or "numpy").

        Returns:
            Dictionary containing manifest data and processing info.
        """
        audio_path = Path(audio_path)

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
