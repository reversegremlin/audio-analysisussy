"""Tests for the AudioPipeline module."""

import json

import pytest

from chromascope.core.decomposer import DecomposedAudio
from chromascope.core.analyzer import ExtractedFeatures
from chromascope.core.polisher import PolishedFeatures
from chromascope.pipeline import AudioPipeline


class TestAudioPipeline:
    """Tests for the complete pipeline."""

    def test_decompose_step(self, temp_audio_file):
        """decompose() should return DecomposedAudio."""
        pipeline = AudioPipeline(target_fps=60)
        result = pipeline.decompose(temp_audio_file)

        assert isinstance(result, DecomposedAudio)

    def test_analyze_step(self, temp_audio_file):
        """analyze() should return ExtractedFeatures."""
        pipeline = AudioPipeline(target_fps=60)
        decomposed = pipeline.decompose(temp_audio_file)
        result = pipeline.analyze(decomposed)

        assert isinstance(result, ExtractedFeatures)

    def test_polish_step(self, temp_audio_file):
        """polish() should return PolishedFeatures."""
        pipeline = AudioPipeline(target_fps=60)
        decomposed = pipeline.decompose(temp_audio_file)
        features = pipeline.analyze(decomposed)
        result = pipeline.polish(features)

        assert isinstance(result, PolishedFeatures)

    def test_process_full_pipeline(self, temp_audio_file):
        """process() should run complete pipeline."""
        pipeline = AudioPipeline(target_fps=60)
        result = pipeline.process(temp_audio_file)

        assert "manifest" in result
        assert "bpm" in result
        assert "duration" in result
        assert "n_frames" in result
        assert "fps" in result

    def test_process_with_json_output(self, temp_audio_file, tmp_path):
        """process() should write JSON when output_path provided."""
        pipeline = AudioPipeline(target_fps=60)
        output_path = tmp_path / "output.json"

        result = pipeline.process(temp_audio_file, output_path=output_path)

        assert "output_path" in result
        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)
        assert "metadata" in loaded

    def test_process_with_numpy_output(self, temp_audio_file, tmp_path):
        """process() should write NPZ when format=numpy."""
        pipeline = AudioPipeline(target_fps=60)
        output_path = tmp_path / "output.npz"

        result = pipeline.process(
            temp_audio_file,
            output_path=output_path,
            format="numpy",
        )

        assert output_path.exists()

    def test_process_to_manifest(self, temp_audio_file):
        """process_to_manifest() should return manifest dict."""
        pipeline = AudioPipeline(target_fps=60)
        manifest = pipeline.process_to_manifest(temp_audio_file)

        assert "metadata" in manifest
        assert "frames" in manifest

    def test_custom_fps(self, temp_audio_file):
        """Pipeline should respect custom FPS."""
        pipeline = AudioPipeline(target_fps=30)
        result = pipeline.process(temp_audio_file)

        assert result["fps"] == 30
        assert result["manifest"]["metadata"]["fps"] == 30

    def test_manifest_frame_count(self, temp_audio_file):
        """Manifest should have correct number of frames."""
        pipeline = AudioPipeline(target_fps=60)
        result = pipeline.process(temp_audio_file)

        manifest = result["manifest"]
        assert len(manifest["frames"]) == manifest["metadata"]["n_frames"]

    def test_manifest_frames_sequential(self, temp_audio_file):
        """Frame indices should be sequential."""
        pipeline = AudioPipeline(target_fps=60)
        manifest = pipeline.process_to_manifest(temp_audio_file)

        for i, frame in enumerate(manifest["frames"]):
            assert frame["frame_index"] == i

    def test_manifest_times_increasing(self, temp_audio_file):
        """Frame times should be monotonically increasing."""
        pipeline = AudioPipeline(target_fps=60)
        manifest = pipeline.process_to_manifest(temp_audio_file)

        times = [f["time"] for f in manifest["frames"]]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]

    def test_values_in_range(self, temp_audio_file):
        """All normalized values should be in [0, 1]."""
        pipeline = AudioPipeline(target_fps=60)
        manifest = pipeline.process_to_manifest(temp_audio_file)

        value_keys = [
            "percussive_impact",
            "harmonic_energy",
            "global_energy",
            "low_energy",
            "mid_energy",
            "high_energy",
            "spectral_brightness",
        ]

        for frame in manifest["frames"]:
            for key in value_keys:
                assert 0.0 <= frame[key] <= 1.0, f"{key} out of range: {frame[key]}"
