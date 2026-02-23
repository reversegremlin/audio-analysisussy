"""Tests for the ManifestExporter module."""

import json

import numpy as np
import pytest

from chromascope.core.analyzer import FeatureAnalyzer
from chromascope.core.decomposer import AudioDecomposer
from chromascope.core.polisher import SignalPolisher
from chromascope.io.exporter import ManifestExporter


class TestManifestExporter:
    """Tests for manifest serialization."""

    @pytest.fixture
    def polished_features(self, mixed_signal):
        """Get polished features for testing."""
        y, sr = mixed_signal
        decomposer = AudioDecomposer()
        decomposed = decomposer.separate(y, sr)
        analyzer = FeatureAnalyzer(target_fps=60)
        features = analyzer.analyze(decomposed)
        polisher = SignalPolisher(fps=60)
        return polisher.polish(features), features.temporal.bpm, decomposed.duration

    def test_build_manifest_structure(self, polished_features):
        """Manifest should have correct top-level structure."""
        polished, bpm, duration = polished_features
        exporter = ManifestExporter()

        manifest = exporter.build_manifest(polished, bpm, duration)

        assert "metadata" in manifest
        assert "frames" in manifest

    def test_metadata_fields(self, polished_features):
        """Metadata should contain required fields."""
        polished, bpm, duration = polished_features
        exporter = ManifestExporter()

        manifest = exporter.build_manifest(polished, bpm, duration)
        meta = manifest["metadata"]

        assert "bpm" in meta
        assert "duration" in meta
        assert "fps" in meta
        assert "n_frames" in meta
        assert "version" in meta

    def test_frame_structure(self, polished_features):
        """Each frame should have required fields."""
        polished, bpm, duration = polished_features
        exporter = ManifestExporter()

        manifest = exporter.build_manifest(polished, bpm, duration)
        frame = manifest["frames"][0]

        required_fields = [
            "frame_index",
            "time",
            "is_beat",
            "is_onset",
            "percussive_impact",
            "harmonic_energy",
            "global_energy",
            "spectral_flux",
            "sub_bass",
            "bass",
            "low_mid",
            "mid",
            "high_mid",
            "presence",
            "brilliance",
            "low_energy",
            "mid_energy",
            "high_energy",
            "spectral_brightness",
            "spectral_flatness",
            "spectral_rolloff",
            "zero_crossing_rate",
            "dominant_chroma",
            "chroma_values",
            "impact",
            "fluidity",
            "brightness",
            "pitch_hue",
            "texture",
            "sharpness",
        ]

        for field in required_fields:
            assert field in frame, f"Missing field: {field}"

    def test_frame_count_matches(self, polished_features):
        """Number of frames should match n_frames."""
        polished, bpm, duration = polished_features
        exporter = ManifestExporter()

        manifest = exporter.build_manifest(polished, bpm, duration)

        assert len(manifest["frames"]) == manifest["metadata"]["n_frames"]
        assert len(manifest["frames"]) == polished.n_frames

    def test_chroma_values_all_notes(self, polished_features):
        """chroma_values should have all 12 notes."""
        polished, bpm, duration = polished_features
        exporter = ManifestExporter()

        manifest = exporter.build_manifest(polished, bpm, duration)
        chroma = manifest["frames"][0]["chroma_values"]

        expected_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for note in expected_notes:
            assert note in chroma

    def test_dominant_chroma_is_note_name(self, polished_features):
        """dominant_chroma should be a valid note name."""
        polished, bpm, duration = polished_features
        exporter = ManifestExporter()

        manifest = exporter.build_manifest(polished, bpm, duration)

        valid_notes = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
        for frame in manifest["frames"]:
            assert frame["dominant_chroma"] in valid_notes

    def test_export_json(self, polished_features, tmp_path):
        """Should write valid JSON file."""
        polished, bpm, duration = polished_features
        exporter = ManifestExporter()

        output_path = tmp_path / "manifest.json"
        result_path = exporter.export_json(polished, bpm, duration, output_path)

        assert result_path.exists()

        # Should be valid JSON
        with open(result_path) as f:
            loaded = json.load(f)

        assert "metadata" in loaded
        assert "frames" in loaded

    def test_export_numpy(self, polished_features, tmp_path):
        """Should write valid NPZ file."""
        polished, bpm, duration = polished_features
        exporter = ManifestExporter()

        output_path = tmp_path / "manifest.npz"
        result_path = exporter.export_numpy(polished, output_path)

        assert result_path.exists()

        # Should load correctly
        data = np.load(result_path)
        assert "is_beat" in data
        assert "percussive_impact" in data
        assert "chroma" in data
        assert "sub_bass" in data
        assert "spectral_flux" in data

    def test_precision_parameter(self, polished_features):
        """Precision should limit decimal places."""
        polished, bpm, duration = polished_features
        exporter = ManifestExporter(precision=2)

        manifest = exporter.build_manifest(polished, bpm, duration)
        frame = manifest["frames"][0]

        # Check that values have at most 2 decimal places
        value = frame["percussive_impact"]
        rounded = round(value, 2)
        assert value == rounded

    def test_to_dict_matches_build_manifest(self, polished_features):
        """to_dict should return same as build_manifest."""
        polished, bpm, duration = polished_features
        exporter = ManifestExporter()

        manifest1 = exporter.build_manifest(polished, bpm, duration)
        manifest2 = exporter.to_dict(polished, bpm, duration)

        assert manifest1 == manifest2
