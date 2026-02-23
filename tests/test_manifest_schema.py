"""Tests for manifest schema versioning and visual primitives."""

from chromascope.core.decomposer import AudioDecomposer
from chromascope.core.analyzer import FeatureAnalyzer
from chromascope.core.polisher import SignalPolisher
from chromascope.io.exporter import ManifestExporter


def _build_manifest_from_mixed_signal(mixed_signal):
    """Helper to run the full analysis stack on the mixed_signal fixture."""
    y, sr = mixed_signal
    decomposer = AudioDecomposer()
    decomposed = decomposer.separate(y, sr)

    analyzer = FeatureAnalyzer(target_fps=60)
    features = analyzer.analyze(decomposed)

    polisher = SignalPolisher(fps=60)
    polished = polisher.polish(features)

    exporter = ManifestExporter()
    manifest = exporter.build_manifest(
        polished,
        bpm=features.temporal.bpm,
        duration=decomposed.duration,
    )
    return manifest, polished


def test_metadata_includes_schema_version(mixed_signal):
    """Metadata should include an explicit schema_version field."""
    manifest, _ = _build_manifest_from_mixed_signal(mixed_signal)
    meta = manifest["metadata"]

    assert "schema_version" in meta
    # Phase-1 audio intelligence bumped the schema to 2.0
    assert isinstance(meta["schema_version"], str)
    assert meta["schema_version"] in ("1.1", "2.0") or meta["schema_version"].startswith("2.")


def test_frame_includes_visual_primitives(mixed_signal):
    """
    Each frame should expose high-level visual primitives derived
    from the underlying polished features.
    """
    manifest, _ = _build_manifest_from_mixed_signal(mixed_signal)
    frame = manifest["frames"][0]

    # Primitive fields should be present
    for field in ("impact", "fluidity", "brightness", "pitch_hue", "texture"):
        assert field in frame, f"Missing primitive field: {field}"

    # Primitives should be consistent with underlying fields
    assert frame["impact"] == frame["percussive_impact"]
    assert frame["fluidity"] == frame["harmonic_energy"]
    assert frame["brightness"] == frame["spectral_brightness"]

    # pitch_hue should be normalized to [0.0, 1.0]
    assert 0.0 <= frame["pitch_hue"] <= 1.0

    # texture should live in [0.0, 1.0] and relate to mid/high bands
    assert 0.0 <= frame["texture"] <= 1.0

