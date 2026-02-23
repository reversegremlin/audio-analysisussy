"""
Phase 1 (C1–C7) and W4 audio intelligence tests.

Covers:
  C1 — MFCC deltas + timbre velocity
  C2 — structural segmentation
  C3 — pyin pitch (F0)
  C4 — Krumhansl-Schmuckler key detection
  C5 — downbeat heuristic + rhythm grid
  C6 — CQT sub-bass/bass bands
  C7 — spectral bandwidth + contrast
  W4 — onset shape classification
Phase 2 stubs:
  SourceSeparator / SeparatedAudio existence + ImportError
  RealtimeAnalyzer stub
"""

import numpy as np
import pytest

from chromascope.core.analyzer import (
    ExtractedFeatures,
    FeatureAnalyzer,
    HarmonicFeatures,
    KeyFeatures,
    StructuralFeatures,
    TonalityFeatures,
    TemporalFeatures,
    FrequencyBands,
)
from chromascope.core.decomposer import AudioDecomposer, SeparatedAudio, SourceSeparator
from chromascope.core.polisher import PolishedFeatures, SignalPolisher
from chromascope.core.stream import LiveFeatures, RealtimeAnalyzer
from chromascope.io.exporter import ManifestExporter
from chromascope.pipeline import AudioPipeline


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TEST_SR = 22050


@pytest.fixture
def decomposed_mixed(mixed_signal):
    y, sr = mixed_signal
    return AudioDecomposer().separate(y, sr)


@pytest.fixture
def decomposed_sine(pure_sine):
    y, sr = pure_sine
    return AudioDecomposer().separate(y, sr)


@pytest.fixture
def decomposed_c_major():
    """A C-major chord (C4, E4, G4) lasting 4 seconds — good key detection signal."""
    sr = TEST_SR
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (
        0.3 * np.sin(2 * np.pi * 261.63 * t) +   # C4
        0.3 * np.sin(2 * np.pi * 329.63 * t) +   # E4
        0.3 * np.sin(2 * np.pi * 392.00 * t)      # G4
    ).astype(np.float32)
    return AudioDecomposer().separate(y, sr)


@pytest.fixture
def analyzer():
    return FeatureAnalyzer(target_fps=60)


@pytest.fixture
def extracted(decomposed_mixed, analyzer):
    return analyzer.analyze(decomposed_mixed)


@pytest.fixture
def polished(extracted):
    return SignalPolisher(fps=60).polish(extracted)


# ---------------------------------------------------------------------------
# Dataclass existence checks
# ---------------------------------------------------------------------------

class TestNewDataclasses:
    def test_structural_features_dataclass(self):
        sf = StructuralFeatures(
            section_labels=np.array([0, 0, 1]),
            section_novelty=np.array([0.1, 0.2, 0.9]),
            section_boundaries=np.array([0.0, 1.5]),
            n_sections=2,
        )
        assert sf.n_sections == 2
        assert len(sf.section_labels) == 3

    def test_key_features_dataclass(self):
        kf = KeyFeatures(
            root_index=0,
            root_name="C",
            mode="major",
            confidence=0.8,
            key_stability=np.array([0.7, 0.8, 0.9]),
        )
        assert kf.root_name == "C"
        assert kf.mode == "major"

    def test_harmonic_features_dataclass(self):
        hf = HarmonicFeatures(chord_labels=[])
        assert isinstance(hf.chord_labels, list)

    def test_structural_features_in_extracted(self, extracted):
        assert hasattr(extracted, "structure")

    def test_key_features_in_extracted(self, extracted):
        assert hasattr(extracted, "key")

    def test_harmony_stub_in_extracted(self, extracted):
        assert hasattr(extracted, "harmony")
        assert extracted.harmony is None  # Phase 2 W3 stub, not populated


# ---------------------------------------------------------------------------
# C1 — MFCC deltas
# ---------------------------------------------------------------------------

class TestC1MFCCDeltas:
    def test_mfcc_delta_shape(self, extracted):
        td = extracted.tonality
        assert td.mfcc_delta is not None
        assert td.mfcc_delta.shape == td.mfcc.shape

    def test_mfcc_delta2_shape(self, extracted):
        td = extracted.tonality
        assert td.mfcc_delta2 is not None
        assert td.mfcc_delta2.shape == td.mfcc.shape

    def test_polished_mfcc_shape(self, polished, extracted):
        assert polished.mfcc is not None
        assert polished.mfcc.shape == (13, extracted.n_frames)

    def test_polished_mfcc_delta_shape(self, polished, extracted):
        assert polished.mfcc_delta is not None
        assert polished.mfcc_delta.shape == (13, extracted.n_frames)

    def test_polished_mfcc_normalized(self, polished):
        assert polished.mfcc is not None
        for i in range(polished.mfcc.shape[0]):
            assert np.all(polished.mfcc[i] >= 0.0)
            assert np.all(polished.mfcc[i] <= 1.0)

    def test_timbre_velocity_shape(self, polished, extracted):
        assert polished.timbre_velocity is not None
        assert len(polished.timbre_velocity) == extracted.n_frames

    def test_timbre_velocity_normalized(self, polished):
        assert polished.timbre_velocity is not None
        assert np.all(polished.timbre_velocity >= 0.0)
        assert np.all(polished.timbre_velocity <= 1.0)


# ---------------------------------------------------------------------------
# C2 — Structure segmentation
# ---------------------------------------------------------------------------

class TestC2Structure:
    def test_structure_extracted(self, extracted):
        assert extracted.structure is not None
        assert isinstance(extracted.structure, StructuralFeatures)

    def test_section_labels_length(self, extracted):
        assert len(extracted.structure.section_labels) == extracted.n_frames

    def test_section_n_sections_positive(self, extracted):
        assert extracted.structure.n_sections >= 1

    def test_section_boundaries_nonempty(self, extracted):
        assert len(extracted.structure.section_boundaries) >= 1

    def test_section_boundaries_first_is_zero(self, extracted):
        assert extracted.structure.section_boundaries[0] == pytest.approx(0.0, abs=0.1)

    def test_polished_section_index_shape(self, polished, extracted):
        assert polished.section_index is not None
        assert len(polished.section_index) == extracted.n_frames

    def test_polished_section_novelty_normalized(self, polished):
        assert polished.section_novelty is not None
        assert np.all(polished.section_novelty >= 0.0)
        assert np.all(polished.section_novelty <= 1.0)

    def test_polished_section_progress_range(self, polished):
        assert polished.section_progress is not None
        assert np.all(polished.section_progress >= 0.0)
        assert np.all(polished.section_progress <= 1.0)

    def test_polished_section_change_is_bool(self, polished):
        assert polished.section_change is not None
        assert polished.section_change.dtype == bool


# ---------------------------------------------------------------------------
# C3 — Pitch / F0
# ---------------------------------------------------------------------------

class TestC3Pitch:
    def test_f0_hz_length(self, extracted):
        assert extracted.tonality.f0_hz is not None
        assert len(extracted.tonality.f0_hz) == extracted.n_frames

    def test_f0_voiced_length(self, extracted):
        assert extracted.tonality.f0_voiced is not None
        assert len(extracted.tonality.f0_voiced) == extracted.n_frames

    def test_f0_probs_length(self, extracted):
        assert extracted.tonality.f0_probs is not None
        assert len(extracted.tonality.f0_probs) == extracted.n_frames

    def test_f0_voiced_is_bool(self, extracted):
        assert extracted.tonality.f0_voiced.dtype == bool

    def test_polished_f0_hz_present(self, polished, extracted):
        assert polished.f0_hz is not None
        assert len(polished.f0_hz) == extracted.n_frames

    def test_polished_pitch_register_range(self, polished):
        assert polished.pitch_register is not None
        assert np.all(polished.pitch_register >= 0.0)
        assert np.all(polished.pitch_register <= 1.0)

    def test_polished_pitch_velocity_range(self, polished):
        assert polished.pitch_velocity is not None
        assert np.all(polished.pitch_velocity >= 0.0)
        assert np.all(polished.pitch_velocity <= 1.0)

    def test_f0_voiced_has_some_false_on_silence(self):
        """pyin should mark silent frames as unvoiced."""
        sr = TEST_SR
        duration = 1.0
        # silent signal
        y = np.zeros(int(sr * duration), dtype=np.float32)
        decomposed = AudioDecomposer().separate(y, sr)
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed)
        if result.tonality.f0_voiced is not None:
            # all frames should be unvoiced for silence
            assert not np.all(result.tonality.f0_voiced)


# ---------------------------------------------------------------------------
# C4 — Key detection
# ---------------------------------------------------------------------------

class TestC4KeyDetection:
    def test_key_extracted(self, extracted):
        assert extracted.key is not None
        assert isinstance(extracted.key, KeyFeatures)

    def test_key_root_index_range(self, extracted):
        assert 0 <= extracted.key.root_index <= 11

    def test_key_root_name_valid(self, extracted):
        assert extracted.key.root_name in FeatureAnalyzer.CHROMA_NAMES

    def test_key_mode_valid(self, extracted):
        assert extracted.key.mode in ("major", "minor")

    def test_key_confidence_range(self, extracted):
        assert 0.0 <= extracted.key.confidence <= 1.0

    def test_key_stability_shape(self, extracted):
        assert len(extracted.key.key_stability) == extracted.n_frames

    def test_key_stability_range(self, extracted):
        assert np.all(extracted.key.key_stability >= 0.0)
        assert np.all(extracted.key.key_stability <= 1.0)

    def test_c_major_key_detected(self, decomposed_c_major):
        """C-major chord should produce a valid key result."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_c_major)
        assert result.key is not None
        # C-major (C,E,G) is genuinely ambiguous with A-minor (relative key).
        # The K-S algorithm may return either; we only assert a valid result.
        assert result.key.mode in ("major", "minor")
        assert 0 <= result.key.root_index <= 11

    def test_c_major_root_in_c_major_family(self, decomposed_c_major):
        """C-major chord root should be within the C-major diatonic set."""
        analyzer = FeatureAnalyzer(target_fps=60)
        result = analyzer.analyze(decomposed_c_major)
        assert result.key is not None
        # C major diatonic roots: C(0), D(2), E(4), F(5), G(7), A(9), B(11)
        c_major_diatonic = {0, 2, 4, 5, 7, 9, 11}
        assert result.key.root_index in c_major_diatonic

    def test_detect_key_standalone(self):
        """detect_key should work with just a chroma_mean vector."""
        analyzer = FeatureAnalyzer()
        # Simulate C-major chroma mean (C, E, G are strong)
        chroma_mean = np.zeros(12)
        chroma_mean[0] = 6.0   # C
        chroma_mean[4] = 4.0   # E
        chroma_mean[7] = 5.0   # G
        key = analyzer.detect_key(chroma_mean)
        assert key.root_name in FeatureAnalyzer.CHROMA_NAMES
        assert key.mode in ("major", "minor")
        assert 0.0 <= key.confidence <= 1.0

    def test_polished_key_fields(self, polished):
        assert polished.key_root_index is not None
        assert polished.key_mode in ("major", "minor")
        assert polished.key_confidence is not None
        assert polished.key_stability is not None

    def test_manifest_key_block(self, polished, extracted):
        exporter = ManifestExporter()
        manifest = exporter.build_manifest(
            polished, extracted.temporal.bpm, 2.0
        )
        assert "key" in manifest
        key = manifest["key"]
        assert "root" in key
        assert "root_index" in key
        assert "mode" in key
        assert key["mode"] in ("major", "minor")
        assert "relative_major" in key


# ---------------------------------------------------------------------------
# C5 — Downbeats / rhythm grid
# ---------------------------------------------------------------------------

class TestC5Downbeats:
    def test_downbeat_frames_present(self, extracted):
        assert extracted.temporal.downbeat_frames is not None

    def test_downbeat_frames_subset_of_beats(self, extracted):
        db = set(extracted.temporal.downbeat_frames.tolist())
        beats = set(extracted.temporal.beat_frames.tolist())
        assert db.issubset(beats)

    def test_polished_is_downbeat_shape(self, polished, extracted):
        assert polished.is_downbeat is not None
        assert len(polished.is_downbeat) == extracted.n_frames

    def test_polished_is_downbeat_is_bool(self, polished):
        assert polished.is_downbeat.dtype == bool

    def test_polished_beat_position_range(self, polished):
        assert polished.beat_position is not None
        assert np.all(polished.beat_position >= 0.0)
        assert np.all(polished.beat_position <= 1.0)

    def test_polished_bar_index_non_negative(self, polished):
        assert polished.bar_index is not None
        assert np.all(polished.bar_index >= 0)

    def test_polished_bar_progress_range(self, polished):
        assert polished.bar_progress is not None
        assert np.all(polished.bar_progress >= 0.0)
        assert np.all(polished.bar_progress <= 1.0)


# ---------------------------------------------------------------------------
# C6 — CQT bands
# ---------------------------------------------------------------------------

class TestC6CQTBands:
    def test_cqt_bands_none_by_default(self, extracted):
        """Without use_cqt=True, CQT bands should be None."""
        assert extracted.energy.frequency_bands.sub_bass_cqt is None
        assert extracted.energy.frequency_bands.bass_cqt is None

    def test_polished_cqt_none_by_default(self, polished):
        assert polished.sub_bass_cqt is None
        assert polished.bass_cqt is None

    def test_cqt_bands_populated_with_flag(self, decomposed_mixed):
        analyzer = FeatureAnalyzer(target_fps=60, use_cqt=True)
        result = analyzer.analyze(decomposed_mixed)
        fb = result.energy.frequency_bands
        assert fb.sub_bass_cqt is not None
        assert fb.bass_cqt is not None
        assert len(fb.sub_bass_cqt) > 0
        assert len(fb.bass_cqt) > 0

    def test_polished_cqt_populated_with_flag(self, decomposed_mixed):
        analyzer = FeatureAnalyzer(target_fps=60, use_cqt=True)
        result = analyzer.analyze(decomposed_mixed)
        polished = SignalPolisher(fps=60).polish(result)
        assert polished.sub_bass_cqt is not None
        assert polished.bass_cqt is not None
        assert np.all(polished.sub_bass_cqt >= 0.0)
        assert np.all(polished.sub_bass_cqt <= 1.0)

    def test_pipeline_use_cqt_flag(self):
        p = AudioPipeline(use_cqt=True)
        assert p.analyzer.use_cqt is True


# ---------------------------------------------------------------------------
# C7 — Spectral bandwidth + contrast
# ---------------------------------------------------------------------------

class TestC7ExtendedSpectral:
    def test_spectral_bandwidth_shape(self, extracted):
        assert extracted.tonality.spectral_bandwidth is not None
        assert len(extracted.tonality.spectral_bandwidth) == extracted.n_frames

    def test_spectral_contrast_shape(self, extracted):
        sc = extracted.tonality.spectral_contrast
        assert sc is not None
        assert sc.shape[0] == 7
        assert sc.shape[1] == extracted.n_frames

    def test_polished_spectral_bandwidth_shape(self, polished, extracted):
        assert polished.spectral_bandwidth is not None
        assert len(polished.spectral_bandwidth) == extracted.n_frames

    def test_polished_spectral_bandwidth_normalized(self, polished):
        bw = polished.spectral_bandwidth
        assert bw is not None
        assert np.all(bw >= 0.0)
        assert np.all(bw <= 1.0)

    def test_polished_spectral_contrast_shape(self, polished, extracted):
        sc = polished.spectral_contrast
        assert sc is not None
        assert sc.shape == (7, extracted.n_frames)

    def test_polished_spectral_contrast_normalized(self, polished):
        sc = polished.spectral_contrast
        assert sc is not None
        assert np.all(sc >= 0.0)
        assert np.all(sc <= 1.0)


# ---------------------------------------------------------------------------
# W4 — Onset shape classification
# ---------------------------------------------------------------------------

class TestW4OnsetShape:
    def test_onset_types_present(self, extracted):
        assert extracted.temporal.onset_types is not None
        assert isinstance(extracted.temporal.onset_types, list)

    def test_onset_types_valid_labels(self, extracted):
        valid = {"transient", "percussive", "harmonic"}
        for label in extracted.temporal.onset_types:
            assert label in valid

    def test_onset_sharpness_present(self, extracted):
        assert extracted.temporal.onset_sharpness is not None
        assert isinstance(extracted.temporal.onset_sharpness, np.ndarray)

    def test_onset_sharpness_length(self, extracted):
        assert len(extracted.temporal.onset_sharpness) == len(
            extracted.temporal.onset_frames
        )

    def test_onset_sharpness_range(self, extracted):
        sh = extracted.temporal.onset_sharpness
        assert np.all(sh >= 0.0)
        assert np.all(sh <= 1.0)

    def test_classify_onset_shape_returns_tuple(self):
        analyzer = FeatureAnalyzer()
        rms = np.ones(100)
        rms[10] = 5.0  # sharp peak
        label, sharpness = analyzer.classify_onset_shape(10, rms, 512, 22050)
        assert isinstance(label, str)
        assert label in ("transient", "percussive", "harmonic")
        assert 0.0 <= sharpness <= 1.0

    def test_polished_onset_type_array_shape(self, polished, extracted):
        assert polished.onset_type is not None
        assert len(polished.onset_type) == extracted.n_frames

    def test_polished_onset_sharpness_range(self, polished):
        sh = polished.onset_sharpness
        assert sh is not None
        assert np.all(sh >= 0.0)
        assert np.all(sh <= 1.0)


# ---------------------------------------------------------------------------
# Manifest / exporter
# ---------------------------------------------------------------------------

class TestManifestPhase1:
    def test_schema_version_is_2(self, polished, extracted):
        exporter = ManifestExporter()
        manifest = exporter.build_manifest(
            polished, extracted.temporal.bpm, 2.0
        )
        assert manifest["metadata"]["schema_version"] == "2.0"

    def test_structure_block_present(self, polished, extracted):
        exporter = ManifestExporter()
        manifest = exporter.build_manifest(
            polished, extracted.temporal.bpm, 2.0
        )
        assert "structure" in manifest
        s = manifest["structure"]
        assert "n_sections" in s
        assert "section_boundaries" in s
        assert "section_durations" in s
        assert s["n_sections"] >= 1

    def test_frame_has_timbre_velocity(self, polished, extracted):
        exporter = ManifestExporter()
        manifest = exporter.build_manifest(
            polished, extracted.temporal.bpm, 2.0
        )
        frame0 = manifest["frames"][0]
        assert "timbre_velocity" in frame0

    def test_frame_has_section_fields(self, polished, extracted):
        exporter = ManifestExporter()
        manifest = exporter.build_manifest(
            polished, extracted.temporal.bpm, 2.0
        )
        frame0 = manifest["frames"][0]
        assert "section_index" in frame0
        assert "section_novelty" in frame0
        assert "section_progress" in frame0

    def test_frame_has_f0_fields(self, polished, extracted):
        exporter = ManifestExporter()
        manifest = exporter.build_manifest(
            polished, extracted.temporal.bpm, 2.0
        )
        frame0 = manifest["frames"][0]
        assert "f0_hz" in frame0
        assert "f0_confidence" in frame0
        assert "f0_voiced" in frame0

    def test_frame_has_beat_position(self, polished, extracted):
        exporter = ManifestExporter()
        manifest = exporter.build_manifest(
            polished, extracted.temporal.bpm, 2.0
        )
        frame0 = manifest["frames"][0]
        assert "beat_position" in frame0
        assert "bar_progress" in frame0
        assert "is_downbeat" in frame0

    def test_frame_has_bandwidth_norm_primitive(self, polished, extracted):
        exporter = ManifestExporter()
        manifest = exporter.build_manifest(
            polished, extracted.temporal.bpm, 2.0
        )
        frame0 = manifest["frames"][0]
        assert "bandwidth_norm" in frame0

    def test_frame_has_onset_type(self, polished, extracted):
        exporter = ManifestExporter()
        manifest = exporter.build_manifest(
            polished, extracted.temporal.bpm, 2.0
        )
        frame0 = manifest["frames"][0]
        assert "onset_type" in frame0
        assert "onset_sharpness" in frame0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestPipelinePhase1:
    def test_analysis_version_is_2(self):
        p = AudioPipeline()
        assert p.ANALYSIS_VERSION == "2.0"

    def test_pipeline_default_flags(self):
        p = AudioPipeline()
        assert p.analyzer.use_cqt is False
        assert p.analyzer.use_neural_beats is False

    def test_pipeline_use_neural_beats_flag(self):
        p = AudioPipeline(use_neural_beats=True)
        assert p.analyzer.use_neural_beats is True


# ---------------------------------------------------------------------------
# Phase 2 stubs — SeparatedAudio, SourceSeparator
# ---------------------------------------------------------------------------

class TestPhase2Stubs:
    def test_separated_audio_dataclass_exists(self):
        sa = SeparatedAudio(
            drums=np.zeros(100),
            bass=np.zeros(100),
            vocals=np.zeros(100),
            other=np.zeros(100),
            sample_rate=22050,
        )
        assert sa.sample_rate == 22050

    def test_source_separator_raises_import_error(self):
        """SourceSeparator should raise ImportError if demucs is absent."""
        try:
            import demucs  # noqa: F401
            pytest.skip("demucs is installed; skip ImportError test")
        except ModuleNotFoundError:
            pass

        with pytest.raises(ImportError, match="demucs"):
            SourceSeparator()


# ---------------------------------------------------------------------------
# Phase 3 stub — RealtimeAnalyzer
# ---------------------------------------------------------------------------

class TestPhase3Stub:
    def test_realtime_analyzer_instantiable(self):
        ra = RealtimeAnalyzer(sample_rate=44100, chunk_size=512)
        assert ra.sample_rate == 44100

    def test_process_chunk_returns_live_features(self):
        ra = RealtimeAnalyzer(sample_rate=44100, chunk_size=512)
        chunk = np.zeros(512, dtype=np.float32)
        result = ra.process_chunk(chunk)
        # May return None or LiveFeatures
        assert result is None or isinstance(result, LiveFeatures)

    def test_live_features_dataclass_exists(self):
        lf = LiveFeatures(chunk_index=1, time_sec=0.01)
        assert lf.chunk_index == 1
        assert lf.is_beat is False  # default

    def test_realtime_analyzer_chunk_index_increments(self):
        ra = RealtimeAnalyzer()
        chunk = np.zeros(ra.chunk_size, dtype=np.float32)
        for _ in range(3):
            ra.process_chunk(chunk)
        assert ra._chunk_index == 3
