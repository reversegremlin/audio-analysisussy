# Audio Intelligence Proposal
## From Signal Statistics to Musical Understanding

**Status:** Proposal
**Date:** 2026-02-22
**Scope:** `src/chromascope/core/` pipeline + manifest schema + Ableton integration path

---

## Executive Summary

The current pipeline is excellent at measuring **what the signal is doing** — it reliably extracts RMS energy, frequency bands, spectral shape, and basic beat timing. What it cannot do is understand **what the music is doing**. The gap between "sub-bass RMS is 0.72" and "the kick just dropped on beat 1 of the chorus in the key of F minor" is the gap between a visualizer that reacts and one that *understands*.

This proposal defines three phases — **Crawl, Walk, Run** — to close that gap progressively. Each phase is independently deployable with no breaking changes to the existing manifest schema (new fields are additive). The total outcome is a pipeline that extracts musical structure, harmony, melody contour, source identity, emotion, and eventually streams into Ableton in real time.

---

## Current State Snapshot

### Pipeline Stages
| Stage | File | What It Does |
|---|---|---|
| Decompose | `core/decomposer.py` | Loads audio at 22050 Hz mono → HPSS → harmonic + percussive |
| Analyze | `core/analyzer.py` | Extracts 7-band RMS, chroma (12), centroid, flatness, rolloff, ZCR, MFCC(13), beats, onsets |
| Polish | `core/polisher.py` | Normalizes → attack/release envelopes → 0–1 scalars |
| Export | `io/exporter.py` | Writes per-frame JSON; computes 6 derived "primitives" |

### What the Manifest Currently Carries (per frame)
```
Energy:    percussive_impact, harmonic_energy, global_energy, spectral_flux
Frequency: sub_bass, bass, low_mid, mid, high_mid, presence, brilliance
Texture:   spectral_brightness, spectral_flatness, spectral_rolloff, zero_crossing_rate
Tonality:  chroma_values (12 bins), dominant_chroma
Triggers:  is_beat, is_onset
Primitives: impact, fluidity, brightness, pitch_hue, texture, sharpness
```

### Known Gaps in the Current Implementation

1. **MFCCs computed but never exported** — `analyzer.py:365` computes 13 MFCC coefficients; `polisher.py` never touches them; `exporter.py` never writes them.
2. **Primitives over-collapse information** — `texture = (flatness + zcr + presence + brilliance) / 4.0` averages four unrelated measurements into one number. The individual signals are more valuable.
3. **Sub-bass resolution is critically coarse** — at sr=22050, n_fft=2048, frequency bin width ≈ 10.8 Hz. The entire 20–60 Hz sub-bass band contains ~4 FFT bins.
4. **No song structure** — no section labels, no novelty curve, no notion of verse/chorus/drop.
5. **No pitch tracking** — chroma gives pitch class energy but no fundamental frequency, no melodic direction.
6. **No key or chord detection** — we know "C and G are loud this frame" but not "this is a G7 resolving to C major."
7. **No downbeat tracking** — beats are tracked but not grouped into bars; beat 1 is musically the most important moment.
8. **Stereo information discarded** — `mono=True` in `decomposer.py:67`. Width, panning, and mid/side content are lost before analysis begins.
9. **Source identity unknown** — kick drum and synth bass both show up in sub-bass/bass bands; they are visually indistinguishable.

---

## The Format Question: MP3 / WAV / Raw

### What MP3 Costs You

MP3's MDCT codec operates on 576–1152 sample windows (~26–52 ms at 44100 Hz). These windows **temporally smear transients** before your onset detector or beat tracker ever sees the audio. A sharp snare hit becomes a slightly blurred transient. This degrades beat and onset detection accuracy — measurably, not negligibly.

Additional costs:
- Pre-echo artifacts (ringing before transients) can create phantom onsets
- Low-frequency phase distortion affects the sub-bass analysis
- High-frequency rolloff (aggressive at 128kbps, acceptable at 320kbps) is mostly moot since you downsample to 22050 Hz anyway — but the transient smearing in time domain is not

### Recommendation by Format

| Format | Quality | Notes |
|---|---|---|
| WAV / AIFF | Best | No codec damage. Use for analysis when possible. |
| FLAC | Best | Lossless compressed. Same analysis quality as WAV. |
| MP3 320kbps | Good | Acceptable. Most damage above 16 kHz which you discard. Transient smearing is real but modest. |
| MP3 128kbps | Degraded | Noticeable transient smearing, high-frequency rolloff to ~16 kHz. Avoid for analysis. |
| AAC 256kbps | Good | Better psychoacoustic model than MP3; less transient damage at equivalent bitrate. |

### For Ableton Integration

When wired into Max for Live, you read directly from Ableton's audio buffer as float32 PCM at the project's native sample rate (44100 or 48000 Hz). No codec. This is the cleanest possible source — all of the format question becomes irrelevant. The real challenge at that point is **latency**, not quality.

**Interim recommendation:** move to WAV or FLAC for your test audio now. The gain on beat detection alone is worth it.

---

## The Core Gap: Signal Statistics vs. Musical Understanding

Here is the fundamental framing for all three phases:

```
Current pipeline asks:  "How much energy is in the 60–250 Hz band right now?"
Musical understanding:  "Is that the kick or the bass guitar, is it on a downbeat,
                         what chord is it supporting, and are we in the chorus?"
```

Every phase below moves further along this spectrum. The features don't replace each other — they layer. A Crawl-phase song structure label combined with a Walk-phase chord quality combined with a Run-phase emotional valence score gives you a complete musical moment description.

---

## Phase 1 — Crawl

**Theme:** Go deeper with what we already have. All improvements use existing librosa/scipy. No new dependencies. No breaking manifest changes (all additions).

**Estimated effort:** 2–3 weeks

### C1 — Fix the MFCC Export Gap

**Files:** `core/polisher.py`, `core/analyzer.py`, `io/exporter.py`

MFCCs (13 coefficients) are computed in `analyzer.py:365` and then silently discarded — they never enter `PolishedFeatures` and never appear in the manifest. Fix this:

- Add MFCC delta and delta-delta (`librosa.feature.delta`) in `analyzer.py`
- Add `mfcc`, `mfcc_delta`, `mfcc_delta2` arrays to `PolishedFeatures` dataclass
- Export all 39 values (13 × 3) to manifest as `mfcc_mean` (13-element vector) per frame
- The delta and delta-delta capture **timbre velocity** — how fast the sound character is changing, independent of loudness

**Visual mapping:** `mfcc_delta` magnitude → visual morphing speed; sharp MFCC changes = texture transitions.

**TODO:**
- [ ] Add `mfcc_delta` and `mfcc_delta2` computation to `FeatureAnalyzer.extract_tonality()`
- [ ] Add `mfcc`, `mfcc_delta`, `mfcc_delta2` to `PolishedFeatures` dataclass
- [ ] Apply normalization (but not envelope smoothing) in `SignalPolisher.polish()`
- [ ] Export MFCC arrays to manifest JSON as compact arrays (not per-bin dict)
- [ ] Add `timbre_velocity` primitive (L2 norm of `mfcc_delta` at each frame)
- [ ] Update `export_numpy()` to include MFCC arrays
- [ ] Bump `ANALYSIS_VERSION` to invalidate cache

---

### C2 — Song Structure Detection

**Files:** `core/analyzer.py` (new `StructuralFeatures` dataclass), `core/polisher.py`, `io/exporter.py`

This is the highest-impact single feature that requires no new dependencies. The song structure shapes the entire macro-arc of the visual experience. Without it, every section looks the same.

**Implementation:**

```python
# In analyzer.py — uses librosa.segment
import librosa.segment

def extract_structure(self, decomposed, hop_length):
    # 1. Compute recurrence matrix from MFCC
    mfcc = librosa.feature.mfcc(y=decomposed.original, sr=sr, hop_length=hop_length)
    R = librosa.segment.recurrence_matrix(mfcc, mode='affinity', sparse=True)

    # 2. Novelty curve — peaks = structural boundaries
    novelty = librosa.segment.path_enhance(R, n=9)
    # novelty[i] ≈ 0.0 (mid-section) → 1.0 (section boundary)

    # 3. Segment boundaries via agglomerative clustering on chroma + MFCC
    bounds = librosa.segment.agglomerative(
        np.vstack([mfcc, chroma]), k=8
    )
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length)

    # 4. Label each frame with its section index
    section_labels = np.searchsorted(bounds, np.arange(n_frames))
```

**New manifest fields (per frame):**
```json
"section_index": 2,
"section_novelty": 0.83,
"section_progress": 0.41
```

**New manifest fields (top-level metadata):**
```json
"structure": {
  "n_sections": 7,
  "section_boundaries": [0.0, 18.4, 36.1, 52.8, ...],
  "section_durations": [18.4, 17.7, 16.7, ...]
}
```

**Visual mapping:**
- `section_novelty` → trigger palette crossfade / style transition when > threshold
- `section_index` → visual renderer picks from a per-section color palette slot
- `section_progress` → drives gradual tension builds within a section (intro builds to chorus)

**TODO:**
- [ ] Add `StructuralFeatures` dataclass to `core/analyzer.py`
- [ ] Implement `FeatureAnalyzer.extract_structure()` using `librosa.segment`
- [ ] Add structural features to `ExtractedFeatures` dataclass
- [ ] Add `section_index`, `section_novelty`, `section_progress` to `PolishedFeatures`
- [ ] Export structural fields to manifest JSON
- [ ] Add `structure` block to manifest metadata (boundary timestamps)
- [ ] Add `section_change` boolean trigger to per-frame data (True on boundaries)
- [ ] Bump `ANALYSIS_VERSION`

---

### C3 — Pitch Tracking (F0 / Melody Contour)

**Files:** `core/analyzer.py`, `core/polisher.py`, `io/exporter.py`

Chroma tells you which pitch classes are present, not the fundamental frequency or melodic direction. A rising melody vs a falling one is one of the most expressive musical signals and we are completely blind to it.

**Implementation using `librosa.pyin`:**

```python
# Run on harmonic component for cleaner pitch
f0, voiced_flag, voiced_probs = librosa.pyin(
    decomposed.harmonic,
    fmin=librosa.note_to_hz('C2'),   # ~65 Hz
    fmax=librosa.note_to_hz('C7'),   # ~2093 Hz
    sr=sr,
    hop_length=hop_length,
)
# f0 is NaN for unvoiced frames; voiced_flag is boolean

# Pitch velocity (semitones per second)
f0_midi = librosa.hz_to_midi(np.where(voiced_flag, f0, np.nan))
f0_velocity = np.gradient(np.nan_to_num(f0_midi))

# Octave register (0=low, 1=high within 0–1 range)
f0_register = np.clip((np.nan_to_num(f0) - 65) / (2093 - 65), 0, 1)
```

**New manifest fields (per frame):**
```json
"f0_hz": 440.0,
"f0_confidence": 0.91,
"f0_voiced": true,
"pitch_velocity": 2.3,
"pitch_register": 0.62
```

**Visual mapping:**
- `pitch_velocity` (positive = rising, negative = falling) → scale expansion/contraction, vertical motion direction
- `f0_confidence` → visual sharpness / focus (confident pitch = sharp geometry, uncertain = diffuse)
- `pitch_register` → vertical positioning of visual elements, brightness modulation
- `f0_voiced` → toggle between tonal rendering mode and noisy/percussive rendering mode

**TODO:**
- [ ] Add `f0_hz`, `f0_voiced`, `f0_probs` to `TonalityFeatures` dataclass
- [ ] Implement `pyin` call in `FeatureAnalyzer.extract_tonality()`
- [ ] Compute `pitch_velocity` (gradient of MIDI note number on voiced frames)
- [ ] Compute `pitch_register` (normalized 0–1 over musical range)
- [ ] Add all pitch fields to `PolishedFeatures`
- [ ] Apply confidence-weighted smoothing (don't smooth through unvoiced gaps)
- [ ] Export to manifest
- [ ] Bump `ANALYSIS_VERSION`

---

### C4 — Key and Mode Detection

**Files:** `core/analyzer.py`, `io/exporter.py`

We already compute chroma at every frame. Key detection is a simple profile match on the mean chroma vector. This is ~5 lines of code for enormous interpretive value.

**Implementation:**

```python
# Krumhansl-Schmuckler key profiles (major + minor, all 12 roots)
# librosa doesn't expose key detection directly; implement profile correlation

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def detect_key(chroma_mean):
    # Correlate mean chroma against all 24 key profiles (12 major + 12 minor)
    scores = []
    for shift in range(12):
        major_score = np.corrcoef(np.roll(MAJOR_PROFILE, shift), chroma_mean)[0, 1]
        minor_score = np.corrcoef(np.roll(MINOR_PROFILE, shift), chroma_mean)[0, 1]
        scores.append((shift, 'major', major_score))
        scores.append((shift, 'minor', minor_score))
    best = max(scores, key=lambda x: x[2])
    return best  # (root_index, mode, confidence)
```

**New manifest fields (top-level metadata):**
```json
"key": {
  "root": "F",
  "root_index": 5,
  "mode": "minor",
  "confidence": 0.87,
  "relative_major": "Ab"
}
```

**New manifest fields (per frame — for tracking key changes):**
```json
"key_stability": 0.94
```

**Visual mapping:**
- `root_index` → global hue bias for the entire piece (each of 12 keys maps to a 30° hue slice)
- `mode` → `major` = warm/bright palette bias, `minor` = cool/dark, `dorian/lydian` = exotic mid-range
- `key_stability` → low stability = key change approaching → trigger palette morph

**TODO:**
- [ ] Implement Krumhansl-Schmuckler key detection in `core/analyzer.py`
- [ ] Add section-level key detection (run on 8-second windows to catch key changes)
- [ ] Add `key_root`, `key_mode`, `key_confidence` to `ExtractedFeatures`
- [ ] Export key block to manifest metadata
- [ ] Add per-frame `key_stability` (sliding window key confidence)
- [ ] Bump `ANALYSIS_VERSION`

---

### C5 — Downbeat Tracking (Bar-Level)

**Files:** `core/analyzer.py`, `core/polisher.py`, `io/exporter.py`

Beats are tracked; bars are not. Beat 1 of each 4-beat bar is the musically most significant moment — where phrases resolve, chords change, energy resets. Currently indistinguishable from beats 2, 3, 4.

**Implementation (librosa heuristic approach):**

```python
# Group beat_frames into bars (assume 4/4 as default, confirm via beat spacing)
# Every 4th beat frame is a downbeat candidate
beat_frames = features.temporal.beat_frames

# Use onset strength to confirm downbeat position within each group of 4
# (downbeat usually has highest onset strength in its bar)
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

# Accumulate downbeat scores per beat position mod 4
bar_phase = np.argmax(
    [onset_env[beat_frames[i::4]].mean() for i in range(4)]
)
downbeat_frames = beat_frames[bar_phase::4]
```

**New manifest fields (per frame):**
```json
"is_downbeat": true,
"beat_position": 1,
"bar_index": 12,
"bar_progress": 0.0
```

**Visual mapping:**
- `is_downbeat` → major visual reset trigger (stronger than `is_beat`)
- `bar_progress` (0→1 over each 4-beat bar) → drives cyclic animations that complete every bar
- `beat_position` (1–4) → different visual accents per beat within the bar (1=big, 2=medium, 3=medium, 4=anticipation)
- `bar_index` → use modulo patterns (every 4 bars, every 8 bars) for large-scale phrase structure

**TODO:**
- [ ] Implement downbeat detection in `FeatureAnalyzer.extract_temporal()`
- [ ] Add `downbeat_frames`, `downbeat_times` to `TemporalFeatures`
- [ ] Add `is_downbeat`, `beat_position`, `bar_index`, `bar_progress` to `PolishedFeatures`
- [ ] Export all bar-level fields to manifest
- [ ] Bump `ANALYSIS_VERSION`

---

### C6 — CQT-Based Sub-Bass (Replace FFT for Low Frequencies)

**Files:** `core/analyzer.py`

The current FFT-based bandpass filter for sub-bass (20–60 Hz) has ~4 bins at n_fft=2048 / sr=22050. This is too coarse to distinguish kick body (50–80 Hz) from sub rumble (20–40 Hz). The Constant-Q Transform uses logarithmic frequency spacing — better low-frequency resolution at the same computational cost.

```python
# Replace _bandpass_rms for sub_bass and bass bands
CQT = librosa.cqt(
    y=decomposed.original,
    sr=sr,
    hop_length=hop_length,
    fmin=librosa.note_to_hz('C1'),  # ~33 Hz
    n_bins=48,  # 4 octaves × 12 bins/octave
    bins_per_octave=12,
)
cqt_magnitude = np.abs(CQT)

# Extract sub-bass from CQT bins covering 20–60 Hz
sub_bass_cqt = cqt_magnitude[:8, :].mean(axis=0)   # C1–B1 (~33–62 Hz)
bass_cqt = cqt_magnitude[8:20, :].mean(axis=0)      # C2–B3 (~65–247 Hz)
```

**Visual mapping:** Better sub-bass = more precise kick drum detection, cleaner "pulse" animations.

**TODO:**
- [ ] Add CQT computation to `_extract_frequency_bands()` for sub-bass and bass bins
- [ ] Add `sub_bass_cqt` and `bass_cqt` as alternative fields alongside existing bandpass versions
- [ ] Benchmark runtime (CQT is slower than bandpass RMS)
- [ ] Evaluate whether CQT replacement should be default or opt-in config flag
- [ ] Bump `ANALYSIS_VERSION`

---

### C7 — Spectral Bandwidth + Contrast

**Files:** `core/analyzer.py`, `core/polisher.py`, `io/exporter.py`

**Bandwidth** measures whether energy is concentrated around the centroid (pure sine tone, sharp chord) or spread across the spectrum (noise burst, dense cluster). Combined with centroid, it fully characterizes the "shape" of the spectrum.

**Spectral contrast** measures the difference between peaks and valleys in each frequency band — captures the "sharpness" of resonant peaks, tells you whether you're hearing discrete harmonics vs. noise.

```python
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
# contrast shape: (7, n_frames) — one value per sub-band
```

**Visual mapping:**
- `spectral_bandwidth` low → laser/geometric shapes (pure tone), high → cloud/fluid shapes
- `spectral_contrast` high → crystalline, structured visuals; low → diffuse, blended

**TODO:**
- [ ] Add `spectral_bandwidth` and `spectral_contrast` to `TonalityFeatures`
- [ ] Export to manifest
- [ ] Add `bandwidth_norm` to primitives (replaces or augments current `texture` calculation)
- [ ] Bump `ANALYSIS_VERSION`

---

### Phase 1 Dependency Changes

```
None — all pure librosa/scipy, already installed.
```

### Phase 1 Manifest Schema Change Summary

New per-frame fields: `mfcc` (13-array), `timbre_velocity`, `section_index`, `section_novelty`, `section_progress`, `section_change`, `f0_hz`, `f0_confidence`, `f0_voiced`, `pitch_velocity`, `pitch_register`, `key_stability`, `is_downbeat`, `beat_position`, `bar_index`, `bar_progress`, `spectral_bandwidth`, `spectral_contrast` (7-array)

New metadata fields: `structure.n_sections`, `structure.section_boundaries`, `key.root`, `key.mode`, `key.confidence`

Bump `ANALYSIS_VERSION` to `2.0` on Phase 1 completion to invalidate all caches.

---

## Phase 2 — Walk

**Theme:** New dependencies unlock source identity and neural-quality beat/chord extraction. These are the features that take you from "louder = more intense" to "this is what's playing and what it means musically."

**Estimated effort:** 3–5 weeks
**New dependencies:** `demucs`, `madmom`, `autochord` (or `chord-recognition`)

---

### W1 — Source Separation (Demucs)

**Files:** `core/decomposer.py` (new `SeparatedAudio` dataclass), `core/analyzer.py`, `pipeline.py`

This is the single largest leap in musical understanding. Demucs separates the mix into isolated stems. Each stem gets its own feature extraction pass. The visual result: kick drum drives one thing, bass guitar drives another, vocals drive something else entirely. Right now they all bleed into each other.

**Stems:**
- `drums` — kick, snare, hi-hat, cymbals isolated
- `bass` — bass guitar, sub synth isolated
- `vocals` — lead/backing vocals isolated
- `other` — guitar, keys, synths, everything else

**Implementation sketch:**

```python
# In core/decomposer.py
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

@dataclass
class SeparatedAudio:
    drums: np.ndarray
    bass: np.ndarray
    vocals: np.ndarray
    other: np.ndarray
    original: np.ndarray
    sample_rate: int
    duration: float

class SourceSeparator:
    MODEL_NAME = "htdemucs"  # best quality; htdemucs_ft is faster

    def __init__(self, device="cpu"):
        self.model = get_model(self.MODEL_NAME)
        self.model.eval()
        self.device = device

    def separate(self, audio_path, sr=44100):
        # Demucs operates at 44100 Hz; pass native SR here, not 22050
        y, sr = librosa.load(audio_path, sr=44100, mono=False)
        # ... apply model, return SeparatedAudio
```

**Architecture impact:** The pipeline needs a second decomposition path. Current HPSS stays for fallback; Demucs replaces it as the primary separator when installed.

**New manifest fields (per frame, per stem):**
```json
"drums_energy":   0.82,
"drums_impact":   0.91,
"bass_energy":    0.54,
"bass_note_hz":   110.0,
"vocals_energy":  0.31,
"vocals_voiced":  true,
"other_energy":   0.67
```

**Visual mapping:**
- `drums_energy` → pure percussive flash (no bass bleed contaminating the hit)
- `bass_note_hz` → MIDI pitch of bass note → drives low-frequency hue/scale
- `vocals_voiced` + `vocals_energy` → when vocals are active, shape complexity increases; vocal pitch tracks hue
- `other_energy` → sustained ambient field (synth pads, guitar sustain) → background layer intensity

**Performance note:** Demucs htdemucs runs at ~3× realtime on CPU for a 3-minute song (≈60s). The htdemucs_ft (fine-tuned) variant is ~2× realtime. GPU reduces this to ~0.3× realtime. This is offline-only (pre-render) in Phase 2; real-time streaming is Phase 3.

**TODO:**
- [ ] Add `demucs` to `pyproject.toml` as optional extra: `pip install -e ".[separation]"`
- [ ] Implement `SourceSeparator` class in `core/decomposer.py`
- [ ] Add `SeparatedAudio` dataclass
- [ ] Add `use_demucs: bool` flag to `AudioPipeline.__init__()`
- [ ] Add per-stem feature extraction in `FeatureAnalyzer.analyze()`
- [ ] Add stem energy fields to `PolishedFeatures`
- [ ] Export stem fields to manifest
- [ ] Add tests in `tests/core/test_source_separation.py`
- [ ] Update cache key to include `use_demucs` flag
- [ ] Benchmark on representative track; document expected runtime

---

### W2 — Neural Beat and Downbeat Tracking (madmom)

**Files:** `core/analyzer.py`

Madmom uses an RNN trained on thousands of songs. On complex rhythms — swing, polyrhythm, rubato, odd meters, electronic music with heavy sidechaining — it is dramatically more accurate than librosa's statistical tracker. It also natively returns **downbeat positions** which librosa doesn't expose cleanly.

```python
# Replaces librosa beat tracking in TemporalFeatures extraction
import madmom.features

def extract_temporal_neural(self, audio_path, sr, hop_length):
    # madmom beat tracker (RNN)
    beat_proc = madmom.features.beats.RNNBeatProcessor()
    beat_times = madmom.features.beats.BeatTrackingProcessor(fps=100)(
        beat_proc(str(audio_path))
    )

    # madmom downbeat tracker
    downbeat_proc = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeats = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4]  # handles 3/4 and 4/4
    )(downbeat_proc(str(audio_path)))

    # downbeats shape: (N, 2) — [[time, beat_position], ...]
    # beat_position == 1 means downbeat
```

**New temporal metadata:**
```json
"time_signature": "4/4",
"time_signature_confidence": 0.94,
"swing_ratio": 0.62,
"tempo_stability": 0.88
```

**Visual mapping:**
- More accurate beats = every beat-driven animation hits exactly on time
- `time_signature` → shapes with 3 vs 4 axes match the meter
- `swing_ratio` → whether off-beat animations lag (jazz feel) or quantize (electronic)

**TODO:**
- [ ] Add `madmom` to `pyproject.toml` optional extra: `pip install -e ".[neural-beats]"`
- [ ] Implement `extract_temporal_neural()` in `FeatureAnalyzer`
- [ ] Add `use_neural_beats: bool` flag to `AudioPipeline`
- [ ] Add `time_signature`, `swing_ratio`, `tempo_stability` to `TemporalFeatures`
- [ ] Export to manifest metadata
- [ ] Update downbeat detection in `PolishedFeatures` (replace C5 heuristic with madmom output)
- [ ] Add fallback to librosa if madmom not installed
- [ ] Tests in `tests/core/test_neural_beats.py`

---

### W3 — Chord Detection

**Files:** `core/analyzer.py` (new `HarmonicFeatures` dataclass), `core/polisher.py`, `io/exporter.py`

Chord identity is the single most untapped dimension. Right now we know "C, E, G are loud" but not "this is a C major chord." The jump to chord labels unlocks: tension/resolution arcs, root motion (how chords connect), quality palette (major=open, minor=closed, diminished=tense, augmented=unsettled).

```python
# Beat-synchronous chord detection using autochord or a TCN model
import autochord

chords = autochord.recognize(audio_path)
# Returns: list of (start_time, end_time, chord_label)
# e.g., [(0.0, 1.86, "C:maj"), (1.86, 3.71, "A:min"), ...]

# Parse chord labels into structured fields
def parse_chord(label):
    # "A:min7" -> root="A", quality="min", extension="7"
    # "N" -> no chord (silence/noise)
    ...

# Chord tension score: map qualities to 0.0 (resolved) → 1.0 (dissonant)
TENSION = {"maj": 0.0, "min": 0.2, "dom7": 0.5, "dim": 0.8, "aug": 0.9, "N": 0.1}
```

**New manifest fields (per frame):**
```json
"chord_root": "A",
"chord_root_index": 9,
"chord_quality": "min",
"chord_extension": "7",
"chord_tension": 0.45,
"chord_is_change": false
```

**New manifest fields (top-level metadata):**
```json
"chord_sequence": [
  {"start": 0.0, "end": 1.86, "chord": "C:maj", "tension": 0.0},
  {"start": 1.86, "end": 3.71, "chord": "A:min", "tension": 0.2}
]
```

**Visual mapping:**
- `chord_tension` → visual complexity, number of sides in polygons, fractal depth
- `chord_root_index` → hue shift relative to key tonic (interval = color relationship)
- `chord_is_change` → trigger shape morph at chord boundary
- `chord_quality` → major = open/symmetric shapes, minor = asymmetric/rotated, dim = jagged, aug = stretched

**TODO:**
- [ ] Add `autochord` to optional extras: `pip install -e ".[harmony]"`
- [ ] Implement `FeatureAnalyzer.extract_harmony()` method
- [ ] Add `HarmonicFeatures` dataclass (chord sequence, per-frame labels)
- [ ] Add chord fields to `PolishedFeatures`
- [ ] Add `chord_tension` to derived primitives (replaces heuristic `texture` calculation)
- [ ] Export chord sequence to manifest metadata
- [ ] Export per-frame chord fields
- [ ] Tests in `tests/core/test_chord_detection.py`

---

### W4 — Onset Envelope Shape (Attack Profiling)

**Files:** `core/analyzer.py`

Currently `is_onset` is a boolean trigger. There's no information about the character of the onset — whether it's a sharp crack (snare), a hard thud (kick), a metallic shimmer (hi-hat), or a soft swell (string section entering). The shape of the onset envelope tells you which.

```python
# For each detected onset, analyze the local amplitude envelope shape
# around the onset frame

def classify_onset_shape(onset_frame, rms, hop_length, sr):
    window = rms[max(0, onset_frame-5) : onset_frame+20]
    if len(window) < 5:
        return "unknown"

    attack_slope = (window[2] - window[0]) / 2  # frames 0-2
    decay_rate   = (window[2] - window[-1]) / len(window)
    peak_ratio   = window.max() / (window.mean() + 1e-6)

    if attack_slope > 0.15 and decay_rate > 0.04:
        return "transient"    # kick, snare
    elif attack_slope > 0.08 and peak_ratio > 2.5:
        return "metallic"     # hi-hat, cymbal
    elif attack_slope < 0.04:
        return "swell"        # string section, pad
    else:
        return "pluck"        # guitar, piano
```

**New manifest fields (per onset frame only, sparse encoding):**
```json
"onset_type": "transient",
"onset_attack_slope": 0.23,
"onset_sharpness": 0.87
```

**Visual mapping:**
- `transient` → hard flash, geometric spike
- `metallic` → shimmer effect, thin bright lines
- `swell` → slow bloom expansion
- `pluck` → medium attack with sustain-driven fade

**TODO:**
- [ ] Implement `classify_onset_shape()` in `FeatureAnalyzer`
- [ ] Add onset classification to `TemporalFeatures`
- [ ] Export `onset_type` and `onset_sharpness` to per-frame manifest data
- [ ] Add to `PolishedFeatures`

---

### Phase 2 Dependency Changes

```toml
# pyproject.toml additions (optional extras)
[project.optional-dependencies]
separation = ["demucs>=4.0.0"]
neural-beats = ["madmom>=0.16.1"]
harmony = ["autochord>=0.1.4"]
analysis-full = ["demucs>=4.0.0", "madmom>=0.16.1", "autochord>=0.1.4"]
```

Install the full analysis suite: `pip install -e ".[analysis-full]"`

---

## Phase 3 — Run

**Theme:** Frontier ML, real-time streaming, and Ableton integration. This is where the visualizer stops reacting to audio and starts responding to music.

**Estimated effort:** 6–10 weeks
**New dependencies:** `crepe`, `essentia-tensorflow`, `transformers` (for MERT), `python-osc`, `ableton-link`

---

### R1 — CREPE: Neural Pitch Estimation

PYIN (Phase 1, C3) is good. CREPE is better — a CNN trained specifically for pitch estimation, significantly more accurate on polyphonic material and in the presence of harmonic noise. It runs as inference, not a signal processing algorithm.

```python
import crepe
time, frequency, confidence, activation = crepe.predict(
    audio,
    sr,
    viterbi=True,     # Viterbi decoding for smooth pitch tracks
    step_size=1000/fps,   # align to target FPS
)
```

CREPE outputs a 360-dimensional activation vector at each frame — one value per cent of pitch resolution from A0 to B7. This activation can itself be used as a visual driver (the "shape" of the pitch activation = harmonic richness).

**TODO:**
- [ ] Add `crepe` to optional extras: `pip install -e ".[pitch]"`
- [ ] Implement `extract_pitch_neural()` as upgrade path from pyin
- [ ] Export CREPE activation vector (sampled to ~24 bins) as `pitch_spectrum` per frame
- [ ] Add `use_crepe: bool` flag to `AudioPipeline`

---

### R2 — Emotion and Mood Embeddings (Essentia / MERT)

Section-level emotional arc. Slow to compute (per 5–10 second segment) but gives you the macro emotional shape of the piece: is this section building or releasing? Dark or bright? Energetic or contemplative?

**Option A — Essentia TensorFlow (fast, pre-trained):**
```python
import essentia.standard as es
# Valence (negative → positive): -1.0 → 1.0
# Arousal (calm → energetic): -1.0 → 1.0
valence, arousal = es.TensorflowPredictMusiCNN()(audio_segment)
```

**Option B — MERT (HuggingFace, higher ceiling):**
```python
from transformers import AutoModel, Wav2Vec2FeatureExtractor
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M")
# Extract 1024-dimensional embeddings per 5-second window
# Apply UMAP to compress to 3D → map to (hue_bias, complexity, luminance)
```

**New manifest fields (per section in structure block):**
```json
"sections": [
  {
    "index": 0,
    "start": 0.0,
    "end": 18.4,
    "arousal": 0.32,
    "valence": -0.15,
    "mood": "melancholic",
    "embedding_3d": [0.21, 0.67, 0.44]
  }
]
```

**Visual mapping:**
- `arousal` → global animation speed, particle density, visual complexity ceiling
- `valence` negative → cool/dark palette, contractive shapes; positive → warm/bright, expansive
- `embedding_3d` → UMAP-reduced musical position drives slow color temperature drift across the piece

**TODO:**
- [ ] Add optional Essentia-TF or MERT to extras
- [ ] Implement segment-level emotion extraction
- [ ] Store per-section emotion in manifest structure block
- [ ] Add per-frame `arousal_smoothed` and `valence_smoothed` (lerped from section values)
- [ ] Tests for emotion extraction module

---

### R3 — Real-Time Streaming Mode

**Files:** New `core/stream.py`, new `core/realtime_analyzer.py`

The offline pipeline processes a whole file. Real-time mode processes a sliding buffer. The architecture is fundamentally different — you cannot know the future, so normalization, beat tracking, and structure detection must be online algorithms.

**Key design decisions:**
- Buffer size: 2048–4096 samples (43–93ms at 44100 Hz) for per-frame features
- Beat tracking must switch to `madmom.features.beats.OnlineBeatTrackingProcessor`
- Normalization must switch from global min/max to a rolling window with exponential decay
- Structure detection is deferred (5-second look-ahead buffer minimum)
- Emit features as they become available via callback or OSC

```python
class RealtimeAnalyzer:
    """
    Streaming audio feature extractor for live/Ableton use.
    Processes fixed-size audio chunks and emits feature dicts.
    """
    def __init__(self, sr=44100, fps=60, osc_port=9000):
        self.sr = sr
        self.chunk_size = sr // fps  # samples per frame at target FPS
        self.beat_proc = madmom.features.beats.OnlineBeatTrackingProcessor(fps=100)
        self._setup_osc(osc_port)

    def process_chunk(self, chunk: np.ndarray) -> dict:
        """Process one audio frame; return feature dict."""
        ...

    def _emit_osc(self, features: dict):
        """Send features as OSC bundle to connected clients (e.g., frontend)."""
        ...
```

**TODO:**
- [ ] Design streaming feature dict schema (subset of offline manifest)
- [ ] Implement `RealtimeAnalyzer` with rolling-window normalization
- [ ] Implement OSC emitter for live feature broadcast
- [ ] Add `python-osc` to extras
- [ ] Test with simulated real-time audio stream
- [ ] Document latency characteristics per feature type

---

### R4 — Max for Live / Ableton Integration

**Architecture options (choose one):**

**Option A — Audio sidechain via Jack/ASIO:**
- Chromascope runs as a standalone process
- Ableton routes audio out to a virtual audio device
- Chromascope reads from that device in real-time
- Pro: works with any DAW, no M4L required
- Con: routing complexity, OS-level audio configuration

**Option B — Max for Live device (recommended):**
- A `.amxd` M4L device in Ableton routes audio and MIDI clock to a Python subprocess
- Python subprocess runs `RealtimeAnalyzer` and serves features via OSC or WebSocket
- Frontend receives feature stream and animates
- Pro: native Ableton integration, BPM sync via Link, MIDI trigger support
- Con: requires Max for Live license

**Option C — Ableton Link + audio file handoff:**
- Ableton exports audio clips to a watch folder
- Chromascope detects new files, runs offline analysis, loads manifest
- Ableton Link provides precise BPM sync for playback alignment
- Pro: uses the full offline analysis pipeline including Demucs and MERT
- Con: latency (export → analysis → load), not true real-time

**Recommendation:** Start with Option C (most reliable, uses best analysis), build toward Option B (best experience).

**TODO:**
- [ ] Implement watch-folder mode in `pipeline.py`
- [ ] Add `ableton-link` library for BPM sync
- [ ] Design M4L device template (Max patch) for audio routing
- [ ] Add WebSocket server option alongside OSC in `RealtimeAnalyzer`
- [ ] Document the full Ableton setup in `docs/guides/ABLETON_INTEGRATION.md`

---

## Manifest Schema Evolution

| Version | What Changed |
|---|---|
| `1.1` (current) | 7-band energy, chroma, spectral features, 6 primitives |
| `2.0` (Crawl) | + MFCC delta, song structure, pitch tracking, key/mode, downbeats, CQT sub-bass, bandwidth |
| `3.0` (Walk) | + stem energy fields (drums/bass/vocals/other), neural beats, chord identity, chord tension, onset shape |
| `4.0` (Run) | + CREPE pitch spectrum, per-section emotion (arousal/valence), MERT embedding_3d |

All schema changes are additive (new fields only). Old consumers continue to work. Schema version is in `manifest.metadata.schema_version`.

---

## Visual Mapping Reference

This table answers: "if I implement this feature, what does it drive in the visuals?"

| Feature | Phase | Visual Parameter |
|---|---|---|
| `section_index` | C2 | Per-section color palette slot; resets on change |
| `section_novelty` | C2 | Palette crossfade trigger when > 0.7 |
| `section_progress` | C2 | Cyclic tension build within each section |
| `f0_hz` | C3 | Lead element hue |
| `pitch_velocity` | C3 | Visual expansion (rising) / contraction (falling) speed |
| `pitch_register` | C3 | Vertical element positioning, brightness |
| `key_root_index` | C4 | Global hue bias for entire piece (30°/step × 12 keys) |
| `key_mode` | C4 | Warm/bright (major) vs. cool/dark (minor) palette bias |
| `is_downbeat` | C5 | Major visual reset; stronger than beat flash |
| `bar_progress` | C5 | Cyclic animations that complete each bar |
| `beat_position` | C5 | Per-beat accent strength (1=strong, 2=medium, 3=medium, 4=anticipation) |
| `spectral_bandwidth` | C7 | Low=laser/geometric, high=cloud/diffuse |
| `timbre_velocity` | C1 | Texture morph speed |
| `drums_energy` | W1 | Pure percussion flash (no bleed) |
| `bass_note_hz` | W1 | Low-frequency hue/scale driver |
| `vocals_energy` | W1 | Shape complexity when voice is active |
| `chord_tension` | W3 | Visual complexity, fractal depth, polygon sides |
| `chord_root_index` | W3 | Hue shift relative to tonic |
| `chord_is_change` | W3 | Shape morph trigger at chord boundary |
| `chord_quality` | W3 | open/symmetric (maj), asymmetric (min), jagged (dim), stretched (aug) |
| `onset_type` | W4 | Hard flash vs. shimmer vs. swell vs. pluck |
| `arousal` | R2 | Global animation speed ceiling, particle density |
| `valence` | R2 | Palette temperature: negative=cool/dark, positive=warm/bright |
| `embedding_3d` | R2 | Slow color temperature drift as piece moves through musical space |

---

## Full TODO Checklist

### Phase 1 — Crawl

**C1: MFCC Export**
- [ ] Add `mfcc_delta`, `mfcc_delta2` to `FeatureAnalyzer.extract_tonality()`
- [ ] Add MFCC fields to `PolishedFeatures`
- [ ] Export MFCC arrays to manifest
- [ ] Add `timbre_velocity` primitive
- [ ] Update `export_numpy()`
- [ ] Tests for MFCC delta extraction
- [ ] Bump `ANALYSIS_VERSION` → `2.0`

**C2: Song Structure**
- [ ] Add `StructuralFeatures` dataclass
- [ ] Implement `extract_structure()` using `librosa.segment`
- [ ] Add `section_index`, `section_novelty`, `section_progress`, `section_change` to `PolishedFeatures`
- [ ] Export per-frame fields + `structure` metadata block
- [ ] Tests: verify section count > 1 on multi-minute track

**C3: Pitch Tracking**
- [ ] Add `f0_hz`, `f0_voiced`, `f0_probs` to `TonalityFeatures`
- [ ] Implement `librosa.pyin()` call in `extract_tonality()`
- [ ] Compute `pitch_velocity`, `pitch_register`
- [ ] Add confidence-weighted smoothing
- [ ] Export all pitch fields
- [ ] Tests: verify `f0_voiced` False on silent sections

**C4: Key/Mode Detection**
- [ ] Implement Krumhansl-Schmuckler profile correlator
- [ ] Add section-level key detection (8-second windows)
- [ ] Add `key_stability` per-frame signal
- [ ] Export `key` block to manifest metadata
- [ ] Tests: verify major/minor classification on known-key test tracks

**C5: Downbeat Tracking**
- [ ] Implement downbeat detection heuristic in `extract_temporal()`
- [ ] Add `is_downbeat`, `beat_position`, `bar_index`, `bar_progress`
- [ ] Export to manifest
- [ ] Tests: verify `beat_position` cycles 1–4

**C6: CQT Sub-Bass**
- [ ] Add CQT computation to `_extract_frequency_bands()`
- [ ] Add `sub_bass_cqt`, `bass_cqt` fields
- [ ] Benchmark CQT vs. bandpass runtime
- [ ] Add `use_cqt: bool` config flag (default False, opt-in)
- [ ] Tests: verify CQT sub-bass > 0 on kick-heavy test track

**C7: Bandwidth + Contrast**
- [ ] Add `spectral_bandwidth` and `spectral_contrast` extraction
- [ ] Export to manifest
- [ ] Add `bandwidth_norm` to primitives
- [ ] Tests

---

### Phase 2 — Walk

**W1: Demucs Source Separation**
- [ ] Add `demucs` optional extra to `pyproject.toml`
- [ ] Implement `SourceSeparator` class in `core/decomposer.py`
- [ ] Add `SeparatedAudio` dataclass
- [ ] Add `use_demucs: bool` to `AudioPipeline`
- [ ] Per-stem feature extraction in `FeatureAnalyzer.analyze()`
- [ ] Add stem fields to `PolishedFeatures`
- [ ] Export stem fields to manifest
- [ ] Update cache key for `use_demucs` flag
- [ ] Tests in `tests/core/test_source_separation.py`
- [ ] Document expected runtime (CPU/GPU)

**W2: Madmom Neural Beat Tracking**
- [ ] Add `madmom` optional extra to `pyproject.toml`
- [ ] Implement `extract_temporal_neural()`
- [ ] Add `use_neural_beats: bool` to `AudioPipeline`
- [ ] Add `time_signature`, `swing_ratio`, `tempo_stability` to `TemporalFeatures`
- [ ] Implement fallback to librosa if madmom not installed
- [ ] Export to manifest metadata
- [ ] Tests in `tests/core/test_neural_beats.py`

**W3: Chord Detection**
- [ ] Add `autochord` optional extra to `pyproject.toml`
- [ ] Implement `extract_harmony()` method
- [ ] Add `HarmonicFeatures` dataclass
- [ ] Add chord fields to `PolishedFeatures`
- [ ] Add `chord_tension` to derived primitives
- [ ] Export chord sequence to metadata + per-frame fields
- [ ] Tests in `tests/core/test_chord_detection.py`

**W4: Onset Envelope Shape**
- [ ] Implement `classify_onset_shape()` in `FeatureAnalyzer`
- [ ] Add onset type/sharpness to `TemporalFeatures`
- [ ] Export to per-frame manifest data
- [ ] Tests: transient vs. swell classification on synthetic signals

---

### Phase 3 — Run

**R1: CREPE Pitch**
- [ ] Add `crepe` optional extra
- [ ] Implement `extract_pitch_neural()`
- [ ] Add `use_crepe: bool` flag
- [ ] Export CREPE activation spectrum (24-bin summary) per frame
- [ ] Tests

**R2: Emotion Embeddings**
- [ ] Evaluate Essentia-TF vs. MERT for accuracy/speed tradeoff
- [ ] Implement segment-level extraction
- [ ] Add per-section emotion fields to `structure` metadata block
- [ ] Add per-frame `arousal_smoothed`, `valence_smoothed` (lerped)
- [ ] UMAP 3D projection for `embedding_3d`
- [ ] Tests

**R3: Real-Time Streaming**
- [ ] Design streaming feature dict schema
- [ ] Implement `RealtimeAnalyzer` with rolling normalization
- [ ] Add OSC emitter (`python-osc`)
- [ ] Document latency per feature type
- [ ] Integration tests with simulated stream

**R4: Ableton Integration**
- [ ] Implement watch-folder mode in `pipeline.py`
- [ ] Add Ableton Link BPM sync
- [ ] Design M4L device template
- [ ] Add WebSocket server option
- [ ] Write `docs/guides/ABLETON_INTEGRATION.md`

---

## Recommended Immediate Next Steps

While you work on getting better source audio (WAV/FLAC):

1. **Fix the MFCC export gap (C1)** — it's a 30-minute task and recovers lost information that's already being computed. Zero risk.
2. **Implement song structure detection (C2)** — this is the single feature with the highest visual impact-to-effort ratio. Everything downstream changes when the visualizer knows it's in the chorus.
3. **Add key/mode to manifest metadata (C4)** — 50 lines of code, instant palette mapping payoff.
4. **Add downbeat tracking (C5)** — bar-level timing fundamentally changes how animations feel (they complete at bar boundaries instead of floating arbitrarily).

All four are pure librosa, zero new dependencies, and can be done in approximately one focused week.

---

*This document supersedes the "Medium Term" and "Long Term" items in `docs/project/roadmap.md` for the audio analysis domain. The roadmap.md items should be updated to reference this proposal.*

---

## Retrofit Plan: All Existing Code and Experiments

Every renderer, visualizer, and consumer of the manifest must be able to use new fields as they arrive — and must continue to work correctly against old manifests (e.g., cached files from before a schema bump). This section defines the exact contract, the extension pattern, and a per-component checklist.

---

### The Backward Compatibility Guarantee

The entire pipeline is already safe by construction: every manifest field is read using `frame_data.get("field_name", default)`. This means:

- **Old manifest + new renderer** → new renderer gracefully falls back to defaults for missing fields. Zero breakage.
- **New manifest + old renderer** → old renderer ignores unknown fields. Zero breakage.
- **New manifest + new renderer** → new fields are consumed, full new behavior.

**This is the invariant to preserve in all future code.** Every `update()` method in every renderer must access manifest fields exclusively via `.get()` with a sensible default — never via `frame_data["field_name"]` (KeyError risk).

---

### The Extension Contract: `BaseVisualizer`

`BaseVisualizer._smooth_audio()` is the single point where per-frame manifest fields become per-renderer smoothed state. All renderers inherit from it. When new manifest fields are added, the base class must be extended to smooth them — so all renderers get the benefit without being individually modified.

**Current smoothed state (baseline):**
```python
# base.py — _smooth_audio() reads these manifest fields:
global_energy, percussive_impact, harmonic_energy,
low_energy, high_energy, spectral_flux, spectral_flatness,
sharpness, sub_bass, brilliance, spectral_centroid
```

**Extension pattern for each new field:**

```python
# base.py — _smooth_audio() additions per phase

# Phase 1 additions:
self._smooth_section_novelty = self._lerp(
    self._smooth_section_novelty, frame_data.get("section_novelty", 0.0), slow
)
self._smooth_pitch_velocity = self._lerp(
    self._smooth_pitch_velocity, frame_data.get("pitch_velocity", 0.0), fast
)
self._smooth_f0_hz = self._lerp(
    self._smooth_f0_hz, frame_data.get("f0_hz", 0.0), med
)
self._smooth_bar_progress = self._lerp(
    self._smooth_bar_progress, frame_data.get("bar_progress", 0.0), fast
)
self._smooth_bandwidth = self._lerp(
    self._smooth_bandwidth, frame_data.get("spectral_bandwidth", 0.5), slow
)
self._smooth_timbre_velocity = self._lerp(
    self._smooth_timbre_velocity, frame_data.get("timbre_velocity", 0.0), med
)

# Phase 2 additions:
self._smooth_drums_energy = self._lerp(
    self._smooth_drums_energy, frame_data.get("drums_energy", 0.0), fast
)
self._smooth_bass_energy = self._lerp(
    self._smooth_bass_energy, frame_data.get("bass_energy", 0.0), slow
)
self._smooth_vocals_energy = self._lerp(
    self._smooth_vocals_energy, frame_data.get("vocals_energy", 0.0), slow
)
self._smooth_chord_tension = self._lerp(
    self._smooth_chord_tension, frame_data.get("chord_tension", 0.0), med
)
```

**All new smoothed fields must also be added to `_smooth_audio_dict()`** so they flow through to `VisualPolisher.apply()` automatically.

---

### `VisualPolisher` Retrofit

`VisualPolisher.apply()` in `base.py` is the universal styling engine that takes `smoothed` dict → produces RGB output. New fields become available in `smoothed` automatically once added to `_smooth_audio_dict()`. However, the polisher itself should be upgraded to use the richer information:

**Phase 1 upgrades to `VisualPolisher.apply()`:**
```python
# Hue base: currently uses pitch_hue + centroid*0.2
# Upgrade: pitch_hue drives base, key_root biases it, pitch_velocity shifts it
hue_base = frame_data.get("pitch_hue", 0.0)
key_root = frame_data.get("key_root_index", 0) / 12.0
pitch_vel = np.clip(smoothed.get("pitch_velocity", 0.0) * 0.05, -0.1, 0.1)
hue_base = (hue_base * 0.7 + key_root * 0.3 + pitch_vel) % 1.0

# Saturation: currently harmonic + brilliance
# Upgrade: also modulated by chord_tension (dissonance → desaturate slightly)
chord_tension = smoothed.get("chord_tension", 0.0)
sat = 0.85 * (0.8 + smoothed["harmonic"] * 0.2 + smoothed["brilliance"] * 0.5)
sat = sat * (1.0 - chord_tension * 0.3)  # tense chords slightly desaturate

# Glow: currently percussive + flux
# Upgrade: also boosted by section_novelty (section changes = flash)
section_novelty = smoothed.get("section_novelty", 0.0)
glow_int = cfg.glow_intensity * (1.0 + smoothed["percussive"] * 0.3
                                  + smoothed["flux"] * 0.4
                                  + section_novelty * 0.6)
```

**Phase 2 upgrade:**
```python
# Chromatic aberration: currently percussive + sharpness
# Upgrade: also boosted by drums_energy (clean drum hit = more CA) and chord change
drums = smoothed.get("drums_energy", 0.0)
chord_change = float(frame_data.get("chord_is_change", False))
ab_offset = int(cfg.aberration_offset * (1.0
    + smoothed["percussive"] * 1.5
    + smoothed["sharpness"] * 3.0
    + drums * 1.0
    + chord_change * 2.0))
```

**TODO for VisualPolisher:**
- [ ] Add `key_root_index`-driven hue bias to palette mapping
- [ ] Add `chord_tension` modulation to saturation
- [ ] Add `section_novelty` boost to glow
- [ ] Add `is_downbeat` → stronger vignette pulse than `is_beat`
- [ ] Add `pitch_velocity` to hue animation speed
- [ ] Phase 2: add `drums_energy` and `chord_is_change` to chromatic aberration

---

### Per-Renderer Retrofit Guide

Each renderer overrides `_smooth_audio()` or reads directly from `frame_data` in `update()`. Below is what each renderer gains from new fields and what needs updating.

---

#### `experiment/attractor.py` — AttractorRenderer

Currently uses (from `_smooth_audio` override): `sub_bass`, `percussive_impact`, `brilliance`, `spectral_flux`, `harmonic_energy`, `global_energy`, `pitch_hue`, `spectral_centroid`, `spectral_flatness`, `is_beat`.

**Phase 1 additions:**
| New field | Mapping |
|---|---|
| `is_downbeat` | Larger shockwave than `is_beat`; reseed 30–50% of particles (vs. 12–40% on beat) |
| `section_change` | Full palette crossfade; reset trail decay to maximum for 1 second |
| `pitch_velocity` | Modulate camera elevation arc speed (rising pitch → camera rising) |
| `key_mode` | At init/section-change: if `minor` → use `void_fire` palette; `major` → `neon_aurora` |
| `f0_hz` | Drive Rössler c parameter directly instead of using `global_energy` (more musical) |
| `bar_progress` | Modulate Lorenz β cyclically with bar (slight breathing on the bar level) |

**Phase 2 additions:**
| New field | Mapping |
|---|---|
| `drums_energy` | Replace `percussive_impact` as the Lorenz ρ driver (cleaner, no bleed) |
| `bass_energy` | Drive scale pulse (the "cloud breathes" effect was sub_bass; bass is cleaner) |
| `chord_tension` | Increase Lorenz σ chaos range — high tension = wilder attractor |
| `chord_is_change` | Small reseed burst at chord boundaries |

**TODO:**
- [ ] Add `_smooth_downbeat`, `_smooth_section_change`, `_smooth_pitch_velocity`, `_smooth_bar_progress` to attractor's `_smooth_audio()` override
- [ ] Wire `is_downbeat` → shockwave system (separate from beat shockwave, larger magnitude)
- [ ] Wire `section_change` → `_trail_decay` reset + palette hot-swap
- [ ] Wire `pitch_velocity` → camera elevation
- [ ] Phase 2: replace `percussive_impact` → `drums_energy` for ρ; `sub_bass` → `bass_energy` for scale; add `chord_tension` → σ chaos

---

#### `experiment/chemical.py` — ChemicalRenderer

Currently uses: `sub_bass`, `percussive_impact`, `global_energy`, `low_energy`, `high_energy`, `brilliance`, `spectral_flatness`, `harmonic_energy`, `is_beat`.

**Phase 1 additions:**
| New field | Mapping |
|---|---|
| `is_downbeat` | Supersized reagent injection (2× normal beat injection) → massive crystallization event |
| `section_novelty` | Trigger full reagent flush + re-nucleation when > 0.8 |
| `spectral_bandwidth` | Low bandwidth → tight crystal lattice; high → amorphous growth patterns |
| `f0_hz` | Drive crystal hue in `synth_chem` style (pitch → color of crystal) |
| `key_mode` | `minor` → potassium palette bias; `major` → copper/sodium bias |

**Phase 2 additions:**
| New field | Mapping |
|---|---|
| `bass_energy` | Replace `low_energy` for injection (cleaner bass-driven injection pulses) |
| `drums_energy` | Replace `percussive_impact` for nucleation (clean drum hit = nucleation burst) |
| `chord_tension` | Drive crystal edge sharpness — dissonant chords = angular crystals |
| `onset_type` | `transient` → sharp angular nucleation; `swell` → smooth circular growth front |

**TODO:**
- [ ] Add `_smooth_section_novelty`, `_smooth_bandwidth` to chemical's `_smooth_audio()` override
- [ ] Wire `is_downbeat` → 2× reagent injection
- [ ] Wire `section_novelty` → reagent flush threshold
- [ ] Wire `spectral_bandwidth` → crystal lattice ordering parameter
- [ ] Phase 2: `drums_energy` → nucleation, `bass_energy` → injection, `chord_tension` → edge sharpness

---

#### `experiment/fractal.py` + `experiment/kaleidoscope.py` — FractalKaleidoscopeRenderer

Currently uses manifest fields via `frame_data.get()` in `renderer.py`. The fractal renderer reads `sub_bass`, `brilliance`, `spectral_flux`, `global_energy`, `harmonic_energy`, `is_beat`, `pitch_hue`.

**Phase 1 additions:**
| New field | Mapping |
|---|---|
| `section_index` | Modulo 4 → select different Julia set constant family per section |
| `section_novelty` | Trigger zoom jump / Julia constant hop at section boundaries |
| `pitch_velocity` | Drive Julia constant velocity (rising pitch → c moves faster through parameter space) |
| `bar_progress` | Drive polar mirror rotation rate (rotation completes at bar boundaries) |
| `f0_hz` | Use as seed for Julia constant selection (pitch → shape family) |
| `spectral_bandwidth` | Narrow → tight fractal detail; wide → smooth blobs |

**Phase 2 additions:**
| New field | Mapping |
|---|---|
| `chord_tension` | Drive Mandelbrot zoom depth (tense chord = deeper zoom = more complexity) |
| `chord_root_index` | Shift hue bias of fractal coloring |
| `onset_type` | `transient` → hard color inversion flash; `swell` → slow warp bloom |

**TODO:**
- [ ] Add section-driven Julia constant family selection to `fractal.py`
- [ ] Wire `pitch_velocity` to Julia c drift speed in `renderer.py`
- [ ] Wire `bar_progress` to `flow_field_warp` time parameter
- [ ] Phase 2: `chord_tension` → max_iter modulation; `chord_root_index` → hue offset

---

#### `experiment/attractor.py` (MixedConfig / dual mode)

The `MixedConfig` inherits from both `AttractorConfig` and `ChemicalConfig`. No separate retrofit needed — changes to both base configs propagate automatically. Verify in tests that `MixedConfig` passes all new fields through to both renderers.

**TODO:**
- [ ] Verify MixedConfig processes new fields in integration test
- [ ] Confirm `chord_tension` drives both Lorenz σ (attractor path) AND crystal edge sharpness (chemical path) simultaneously

---

#### `experiment/decay.py` — DecayRenderer

Uses `is_beat`, `percussive_impact`, `global_energy`, `harmonic_energy`, `sub_bass`, `brilliance`, `spectral_flatness`.

**Phase 1 additions:**
| New field | Mapping |
|---|---|
| `is_downbeat` | Mega-burst: 4× normal beat spawn count |
| `pitch_velocity` | Rising pitch → particles spawn with upward velocity bias |
| `section_novelty` | Trigger color palette swap |
| `f0_hz` | Drive particle hue when voices are active |

**Phase 2 additions:**
| New field | Mapping |
|---|---|
| `onset_type` | `transient` → tight burst; `swell` → wide diffuse bloom; `metallic` → shimmer ring |
| `drums_energy` | Replace `percussive_impact` for spawn rate |
| `chord_tension` | Drive particle interaction force (repulsion vs. attraction) |

**TODO:**
- [ ] Add `_smooth_pitch_velocity`, `_smooth_f0_hz` to decay's `_smooth_audio()` override
- [ ] Wire `is_downbeat` to mega-burst spawn
- [ ] Phase 2: onset type → spawn shape; `drums_energy` → spawn rate; `chord_tension` → interaction force

---

#### `experiment/solar.py` — SolarRenderer

Uses `global_energy`, `harmonic_energy`, `sub_bass`, `brilliance`, `spectral_flux`, `is_beat`.

**Phase 1 additions:**
| New field | Mapping |
|---|---|
| `section_index` | Switch planetary configuration (different planet count / arrangement per section) |
| `bar_progress` | Drive orbital phase clock (keeps orbits bar-synchronized) |
| `key_mode` | `major` → warm star color; `minor` → blue dwarf; `diminished` → red giant |
| `pitch_register` | Drive solar flare height |

**Phase 2 additions:**
| New field | Mapping |
|---|---|
| `chord_tension` | Drive solar activity (flare frequency, coronal ejection probability) |
| `bass_energy` | Drive magnetic field visualization strength (replaces `sub_bass`) |

**TODO:**
- [ ] Wire `bar_progress` to orbital mechanics phase
- [ ] Wire `section_index` to planet configuration
- [ ] Phase 2: `chord_tension` → solar activity, `bass_energy` → magnetic field

---

#### `visualizers/kaleidoscope.py` — Python Kaleidoscope Renderer (Geometric only)

The Python renderer currently only renders the Geometric style and reads a flat set of fields.

**Retrofit:** Add `.get()` fallback reads for all new Phase 1 fields so cached manifests and fresh manifests both work. No behavior changes required — this renderer is slated for deprecation in favor of the experiment renderers.

**TODO:**
- [ ] Audit all `frame_data["field"]` accesses → replace with `frame_data.get("field", default)`
- [ ] Confirm renderer handles both `schema_version: "1.1"` and `schema_version: "2.0"` manifests

---

### Frontend Retrofit (`frontend/app.js`)

`app.js` is ~6500 lines and is the primary consumer of the manifest JSON in the web UI. It reads per-frame data from the loaded manifest and maps fields to the 12 visualization styles.

**The contract:** `app.js` should follow the same `.get(field, default)` discipline as the Python renderers. In JS this means:

```javascript
// WRONG — crashes on old manifests:
const tension = frameData.chord_tension;

// RIGHT — safe across all schema versions:
const tension = frameData.chord_tension ?? 0.0;
const sectionIdx = frameData.section_index ?? 0;
const f0Hz = frameData.f0_hz ?? null;
const isDownbeat = frameData.is_downbeat ?? false;
```

**Phase 1 frontend wiring (per style group):**

| New field | All styles | Style-specific |
|---|---|---|
| `section_index` | Global palette index (CSS custom property `--section-hue`) | All styles: cycle accent color every section |
| `section_novelty` | Background flash opacity | Glass: refraction burst; Flower: petal count change |
| `is_downbeat` | Stronger rotation impulse than beat | Orrery: gravitational shockwave; Circuit: full board reset |
| `bar_progress` | N/A | Spiral: revolution sync; Fibonacci: Phi spiral phase lock |
| `pitch_velocity` | N/A | Fluid: upward/downward flow direction; Flower: petal extension direction |
| `f0_hz` | N/A | Orrery: planet orbital resonance frequency; Quark: wavefunction node position |
| `chord_tension` | N/A | DMT: symmetry complexity; Sacred: interlocking ratio |
| `key_mode` | Global palette temperature bias | Mycelial: warm (major) vs. cool (minor) branching colors |

**`smoothedValues` object in `app.js`:**

`app.js` maintains a `smoothedValues` object with its own lerp smoothing (mirroring `BaseVisualizer._smooth_audio()`). New fields need to be added here with per-field smoothing constants:

```javascript
// In the smoothedValues update loop (wherever smoothedValues is populated):
smoothedValues.sectionNovelty = lerp(
    smoothedValues.sectionNovelty ?? 0.0,
    frame.section_novelty ?? 0.0,
    0.08  // slow — section novelty should linger
);
smoothedValues.chordTension = lerp(
    smoothedValues.chordTension ?? 0.0,
    frame.chord_tension ?? 0.0,
    0.12  // medium — chord changes matter but shouldn't spike
);
smoothedValues.pitchVelocity = lerp(
    smoothedValues.pitchVelocity ?? 0.0,
    frame.pitch_velocity ?? 0.0,
    0.3   // faster — pitch direction is a real-time signal
);
smoothedValues.barProgress = frame.bar_progress ?? 0.0;  // no smoothing, it's already a ramp
smoothedValues.isDownbeat = frame.is_downbeat ?? false;
```

**TODO for `app.js`:**
- [ ] Audit all `frameData.field_name` direct accesses → replace with `?? default` null-coalescing
- [ ] Add new fields to `smoothedValues` initialization block (with defaults so startup is safe)
- [ ] Add `section_index` handler in the manifest-load path → set CSS custom property for global palette
- [ ] Add `is_downbeat` handler as a stronger beat trigger in all 12 `render*Style()` methods
- [ ] Add `bar_progress` to the rotation/animation systems that currently use raw time
- [ ] Wire `chord_tension` → shape complexity modulation in DMT, Sacred, Geometric styles
- [ ] Wire `pitch_velocity` → directional flow in Fluid, Flower, Spiral styles
- [ ] Phase 2: Wire `drums_energy`, `bass_energy` as clean replacements for legacy band reads
- [ ] Add schema version check at manifest load: log warning if `schema_version < "2.0"` and fields are missing

---

### Schema Version Negotiation

Every renderer and the frontend should check `manifest.metadata.schema_version` at load time and adapt accordingly:

```python
# Python (BaseVisualizer or pipeline consumer)
SUPPORTED_SCHEMA_MIN = "1.1"
SUPPORTED_SCHEMA_MAX = "4.0"

schema_ver = manifest.get("metadata", {}).get("schema_version", "1.1")
if schema_ver < SUPPORTED_SCHEMA_MIN:
    raise ValueError(f"Manifest schema {schema_ver} too old; re-analyze audio.")
# Features beyond your version → silently ignored via .get() defaults
```

```javascript
// JavaScript (app.js manifest loader)
const schemaVer = manifest.metadata?.schema_version ?? "1.1";
const HAS_STRUCTURE = schemaVer >= "2.0";
const HAS_STEMS     = schemaVer >= "3.0";
const HAS_EMOTION   = schemaVer >= "4.0";

// Feature flags used throughout render loop:
if (HAS_STRUCTURE) { /* use section_index, section_novelty, etc. */ }
```

---

### Retrofit TODO Checklist (All Consumers)

**`base.py` (`BaseVisualizer` + `VisualPolisher`)**
- [ ] Add Phase 1 fields to `_smooth_audio()`: `section_novelty`, `pitch_velocity`, `f0_hz`, `bar_progress`, `spectral_bandwidth`, `timbre_velocity`
- [ ] Add all new smoothed fields to `_smooth_audio_dict()`
- [ ] Initialize all new smoothed fields in `__init__()` with safe defaults
- [ ] Upgrade `VisualPolisher.apply()` with Phase 1 hue/saturation/glow logic
- [ ] Add `is_downbeat` vignette pulse to `VisualPolisher`
- [ ] Phase 2: Add `drums_energy`, `bass_energy`, `chord_tension` to `_smooth_audio()`
- [ ] Phase 2: Upgrade `VisualPolisher.apply()` CA + saturation with chord/stem fields

**`experiment/attractor.py`**
- [ ] Phase 1: downbeat shockwave, section palette swap, pitch_velocity → camera
- [ ] Phase 2: drums/bass stem replacement, chord_tension → Lorenz σ

**`experiment/chemical.py`**
- [ ] Phase 1: downbeat injection, section flush, bandwidth → lattice order
- [ ] Phase 2: stem replacements, chord_tension → edge sharpness, onset_type → growth shape

**`experiment/fractal.py` + `experiment/kaleidoscope.py`**
- [ ] Phase 1: section_index → Julia family, pitch_velocity → c drift, bar_progress → warp
- [ ] Phase 2: chord_tension → zoom depth, chord_root_index → hue

**`experiment/decay.py`**
- [ ] Phase 1: downbeat mega-burst, pitch_velocity → spawn direction
- [ ] Phase 2: onset_type → burst shape, drums_energy → spawn rate

**`experiment/solar.py`**
- [ ] Phase 1: bar_progress → orbital sync, key_mode → star color
- [ ] Phase 2: chord_tension → solar activity

**`visualizers/kaleidoscope.py`** (legacy Python renderer)
- [ ] Audit all direct dict accesses; replace with `.get()` + defaults
- [ ] Confirm works with both schema_version 1.1 and 2.0

**`frontend/app.js`**
- [ ] Audit all `frameData.field` direct accesses → add `?? default`
- [ ] Add new fields to `smoothedValues` init and update loop
- [ ] Add `section_index` global palette handler
- [ ] Add `is_downbeat` as stronger beat trigger in all 12 styles
- [ ] Add `bar_progress` to animation systems
- [ ] Wire Phase 1 fields to style-specific visual parameters (see mapping table above)
- [ ] Add schema version feature flags at manifest load

**`io/exporter.py`**
- [ ] Every new `PolishedFeatures` field must be added to both `_build_frame()` and `export_numpy()`
- [ ] Bump `schema_version` at each phase boundary: `"2.0"` (Crawl), `"3.0"` (Walk), `"4.0"` (Run)

**`pipeline.py`**
- [ ] Cache key must include all new config flags: `use_demucs`, `use_neural_beats`, `use_crepe`, `use_cqt`
- [ ] Bump `ANALYSIS_VERSION` at each phase boundary to match `schema_version`

---

*This document supersedes the "Medium Term" and "Long Term" items in `docs/project/roadmap.md` for the audio analysis domain. The roadmap.md items should be updated to reference this proposal.*
