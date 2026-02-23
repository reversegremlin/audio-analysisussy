# Audio Intelligence Proposal
## From Signal Statistics to Musical Understanding

**Status:** Phase 1 Complete · Phase 2 Partial (W4 only) · Attractor Retrofitted
**Date:** 2026-02-22 · **Updated:** 2026-02-23
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

### What the Manifest Currently Carries (per frame) — Schema 2.0

```
Energy:    percussive_impact, harmonic_energy, global_energy, spectral_flux
Frequency: sub_bass, bass, low_mid, mid, high_mid, presence, brilliance
           sub_bass_cqt, bass_cqt  [opt-in: use_cqt=True]
Texture:   spectral_brightness, spectral_flatness, spectral_rolloff, zero_crossing_rate
           spectral_bandwidth, spectral_contrast (7 bins)
Tonality:  chroma_values (12 bins), dominant_chroma
           mfcc (13), mfcc_delta (13), mfcc_delta2 (13), timbre_velocity
Pitch:     f0_hz, f0_voiced, f0_probs, pitch_register
Timing:    is_beat, is_onset, is_downbeat, beat_position, bar_index, bar_progress
           onset_type ("transient"/"percussive"/"harmonic"), onset_sharpness
Structure: section_index, section_novelty, section_progress, section_change
Harmony:   key_stability
Primitives: impact, fluidity, brightness, pitch_hue, texture, sharpness
```

**Manifest metadata additions (schema 2.0):**
```json
"structure": { "n_sections": 8, "section_boundaries": [...], "section_durations": [...] },
"key":       { "root": "C", "root_index": 0, "mode": "major", "confidence": 0.87 }
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

## Phase 1 — Crawl ✓ COMPLETE

**Theme:** Go deeper with what we already have. All improvements use existing librosa/scipy. No new dependencies. No breaking manifest changes (all additions).

**Status:** All C1–C7 features implemented. `ANALYSIS_VERSION = "2.0"`, `schema_version = "2.0"`. 87 tests passing. OOM fix applied (pyin runs at 4× coarser hop, output upsampled via nearest-neighbour; `chroma_stft` replaces `chroma_cqt` in structure extraction).

### C1 — Fix the MFCC Export Gap

**Files:** `core/polisher.py`, `core/analyzer.py`, `io/exporter.py`

MFCCs (13 coefficients) are computed in `analyzer.py:365` and then silently discarded — they never enter `PolishedFeatures` and never appear in the manifest. Fix this:

- Add MFCC delta and delta-delta (`librosa.feature.delta`) in `analyzer.py`
- Add `mfcc`, `mfcc_delta`, `mfcc_delta2` arrays to `PolishedFeatures` dataclass
- Export all 39 values (13 × 3) to manifest as `mfcc_mean` (13-element vector) per frame
- The delta and delta-delta capture **timbre velocity** — how fast the sound character is changing, independent of loudness

**Visual mapping:** `mfcc_delta` magnitude → visual morphing speed; sharp MFCC changes = texture transitions.

**TODO:**
- [x] Add `mfcc_delta` and `mfcc_delta2` computation to `FeatureAnalyzer.extract_tonality()`
- [x] Add `mfcc`, `mfcc_delta`, `mfcc_delta2` to `PolishedFeatures` dataclass
- [x] Apply normalization (but not envelope smoothing) in `SignalPolisher.polish()`
- [x] Export MFCC arrays to manifest JSON as compact arrays (not per-bin dict)
- [x] Add `timbre_velocity` primitive (L2 norm of `mfcc_delta` at each frame)
- [x] Update `export_numpy()` to include MFCC arrays
- [x] Bump `ANALYSIS_VERSION` to invalidate cache

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
- [x] Add `StructuralFeatures` dataclass to `core/analyzer.py`
- [x] Implement `FeatureAnalyzer.extract_structure()` using `librosa.segment`
- [x] Add structural features to `ExtractedFeatures` dataclass
- [x] Add `section_index`, `section_novelty`, `section_progress` to `PolishedFeatures`
- [x] Export structural fields to manifest JSON
- [x] Add `structure` block to manifest metadata (boundary timestamps)
- [x] Add `section_change` boolean trigger to per-frame data (True on boundaries)
- [x] Bump `ANALYSIS_VERSION`

> **Implementation note:** `extract_structure()` uses `chroma_stft` instead of the proposed `chroma_cqt` to avoid the additional memory allocation of a full CQT pass (OOM mitigation).

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
- [x] Add `f0_hz`, `f0_voiced`, `f0_probs` to `TonalityFeatures` dataclass
- [x] Implement `pyin` call in `FeatureAnalyzer.extract_tonality()`
- [x] Compute `pitch_register` (normalized 0–1 over musical range)
- [x] Add all pitch fields to `PolishedFeatures`
- [x] Apply confidence-weighted smoothing (don't smooth through unvoiced gaps)
- [x] Export to manifest
- [x] Bump `ANALYSIS_VERSION`

> **Implementation note:** `pyin` runs at `hop_length * 4` internally to prevent OOM on long tracks (was crashing Crostini VM at 60fps on ~5-7 min files). Output is resampled back to `n_ref` frames via nearest-neighbour. `pitch_velocity` (gradient) is not currently exported — `pitch_register` (log Hz → [0,1]) is the primary per-frame signal.

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
- [x] Implement Krumhansl-Schmuckler key detection in `core/analyzer.py`
- [x] Add section-level key detection (run on 8-second windows to catch key changes)
- [x] Add `key_root`, `key_mode`, `key_confidence` to `ExtractedFeatures`
- [x] Export key block to manifest metadata
- [x] Add per-frame `key_stability` (sliding window key confidence)
- [x] Bump `ANALYSIS_VERSION`

> **Implementation note:** K-S on pure major triads (e.g., C,E,G) is ambiguous with the relative minor (A minor). Test suite uses diatonic set membership checks rather than exact root assertions for this reason.

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
- [x] Implement downbeat detection in `FeatureAnalyzer.extract_temporal()`
- [x] Add `downbeat_frames`, `downbeat_times` to `TemporalFeatures`
- [x] Add `is_downbeat`, `beat_position`, `bar_index`, `bar_progress` to `PolishedFeatures`
- [x] Export all bar-level fields to manifest
- [x] Bump `ANALYSIS_VERSION`

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
- [x] Add CQT computation to `_extract_frequency_bands()` for sub-bass and bass bins
- [x] Add `sub_bass_cqt` and `bass_cqt` as alternative fields alongside existing bandpass versions
- [x] Benchmark runtime (CQT is slower than bandpass RMS)
- [x] Evaluate whether CQT replacement should be default or opt-in config flag → **decided: opt-in** (`use_cqt=False` default)
- [x] Bump `ANALYSIS_VERSION`

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
- [x] Add `spectral_bandwidth` and `spectral_contrast` to `TonalityFeatures`
- [x] Export to manifest
- [x] Add `bandwidth_norm` to primitives (replaces or augments current `texture` calculation)
- [x] Bump `ANALYSIS_VERSION`

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

## Phase 2 — Walk (Partial: W4 complete, W1/W2/W3 stubs only)

**Theme:** New dependencies unlock source identity and neural-quality beat/chord extraction. These are the features that take you from "louder = more intense" to "this is what's playing and what it means musically."

**Status:** W4 (onset shape) fully implemented. W1/W2/W3 have dataclass stubs and config flags but no computation — they raise `ImportError` (W1) or silently return `None`/fall back to librosa (W2/W3) when optional dependencies are absent.
**New dependencies:** `demucs`, `madmom`, `autochord` (or `chord-recognition`)

### Recommended Implementation Order

**W2 → W3 → W1.** Do not start with W1.

- **W2 (madmom) first:** ~12–16 hours. Offline, no GPU, fast inference (~20× realtime on CPU). Immediately improves beat/downbeat accuracy for every renderer. Replaces the C5 heuristic with a trained RNN. Low risk, high reward.
- **W3 (autochord) second:** ~20–24 hours. `chord_tension` is the single highest-value new visual driver — it gives every renderer a new musical dimension (complexity, hue shift, shape morph). Medium difficulty.
- **W1 (demucs) last:** ~28–36 hours. Highest effort, GPU dependency, 1–2 GB model download. Worth it for the isolation quality, but do it after the others are stable.

Renderer retrofits can begin as soon as W2 is merged. Each renderer uses `frame_data.get("chord_tension", 0.0)` etc., so W3/W1 fields simply return the default until those land.

**Total estimate:** 88–115 hours (~2–3 weeks focused). See detailed breakdown per work item below.

---

### W1 — Source Separation (Demucs)

**Files:** `core/decomposer.py`, `core/analyzer.py`, `core/polisher.py`, `io/exporter.py`, `pipeline.py`
**Estimate:** ~385–445 new lines across 6 files. ~28–36 hours including testing.
**Do this last.** Highest effort, hardest dependency, but also the biggest isolation payoff.

This is the single largest leap in musical understanding. Demucs separates the mix into isolated stems. Each stem gets its own feature extraction pass. The visual result: kick drum drives one thing, bass guitar drives another, vocals drive something else entirely. Right now they all bleed into each other.

**Stems:**
- `drums` — kick, snare, hi-hat, cymbals isolated
- `bass` — bass guitar, sub synth isolated
- `vocals` — lead/backing vocals isolated
- `other` — guitar, keys, synths, everything else

**Implementation sketch:**

```python
# In core/decomposer.py — SourceSeparator.separate() implementation
# (Stub already exists; this is what replaces the NotImplementedError)

def separate(self, audio_path: str) -> SeparatedAudio:
    from demucs.apply import apply_model
    import torch

    # Demucs REQUIRES stereo at 44100 Hz. Load separately from the main pipeline.
    y_stereo, _ = librosa.load(audio_path, sr=44100, mono=False)
    if y_stereo.ndim == 1:
        y_stereo = np.stack([y_stereo, y_stereo])  # mono → fake stereo

    waveform = torch.from_numpy(y_stereo).unsqueeze(0)  # (1, 2, N)
    with torch.no_grad():
        sources = apply_model(self.model, waveform, device=self.device)
    # sources shape: (1, 4, 2, N) → drums/bass/vocals/other

    stems = {}
    for i, name in enumerate(self.model.sources):  # ["drums","bass","vocals","other"]
        stem_mono = sources[0, i].mean(dim=0).numpy()  # stereo → mono
        stems[name] = librosa.resample(stem_mono, orig_sr=44100, target_sr=22050)

    return SeparatedAudio(
        drums=stems["drums"], bass=stems["bass"],
        vocals=stems["vocals"], other=stems["other"],
        sample_rate=22050,
        duration=len(stems["drums"]) / 22050,
    )
```

```python
# In core/analyzer.py — add _analyze_stem() helper
def _analyze_stem(self, y: np.ndarray, name: str) -> dict:
    """Compute RMS energy + pyin F0 (bass only) for a single separated stem."""
    rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
    energy = self._normalize(rms)

    result = {f"{name}_energy": energy}
    if name == "bass":
        f0, voiced, _ = librosa.pyin(y, fmin=30.0, fmax=300.0, sr=self.sr,
                                      hop_length=self.hop_length * 4)
        result["bass_note_hz"] = np.where(voiced, f0, 0.0)
    if name == "vocals":
        f0, voiced, _ = librosa.pyin(y, fmin=80.0, fmax=1100.0, sr=self.sr,
                                      hop_length=self.hop_length * 4)
        result["vocals_voiced"] = voiced
    return result
```

**Architecture impact:** Current HPSS stays as fallback when `use_demucs=False`. Demucs replaces it as the primary separator when installed. The main pipeline call is:
```python
if self.use_demucs and SourceSeparator is available:
    separated = separator.separate(audio_path)
    stem_features = analyzer.analyze_stems(separated)
else:
    # existing HPSS path unchanged
```

**New manifest fields (per frame):**
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

**Hardware note (Chromebook):** CPU-only, no GPU. `htdemucs` at 3× realtime means a 5-minute track takes ~15 minutes to separate. Acceptable for offline pre-render (cache means you pay once). Do not attempt real-time use. `htdemucs_ft` (fine-tuned, faster) is an option at some quality cost: ~2× realtime (~10 min). First run also downloads ~1–2 GB model weights.

**Critical gotchas:**
1. **Stereo at 44100 Hz required** — the pipeline loads mono at 22050 Hz. `SourceSeparator.separate()` must do its own `librosa.load(path, sr=44100, mono=False)` call, independent of the main load.
2. **Fake stereo fallback** — if input is mono, duplicate the channel before passing to `apply_model`. Demucs will still work but stereo width cues are lost.
3. **Cache the loaded model** — `get_model("htdemucs")` takes ~5–10 seconds. Cache in `SourceSeparator.__init__()`, not per-call.
4. **GPU memory** — `htdemucs` uses ~4 GB VRAM. Auto-detect at init: if `torch.cuda.is_available()` and VRAM < 6 GB, fall back to CPU and log a warning.
5. **Resample stems after separation** — Demucs outputs at 44100 Hz. Resample each stem to 22050 Hz before passing to `_analyze_stem()` so hop lengths align.
6. **ANALYSIS_VERSION → "3.0"** on W1 completion to invalidate stem-less caches.

**TODO (stub status):**
- [x] Add `demucs` to `pyproject.toml` as optional extra: `pip install -e ".[separation]"`
- [x] Add `SeparatedAudio` dataclass (`core/decomposer.py`)
- [x] Add `SourceSeparator` class stub — raises `ImportError` if demucs not installed
- [x] Add `use_demucs: bool` flag to `AudioPipeline.__init__()`
- [ ] **`decomposer.py`** — Implement `SourceSeparator.separate()` (~60–80 lines)
- [ ] **`analyzer.py`** — Add `_analyze_stem(y, name)` helper (~40 lines); add `analyze_stems(separated)` call in main `analyze()` path (~30 lines)
- [ ] **`analyzer.py`** — pyin on bass stem for `bass_note_hz` (reuse pyin 4× hop pattern from C3, ~20 lines)
- [ ] **`polisher.py`** — Add 7–8 stem fields to `PolishedFeatures` dataclass + envelope smoothing (~40 lines)
- [ ] **`exporter.py`** — Add stem fields to `_build_frame()` and `export_numpy()` (~30 lines)
- [ ] **`pipeline.py`** — Wire `use_demucs` flag through to separator instantiation; update cache key (~15 lines)
- [ ] **`tests/core/test_source_separation.py`** — Synthetic multi-track mix, SNR assertion, stem energy isolation tests (~150 lines)
- [ ] Benchmark `htdemucs` vs `htdemucs_ft` on representative track; add note to README

---

### W2 — Neural Beat and Downbeat Tracking (madmom)

**Files:** `core/analyzer.py`, `core/polisher.py`, `io/exporter.py`
**Estimate:** ~300–320 new lines across 3 files. ~12–16 hours.
**Start here.** Fastest win, lowest risk, immediate improvement.

Madmom uses an RNN trained on thousands of songs. On complex rhythms — swing, polyrhythm, rubato, odd meters, electronic music with heavy sidechaining — it is dramatically more accurate than librosa's statistical tracker. It also natively returns **downbeat positions** which librosa doesn't expose cleanly.

**Implementation:**

```python
# In core/analyzer.py — new method FeatureAnalyzer.extract_temporal_neural()

def extract_temporal_neural(self, audio_path: str, sr: int, hop_length: int) -> TemporalFeatures:
    import madmom.features.beats
    import madmom.features.downbeats

    # madmom requires a file path (decodes via ffmpeg internally)
    # it does NOT accept numpy arrays
    beat_proc = madmom.features.beats.RNNBeatProcessor()
    beat_times = madmom.features.beats.BeatTrackingProcessor(fps=100)(
        beat_proc(str(audio_path))
    )  # → (N,) float array of beat times in seconds

    downbeat_proc = madmom.features.downbeats.RNNDownBeatProcessor()
    downbeats = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4]  # handles 3/4 and 4/4
    )(downbeat_proc(str(audio_path)))
    # downbeats shape: (N, 2) — [[time, beat_position], ...]
    # beat_position == 1.0 means downbeat; 2.0/3.0/4.0 = beats 2–4

    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    downbeat_mask = downbeats[:, 1] == 1.0
    downbeat_frames = librosa.time_to_frames(
        downbeats[downbeat_mask, 0], sr=sr, hop_length=hop_length
    )

    # Derive time signature from typical beats-per-bar detected
    # DBNDownBeatTrackingProcessor uses beats_per_bar=[3, 4]; infer from output
    bar_sizes = np.diff(np.where(downbeat_mask)[0])
    time_sig_num = int(np.round(np.median(bar_sizes))) if len(bar_sizes) > 0 else 4
    time_signature = f"{time_sig_num}/4"

    # Swing ratio: measure off-beat timing relative to strict grid
    # 0.5 = perfectly quantized, 0.67 = triplet swing, 0.55 = light swing
    if len(beat_times) >= 4:
        even_beats = beat_times[0::2]
        odd_beats  = beat_times[1::2]
        min_len = min(len(even_beats) - 1, len(odd_beats))
        if min_len > 0:
            ratios = (odd_beats[:min_len] - even_beats[:min_len]) / \
                     (even_beats[1:min_len+1] - even_beats[:min_len])
            swing_ratio = float(np.median(ratios))
        else:
            swing_ratio = 0.5
    else:
        swing_ratio = 0.5

    # Tempo stability: std dev of inter-beat interval normalized by mean
    ibi = np.diff(beat_times)
    tempo_stability = float(1.0 - np.clip(np.std(ibi) / (np.mean(ibi) + 1e-6), 0, 1))

    return TemporalFeatures(
        beat_frames=beat_frames,
        beat_times=beat_times,
        downbeat_frames=downbeat_frames,
        time_signature=time_signature,
        swing_ratio=swing_ratio,
        tempo_stability=tempo_stability,
        # ... existing fields unchanged
    )
```

```python
# In core/analyzer.py — integrate into extract_temporal():
def extract_temporal(self, audio_path, ...):
    if self.use_neural_beats:
        try:
            return self.extract_temporal_neural(audio_path, sr, hop_length)
        except ImportError:
            pass  # fall through to librosa heuristic
    # existing librosa path (C5 heuristic) unchanged
    ...
```

**New manifest metadata fields:**
```json
"time_signature": "4/4",
"swing_ratio": 0.62,
"tempo_stability": 0.88
```

**Visual mapping:**
- More accurate beats = every beat-driven animation lands on time
- `time_signature` → 3/4 meter could use triangular symmetry; 4/4 uses square/fourfold
- `swing_ratio` > 0.58 → jazz/blues feel; animate off-beats with slight lag
- `swing_ratio` < 0.52 → strict electronic quantization

**Critical gotchas:**
1. **Requires audio file path, not numpy array** — madmom calls ffmpeg internally. Pass `str(audio_path)`, not `y`. This is the biggest API surprise.
2. **madmom always operates at 100 FPS internally** — output beat times are in seconds and need to be converted to your hop-length frame grid via `librosa.time_to_frames()`.
3. **TensorFlow compilation on first call** — ~2–5 seconds on the very first invocation. This is a one-time JIT cost per process startup, not per track.
4. **`beats_per_bar=[3, 4]` is statically specified** — DBN tries both. Output tells you which won; use the median bar size to infer time signature.
5. **Downbeat times are separate from beat times** — madmom returns two separate arrays. You need to merge them carefully to reconstruct the full bar grid.

**TODO (stub status):**
- [x] Add `madmom` to `pyproject.toml` optional extra: `pip install -e ".[neural-beats]"`
- [x] Add `use_neural_beats: bool` flag to `AudioPipeline`
- [x] Fallback to librosa C5 heuristic when madmom absent
- [ ] **`analyzer.py`** — Implement `extract_temporal_neural()` as shown above (~100 lines)
- [ ] **`analyzer.py`** — Add `time_signature: str`, `swing_ratio: float`, `tempo_stability: float` to `TemporalFeatures` dataclass (~5 lines)
- [ ] **`analyzer.py`** — Call `extract_temporal_neural()` when `use_neural_beats=True` in `extract_temporal()` (~10 lines)
- [ ] **`polisher.py`** — Pass `time_signature`, `swing_ratio`, `tempo_stability` through `PolishedFeatures` (~10 lines)
- [ ] **`exporter.py`** — Write `time_signature`, `swing_ratio`, `tempo_stability` to manifest metadata block (~15 lines)
- [ ] **`pipeline.py`** — No changes needed (flag already wired)
- [ ] **`tests/core/test_neural_beats.py`** — Synthetic 4/4 click track, assert beat times within ±50ms; 3/4 click, assert time_signature; swing track, assert swing_ratio > 0.55 (~100 lines)

---

### W3 — Chord Detection (autochord)

**Files:** `core/analyzer.py`, `core/polisher.py`, `io/exporter.py`
**Estimate:** ~375–420 new lines across 4 files. ~20–24 hours.
**Do this second.** `chord_tension` is the single highest-value new visual driver — it gives every renderer a new musical dimension (complexity, hue shift, shape morph) at medium effort.

Chord identity is the single most untapped dimension. Right now we know "C, E, G are loud" but not "this is a C major chord." The jump to chord labels unlocks: tension/resolution arcs, root motion (how chords connect), quality palette (major=open, minor=closed, diminished=tense, augmented=unsettled).

**Implementation:**

```python
# In core/analyzer.py — constants and helpers

import re

NOTE_INDEX = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11
}

CHORD_TENSION = {
    "maj":   0.0,   "min":   0.2,  "dom7": 0.5,
    "min7":  0.35,  "maj7":  0.15, "dim":  0.8,
    "aug":   0.9,   "sus2":  0.1,  "sus4": 0.15,
    "hdim7": 0.7,   "dim7":  0.75, "N":    0.0,
}

def _parse_chord(label: str) -> tuple[str, str, str]:
    """Parse chord label into (root, quality, extension).
    "A:min7"  -> ("A",  "min",  "7")
    "G:dom7"  -> ("G",  "dom7", "")
    "N"       -> ("N",  "N",    "")   # no chord
    "C"       -> ("C",  "maj",  "")   # implied major
    """
    if label == "N":
        return ("N", "N", "")
    parts = label.split(":", 1)
    root = parts[0]
    if len(parts) == 1:
        return (root, "maj", "")
    m = re.match(r"([a-zA-Z]+)(\d*)", parts[1])
    quality = m.group(1) if m else parts[1]
    extension = m.group(2) if m else ""
    return (root, quality, extension)
```

```python
# In core/analyzer.py — FeatureAnalyzer.extract_harmony()

def extract_harmony(
    self, audio_path: str, n_frames: int, sr: int, hop_length: int
) -> HarmonicFeatures:
    import autochord

    # autochord returns segment-level list: [(start_t, end_t, label), ...]
    chords = autochord.recognize(str(audio_path))

    # Build per-frame arrays (all frames in a segment get the same label)
    chord_root      = ["N"] * n_frames
    chord_quality   = ["N"] * n_frames
    chord_extension = [""]  * n_frames
    chord_tension   = np.zeros(n_frames, dtype=np.float32)
    chord_root_idx  = np.zeros(n_frames, dtype=np.int32)
    chord_is_change = np.zeros(n_frames, dtype=bool)

    prev_label = None
    for (start_t, end_t, label) in chords:
        start_f = int(np.clip(
            librosa.time_to_frames(start_t, sr=sr, hop_length=hop_length),
            0, n_frames - 1
        ))
        end_f = int(np.clip(
            librosa.time_to_frames(end_t, sr=sr, hop_length=hop_length),
            0, n_frames
        ))

        root, quality, ext = _parse_chord(label)
        tension  = CHORD_TENSION.get(quality, 0.3)  # fallback for unknown qualities
        root_idx = NOTE_INDEX.get(root, 0)

        chord_root[start_f:end_f]      = [root]      * (end_f - start_f)
        chord_quality[start_f:end_f]   = [quality]   * (end_f - start_f)
        chord_extension[start_f:end_f] = [ext]        * (end_f - start_f)
        chord_tension[start_f:end_f]   = tension
        chord_root_idx[start_f:end_f]  = root_idx

        if prev_label is not None and label != prev_label and start_f < n_frames:
            chord_is_change[start_f] = True
        prev_label = label

    # 3-frame median smoothing on root_idx to suppress single-frame jitter
    # (autochord can produce 1-frame stray labels at segment boundaries)
    from scipy.ndimage import median_filter
    chord_root_idx = median_filter(chord_root_idx, size=3).astype(np.int32)

    # Compact chord_sequence for manifest metadata
    chord_sequence = [
        {"start": s, "end": e, "chord": lbl,
         "tension": CHORD_TENSION.get(_parse_chord(lbl)[1], 0.3)}
        for s, e, lbl in chords
    ]

    return HarmonicFeatures(
        chord_root=chord_root,
        chord_root_index=chord_root_idx,
        chord_quality=chord_quality,
        chord_extension=chord_extension,
        chord_tension=chord_tension,
        chord_is_change=chord_is_change,
        chord_sequence=chord_sequence,
    )
```

```python
# In core/analyzer.py — integrate into analyze():
def analyze(self, decomposed: DecomposedAudio) -> ExtractedFeatures:
    ...
    harmonic = None
    if self.use_harmony:
        try:
            harmonic = self.extract_harmony(
                decomposed.audio_path, n_frames=n_frames,
                sr=self.sr, hop_length=self.hop_length
            )
        except ImportError:
            pass  # autochord not installed — harmonic stays None, fields absent from manifest
    ...
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

**Critical gotchas:**
1. **Accuracy ~65–75% on real mixes** — autochord works well on clean piano/guitar recordings but degrades on dense electronic or heavily distorted material. The `chord_tension` shape is still musically useful even when the specific label is wrong (a wrong chord that's still in the right tension region looks fine visually). Don't over-optimize for label accuracy.
2. **Segment-level only, not per-frame** — autochord returns time-windowed labels, not per-frame data. All frames within a segment get the same label. This is expected and correct: chords don't change faster than ~0.5 seconds in real music. Do not attempt to interpolate tension between segment boundaries — the step change is musically correct.
3. **Enharmonic ambiguity (C:maj vs A:min)** — These share nearly identical pitch content. autochord may flip between them on clean root-position chords when the bass is unclear. The K-S key context from Phase 1 (`key.root`, `key.mode`) can post-hoc disambiguate: if `key_root="C"` and `key_mode="major"`, prefer "C:maj" over "A:min" for the same frames. This is a nice-to-have refinement, not blocking.
4. **"N" (no-chord) segments** — silence, percussion-only sections, and inharmonic bursts return `"N"`. This is valid. Set `chord_root="N"`, `chord_root_index=0`, `chord_tension=0.0`. Never pass `"N"` through `NOTE_INDEX` lookup or it will KeyError.
5. **Unknown quality strings** — autochord occasionally emits non-standard labels (e.g., `"maj6"`, `"min9"`, `"add9"`, `"11"`). `CHORD_TENSION.get(quality, 0.3)` handles these gracefully. Log at DEBUG level so the user can extend CHORD_TENSION if needed.
6. **`audio_path` must be stored on DecomposedAudio** — `extract_harmony()` needs to call `autochord.recognize(path)` but by the time analysis runs, we only have the numpy arrays. Add `audio_path: str | None = None` to `DecomposedAudio` in `decomposer.py` and populate it in `decompose_file()`.

**TODO (stub status):**
- [x] Add `autochord` to `pyproject.toml` optional extras: `pip install -e ".[harmony]"`
- [x] Add `HarmonicFeatures` dataclass stub (returns `None` fields when autochord absent)
- [x] Add `use_neural_beats` / `use_harmony` flags reserved in `AudioPipeline`
- [ ] **`decomposer.py`** — Add `audio_path: str | None` field to `DecomposedAudio`; populate in `decompose_file()` (~5 lines)
- [ ] **`analyzer.py`** — Add `NOTE_INDEX`, `CHORD_TENSION` constants + `_parse_chord()` helper (~45 lines)
- [ ] **`analyzer.py`** — Populate `HarmonicFeatures` dataclass fields: `chord_root`, `chord_root_index`, `chord_quality`, `chord_extension`, `chord_tension`, `chord_is_change`, `chord_sequence` (~15 lines)
- [ ] **`analyzer.py`** — Implement `extract_harmony()` as shown above (~70 lines)
- [ ] **`analyzer.py`** — Wire `use_harmony` flag into `analyze()` with ImportError fallback (~10 lines)
- [ ] **`polisher.py`** — Add `chord_tension`, `chord_root_index`, `chord_quality`, `chord_is_change` to `PolishedFeatures` dataclass (~10 lines); add `chord_tension` EMA smoothing (α=0.12) in `polish()` (~20 lines); pass remaining chord fields through unchanged (~15 lines)
- [ ] **`exporter.py`** — Add chord fields to `_build_frame()` (~25 lines); add `chord_sequence` to manifest metadata block (~15 lines); add chord fields to `export_numpy()` schema (~5 lines)
- [ ] **`tests/core/test_chord_detection.py`** — Mock `autochord.recognize` with a known I–IV–V–I progression; assert chord_root arrays, tension values, is_change triggers, "N" handling, unknown-quality fallback (~130 lines)

---

### W4 — Onset Envelope Shape (Attack Profiling) ✓ COMPLETE

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
- `percussive` → standard beat-driven flash
- `harmonic` → soft bloom, no reseed in attractor

> **Implementation note:** The actual implementation uses 3 onset types — `"transient"`, `"percussive"`, `"harmonic"` — derived from HPSS ratios and RMS envelope shape, rather than the 4 proposed above (`transient`/`metallic`/`swell`/`pluck`). The HPSS-based approach is more reliable on mixed material.

**TODO:**
- [x] Implement `classify_onset_shape()` in `FeatureAnalyzer`
- [x] Add onset classification to `TemporalFeatures`
- [x] Export `onset_type` and `onset_sharpness` to per-frame manifest data
- [x] Add to `PolishedFeatures`

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
| `1.1` | 7-band energy, chroma, spectral features, 6 primitives |
| `2.0` ✓ **current** | + MFCC delta/delta2, timbre_velocity, song structure (C2), pitch tracking/register (C3), key/mode (C4), downbeats/bar grid (C5), CQT sub-bass opt-in (C6), spectral bandwidth/contrast (C7), onset shape 3-type (W4) |
| `3.0` | + stem energy fields (drums/bass/vocals/other), neural beats, chord identity, chord tension |
| `4.0` | + CREPE pitch spectrum, per-section emotion (arousal/valence), MERT embedding_3d |

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

### Phase 1 — Crawl ✓ ALL COMPLETE

**C1: MFCC Export**
- [x] Add `mfcc_delta`, `mfcc_delta2` to `FeatureAnalyzer.extract_tonality()`
- [x] Add MFCC fields to `PolishedFeatures`
- [x] Export MFCC arrays to manifest
- [x] Add `timbre_velocity` primitive
- [x] Update `export_numpy()`
- [x] Tests for MFCC delta extraction
- [x] Bump `ANALYSIS_VERSION` → `2.0`

**C2: Song Structure**
- [x] Add `StructuralFeatures` dataclass
- [x] Implement `extract_structure()` using `librosa.segment`
- [x] Add `section_index`, `section_novelty`, `section_progress`, `section_change` to `PolishedFeatures`
- [x] Export per-frame fields + `structure` metadata block
- [x] Tests: verify section count > 1 on multi-minute track

**C3: Pitch Tracking**
- [x] Add `f0_hz`, `f0_voiced`, `f0_probs` to `TonalityFeatures`
- [x] Implement `librosa.pyin()` call in `extract_tonality()` (at 4× hop for OOM safety)
- [x] Compute `pitch_register` (normalized 0–1); `pitch_velocity` deferred
- [x] Add confidence-weighted smoothing
- [x] Export all pitch fields
- [x] Tests: verify `f0_voiced` False on silent sections

**C4: Key/Mode Detection**
- [x] Implement Krumhansl-Schmuckler profile correlator
- [x] Add section-level key detection (8-second windows)
- [x] Add `key_stability` per-frame signal
- [x] Export `key` block to manifest metadata
- [x] Tests: verify major/minor classification on known-key test tracks

**C5: Downbeat Tracking**
- [x] Implement downbeat detection heuristic in `extract_temporal()`
- [x] Add `is_downbeat`, `beat_position`, `bar_index`, `bar_progress`
- [x] Export to manifest
- [x] Tests: verify `beat_position` cycles 1–4

**C6: CQT Sub-Bass**
- [x] Add CQT computation to `_extract_frequency_bands()`
- [x] Add `sub_bass_cqt`, `bass_cqt` fields
- [x] Benchmark CQT vs. bandpass runtime
- [x] Add `use_cqt: bool` config flag (default False, opt-in)
- [x] Tests: verify CQT sub-bass > 0 on kick-heavy test track

**C7: Bandwidth + Contrast**
- [x] Add `spectral_bandwidth` and `spectral_contrast` extraction
- [x] Export to manifest
- [x] Add `bandwidth_norm` to primitives
- [x] Tests

---

### Phase 2 — Walk (partial)

**W1: Demucs Source Separation** *(stub)*
- [x] Add `demucs` optional extra to `pyproject.toml`
- [x] Add `SeparatedAudio` dataclass
- [x] Add `SourceSeparator` stub (raises `ImportError` when demucs absent)
- [x] Add `use_demucs: bool` to `AudioPipeline`
- [ ] Implement actual stem separation in `SourceSeparator.separate()`
- [ ] Per-stem feature extraction in `FeatureAnalyzer.analyze()`
- [ ] Add stem fields to `PolishedFeatures`
- [ ] Export stem fields to manifest
- [ ] Update cache key for `use_demucs` flag
- [ ] Tests in `tests/core/test_source_separation.py`
- [ ] Document expected runtime (CPU/GPU)

**W2: Madmom Neural Beat Tracking** *(stub)*
- [x] Add `madmom` optional extra to `pyproject.toml`
- [x] Add `use_neural_beats: bool` to `AudioPipeline`
- [x] Fallback to librosa C5 heuristic when madmom absent
- [ ] Implement `extract_temporal_neural()` with actual madmom RNN processor
- [ ] Add `time_signature`, `swing_ratio`, `tempo_stability` to `TemporalFeatures`
- [ ] Export to manifest metadata
- [ ] Tests in `tests/core/test_neural_beats.py`

**W3: Chord Detection** *(stub)*
- [x] Add `autochord` optional extra to `pyproject.toml`
- [x] Add `HarmonicFeatures` dataclass stub (fields all `None` when absent)
- [ ] Implement `extract_harmony()` method
- [ ] Add chord fields to `PolishedFeatures`
- [ ] Add `chord_tension` to derived primitives
- [ ] Export chord sequence to metadata + per-frame fields
- [ ] Tests in `tests/core/test_chord_detection.py`

**W4: Onset Envelope Shape** ✓ COMPLETE
- [x] Implement `classify_onset_shape()` in `FeatureAnalyzer` (3 types: transient/percussive/harmonic)
- [x] Add onset type/sharpness to `TemporalFeatures`
- [x] Export to per-frame manifest data
- [x] Tests: onset shape classification on synthetic signals

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

## Recommended Next Steps

Phase 1 is complete. The highest-impact remaining work in priority order:

1. **Retrofit remaining experiment renderers** — chemical, fractal/kaleidoscope, decay, solar all have Phase 1 wiring plans defined in the Retrofit Guide below. The attractor is done. Each renderer retrofit is 1–2 hours. chemical.py is the most impactful (section flush + bandwidth → crystal lattice order).
2. **Wire Phase 1 fields into `frontend/app.js`** — `section_index`, `is_downbeat`, `bar_progress`, `key_mode` have specific per-style mappings defined in the Frontend Retrofit section. The manifest already carries all these fields; the frontend just isn't reading them yet.
3. **Implement W3 chord detection** — the highest-value Phase 2 feature. Unlocks tension arcs, hue shift relative to tonic, shape morph on chord change. Requires `pip install autochord`.
4. **Implement W1 source separation (Demucs)** — eliminates bass/kick bleed from all band readings. Transforms the attractor's ρ and scale drivers. Requires GPU or patience (CPU ~3×RT).
5. **Upgrade to WAV/FLAC source audio** — still the best single improvement for beat tracking and onset detection accuracy. Zero code changes.

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

#### `experiment/attractor.py` — AttractorRenderer ✓ Phase 1 COMPLETE

Currently uses (from `_smooth_audio` override): `sub_bass`, `percussive_impact`, `brilliance`, `spectral_flux`, `harmonic_energy`, `global_energy`, `pitch_hue`, `spectral_centroid`, `spectral_flatness`, `is_beat`, **plus all Phase 1 fields below**.

**Phase 1 additions (all wired — commit `ad53d84`):**
| New field | Mapping |
|---|---|
| `is_downbeat` | 1.6× flash strength + 1.8× camera kick vs regular beat shockwave |
| `section_change` | Golden-ratio hue jump, 40% mass reseed, bloom flash |
| `pitch_register` | Camera elevation target: `0.05 + pitch_register * 0.55` (was harmonic-based) |
| `key_stability` | Trail decay stability bonus (`+ key_stability * 0.04`); Rössler c chaos term |
| `timbre_velocity` | Lorenz σ turbulence (`+ timbre_velocity * 0.3 * s`) |
| `beat_position` | Bar breathing: `cos(beat_position * 2π) * 0.04` scale pulse |
| `onset_type` | `"harmonic"` skips reseed; `"transient"` adds azimuth micro-jolt |

**Phase 2 additions (planned):**
| New field | Mapping |
|---|---|
| `drums_energy` | Replace `percussive_impact` as the Lorenz ρ driver (cleaner, no bleed) |
| `bass_energy` | Drive scale pulse (the "cloud breathes" effect was sub_bass; bass is cleaner) |
| `chord_tension` | Increase Lorenz σ chaos range — high tension = wilder attractor |
| `chord_is_change` | Small reseed burst at chord boundaries |

**TODO:**
- [x] Add `_smooth_pitch_register`, `_smooth_key_stability`, `_smooth_timbre_velocity`, `_smooth_bandwidth` to attractor's `_smooth_audio()` override
- [x] Wire `is_downbeat` → shockwave system (1.6× flash, 1.8× camera kick)
- [x] Wire `section_change` → hue jump + mass reseed + bloom
- [x] Wire `pitch_register` → camera elevation
- [x] Wire `key_stability` → trail decay + Rössler c
- [x] Wire `timbre_velocity` → Lorenz σ turbulence
- [x] Wire `beat_position` → bar breathing scale pulse
- [x] Wire `onset_type` → reseed skip / azimuth jolt
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

**`base.py` (`BaseVisualizer` + `VisualPolisher`)** — *pending*
- [ ] Add Phase 1 fields to `_smooth_audio()`: `section_novelty`, `pitch_velocity`, `f0_hz`, `bar_progress`, `spectral_bandwidth`, `timbre_velocity`
- [ ] Add all new smoothed fields to `_smooth_audio_dict()`
- [ ] Initialize all new smoothed fields in `__init__()` with safe defaults
- [ ] Upgrade `VisualPolisher.apply()` with Phase 1 hue/saturation/glow logic
- [ ] Add `is_downbeat` vignette pulse to `VisualPolisher`
- [ ] Phase 2: Add `drums_energy`, `bass_energy`, `chord_tension` to `_smooth_audio()`
- [ ] Phase 2: Upgrade `VisualPolisher.apply()` CA + saturation with chord/stem fields

**`experiment/attractor.py`** ✓ Phase 1 DONE (commit `ad53d84`)
- [x] Phase 1: downbeat shockwave (1.6×), section hue jump + reseed, pitch_register → camera elevation, key_stability → trail/chaos, timbre_velocity → σ turbulence, bar breathing, onset_type differentiation
- [ ] Phase 2: drums/bass stem replacement, chord_tension → Lorenz σ

**`experiment/chemical.py`** — *Phase 1 pending*
- [ ] Phase 1: downbeat injection (2×), section novelty flush, bandwidth → lattice order
- [ ] Phase 2: stem replacements, chord_tension → edge sharpness, onset_type → growth shape

**`experiment/fractal.py` + `experiment/kaleidoscope.py`** — *Phase 1 pending*
- [ ] Phase 1: section_index → Julia family, pitch_velocity → c drift, bar_progress → warp
- [ ] Phase 2: chord_tension → zoom depth, chord_root_index → hue

**`experiment/decay.py`** — *Phase 1 pending*
- [ ] Phase 1: downbeat mega-burst, pitch_velocity → spawn direction
- [ ] Phase 2: onset_type → burst shape, drums_energy → spawn rate

**`experiment/solar.py`** — *Phase 1 pending*
- [ ] Phase 1: bar_progress → orbital sync, key_mode → star color
- [ ] Phase 2: chord_tension → solar activity

**`visualizers/kaleidoscope.py`** (legacy Python renderer) — *pending*
- [ ] Audit all direct dict accesses; replace with `.get()` + defaults
- [ ] Confirm works with both schema_version 1.1 and 2.0

**`frontend/app.js`** — *Phase 1 pending*
- [ ] Audit all `frameData.field` direct accesses → add `?? default`
- [ ] Add new fields to `smoothedValues` init and update loop
- [ ] Add `section_index` global palette handler
- [ ] Add `is_downbeat` as stronger beat trigger in all 12 styles
- [ ] Add `bar_progress` to animation systems
- [ ] Wire Phase 1 fields to style-specific visual parameters (see mapping table above)
- [ ] Add schema version feature flags at manifest load

**`io/exporter.py`** ✓ Phase 1 DONE
- [x] All Phase 1 `PolishedFeatures` fields added to `_build_frame()` and `export_numpy()`
- [x] `schema_version = "2.0"` on Phase 1 completion
- [ ] Bump `schema_version` → `"3.0"` on Walk completion, `"4.0"` on Run

**`pipeline.py`** ✓ Phase 1 DONE
- [x] `ANALYSIS_VERSION = "2.0"` bumped
- [x] Cache key includes `use_cqt` flag
- [x] `use_demucs`, `use_neural_beats` flags exist
- [ ] Add `use_crepe` flag when R1 is implemented

---

*This document supersedes the "Medium Term" and "Long Term" items in `docs/project/roadmap.md` for the audio analysis domain. The roadmap.md items should be updated to reference this proposal.*
