# Chromascope

Audio analysis engine for reactive generative art. Chromascope analyzes music
down to the level of musical structure — song sections, key, pitch contour,
downbeats, onset character — and produces a versioned, renderer-agnostic
manifest that drives visualization without the renderer touching any DSP logic.

![Chromascope Teaser](docs/assets/preview.gif)

---

## Web Interface

11 real-time visualization styles driven by the Web Audio API. Load a manifest
for frame-accurate playback, or connect live audio.

Styles: Geometric, Glass, Flower, Spiral, Circuit, Fibonacci, DMT, Sacred,
Mycelial, Fluid, Orrery (binary star + 8-planet N-body), Quark.

```bash
cd frontend && python server.py
# open http://localhost:8000
```

---

## Audio Intelligence (Schema 2.0)

Most visualizers bind an FFT snapshot directly to a visual frame. Chromascope
builds a **musical manifest** — a versioned, frame-accurate JSON document that
answers musical questions.

### What the manifest knows

| Domain | Fields |
|--------|--------|
| **Song structure** | `section_index`, `section_novelty`, `section_progress`, `section_change` |
| **Key & harmony** | `key.root`, `key.mode`, `key.confidence`, `key_stability` per frame |
| **Pitch tracking** | `f0_hz`, `f0_voiced`, `f0_probs`, `pitch_register` |
| **Bar grid** | `is_downbeat`, `beat_position`, `bar_index`, `bar_progress` |
| **Onset shape** | `onset_type` (transient / percussive / harmonic), `onset_sharpness` |
| **Timbre** | `mfcc`, `mfcc_delta`, `mfcc_delta2`, `timbre_velocity` |
| **Spectrum** | `spectral_bandwidth`, `spectral_contrast` (7 bands), `sub_bass_cqt`, `bass_cqt` |
| **Energy** | `percussive_impact`, `harmonic_energy`, `sub_bass`–`brilliance` (7 bands) |
| **Primitives** | `impact`, `fluidity`, `brightness`, `pitch_hue`, `texture`, `sharpness` |

The manifest is renderer-agnostic. One analysis pass powers all renderers.
Manifests are cached by content hash; re-renders are instant.

See [`docs/MANIFEST.md`](docs/MANIFEST.md) for a full annotated example and
field reference.

---

## Pipeline

```
Audio file
    │
    ▼
Decompose   HPSS: harmonic + percussive streams
    │
    ▼
Analyze     40+ features @ target FPS, schema 2.0
    │
    ▼
Polish      Attack/release envelopes, normalization, smoothing
    │
    ▼
Manifest    Versioned JSON, cached by content hash
```

Separating analysis from rendering means:
- Re-render with different visual parameters without re-analyzing audio
- Multiple renderers consume the same manifest
- Renderer code has no DSP logic — it reads normalized [0,1] control signals
  and event triggers

---

## Installation

```bash
git clone https://github.com/nadiabellamorris/chromascope.git
cd chromascope
pip install -e .                    # core pipeline + analysis
pip install -e ".[analysis-full]"   # + Demucs stems, madmom beats, autochord
```

**Requirements:** Python ≥ 3.10, librosa, numpy, scipy

---

## Analyze audio into a manifest

```bash
chromascope track.mp3 --output manifest.json
```

The manifest is automatically cached at `~/.cache/chromascope/manifests/`.
Subsequent renders skip analysis entirely.

---

## Project Structure

```
src/chromascope/
├── core/
│   ├── decomposer.py   HPSS decomposition + SeparatedAudio stub (Demucs path)
│   ├── analyzer.py     Feature extraction (40+ fields, schema 2.0)
│   ├── polisher.py     Envelope shaping + normalization
│   └── stream.py       RealtimeAnalyzer stub (Phase 3)
├── io/
│   └── exporter.py     Manifest serialization (JSON + NumPy)
└── pipeline.py         AudioPipeline orchestrator + manifest cache
frontend/
    app.js              11-style real-time renderer (~6500 lines)
```

---

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## Roadmap

**Implemented (schema 2.0):** full Phase 1 audio intelligence — song structure,
pitch tracking, key/mode, downbeat grid, onset shape, timbre velocity, CQT
sub-bass, bandwidth/contrast.

**Planned:** Demucs source separation (drums/bass/vocals as isolated control
channels), neural beat tracking (madmom RNN), chord detection, real-time OSC
streaming, Ableton integration.

See [`docs/project/AUDIO_INTELLIGENCE_PROPOSAL.md`](docs/project/AUDIO_INTELLIGENCE_PROPOSAL.md)
for the full roadmap.
