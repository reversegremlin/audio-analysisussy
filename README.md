# Chromascope: A Distinctive Audio-to-Visual Engine

Chromascope is not a generic “bars-and-waveform” visualizer. It is a **structured audio-to-visual engine** that converts music into a reusable, frame-accurate control language for many different renderers.

At its core, Chromascope treats audio analysis as a **rendering contract**, not a one-off effect.

![Chromascope Teaser](docs/assets/preview.gif)

---

## Why This Engine Is Distinctive

Most visualizers bind one FFT snapshot directly to one visual frame.
Chromascope instead uses a deliberate multi-stage architecture:

1. **Decompose** audio into harmonic and percussive roles.
2. **Extract** rich musical features with timing alignment.
3. **Polish** signals with envelope shaping and normalization.
4. **Serialize** everything into a renderer-agnostic visual manifest.

This design makes visual behavior more musical, more stable, and easier to reuse across very different styles.

---

## Engine Architecture

### Phase A — Harmonic/Percussive Decomposition
Chromascope starts with HPSS (Harmonic-Percussive Source Separation):

- **Percussive stream** drives impact/transient behavior.
- **Harmonic stream** drives flow/tonal behavior.

This avoids common visualizer failure modes where loud melodic content triggers drum-like effects.

### Phase B — Feature Extraction for Visual Control
The engine extracts synchronized, frame-aligned drivers including:

- Beat and onset timing
- Global/harmonic/percussive energy
- Multi-band frequency energy (sub-bass through brilliance)
- Chroma and dominant note identity
- Spectral shape features (flux, centroid/brightness, flatness, rolloff)
- Zero-crossing rate and MFCC timbre descriptors

These are chosen as **control signals** for animation systems, not just analytics outputs.

### Phase C — Signal Polishing Layer
Raw audio features are not yet visually usable.
Chromascope applies:

- Attack/release envelopes (fast rise, controlled decay)
- Feature-wise normalization to `[0, 1]`
- Optional BPM-adaptive envelope scaling

This produces smooth, stable, flicker-resistant motion and parameter changes.

### Phase D — Manifest as a Stable Contract
The final output is a versioned JSON manifest with per-frame values and semantic primitives:

- `impact`
- `fluidity`
- `brightness`
- `pitch_hue`
- `texture`
- `sharpness`

Renderers can consume this contract without being tightly coupled to low-level DSP details.

---

## The Visual Driver Manifest

The manifest is the key abstraction in Chromascope:

- **Frame-accurate timing** at target FPS (default 60)
- **Normalized continuous controls** for deterministic mapping
- **Beat/onset trigger channels** for event-driven visuals
- **Schema/version metadata** for forward evolution

This enables one analysis pass to power multiple rendering systems consistently.

---

## What You Can Build With It

Chromascope ships with both real-time and cinematic pathways:

- **Web interface (`frontend/`)** for interactive experimentation with multiple visual styles.
- **Python experiment framework (`src/chromascope/experiment/`)** for high-fidelity offline rendering.

Because both consume the same audio-control semantics, ideas transfer cleanly from prototyping to final output.

---

## Quick Start

### Installation
```bash
git clone https://github.com/nadiabellamorris/chromascope.git
cd chromascope
pip install -e .
```

### Analyze audio into a visual manifest
```bash
chromascope path/to/audio.wav --output manifest.json
```

### Run the interactive web renderer
```bash
cd frontend
python server.py
# then open http://localhost:8000
```

### Generate cinematic output (example)
```bash
chromascope-decay my_track.wav --output video.mp4 --style dark_nebula
```

---

## Design Principles

- **Musical separation before mapping** (harmonic vs percussive roles)
- **Renderer-agnostic control contract** (manifest-first design)
- **Temporal coherence over raw reactivity** (polishing layer)
- **Modular pipeline components** that can evolve independently

---

Chromascope is best understood as a **sound-to-parameter engine** for visual simulation systems.
It does not just “draw to music”; it builds a durable language that renderers can perform.
