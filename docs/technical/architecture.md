# Architecture: Project Chromascope

## 1. System Vision
A decoupled pipeline that transforms raw audio into a "Visual Driver Manifest" (JSON). The system prioritizes the separation of percussive transients from harmonic textures to create "musical" visuals rather than just frequency-reactive ones.

## 2. The Modular Pipeline
The system is divided into four distinct phases:

### Phase A: Input & Decomposition (`/src/core/decomposer.py`)
- **Input:** Audio file (wav, mp3, flac).
- **Process:** HPSS (Harmonic-Percussive Source Separation).
- **Output:** Two distinct signal arrays (Harmonic/Percussive).

### Phase B: Feature Extraction (`/src/core/analyzer.py`)
- **Input:** Decomposed signals.
- **Features extracted:**
    - `Onsets`: Binary triggers for visuals.
    - `BPM/Beats`: Global pulse.
    - `RMS`: Global and band-specific amplitude.
    - `Spectral Flux`: Measures change in the spectrum (novelty).
    - `Chroma`: 12-bin note intensity.
    - `Spectral Characteristics`: Centroid (brightness), Flatness (noisiness), Rolloff (bandwidth).
    - `Zero Crossing Rate`: High-frequency/noise indicator.

### Phase C: Signal Smoothing (`/src/core/polisher.py`)
- **Input:** Raw feature data.
- **Logic:** Applies Attack/Decay envelopes to prevent visual flickering.
- **Adaptive Envelopes:** Optionally scales release times based on detected BPM.
- **Normalization:** All values mapped to `[0.0 - 1.0]`.

### Phase D: Serialization (`/src/io/exporter.py`)
- **Input:** Polished signals.
- **Output:** `manifest.json` aligned to a specific FPS (default 60).

## 3. Data Contract (The Manifest Schema)
The Visual Driver Manifest is a JSON document with a metadata header and
frame-by-frame data. It is versioned via `schema_version` to allow the
feature set to evolve over time without breaking consumers.

```json
{
  "metadata": {
    "bpm": "float",
    "duration": "float",
    "fps": "int",
    "n_frames": "int",
    "version": "string",         // Exporter/engine version
    "schema_version": "string"   // Manifest schema version, e.g. "1.1"
  },
  "frames": [
    {
      "frame_index": "int",
      "time": "float",
      "is_beat": "bool",
      "is_onset": "bool",

      // Core continuous drivers (polished signals, all normalized [0.0, 1.0])
      "percussive_impact": "float",
      "harmonic_energy": "float",
      "global_energy": "float",
      "spectral_flux": "float",
      
      // 7-Band Energy System
      "sub_bass": "float",
      "bass": "float",
      "low_mid": "float",
      "mid": "float",
      "high_mid": "float",
      "presence": "float",
      "brilliance": "float",

      // Legacy bands
      "low_energy": "float",
      "mid_energy": "float",
      "high_energy": "float",

      // Texture/Spectral features
      "spectral_brightness": "float",
      "spectral_flatness": "float",
      "spectral_rolloff": "float",
      "zero_crossing_rate": "float",

      // Tonality
      "dominant_chroma": "string",        // Note name: C, C#, ..., B
      "chroma_values": { "C": "float", "C#": "float", "...": "float" },

      // Visual primitives: renderer-agnostic semantic controls
      "impact": "float",      // Alias of percussive_impact
      "fluidity": "float",    // Alias of harmonic_energy
      "brightness": "float",  // Alias of spectral_brightness
      "pitch_hue": "float",   // Normalized [0.0, 1.0] mapping of dominant_chroma
      "texture": "float",     // Aggregated flatness/zcr/presence/brilliance
      "sharpness": "float"    // Aggregated flux/rolloff
    }
  ]
}
```

## 4. Architectural Decision Log (ADR)
The Trail: Newest decisions at the top.

ADR-004 (2026-02-17): Expanded feature extraction to include spectral flux, flatness, rolloff, and zero-crossing rate. Upgraded frequency band system from 3 to 7 bands for granular visual control. Added `sharpness` visual primitive.

ADR-003 (2026-02-10): Added explicit manifest schema versioning and a visual
primitives layer (`impact`, `fluidity`, `brightness`, `pitch_hue`, `texture`)
computed from polished features. Introduced shared style presets consumed by
both Python and web renderers.

ADR-002 (2023-10-27): Implemented HPSS separation as the first step to ensure "Drum" visuals don't get triggered by loud "Melody" notes.

ADR-001 (2023-10-27): Decided to use librosa over scipy.fft for initial extraction to leverage high-level MIR (Music Information Retrieval) functions and better beat tracking.


## 5. Testing Infrastructure
- **Framework:** `pytest`
- **Mock Data:** `/tests/assets/` contains short (2-second) reference audio clips:
    - `pure_sine.wav`: For testing frequency/pitch accuracy.
    - `click_track.wav`: For testing beat and onset detection.
    - `white_noise.wav`: For testing normalization and clipping.
- **Coverage Goal:** 90% coverage for the `core/` logic.
