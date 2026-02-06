# Architecture: Project Audio Analysisussy

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
    - `Chroma`: 12-bin note intensity.
    - `Spectral Centroid`: Perceived "brightness."

### Phase C: Signal Smoothing (`/src/core/polisher.py`)
- **Input:** Raw feature data.
- **Logic:** Applies Attack/Decay envelopes to prevent visual flickering.
- **Normalization:** All values mapped to `[0.0 - 1.0]`.

### Phase D: Serialization (`/src/io/exporter.py`)
- **Input:** Polished signals.
- **Output:** `manifest.json` aligned to a specific FPS (default 60).

## 3. Data Contract (The Manifest Schema)
Every frame in the output JSON must follow this structure:
```json
{
  "frame": "int",
  "time": "float",
  "is_beat": "bool",
  "impact": "float",    // From Percussive component
  "fluidity": "float",  // From Harmonic component
  "brightness": "float" // Spectral Centroid
}

## 4. Architectural Decision Log (ADR)
The Trail: Newest decisions at the top.

ADR-001 (2023-10-27): Decided to use librosa over scipy.fft for initial extraction to leverage high-level MIR (Music Information Retrieval) functions and better beat tracking.

ADR-002 (2023-10-27): Implemented HPSS separation as the first step to ensure "Drum" visuals don't get triggered by loud "Melody" notes.


## 5. Testing Infrastructure
- **Framework:** `pytest`
- **Mock Data:** `/tests/assets/` contains short (2-second) reference audio clips:
    - `pure_sine.wav`: For testing frequency/pitch accuracy.
    - `click_track.wav`: For testing beat and onset detection.
    - `white_noise.wav`: For testing normalization and clipping.
- **Coverage Goal:** 90% coverage for the `core/` logic.
