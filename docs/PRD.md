# PRD: Project chromascope

**Objective:** To build a robust Python-based audio analysis engine that extracts high-fidelity musical features for the purpose of driving reactive generative art.

---

## 1. System Overview

The system will take an audio file as input and generate a time-aligned "Visual Driver Manifest" (JSON or NumPy). This manifest will contain discrete triggers (beats, onsets) and continuous signals (volume envelopes, spectral flux) normalized for immediate use in rendering engines.

---

## 2. Functional Requirements

### 2.1 Audio Pre-processing (The "Feel" Separation)

The engine must perform **Harmonic-Percussive Source Separation (HPSS)**.

* **Requirement:** Separate the signal into a `.percussive` array (drums/transients) and a `.harmonic` array (chords/melody).
* **Purpose:** Allows the developer to map "impact" visuals to percussion and "flow" visuals to harmony independently.

### 2.2 Feature Extraction Suite

The developer must implement extraction for the following "Visual Drivers":

1. **Temporal Drivers:**
* **Beat Tracking:** Global BPM and a boolean array of beat timestamps.
* **Onset Detection:** Identification of sudden "attacks" (notes) that aren't necessarily on the beat.


2. **Energy Drivers:**
* **RMS Energy:** Root Mean Square energy for global volume.
* **Frequency Bands:** Sub-division into Low (0–200Hz), Mid (200Hz–4kHz), and High (4kHz+).


3. **Tonality Drivers:**
* **Chroma Features:** A 12-bin representation of the intensity of each semitone (C, C#, D, etc.).
* **Spectral Centroid:** The "center of mass" of the sound (the "brightness" factor).



### 2.3 The "Aesthetic" Smoothing Layer

Raw audio data is too jittery for smooth visuals. The pipeline must include:

* **Envelope Following:** Implementation of attack/release constants.
* *Example:* If a beat hits, the visual value jumps to 1.0 instantly (0ms attack) but fades to 0.0 over 500ms (release) to create a "glow" effect.


* **Normalization:** All output values must be scaled between `[0.0, 1.0]` or `[-1.0, 1.0]`.

---

## 3. Technical Specifications

* **Primary Language:** Python 3.10+
* **Core Libraries:** * `librosa` (Analysis)
* `numpy` (Data handling)
* `scipy.signal` (Filtering/Smoothing)
* `demucs` (Optional: AI-based stem separation for high-end tasks)


* **Alignment:** All data must be sampled to match a user-defined **Target FPS** (e.g., 60fps). If the audio sample rate is 44.1kHz, the developer must calculate the correct `hop_length` to ensure 1 frame of data = 1 frame of video.

---

## 4. Data Output Schema (The "Manifest")

The output should be a structured object (JSON) following this logic:

```json
{
  "metadata": { "bpm": 124, "duration": 180.5, "fps": 60 },
  "frames": [
    {
      "frame_index": 0,
      "is_beat": true,
      "percussive_impact": 0.85,
      "harmonic_energy": 0.42,
      "dominant_chroma": "G#",
      "spectral_brightness": 0.22
    },
    ...
  ]
}

```

---

## 5. Success Criteria

* **Phase 1:** The script can process a 3-minute MP3 in under 30 seconds.
* **Phase 2:** The "is_beat" trigger aligns within +/- 10ms of the actual audible transient.
* **Phase 3:** Visualizations driven by the "harmonic_energy" appear fluid and organic, not flickering or "noisy."

---

> **Note to Developer:** High-quality visualization isn't about raw data; it's about **intent**. Prioritize the separation of percussive transients from harmonic textures, as this is the difference between a generic "bars" visualizer and a professional music video.

