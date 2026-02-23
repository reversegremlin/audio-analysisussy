# The Chromascope Manifest

The manifest is a versioned JSON document that bridges audio analysis and visual rendering. One analysis pass produces it; every renderer consumes it. The key design rule: **renderers contain no DSP code** — they only read normalized [0,1] control signals and boolean event triggers from the manifest.

---

## Structure

```
manifest.json
├── metadata          ← global track info, schema version, structure + key blocks
└── frames[]          ← per-frame array (one entry per frame at target FPS)
    ├── timing        ← frame_index, time
    ├── energy        ← percussive_impact, harmonic_energy, global_energy, spectral_flux
    ├── bands         ← sub_bass, bass, low_mid, mid, high_mid, presence, brilliance
    ├── texture       ← spectral_brightness, spectral_flatness, spectral_rolloff, zcr
    ├── tonality      ← chroma_values (12), dominant_chroma
    ├── primitives    ← impact, fluidity, brightness, pitch_hue, texture, sharpness
    ├── triggers      ← is_beat, is_onset
    └── [schema 2.0]  ← timbre_velocity, section_*, f0_*, pitch_register,
                         key_stability, is_downbeat, beat_position, bar_*,
                         onset_type, onset_sharpness, spectral_bandwidth
```

---

## Annotated Example

Two frames — the first is a downbeat, the second is a quiet inter-beat frame.

```json
{
  "metadata": {
    "version": "2.0",
    "schema_version": "2.0",
    "bpm": 124.3,
    "duration": 214.7,
    "fps": 60,
    "n_frames": 12882,
    "structure": {
      "n_sections": 8,
      "section_boundaries": [0.0, 26.4, 52.8, 79.3, 105.6, 131.9, 158.4, 184.8],
      "section_durations": [26.4, 26.4, 26.5, 26.3, 26.3, 26.5, 26.4, 29.9]
    },
    "key": {
      "root": "F",
      "root_index": 5,
      "mode": "minor",
      "confidence": 0.84,
      "relative_major": "Ab"
    }
  },

  "frames": [

    {
      "frame_index": 5160,
      "time": 86.0,

      // ── Triggers ────────────────────────────────────────────────────────
      "is_beat": true,
      "is_onset": true,

      // ── Core energy ─────────────────────────────────────────────────────
      // All [0, 1]. Attack/release envelopes applied; no raw feature values.
      "percussive_impact": 0.923,   // kick drum landed hard
      "harmonic_energy": 0.541,     // sustained chord layer
      "global_energy": 0.781,
      "spectral_flux": 0.672,       // large spectral change (drop onset)

      // ── 7-band frequency energy ──────────────────────────────────────────
      "sub_bass": 0.887,            // 808/sub hit
      "bass": 0.742,
      "low_mid": 0.431,
      "mid": 0.384,
      "high_mid": 0.261,
      "presence": 0.198,
      "brilliance": 0.134,

      // ── Spectral texture ─────────────────────────────────────────────────
      "spectral_brightness": 0.312, // centroid weighted toward lows on drop
      "spectral_flatness": 0.087,   // tonal (not noisy) — low flatness
      "spectral_rolloff": 0.241,
      "zero_crossing_rate": 0.104,
      "spectral_bandwidth": 0.448,  // [C7] moderate spread

      // ── Tonality ─────────────────────────────────────────────────────────
      "dominant_chroma": "F",
      "chroma_values": {
        "C": 0.121, "C#": 0.043, "D": 0.087, "D#": 0.212,
        "E": 0.031, "F": 0.891,  "F#": 0.044, "G": 0.356,
        "G#": 0.198, "A": 0.067, "A#": 0.312, "B": 0.028
      },

      // ── MFCC / Timbre [C1] ───────────────────────────────────────────────
      // Full mfcc/mfcc_delta/mfcc_delta2 arrays (13 coefficients each) are
      // stored in the .npz archive, not here. timbre_velocity is the
      // L2-norm of mfcc_delta, polished to [0, 1].
      "timbre_velocity": 0.731,     // texture changing fast on the drop

      // ── Song structure [C2] ──────────────────────────────────────────────
      "section_index": 3,           // this is the 4th section (chorus)
      "section_novelty": 0.912,     // high — we just crossed a boundary
      "section_progress": 0.003,    // 0.3% into this section (just started)
      "section_change": true,       // boundary crossed this frame

      // ── Pitch / F0 [C3] ──────────────────────────────────────────────────
      "f0_hz": 349.2,               // F4 — melody note
      "f0_confidence": 0.88,        // voiced, high confidence
      "f0_voiced": true,
      "pitch_register": 0.621,      // upper-mid register (C2–C7 range → [0,1])

      // ── Key stability [C4] ───────────────────────────────────────────────
      "key_stability": 0.831,       // confidently in F minor this frame

      // ── Downbeat grid [C5] ───────────────────────────────────────────────
      "is_downbeat": true,          // beat 1 of a new bar
      "beat_position": 0.0,         // start of the beat cycle
      "bar_index": 64,
      "bar_progress": 0.0,          // start of the bar

      // ── CQT bands [C6, opt-in] ───────────────────────────────────────────
      // null when use_cqt=False (default). Non-null when use_cqt=True.
      "sub_bass_cqt": null,
      "bass_cqt": null,

      // ── Onset shape [W4] ─────────────────────────────────────────────────
      "onset_type": "transient",    // hard attack — kick/snare
      "onset_sharpness": 0.921,     // nearly instantaneous attack

      // ── Semantic primitives ──────────────────────────────────────────────
      // High-level controls distilled from the raw fields above.
      // These are what most renderers should map to visual parameters.
      "impact": 0.923,              // = percussive_impact
      "fluidity": 0.541,            // = harmonic_energy
      "brightness": 0.312,          // = spectral_brightness
      "pitch_hue": 0.455,           // = dominant_chroma_index / 11 (F=5 → 5/11)
      "texture": 0.109,             // = (flatness + zcr + presence + brilliance) / 4
      "sharpness": 0.457,           // = (flux + rolloff) / 2
      "timbre_velocity": 0.731,     // (same as C1 field above)
      "bandwidth_norm": 0.448       // = spectral_bandwidth
    },

    {
      "frame_index": 5180,
      "time": 86.333,

      "is_beat": false,
      "is_onset": false,

      "percussive_impact": 0.312,   // decaying after the kick
      "harmonic_energy": 0.589,     // chord sustain still active
      "global_energy": 0.498,
      "spectral_flux": 0.143,

      "sub_bass": 0.421,
      "bass": 0.502,
      "low_mid": 0.378,
      "mid": 0.411,
      "high_mid": 0.289,
      "presence": 0.234,
      "brilliance": 0.167,

      "spectral_brightness": 0.389,
      "spectral_flatness": 0.112,
      "spectral_rolloff": 0.298,
      "zero_crossing_rate": 0.134,
      "spectral_bandwidth": 0.389,

      "dominant_chroma": "Ab",
      "chroma_values": {
        "C": 0.098, "C#": 0.044, "D": 0.067, "D#": 0.189,
        "E": 0.029, "F": 0.712,  "F#": 0.038, "G": 0.298,
        "G#": 0.734, "A": 0.054, "A#": 0.267, "B": 0.021
      },

      "timbre_velocity": 0.234,     // timbre settling after the hit

      "section_index": 3,
      "section_novelty": 0.312,     // novelty decaying away from boundary
      "section_progress": 0.013,    // 1.3% into the section
      "section_change": false,

      "f0_hz": 207.6,               // Ab3 — bass moving
      "f0_confidence": 0.71,
      "f0_voiced": true,
      "pitch_register": 0.389,

      "key_stability": 0.862,

      "is_downbeat": false,
      "beat_position": 0.333,       // 1/3 of the way through this beat
      "bar_index": 64,
      "bar_progress": 0.083,        // 8.3% through the bar (frame 20 of ~240)

      "sub_bass_cqt": null,
      "bass_cqt": null,

      "onset_type": "harmonic",     // gentle harmonic onset (chord settling)
      "onset_sharpness": 0.189,

      "impact": 0.312,
      "fluidity": 0.589,
      "brightness": 0.389,
      "pitch_hue": 0.727,           // Ab=8 → 8/11 = 0.727
      "texture": 0.162,
      "sharpness": 0.221,
      "timbre_velocity": 0.234,
      "bandwidth_norm": 0.389
    }

  ]
}
```

---

## Field Reference

### Metadata

| Field | Type | Description |
|---|---|---|
| `metadata.version` | string | Engine version (`"2.0"`) |
| `metadata.schema_version` | string | Manifest schema (`"2.0"`) |
| `metadata.bpm` | float | Detected tempo in BPM |
| `metadata.duration` | float | Track duration in seconds |
| `metadata.fps` | int | Analysis frame rate |
| `metadata.n_frames` | int | Total frame count |
| `metadata.structure.n_sections` | int | Number of detected song sections |
| `metadata.structure.section_boundaries` | float[] | Section start times (seconds) |
| `metadata.structure.section_durations` | float[] | Section lengths (seconds) |
| `metadata.key.root` | string | Key root note ("C"…"B") |
| `metadata.key.root_index` | int | Chroma index 0–11 |
| `metadata.key.mode` | string | `"major"` or `"minor"` |
| `metadata.key.confidence` | float\|null | K-S correlation confidence [0,1] |
| `metadata.key.relative_major` | string | Relative major note name |

### Per-Frame Fields

#### Timing
| Field | Type | Range | Notes |
|---|---|---|---|
| `frame_index` | int | 0…n_frames-1 | |
| `time` | float | 0…duration | Seconds |

#### Triggers
| Field | Type | Notes |
|---|---|---|
| `is_beat` | bool | Estimated beat position |
| `is_onset` | bool | Any transient onset |
| `is_downbeat` | bool\|null | Beat 1 of each bar (C5) |
| `section_change` | bool\|null | True only on section boundary frames (C2) |

#### Core Energy — all [0,1], envelope-shaped
| Field | Description |
|---|---|
| `percussive_impact` | HPSS percussive stream energy |
| `harmonic_energy` | HPSS harmonic stream energy |
| `global_energy` | Full-signal RMS |
| `spectral_flux` | Frame-to-frame spectral change magnitude |

#### 7-Band Frequency Energy — all [0,1]
| Field | Approx. Hz Range |
|---|---|
| `sub_bass` | 20–60 Hz |
| `bass` | 60–250 Hz |
| `low_mid` | 250–500 Hz |
| `mid` | 500–2000 Hz |
| `high_mid` | 2000–4000 Hz |
| `presence` | 4000–6000 Hz |
| `brilliance` | 6000+ Hz |

#### Spectral Texture — all [0,1]
| Field | Description |
|---|---|
| `spectral_brightness` | Centroid (weighted avg frequency, normalized) |
| `spectral_flatness` | 0 = tonal/harmonic, 1 = white noise |
| `spectral_rolloff` | Frequency below which 85% of energy sits (normalized) |
| `zero_crossing_rate` | Signal noisiness / presence of high-frequency content |
| `spectral_bandwidth` | Spectral spread around centroid [C7] |

#### Tonality
| Field | Type | Description |
|---|---|---|
| `dominant_chroma` | string | Loudest pitch class this frame |
| `chroma_values` | object | 12 pitch classes, each [0,1] |

#### MFCC / Timbre [C1]
| Field | Type | Range | Description |
|---|---|---|---|
| `timbre_velocity` | float\|null | [0,1] | Speed of timbral change (L2 norm of mfcc_delta). High = rapidly morphing sound character. |

> Full MFCC arrays (13 × n_frames each for mfcc, mfcc_delta, mfcc_delta2) are in the companion `.npz` archive, not in JSON.

#### Song Structure [C2]
| Field | Type | Range | Description |
|---|---|---|---|
| `section_index` | int\|null | 0…n_sections-1 | Which section this frame is in |
| `section_novelty` | float\|null | [0,1] | Peaks at section boundaries, decays away |
| `section_progress` | float\|null | [0,1] | How far through the current section |
| `section_change` | bool\|null | | True only on boundary frames |

#### Pitch Tracking [C3]
| Field | Type | Range | Description |
|---|---|---|---|
| `f0_hz` | float\|null | 65–2093 | Fundamental frequency in Hz. null on unvoiced frames. |
| `f0_confidence` | float\|null | [0,1] | pyin voicing probability |
| `f0_voiced` | bool\|null | | Whether a pitched note was detected |
| `pitch_register` | float\|null | [0,1] | log(f0) mapped to C2–C7 range. Low notes → 0, high → 1. |

#### Key Stability [C4]
| Field | Type | Range | Description |
|---|---|---|---|
| `key_stability` | float\|null | [0,1] | Frame-level confidence in the detected key. Drops as key changes approach. |

#### Bar Grid [C5]
| Field | Type | Range | Description |
|---|---|---|---|
| `beat_position` | float\|null | [0,1) | Phase within the current beat |
| `bar_index` | int\|null | 0… | Bar/measure number since start |
| `bar_progress` | float\|null | [0,1] | Phase within the current 4-beat bar. 0 on downbeat, 1 just before next downbeat. |

#### CQT Bands [C6] — opt-in, null by default
| Field | Type | Range | Description |
|---|---|---|---|
| `sub_bass_cqt` | float\|null | [0,1] | CQT-based sub-bass (better low-frequency resolution). Non-null only with `use_cqt=True`. |
| `bass_cqt` | float\|null | [0,1] | CQT-based bass. |

#### Onset Shape [W4]
| Field | Type | Description |
|---|---|---|
| `onset_type` | string\|null | `"transient"` (hard kick/snare), `"percussive"` (beat without sharp attack), `"harmonic"` (tonal entry, no percussive content) |
| `onset_sharpness` | float\|null | [0,1] — 1.0 = instantaneous attack, 0.0 = slow swell |

#### Semantic Primitives
Pre-computed high-level controls. Renderers that want simplicity can use these exclusively.

| Field | Range | What It Means |
|---|---|---|
| `impact` | [0,1] | How hard is the hit right now |
| `fluidity` | [0,1] | How smooth/sustained is the sound |
| `brightness` | [0,1] | How bright/airy vs dark/heavy |
| `pitch_hue` | [0,1] | Dominant pitch class mapped to a hue (C=0, B≈1) |
| `texture` | [0,1] | How noisy/granular vs clean |
| `sharpness` | [0,1] | How much transient content is present |
| `timbre_velocity` | [0,1] | How fast the sound character is morphing |
| `bandwidth_norm` | [0,1] | How spectrally spread (wide = diffuse, narrow = tonal) |

---

## Consuming the Manifest

### Python

```python
import json

with open("manifest.json") as f:
    manifest = json.load(f)

meta = manifest["metadata"]
print(f"Key: {meta['key']['root']} {meta['key']['mode']}")
print(f"Sections: {meta['structure']['n_sections']}")

for frame in manifest["frames"]:
    t = frame["time"]
    energy = frame["percussive_impact"]
    section = frame.get("section_index") or 0        # null-safe
    downbeat = frame.get("is_downbeat") or False      # null-safe
    onset = frame.get("onset_type") or "percussive"  # null-safe
```

### JavaScript

```javascript
const meta = manifest.metadata;
const schemaVer = meta.schema_version ?? "1.1";
const hasPhase1 = schemaVer >= "2.0";

for (const frame of manifest.frames) {
    const energy   = frame.percussive_impact   ?? 0.0;
    const section  = frame.section_index       ?? 0;
    const downbeat = frame.is_downbeat         ?? false;
    const register = frame.pitch_register      ?? 0.5;
    const onset    = frame.onset_type          ?? "percussive";
    const barProg  = frame.bar_progress        ?? 0.0;
    const stability = frame.key_stability      ?? 1.0;
}
```

> Always use `?? default` (JS) or `.get(key, default)` (Python) when reading per-frame fields. Schema 2.0 fields are `null` when the feature was not computed, and old 1.x manifests simply won't have the key. Direct access crashes on both.

---

## Schema History

| Version | What's in it |
|---|---|
| `1.1` | energy, 7-band frequency, spectral texture, chroma, 6 primitives, beat/onset triggers |
| `2.0` | + timbre_velocity, song structure (C2), pitch tracking (C3), key stability (C4), downbeat grid (C5), CQT bands opt-in (C6), spectral bandwidth (C7), onset shape (W4) |
| `3.0` *(planned)* | + stem energy (drums/bass/vocals/other), neural beats, chord identity + tension |
| `4.0` *(planned)* | + CREPE pitch spectrum, per-section arousal/valence, MERT embedding |

Manifests are cached by content hash at `~/.cache/chromascope/manifests/`. A schema version bump invalidates the cache; re-analysis runs automatically.
