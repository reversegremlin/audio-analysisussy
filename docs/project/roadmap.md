# Roadmap – Chromascope

A living roadmap for planned evolution of the Chromascope analysis and
visualization stack. Items are grouped by theme rather than strict dates.

See [`AUDIO_INTELLIGENCE_PROPOSAL.md`](AUDIO_INTELLIGENCE_PROPOSAL.md) for
the full Phase 1/2/3 breakdown with implementation sketches, gotchas, and
per-item effort estimates. **Phase 1 (schema 2.0) is complete.**

---

## Short Term

- **Wire Phase 1 fields into the JS renderer**: `section_novelty`,
  `pitch_register`, `bar_progress`, `key_mode`, `timbre_velocity`, and
  `spectral_bandwidth` are all in the manifest (schema 2.0) but not yet
  driving visual parameters in `frontend/app.js`. This is free visual
  improvement with no new analysis work required.
- **W2 — Neural beat tracking (madmom)**: Replace the C5 librosa heuristic
  with the madmom RNN beat/downbeat tracker. Produces accurate downbeats,
  time signature, swing ratio, and tempo stability. ~12–16h. Lowest risk of
  the Phase 2 items; improves every renderer immediately. Do this first.
  See proposal §W2 for implementation sketch and gotchas.
- **Schema validation**: Write `docs/manifest.schema.json` (JSON Schema
  draft-07) covering all schema 2.0 fields, and add a `chromascope validate
  <manifest.json>` CLI subcommand. Catches malformed pipeline output early
  and is a prerequisite for multi-renderer work.
- **Manifest consumers**: Expose helper utilities in Python and JavaScript
  for reading the visual primitives from a manifest, so renderers and
  external tools can create genre-specific mappings without parsing raw JSON.
- **Analysis + backend tests**: Add the Phase 2 test suite
  (`test_neural_beats.py`, `test_chord_detection.py`) alongside integration
  tests for `frontend/server.py` (`/api/analyze`, `/api/render`).
- **Studio UX polish**: Wire session save/load helpers into dedicated UI
  controls (preset selector, save/load buttons), and refine progress / error
  messaging for long renders.

---

## Medium Term

- **W3 — Chord detection (autochord)**: ~20–24h. `chord_tension` is the
  single highest-value new visual driver — it gives every renderer a
  complexity/hue/morph dimension tied directly to musical harmony. Do this
  before W1. See proposal §W3 for `CHORD_TENSION` constants, `_parse_chord`,
  and the enharmonic disambiguation note.
- **W1 — Source separation (Demucs)**: ~28–36h. Separates the mix into
  drums / bass / vocals / other stems; each stem gets its own feature
  extraction pass so kick, bass, and vocals can drive independent visual
  layers. Do this last — highest effort, GPU dependency, 1–2 GB model
  download. On Chromebook CPU: ~15 min per 5-min track (pay once, cached).
  See proposal §W1 for the stereo-at-44100 requirement and fake-stereo
  fallback. Bumps schema to 3.0.
- **Adaptive presets**: Genre- or mood-specific analysis + style presets
  (e.g., "Ambient Bloom", "Glitch Grid") bundling pipeline parameters,
  visual mapping, and Studio defaults. Depends on W3 being stable — presets
  need `chord_quality`, `section_labels`, and `key_mode` to be musically
  meaningful.
- **Performance profiling**: Benchmark the Phase 2 pipeline on
  representative tracks. Demucs caching strategy is critical (15 min/track
  without it). Also profile HPSS, MFCC, and Studio canvas drawing hotspots.

---

## Long Term

- **R2 — Emotion embeddings**: Per-section `valence` and `arousal` via
  Essentia TensorFlow (fast path) or MERT (higher ceiling). Drives global
  animation speed ceiling, palette temperature, and slow color drift across
  the piece. Implement before real-time mode — the offline version validates
  the visual mapping before the streaming complexity is added.
- **R3/R4 — Real-time / Ableton integration**: Implement `RealtimeAnalyzer`
  with rolling-window normalization and madmom's `OnlineBeatTrackingProcessor`.
  Start with Option C (watch-folder + Ableton Link for BPM sync) using the
  full offline pipeline; build toward Option B (Max for Live device with OSC
  feature stream) once the watch-folder path is stable. See proposal §R3–R4.
- **Multi-renderer support**: Generalize the manifest consumer API for
  additional renderers (WebGL target first, then Unreal/Unity). The
  `manifest.schema.json` from short-term is the prerequisite — it becomes
  the contract between the analysis pipeline and any renderer.
- **Preset sharing**: Design a portable preset format (analysis config +
  visual mapping config bundled) so users can export / import settings for
  collaboration. Natural evolution of the adaptive presets work.
