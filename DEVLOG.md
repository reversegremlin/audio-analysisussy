# DEVLOG

Chronological log of significant changes to the Chromascope analysis and
visualization pipeline. Newest entries at the top.

## 2026-02-10 – Schema, DSP, Styles, and Studio Session Model

- Introduced an explicit manifest `schema_version` and a small set of
  renderer-agnostic visual primitives (`impact`, `fluidity`, `brightness`,
  `pitch_hue`, `texture`) derived from polished features. Updated
  `docs/architecture.md` and `README.md` to reflect the richer schema.
- Extended the DSP layer with a tempo curve derived from beat spacing,
  MFCC-based timbre features, and optional BPM-adaptive envelopes in the
  `SignalPolisher`. Added focused tests to guard hop-length/frame alignment
  and new features.
- Created shared kaleidoscope style presets (`styles.json`) consumed by both
  the Python renderer and the Studio via `/styles.json`, ensuring that named
  styles map to consistent geometry and motion across runtimes.
- Enhanced the Studio frontend with a lightweight session model
  (`createDefaultSession` / `getSessionSnapshot` / `applySessionSnapshot`)
  and localStorage save/load helpers, plus integration with shared style
  presets.
- Fixed the `/api/analyze` endpoint in `frontend/server.py` to correctly
  persist uploaded audio to a temp file before running the `AudioPipeline`,
  returning structured errors for invalid content types or missing files.

## 2026-02-10 – Video Export UX

- Improved the CLI `render_video` experience with a terminal-friendly text
  progress bar that visualizes overall completion while still supporting
  programmatic `progress_callback` usage.
- Tightened Studio export error reporting in `exportVideo()` so HTTP failures
  from `/api/render` surface backend error details in the UI instead of a
  generic message, making it easier to diagnose export issues from the
  browser alone.

