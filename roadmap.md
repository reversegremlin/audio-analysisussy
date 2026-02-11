# Roadmap – Chromascope

A living roadmap for planned evolution of the Chromascope analysis and
visualization stack. Items are grouped by theme rather than strict dates.

## Short Term

- **Studio UX polish**: Wire the new session save/load helpers into dedicated
  UI controls (preset selector, save/load buttons), and refine progress /
  error messaging for long renders.
- **Manifest consumers**: Expose helper utilities in Python and JavaScript
  for working directly with the new visual primitives, so renderers and
  external tools can more easily create genre-specific mappings.
- **Backend tests**: Add pytest-based integration tests around
  `frontend/server.py` to exercise `/api/analyze` and `/api/render` with the
  test audio fixtures.

## Medium Term

- **Additional DSP features**: Experiment with section/segment detection and
  higher-level “scene” descriptors (e.g., breakdown, chorus, drop) to drive
  macro-structure visual changes.
- **Adaptive presets**: Introduce genre- or mood-specific analysis + style
  presets (e.g., “Ambient Bloom”, “Glitch Grid”) that bundle pipeline
  parameters, visual mapping, and Studio defaults.
- **Performance profiling**: Benchmark end-to-end pipeline performance on
  representative tracks and optimize hotspots (HPSS, MFCC computation,
  Pygame rendering, and Studio canvas drawing).

## Long Term

- **Multi-renderer support**: Generalize the manifest and style system for
  additional renderers (e.g., WebGL, Unreal/Unity integrations) while
  keeping the same visual primitives API.
- **Real-time live mode**: Investigate a low-latency streaming mode that
  computes features and visuals in near real time for live performance.
- **Preset sharing**: Design a portable preset format so users can export /
  import both analysis and visual settings for collaboration.

