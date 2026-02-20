# Chromascope: Project Roadmap & TODO

## 🟢 Phase 1: Core Engine & Experiments [COMPLETE]
- [X] **Basic Renderers**: Fractal, Solar, Kaleidoscope.
- [X] **Radioactive Decay Experiment (`chromascope-decay`)**:
    - [X] Implement Particle Drag (Alpha/Beta/Gamma).
    - [X] Implement Dual-Buffer Vapor (Track/Smoke).
    - [X] Implement **Dynamic Symmetrical Mirror Architecture**.
    - [X] Implement **Matter-Antimatter Overlap Coloring**.
    - [X] Implement Axis-Locked Symmetrical Sliding.
    - [X] Fix Jitter with Sub-Pixel Interpolation.
    - [X] Implement Audio-Reactive Phase Phasing.
- [X] **Global "OPEN UP" Refactor**:
    - [X] Create `BaseVisualizer` Abstract Class.
    - [X] Decouple Global Randomness (Force local `self.rng`).
    - [X] Port Symmetrical Mirror to `SolarRenderer`.
    - [X] Port Symmetrical Mirror to `FractalKaleidoscopeRenderer`.
    - [X] Implement `CrossVisualizerCompositor` (e.g. Solar interfering with Decay).

## 🔵 Phase 2: Performance & Polish [COMPLETE]
- [X] **Numba Acceleration**: JIT-compile particle physics (`decay.py` SoA + `@numba.njit`); vectorized fBm replaces `pnoise3` loop in `solar.py` (100-500× faster at 1080p).
- [X] **Resolution Scaling**: Implement internal low-res rendering for mirrored modes.
- [X] **Vapor Warp 2.0**: `flow_field_warp()` in `kaleidoscope.py` — two independent fBm fields drive dx/dy, giving organic non-radial swirling distortion. Used by `FractalKaleidoscopeRenderer`.

## 🔴 Phase 3: CLI & UX [COMPLETE]
- [X] **Unified CLI**: Add `--mirror` and `--interference` to the main `chromascope` entry point.
- [X] **Real-Time Preview**: `chromascope audio.wav --preview` opens a pygame window with SPACE=pause, ESC=quit, →=step.

---
**All phases complete.**
