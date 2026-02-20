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

## 🔵 Phase 2: Performance & Polish
- [ ] **Numba Acceleration**: JIT-compile particle physics and buffer mixing.
- [X] **Resolution Scaling**: Implement internal low-res rendering for mirrored modes.
- [ ] **Vapor Warp 2.0**: Replace simple Sine distortion with Perlin/Simplex Flow Fields.

## 🔴 Phase 3: CLI & UX
- [X] **Unified CLI**: Add `--mirror` and `--interference` to the main `chromascope` entry point.
- [ ] **Real-Time Preview**: Build a minimal `opencv` or `pygame` viewer for local testing.

---
**Current Priority:** Numba Acceleration for Decay/Solar physics.
