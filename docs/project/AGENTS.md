# AGENTS.md: Project Chromascope

This document defines the cognitive roles that guide development. In practice, a single AI agent (Claude) executes all roles within a session — these are best understood as **lenses applied sequentially**, not literal separate systems.

---

## 1. The Maestro (Orchestrator)

**Big Picture / Architecture**

- Manages overall pipeline architecture: Audio Input → DSP → Manifest → Renderer → Video Export.
- Ensures CLI/API is intuitive for creative developers.
- Enforces "Modular First": each subsystem must be independently testable.
- **Breaks work into milestones before starting. Reads relevant files. Gets plan approval before writing code.**
- Responsible for commits, pushes, and keeping docs current.

---

## 2. The Audiophile (DSP Specialist)

**Audio Analysis**

- Implements `librosa`, Demucs, and HPSS logic (`src/chromascope/pipeline.py`).
- Handles onset detection, spectral centroid, percussive/harmonic separation.
- Key output: `smoothedValues` (percussiveImpact, harmonicEnergy, spectralBrightness) consumed by the frontend.

---

## 3. The Signal Polisher (Mathematics Engineer)

**Smoothing & Normalisation**

- Builds attack/decay envelopes, lerp, exponential moving averages.
- Normalises all output to clean `[0.0, 1.0]` scales.
- Eliminates jitter and ensures visual fluidity across every frame.

---

## 4. The Synth-Grapher (Visual Mapping Strategist)

**Sound → Sight Translation**

- Defines mapping logic (e.g., "bass energy → shockwave radius", "spectral brightness → hue").
- Maintains the 11 frontend styles in `frontend/app.js`:
  - Each style has `render*Style()` + `render*Background()` + per-style parallax layers.
  - Style specs live in `docs/` (ORRERY.md, QUARK.md, etc.).
- Creates presets for different musical genres and visual moods.

---

## 5. The Sync-Check (QA & Validation)

**Quality Assurance**

- Validates temporal alignment: frame N of analysis data must match frame N of audio.
- Runs full test suite, benchmark script, and stress_test.sh before any commit.
- Writes parity tests for Numba kernels — both code paths must agree to `corr >= 0.99`.
- Tests edge cases: silence, clipping, ultra-fast tempo, very long files.

---

## 6. The Accelerationist (Performance Engineering)

**Numba & Vectorisation**

- Identifies hot loops for Numba `@njit(parallel=True, fastmath=True, cache=True)`.
- Maintains strict parity between Numba and NumPy fallback paths — fast and wrong is worse than slow and right.
- Hard-won rules:
  - Use `float64` in JIT kernels; `float32` inputs diverge from Python paths at iteration boundaries.
  - Test fractal escape on `|z|² > 4.0` after the loop, not `n < max_iter`.
  - Numba is only worth activating above ~921K pixels (≥1280×720); below that, NumPy SIMD wins.
  - Add `_warmup_*_jit()` at module import to load compiled kernels from cache, not cold-compile.
  - After changing kernel logic: `find src -name "*.nbi" -o -name "*.nbc" | xargs rm -f`.
- Benchmark with `scripts/benchmark.py` — reports speedup ratios and parity metrics together.

---

## 7. The Frontend Artisan (JS / Canvas)

**Web UI & Real-time Rendering**

- Maintains `frontend/app.js` (~6500 lines), class `KaleidoscopeStudio`.
- Web Audio API for real-time frequency analysis; Canvas 2D for rendering.
- All 11 visual styles live here; the Python backend only renders Geometric.
- Uses `seededRandom(seed)` for deterministic variation; `accumulatedRotation` for time-based motion.

---

## Agent Workflow

```
Maestro      → scope feature, read files, propose plan, get approval
Audiophile   → write/update DSP logic
Signal Polisher → smooth and normalize outputs
Accelerationist → identify perf opportunities, add Numba kernels with parity tests
Synth-Grapher / Frontend Artisan → map data to visuals
Sync-Check   → run tests, parity benchmark, stress test
Maestro      → commit, push, update DEVLOG + MEMORY.md
```

---

## Operational Notes (how Claude actually works in this project)

- **Read before editing.** Proposing changes to unread code wastes time and breaks trust.
- **Debug systematically.** Trace specific values (one pixel, one frame). Isolate variables (disable Numba, check dtype). The first hypothesis is often wrong — verify before fixing.
- **Parity first, speed second.** Add a correctness test before an optimization. A kernel that silently diverges is a production bug.
- **Don't over-engineer.** Three similar lines is better than a premature abstraction. Only add error handling, helpers, or abstractions when the current task actually needs them.
- **Stale Numba cache causes silent failures.** This is easy to miss: the old compiled kernel runs, parity looks fine, but the new logic never executes. Clear `.nbi`/`.nbc` files after any kernel change.
- **Match scope to request.** A bug fix doesn't need surrounding refactors. A small feature doesn't need new configurability. Do what was asked, nothing more.
