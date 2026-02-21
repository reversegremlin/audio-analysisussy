# Software Development Life Cycle (SDLC)

This project has two main subsystems — the **core audio pipeline** (`src/chromascope/`) and the **fractal experiment renderer** (`src/chromascope/experiment/`). Changes should follow the appropriate track below.

---

## Stage 1: Analysis & Impact Assessment

- Read relevant source files before proposing any change. Never modify code you haven't read.
- Review the requirement against `architecture.md` and relevant docs in `docs/`.
- Identify which subsystem is affected:
  - **Audio pipeline**: Decomposer, Analyzer, Polisher, Visualizer (`src/chromascope/`)
  - **Fractal experiment**: `fractal.py`, `kaleidoscope.py`, `colorgrade.py`, `renderer.py`, `encoder.py` (`src/chromascope/experiment/`)
  - **Frontend**: `frontend/app.js` (Canvas 2D, Web Audio API, ~6500 lines)
- For performance-sensitive paths: determine whether Numba acceleration applies (pixel-level loops, fBm fields, fractal iteration).
- Check for synchronization drift risk (FPS vs sample rate alignment).
- Break large changes into milestones. Get plan approval before writing code.

---

## Stage 2: Test Specification (Red)

- Create or extend a test file in `tests/` or `tests/experiment/`.
- Define expected output: shape, dtype, value range, and determinism.
- For Numba-accelerated functions: add a **parity test** comparing the Numba path against the NumPy fallback on identical inputs. Pass criteria: `corr >= 0.99` and `p99_diff <= 0.02`.
- Goal: the test should fail before the implementation exists.

---

## Stage 3: Implementation (Green)

- Write the minimum code to make the test pass.
- Follow the Google Python Style Guide. Apply type hints to all new functions.
- For Numba kernels specifically:
  - Decorate with `@numba.njit(parallel=True, fastmath=True, cache=True)`.
  - Add a `_warmup_*_jit()` function called at the **module bottom** to pre-compile from cache at import time.
  - Use `float64` for all intermediate arithmetic inside kernels. `float32` grids diverge from Python paths at iteration boundaries.
  - Keep post-processing (normalization, clipping, array conversion) in NumPy, **outside** the JIT kernel.
  - **Fractal escape condition**: test `|z|² > 4.0` after the while-loop, not `n < max_iter`. A pixel that first exceeds `|z|=2` on the last iteration has `n == max_iter` but `|z|² > 4` — testing the counter misclassifies it as interior.

---

## Stage 4: Validation & Polishing

Run the full test suite (165 tests as of Feb 2026):
```
/home/nadiabellamorris/chromascope/.venv/bin/python -m pytest tests/ -v
```

For Numba code, run the parity benchmark to confirm correctness and measure speedup:
```
.venv/bin/python scripts/benchmark.py --quick
```

For integration validation, run the stress test (fast mode skips slow renders):
```
bash stress_test.sh --skip-renders --skip-fractal
```

**Numba cache management**: after changing kernel logic, delete stale compiled artifacts or results will be silently wrong:
```
find src/chromascope -name "*.nbi" -o -name "*.nbc" | xargs rm -f
```

- Vectorize non-JIT hot paths with NumPy where possible.
- Verify the Visual Driver Manifest JSON validates against schema.

---

## Stage 5: Documentation & Handover

- Update `DEVLOG.md` with the "Why" and "How."
- Update `roadmap.md`.
- Update `memory/MEMORY.md` if architectural decisions, key file locations, or development patterns changed.
- Run tests one final time. Commit with a descriptive message. Push to `main` only when all tests pass.
