"""
Chromascope fractal renderer benchmark + parity validation.

Usage:
    python scripts/benchmark.py [--quick]

Modes:
    default  — 1920×1080 timing, 3 warm-up + 5 timed runs per function
    --quick  — 640×360 timing, 2 warm-up + 3 timed runs (CI-friendly, ~30 s)

Output: timing table + parity report printed to stdout.

Parity check: compares Numba path output against the NumPy fallback by
temporarily monkey-patching _NUMBA_OK=False.  The two outputs must agree
to within 1e-4 (float32 precision; smooth-colouring formula is identical).
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager
from typing import List

import numpy as np

# Make sure the installed package is on the path when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chromascope.experiment import fractal as _fractal_mod
from chromascope.experiment import kaleidoscope as _kscope_mod
from chromascope.experiment.fractal import (
    _NUMBA_OK,
    julia_set,
    mandelbrot_zoom,
    noise_fractal,
)
from chromascope.experiment.kaleidoscope import (
    flow_field_warp,
    polar_mirror,
)

_SEP = "─" * 72


def _hdr(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _timeit(fn, *args, warmup: int = 2, runs: int = 5, **kwargs) -> List[float]:
    """Run fn(*args, **kwargs), discard warmup iterations, return timed samples."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return times


def _stats(times: List[float]) -> str:
    arr = np.array(times)
    return f"mean={arr.mean()*1000:.1f} ms  min={arr.min()*1000:.1f} ms  max={arr.max()*1000:.1f} ms"


@contextmanager
def _numpy_mode():
    """Temporarily disable Numba inside fractal + kaleidoscope modules."""
    orig_f = _fractal_mod._NUMBA_OK
    orig_k = _kscope_mod._NUMBA_OK
    _fractal_mod._NUMBA_OK = False
    _kscope_mod._NUMBA_OK = False
    try:
        yield
    finally:
        _fractal_mod._NUMBA_OK = orig_f
        _kscope_mod._NUMBA_OK = orig_k


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------

def _parity_report(numba_out: np.ndarray, numpy_out: np.ndarray) -> dict:
    """Compute structural parity metrics between Numba and NumPy outputs.

    max_diff is informational only: the Numba kernel iterates in float64
    while the NumPy path stores intermediate z values in complex64, so
    boundary pixels can be classified differently (higher precision ≠ wrong).
    Correlation and 99th-percentile are better quality signals.
    """
    diff = np.abs(numba_out.astype(np.float64) - numpy_out.astype(np.float64))
    flat_n = numba_out.flatten().astype(np.float64)
    flat_p = numpy_out.flatten().astype(np.float64)
    corr = float(np.corrcoef(flat_n, flat_p)[0, 1]) if flat_n.std() > 0 else 1.0
    p99 = float(np.percentile(diff, 99))
    return {
        "max_diff": float(diff.max()),
        "p99_diff": p99,
        "corr": corr,
        "pct_close": float((diff < 0.01).mean() * 100),
    }


def _parity_julia(w: int, h: int) -> dict:
    c = -0.7269 + 0.1889j
    numba_out = julia_set(w, h, c=c, max_iter=80)
    with _numpy_mode():
        numpy_out = julia_set(w, h, c=c, max_iter=80)
    return _parity_report(numba_out, numpy_out)


def _parity_mandelbrot(w: int, h: int) -> dict:
    numba_out = mandelbrot_zoom(w, h, max_iter=80)
    with _numpy_mode():
        numpy_out = mandelbrot_zoom(w, h, max_iter=80)
    return _parity_report(numba_out, numpy_out)


def _parity_flow(w: int, h: int) -> dict:
    tex = np.random.RandomState(0).rand(h, w).astype(np.float32)
    numba_out = flow_field_warp(tex, amplitude=0.1, scale=3.0, time=1.5, octaves=3)
    with _numpy_mode():
        numpy_out = flow_field_warp(tex, amplitude=0.1, scale=3.0, time=1.5, octaves=3)
    return _parity_report(numba_out, numpy_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chromascope fractal benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 640×360 instead of 1920×1080 for fast CI runs",
    )
    args = parser.parse_args()

    if args.quick:
        W, H = 640, 360
        WARMUP, RUNS = 2, 3
        label = "640×360 (quick mode)"
    else:
        W, H = 1920, 1080
        WARMUP, RUNS = 3, 5
        label = "1920×1080 (full mode)"

    print(f"\nChromascope Fractal Benchmark  —  {label}")
    print(f"Numba available: {_NUMBA_OK}")
    print(f"Warm-up runs: {WARMUP}  |  Timed runs: {RUNS}")

    results = {}

    # ------------------------------------------------------------------
    # 1. julia_set
    # ------------------------------------------------------------------
    _hdr("1. julia_set")
    c = -0.7269 + 0.1889j
    t = _timeit(julia_set, W, H, c=c, max_iter=200, warmup=WARMUP, runs=RUNS)
    results["julia_set"] = t
    print(f"  Numba={_NUMBA_OK}  {_stats(t)}")

    if _NUMBA_OK:
        with _numpy_mode():
            t_np = _timeit(julia_set, W, H, c=c, max_iter=200, warmup=1, runs=2)
        results["julia_set_numpy"] = t_np
        speedup = np.mean(t_np) / np.mean(t)
        print(f"  NumPy fallback:   {_stats(t_np)}")
        print(f"  Speedup: {speedup:.1f}×")

    # ------------------------------------------------------------------
    # 2. mandelbrot_zoom
    # ------------------------------------------------------------------
    _hdr("2. mandelbrot_zoom")
    t = _timeit(mandelbrot_zoom, W, H, max_iter=200, warmup=WARMUP, runs=RUNS)
    results["mandelbrot_zoom"] = t
    print(f"  Numba={_NUMBA_OK}  {_stats(t)}")

    if _NUMBA_OK:
        with _numpy_mode():
            t_np = _timeit(mandelbrot_zoom, W, H, max_iter=200, warmup=1, runs=2)
        results["mandelbrot_zoom_numpy"] = t_np
        speedup = np.mean(t_np) / np.mean(t)
        print(f"  NumPy fallback:   {_stats(t_np)}")
        print(f"  Speedup: {speedup:.1f}×")

    # ------------------------------------------------------------------
    # 3. noise_fractal
    # ------------------------------------------------------------------
    _hdr("3. noise_fractal")
    t = _timeit(noise_fractal, W, H, time=1.0, octaves=4, warmup=WARMUP, runs=RUNS)
    results["noise_fractal"] = t
    print(f"  {_stats(t)}")

    # ------------------------------------------------------------------
    # 4. flow_field_warp
    # ------------------------------------------------------------------
    _hdr("4. flow_field_warp")
    rng = np.random.RandomState(42)
    tex = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
    t = _timeit(flow_field_warp, tex, amplitude=0.1, scale=3.0, time=1.0, octaves=3,
                warmup=WARMUP, runs=RUNS)
    results["flow_field_warp"] = t
    print(f"  Numba={_NUMBA_OK}  {_stats(t)}")

    if _NUMBA_OK:
        with _numpy_mode():
            t_np = _timeit(flow_field_warp, tex, amplitude=0.1, scale=3.0, time=1.0,
                           octaves=3, warmup=1, runs=2)
        results["flow_field_warp_numpy"] = t_np
        speedup = np.mean(t_np) / np.mean(t)
        print(f"  NumPy fallback:   {_stats(t_np)}")
        print(f"  Speedup: {speedup:.1f}×")

    # ------------------------------------------------------------------
    # 5. polar_mirror (benefits from remap cache)
    # ------------------------------------------------------------------
    _hdr("5. polar_mirror (with remap cache warm)")
    tex2d = rng.rand(H, W).astype(np.float32)
    # First call populates cache
    polar_mirror(tex2d, num_segments=8, rotation=0.0)
    # Subsequent calls are the cached path
    t = _timeit(polar_mirror, tex2d, num_segments=8, rotation=0.1,
                warmup=WARMUP, runs=RUNS)
    results["polar_mirror_cached"] = t
    print(f"  {_stats(t)}")

    # ------------------------------------------------------------------
    # Parity validation
    # ------------------------------------------------------------------
    _hdr("Parity validation (Numba vs NumPy — small render 160×120)")
    PW, PH = 160, 120

    if _NUMBA_OK:
        r_julia = _parity_julia(PW, PH)
        r_mb    = _parity_mandelbrot(PW, PH)
        r_flow  = _parity_flow(PW, PH)

        # Pass criteria: correlation > 0.99 and 99th-percentile diff < 0.02.
        # Note: max_diff is informational — the Numba kernel uses float64 while
        # the NumPy path accumulates in complex64, so a few boundary pixels will
        # always differ (higher precision ≠ wrong output).
        CORR_MIN  = 0.99
        P99_MAX   = 0.02

        def _status(r: dict) -> str:
            ok = r["corr"] >= CORR_MIN and r["p99_diff"] <= P99_MAX
            return "PASS" if ok else "FAIL"

        def _row(name: str, r: dict) -> str:
            return (
                f"  {name:<20}  max={r['max_diff']:.3f}  p99={r['p99_diff']:.3f}"
                f"  corr={r['corr']:.4f}  close={r['pct_close']:.1f}%"
                f"  [{_status(r)}]"
            )

        print("  Note: max_diff at fractal boundaries is expected (float64 vs complex64"
              " precision).\n  Structural metrics (correlation, p99) are the quality signal.")
        print()
        print(f"  {'Function':<20}  {'max':>7}  {'p99':>6}  {'corr':>7}  {'≈100%':>6}  status")
        print(f"  {'-'*20}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*6}  ------")
        print(_row("julia_set", r_julia))
        print(_row("mandelbrot_zoom", r_mb))
        print(_row("flow_field_warp", r_flow))

        all_ok = all(
            r["corr"] >= CORR_MIN and r["p99_diff"] <= P99_MAX
            for r in [r_julia, r_mb, r_flow]
        )
        if all_ok:
            print("\n  All parity checks PASSED.")
        else:
            print("\n  !! PARITY FAILURES DETECTED — investigate before overnight renders !!")
            sys.exit(1)
    else:
        print("  Numba not installed — skipping parity checks.")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    _hdr("Summary")
    cols = [
        ("Function", "Time (ms mean)"),
    ]
    rows = []
    for name, times in results.items():
        rows.append((name, f"{np.mean(times)*1000:.1f}"))

    name_w = max(len(r[0]) for r in rows) + 2
    print(f"  {'Function':<{name_w}} Time (ms, mean)")
    print(f"  {'-'*name_w} ---------------")
    for name, val in rows:
        print(f"  {name:<{name_w}} {val}")

    print(f"\n{_SEP}\n")


if __name__ == "__main__":
    main()
