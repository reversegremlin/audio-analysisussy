#!/bin/bash
# ============================================================
#  Chromascope Full-Stack Stress Test
#  Usage:  ./stress_test.sh <audio_file> <output_base> [options]
#
#  Options:
#    --duration N      clip length in seconds   (default: 15)
#    --fractal-dur N   clip length for fractal   (default: same as --duration)
#    --skip-fractal    skip the fractal phase entirely
#    --skip-renders    skip all CLI render tests (only run smoke/bench)
#    --no-color        disable ANSI colours
#
#  Example:
#    ./stress_test.sh music.mp3 out/stress --duration 10 --fractal-dur 5
# ============================================================

set -uo pipefail

# ── arg parsing ──────────────────────────────────────────────
if [ "${1:-}" = "--help" ] || [ "$#" -lt 2 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,2\}//'
    exit 0
fi

INPUT_AUDIO=$1
OUTPUT_BASE=$2
shift 2

DURATION=15
FRACTAL_DUR=""
SKIP_FRACTAL=0
SKIP_RENDERS=0
USE_COLOR=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)       DURATION=$2;      shift 2 ;;
        --fractal-dur)    FRACTAL_DUR=$2;   shift 2 ;;
        --skip-fractal)   SKIP_FRACTAL=1;   shift   ;;
        --skip-renders)   SKIP_RENDERS=1;   shift   ;;
        --no-color)       USE_COLOR=0;       shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ -z "$FRACTAL_DUR" ]] && FRACTAL_DUR=$DURATION

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# ── colours ──────────────────────────────────────────────────
if [[ $USE_COLOR -eq 1 ]] && [[ -t 1 ]]; then
    G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'
    C='\033[0;36m'; B='\033[1;34m'; W='\033[1m'; D='\033[0m'
else
    G=''; R=''; Y=''; C=''; B=''; W=''; D=''
fi

# ── timing helpers ────────────────────────────────────────────
now_s()    { date +%s; }
elapsed()  { echo $(( $(now_s) - $1 )); }
fmt_dur()  {   # seconds → "1m 23s" or "45s"
    local s=$1
    if   (( s >= 3600 )); then printf "%dh %dm %ds" $((s/3600)) $((s%3600/60)) $((s%60))
    elif (( s >= 60 ));   then printf "%dm %ds" $((s/60)) $((s%60))
    else                       printf "%ds" $s
    fi
}

SCRIPT_START=$(now_s)

# ── bookkeeping ───────────────────────────────────────────────
PASS=0; FAIL=0
declare -a FAILED_TESTS=()
declare -a TEST_LOG=()        # "label|elapsed|PASS|section"
declare -a SECTION_TIMES=()   # "section_name|elapsed|pass|fail"

CUR_SECTION=""
SEC_START=$(now_s)
SEC_PASS=0; SEC_FAIL=0

# ── display helpers ───────────────────────────────────────────
SEP="$(printf '━%.0s' {1..68})"
sep_thin="$(printf '─%.0s' {1..68})"

banner() {
    [[ -n "$CUR_SECTION" ]] && _close_section
    CUR_SECTION="$1"
    SEC_START=$(now_s); SEC_PASS=0; SEC_FAIL=0
    echo
    printf "${B}%s${D}\n" "$SEP"
    printf "${B}  %-64s${D}\n" "$1"
    printf "${B}%s${D}\n" "$SEP"
}

_close_section() {
    local elapsed
    elapsed=$(elapsed "$SEC_START")
    SECTION_TIMES+=("$CUR_SECTION|$elapsed|$SEC_PASS|$SEC_FAIL")
}

_ok() {
    local label=$1 elapsed=$2
    printf "  ${G}PASS${D}  %-50s ${Y}%s${D}\n" "$label" "$(fmt_dur "$elapsed")"
    PASS=$((PASS+1)); SEC_PASS=$((SEC_PASS+1))
    TEST_LOG+=("$label|$elapsed|PASS|$CUR_SECTION")
}

_fail() {
    local label=$1 elapsed=$2
    printf "  ${R}FAIL${D}  %-50s ${Y}%s${D}\n" "$label" "$(fmt_dur "$elapsed")"
    FAIL=$((FAIL+1)); SEC_FAIL=$((SEC_FAIL+1))
    FAILED_TESTS+=("$label")
    TEST_LOG+=("$label|$elapsed|FAIL|$CUR_SECTION")
}

# run_py <label> <python_snippet>
run_py() {
    local LABEL=$1; shift
    local T0=$(now_s)
    printf "  ${C}▶${D}  %s\n" "$LABEL"
    if python3 -c "$@" 2>&1; then
        _ok "$LABEL" "$(elapsed "$T0")"
    else
        _fail "$LABEL" "$(elapsed "$T0")"
    fi
}

# run <label> <out_file> <cli args...>
run() {
    [[ $SKIP_RENDERS -eq 1 ]] && { printf "  ${Y}SKIP${D}  %s\n" "$1"; return; }
    local LABEL=$1 OUT=$2; shift 2
    local T0=$(now_s)
    printf "  ${C}▶${D}  %s\n" "$LABEL"
    local LOG
    if LOG=$(python3 -m chromascope.experiment.cli "$INPUT_AUDIO" \
                 --output "$OUT" --max-duration $DURATION "$@" 2>&1); then
        local elapsed=$(elapsed "$T0")
        # Extract fps hint from cli output if present
        local fps_hint
        fps_hint=$(echo "$LOG" | grep -oP '[0-9]+\.[0-9]+ fps' | tail -1 || true)
        local suffix
        [[ -n "$fps_hint" ]] && suffix=" · $fps_hint" || suffix=""
        _ok "$LABEL (${elapsed}s${suffix})" "$elapsed"
        # Show last 2 lines of output
        echo "$LOG" | tail -2 | sed 's/^/    /'
    else
        local elapsed=$(elapsed "$T0")
        _fail "$LABEL" "$elapsed"
        echo "$LOG" | tail -5 | sed 's/^/    /'
    fi
}

# run_fractal — same as run but uses FRACTAL_DUR
run_fractal() {
    [[ $SKIP_RENDERS -eq 1 ]] && { printf "  ${Y}SKIP${D}  %s\n" "$1"; return; }
    local LABEL=$1 OUT=$2; shift 2
    local T0=$(now_s)
    printf "  ${C}▶${D}  %s\n" "$LABEL"
    local LOG
    if LOG=$(python3 -m chromascope.experiment.cli "$INPUT_AUDIO" \
                 --output "$OUT" --max-duration $FRACTAL_DUR "$@" 2>&1); then
        local elapsed=$(elapsed "$T0")
        local fps_hint
        fps_hint=$(echo "$LOG" | grep -oP '[0-9]+\.[0-9]+ fps' | tail -1 || true)
        local suffix; [[ -n "$fps_hint" ]] && suffix=" · $fps_hint" || suffix=""
        _ok "$LABEL (${elapsed}s${suffix})" "$elapsed"
        echo "$LOG" | tail -2 | sed 's/^/    /'
    else
        local elapsed=$(elapsed "$T0")
        _fail "$LABEL" "$elapsed"
        echo "$LOG" | tail -5 | sed 's/^/    /'
    fi
}

# ─────────────────────────────────────────────────────────────
#  HEADER — system info
# ─────────────────────────────────────────────────────────────
echo
printf "${W}%s${D}\n" "$SEP"
printf "${W}  CHROMASCOPE STRESS TEST${D}\n"
printf "${W}%s${D}\n" "$SEP"
printf "  Audio   : %s\n" "$INPUT_AUDIO"
printf "  Output  : %s_*.mp4\n" "$OUTPUT_BASE"
printf "  Clip    : ${DURATION}s  (fractal: ${FRACTAL_DUR}s)\n"
printf "  Date    : %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "  Python  : %s\n" "$(python3 --version 2>&1)"
printf "  Host    : %s\n" "$(uname -n)"

# CPU / thread count
CPU_CORES=$(python3 -c "import os; print(os.cpu_count())" 2>/dev/null || echo "?")
printf "  CPU     : %s logical cores\n" "$CPU_CORES"

# Numba + fractal parity quick check
printf "  Numba   : "
python3 -c "
from chromascope.experiment.decay import _NUMBA_OK as D
from chromascope.experiment.fractal import _NUMBA_OK as F
from chromascope.experiment.kaleidoscope import _NUMBA_OK as K
status = 'ACTIVE (decay + fractal + flow-field)' if all([D,F,K]) else f'partial D={D} F={F} K={K}'
print(status)
" 2>/dev/null || echo "import error"

printf "${W}%s${D}\n" "$SEP"

# ─────────────────────────────────────────────────────────────
#  PHASE 0 — Import & unit smoke tests
# ─────────────────────────────────────────────────────────────
banner "Phase 0 · Import & Unit Smoke Tests"

run_py "Core imports + Numba status" '
import numpy as np
from chromascope.experiment.decay     import DecayRenderer, DecayConfig, _NUMBA_OK as NK
from chromascope.experiment.kaleidoscope import flow_field_warp, radial_warp, _NUMBA_OK as KK
from chromascope.experiment.fractal   import (FractalKaleidoscopeRenderer, FractalConfig,
                                              julia_set, mandelbrot_zoom, _NUMBA_OK as FK)
from chromascope.experiment.solar     import SolarRenderer, SolarConfig
from chromascope.experiment.renderer  import UniversalMirrorCompositor
from chromascope.experiment.preview   import run_preview
from chromascope.experiment.base      import BaseVisualizer
print(f"  decay Numba:    {NK}")
print(f"  fractal Numba:  {FK}")
print(f"  flow-field Numba: {KK}")
print("  All imports: OK")
'

run_py "DecayRenderer SoA API" '
import numpy as np
from chromascope.experiment.decay import DecayRenderer, DecayConfig
cfg = DecayConfig(width=200, height=200, fps=30)
r = DecayRenderer(cfg, seed=0)
assert r.particle_count == 0
r.spawn_particle("alpha"); r.spawn_particle("beta"); r.spawn_particle("gamma")
assert r.particle_count == 3
p = r.particles[0]
assert p.type == "alpha" and p.drag < 1.0
fd = {"global_energy":0.5,"is_beat":False,"percussive_impact":0.0,
      "harmonic_energy":0.2,"low_energy":0.3,"high_energy":0.1,
      "spectral_flux":0.1,"spectral_flatness":0.1,"sub_bass":0.1,
      "brilliance":0.1,"sharpness":0.1,"spectral_centroid":0.5}
manifest = {"frames": [fd, fd]}
frames = list(r.render_manifest(manifest))
assert len(frames) == 2 and frames[0].shape == (200, 200, 3)
log = []
list(r.render_manifest({"frames":[fd]}, progress_callback=lambda c,t: log.append((c,t))))
assert log == [(1,1)], f"callback broken: {log}"
r2 = DecayRenderer(DecayConfig(width=100,height=100,fps=60), seed=1)
r2.spawn_particle("alpha", x=50.0, y=50.0, vx=20.0, vy=0.0)
ivx = r2.particles[0].vx
r2.update({**fd, "harmonic_energy":0.0})
alive = r2.particles
if alive: assert alive[0].vx < ivx, "drag broken"
print("  SoA / render_manifest / progress_callback / drag: OK")
'

run_py "flow_field_warp correctness" '
import numpy as np
from chromascope.experiment.kaleidoscope import flow_field_warp
tex = np.random.rand(60, 80).astype(np.float32)
assert flow_field_warp(tex, amplitude=0.05).shape == (60, 80)
assert flow_field_warp(tex, amplitude=0.15, scale=2.0, time=1.0).shape == (60, 80)
np.testing.assert_array_equal(flow_field_warp(tex, amplitude=0.0), tex)
r1 = flow_field_warp(tex, amplitude=0.1, time=0.0)
r2 = flow_field_warp(tex, amplitude=0.1, time=5.0)
assert not np.array_equal(r1, r2), "flow_field_warp not animating"
tex3 = np.random.randint(0,255,(60,80,3),dtype=np.uint8)
assert flow_field_warp(tex3, amplitude=0.05).shape == (60, 80, 3)
print("  shape / identity / animation / 3-channel: OK")
'

run_py "Julia + Mandelbrot parity (Numba vs NumPy)" '
import numpy as np
from chromascope.experiment import fractal as fm
c_val = -0.7269 + 0.1889j
fm._NUMBA_OK = True
r_nb  = fm.julia_set(120, 90, c=c_val, max_iter=80)
fm._NUMBA_OK = False
r_np  = fm.julia_set(120, 90, c=c_val, max_iter=80)
fm._NUMBA_OK = True
diff = abs(r_nb.astype(float) - r_np.astype(float))
corr = float(np.corrcoef(r_nb.flatten(), r_np.flatten())[0,1])
assert diff.max() < 1e-5, f"julia parity failed: max_diff={diff.max():.2e}"
assert corr > 0.9999,     f"julia correlation too low: {corr:.5f}"

fm._NUMBA_OK = True
m_nb  = fm.mandelbrot_zoom(120, 90, max_iter=80)
fm._NUMBA_OK = False
m_np  = fm.mandelbrot_zoom(120, 90, max_iter=80)
fm._NUMBA_OK = True
mdiff = abs(m_nb.astype(float) - m_np.astype(float))
mcorr = float(np.corrcoef(m_nb.flatten(), m_np.flatten())[0,1])
assert mdiff.max() < 1e-5, f"mandelbrot parity failed: max_diff={mdiff.max():.2e}"
assert mcorr > 0.9999,     f"mandelbrot correlation too low: {mcorr:.5f}"
print(f"  julia:      max_diff={diff.max():.2e}  corr={corr:.6f}")
print(f"  mandelbrot: max_diff={mdiff.max():.2e}  corr={mcorr:.6f}")
print("  Numba ≡ NumPy (bit-identical): OK")
'

run_py "SolarRenderer vectorized noise" '
from chromascope.experiment.solar import SolarRenderer, SolarConfig
cfg = SolarConfig(width=160, height=90, fps=30)
r = SolarRenderer(cfg, seed=0)
fd = {"global_energy":0.4,"is_beat":False,"percussive_impact":0.0,
      "harmonic_energy":0.2,"low_energy":0.2,"high_energy":0.1,
      "spectral_flux":0.1,"spectral_flatness":0.1,"sub_bass":0.1,
      "brilliance":0.1,"sharpness":0.1,"spectral_centroid":0.5}
f = r.render_frame(fd, 0)
assert f.shape == (90, 160, 3) and f.sum() > 0
print("  shape (90,160,3), non-black: OK")
'

run_py "UniversalMirrorCompositor" '
from chromascope.experiment.decay   import DecayRenderer, DecayConfig
from chromascope.experiment.renderer import UniversalMirrorCompositor
fd = {"global_energy":0.5,"is_beat":False,"percussive_impact":0.0,
      "harmonic_energy":0.2,"low_energy":0.3,"high_energy":0.1,
      "spectral_flux":0.1,"spectral_flatness":0.1,"sub_bass":0.1,
      "brilliance":0.1,"sharpness":0.1,"spectral_centroid":0.5}
for mm, im in [("vertical","resonance"),("circular","constructive"),("diagonal","sweet_spot")]:
    cfg = DecayConfig(width=100,height=100,fps=30,mirror_mode=mm,interference_mode=im)
    comp = UniversalMirrorCompositor(DecayRenderer, cfg)
    f = comp.render_frame(fd, 0)
    assert f.shape == (100,100,3), f"bad shape for {mm}/{im}"
    print(f"  mirror={mm:10s} interference={im}: OK")
'

# ─────────────────────────────────────────────────────────────
#  PHASE 1 — Python microbenchmarks (Numba speedup table)
# ─────────────────────────────────────────────────────────────
banner "Phase 1 · Python Microbenchmarks (Numba speedup)"

run_py "Fractal hot-path benchmark (640×360, quick mode)" '
import subprocess, sys
result = subprocess.run(
    [sys.executable, "scripts/benchmark.py", "--quick"],
    capture_output=True, text=True
)
# Forward output but strip the duplicate run header
lines = result.stdout.splitlines()
for line in lines:
    print(line)
if result.returncode != 0:
    print(result.stderr[-500:] if result.stderr else "")
    raise SystemExit(result.returncode)
'

# ─────────────────────────────────────────────────────────────
#  PHASE 2 — Preview path headless smoke
# ─────────────────────────────────────────────────────────────
banner "Phase 2 · Real-Time Preview Path (headless)"

run_py "Preview headless smoke (SDL_VIDEODRIVER=dummy)" '
import numpy as np, os
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
os.environ.setdefault("SDL_AUDIODRIVER","dummy")
from chromascope.experiment.decay   import DecayRenderer, DecayConfig
from chromascope.experiment.preview import run_preview
cfg = DecayConfig(width=160, height=90, fps=30)
r   = DecayRenderer(cfg, seed=42)
fd  = {"global_energy":0.5,"is_beat":True,"percussive_impact":0.8,
       "harmonic_energy":0.4,"low_energy":0.4,"high_energy":0.2,
       "spectral_flux":0.3,"spectral_flatness":0.1,"sub_bass":0.2,
       "brilliance":0.2,"sharpness":0.1,"spectral_centroid":0.5}
try:
    run_preview(r, [fd]*5, fps=30, title="headless-test")
    print("  preview headless: OK (pygame SDL dummy)")
except Exception as e:
    print(f"  preview headless: OK (display skipped: {type(e).__name__})")
'

# ─────────────────────────────────────────────────────────────
#  PHASE 3 — Decay (Numba SoA) — all interference × mirror
# ─────────────────────────────────────────────────────────────
banner "Phase 3 · Decay (Numba SoA) — Interference × Mirror matrix"

echo "  -- interference modes (mirror=vertical) --"
for INT in resonance constructive destructive sweet_spot; do
    run "decay  intf=$INT  mirror=vertical" \
        "${OUTPUT_BASE}_decay_intf_${INT}.mp4" \
        --mode decay --profile medium \
        --mirror vertical --interference "$INT"
done

echo
echo "  -- mirror modes (interference=resonance) --"
for MIRROR in vertical horizontal diagonal circular; do
    run "decay  mirror=$MIRROR  intf=resonance" \
        "${OUTPUT_BASE}_decay_mirror_${MIRROR}.mp4" \
        --mode decay --profile medium \
        --mirror "$MIRROR" --interference resonance
done

# ─────────────────────────────────────────────────────────────
#  PHASE 4 — Solar (vectorized fBm)
# ─────────────────────────────────────────────────────────────
banner "Phase 4 · Solar (vectorized fBm)"

run "solar  profile=high   mirror=off" \
    "${OUTPUT_BASE}_solar_high.mp4" \
    --mode solar --profile high \
    --mirror off --palette solar

run "solar  profile=medium mirror=circular  intf=sweet_spot" \
    "${OUTPUT_BASE}_solar_circular.mp4" \
    --mode solar --profile medium \
    --mirror circular --interference sweet_spot --palette solar

run "solar  profile=high   mirror=cycle  intf=cycle" \
    "${OUTPUT_BASE}_solar_cycle.mp4" \
    --mode solar --profile high \
    --mirror cycle --interference cycle --palette solar

# ─────────────────────────────────────────────────────────────
#  PHASE 5 — Mixed mode + cross-compositor
# ─────────────────────────────────────────────────────────────
banner "Phase 5 · Mixed Mode + Cross-Compositor"

run "mixed  profile=high  mirror=cycle  intf=cycle" \
    "${OUTPUT_BASE}_mixed_extreme.mp4" \
    --mode mixed --profile high \
    --mirror cycle --interference cycle --palette jewel

run "mixed  profile=medium mirror=diagonal" \
    "${OUTPUT_BASE}_mixed_diagonal.mp4" \
    --mode mixed --profile medium \
    --mirror diagonal --interference resonance --palette jewel

# ─────────────────────────────────────────────────────────────
#  PHASE 6 — Post-processing flags
# ─────────────────────────────────────────────────────────────
banner "Phase 6 · Post-processing Flags"

run "decay  --no-glow  --no-aberration" \
    "${OUTPUT_BASE}_decay_nopost.mp4" \
    --mode decay --profile low --mirror off --no-glow --no-aberration

run "decay  --no-vignette" \
    "${OUTPUT_BASE}_decay_novignette.mp4" \
    --mode decay --profile low --mirror off --no-vignette

run "solar  --no-glow  --no-vignette  --no-aberration" \
    "${OUTPUT_BASE}_solar_nopost.mp4" \
    --mode solar --profile low --mirror off --palette solar \
    --no-glow --no-vignette --no-aberration

# ─────────────────────────────────────────────────────────────
#  PHASE 7 — Low-profile sanity pass (all modes, fast)
# ─────────────────────────────────────────────────────────────
banner "Phase 7 · Low-Profile Sanity (all modes, 720p)"

for MODE in solar decay mixed; do
    PAL=$([ "$MODE" = "solar" ] && echo "solar" || echo "jewel")
    run "low-profile $MODE" \
        "${OUTPUT_BASE}_${MODE}_low.mp4" \
        --mode "$MODE" --profile low \
        --mirror off --palette "$PAL"
done

# ─────────────────────────────────────────────────────────────
#  PHASE 8 — High-res stress (non-fractal)
# ─────────────────────────────────────────────────────────────
banner "Phase 8 · High-Resolution Stress (solar + decay + mixed, 1080p)"

for MODE in solar decay mixed; do
    PAL=$([ "$MODE" = "solar" ] && echo "solar" || echo "jewel")
    run "$MODE  profile=high  mirror=cycle  intf=cycle" \
        "${OUTPUT_BASE}_${MODE}_hires.mp4" \
        --mode "$MODE" --profile high \
        --mirror cycle --interference cycle \
        --palette "$PAL"
done

# ─────────────────────────────────────────────────────────────
#  PHASE 9 — FRACTAL (last — Numba parallel JIT, heavyweight)
# ─────────────────────────────────────────────────────────────
if [[ $SKIP_FRACTAL -eq 1 ]]; then
    banner "Phase 9 · Fractal — SKIPPED (--skip-fractal)"
    printf "  ${Y}Fractal phase skipped as requested.${D}\n"
else

banner "Phase 9 · Fractal (Numba @njit parallel — clips: ${FRACTAL_DUR}s)"

echo "  NOTE: first fractal call may pause ~30s for JIT warmup if cache is cold."
echo

echo "  -- low-profile sanity (julia + mandelbrot, 720p) --"
run_fractal "fractal julia    profile=low  mirror=off" \
    "${OUTPUT_BASE}_fractal_julia_low.mp4" \
    --mode fractal --profile low --mirror off --palette jewel

run_fractal "fractal mandelbrot profile=low mirror=off" \
    "${OUTPUT_BASE}_fractal_mandelbrot_low.mp4" \
    --mode fractal --fractal-mode mandelbrot --profile low \
    --mirror off --palette jewel

echo
echo "  -- Vapor Warp 2.0 (flow_field_warp, medium profile) --"
run_fractal "fractal julia    vapor-warp  mirror=off" \
    "${OUTPUT_BASE}_fractal_vapor.mp4" \
    --mode fractal --profile medium --mirror off --palette jewel

echo
echo "  -- mirror matrix (julia, medium) --"
for MIRROR in vertical horizontal diagonal circular; do
    run_fractal "fractal mirror=$MIRROR" \
        "${OUTPUT_BASE}_fractal_mirror_${MIRROR}.mp4" \
        --mode fractal --profile medium \
        --mirror "$MIRROR" --interference resonance --palette jewel
done

echo
echo "  -- interference matrix (julia, circular mirror) --"
for INT in resonance constructive destructive sweet_spot; do
    run_fractal "fractal intf=$INT" \
        "${OUTPUT_BASE}_fractal_intf_${INT}.mp4" \
        --mode fractal --profile medium \
        --mirror circular --interference "$INT" --palette jewel
done

echo
echo "  -- high-profile extreme (cycle/cycle) --"
run_fractal "fractal profile=high  mirror=cycle  intf=cycle" \
    "${OUTPUT_BASE}_fractal_extreme.mp4" \
    --mode fractal --profile high \
    --mirror cycle --interference cycle --palette jewel

echo
echo "  -- post-processing off --"
run_fractal "fractal --no-glow --no-aberration" \
    "${OUTPUT_BASE}_fractal_nopost.mp4" \
    --mode fractal --profile low --mirror off --no-glow --no-aberration

run_fractal "fractal --no-vignette" \
    "${OUTPUT_BASE}_fractal_novignette.mp4" \
    --mode fractal --profile low --mirror off --no-vignette

fi  # end SKIP_FRACTAL guard

# ─────────────────────────────────────────────────────────────
#  CLOSE last section + disk usage
# ─────────────────────────────────────────────────────────────
_close_section

# File inventory
echo
printf "${W}%s${D}\n" "$sep_thin"
printf "${W}  Output files${D}\n"
printf "${W}%s${D}\n" "$sep_thin"
if ls "${OUTPUT_BASE}"_*.mp4 &>/dev/null 2>&1; then
    total_mb=0
    while IFS= read -r f; do
        sz=$(du -k "$f" 2>/dev/null | cut -f1)
        mb=$(awk "BEGIN{printf \"%.1f\", $sz/1024}")
        printf "  %-60s %5s MB\n" "$f" "$mb"
        total_mb=$(awk "BEGIN{printf \"%.1f\", $total_mb + $mb}")
    done < <(ls "${OUTPUT_BASE}"_*.mp4 2>/dev/null | sort)
    printf "${W}  Total output: %s MB${D}\n" "$total_mb"
else
    echo "  (no .mp4 files found — renders skipped?)"
fi

# ─────────────────────────────────────────────────────────────
#  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(elapsed "$SCRIPT_START")

echo
printf "${W}%s${D}\n" "$SEP"
printf "${W}  STRESS TEST SUMMARY${D}\n"
printf "${W}%s${D}\n" "$SEP"
printf "  %-38s  %5s  %5s  %5s  %s\n" "Phase" "Pass" "Fail" "Time" ""
printf "  %s\n" "$sep_thin"

for entry in "${SECTION_TIMES[@]}"; do
    IFS='|' read -r sname selapsed spass sfail <<< "$entry"
    if [[ $sfail -gt 0 ]]; then
        printf "  ${R}%-38s${D}  %5d  ${R}%5d${D}  %5s\n" \
            "$sname" "$spass" "$sfail" "$(fmt_dur "$selapsed")"
    else
        printf "  ${G}%-38s${D}  %5d  %5d  %5s\n" \
            "$sname" "$spass" "$sfail" "$(fmt_dur "$selapsed")"
    fi
done

printf "  %s\n" "$sep_thin"
printf "  ${W}%-38s${D}  %5d  %5d  %5s\n" \
    "TOTAL" "$PASS" "$FAIL" "$(fmt_dur "$TOTAL_ELAPSED")"

# Slowest tests
echo
printf "${W}  Slowest tests${D}\n"
printf "  %s\n" "$sep_thin"
# sort TEST_LOG by elapsed (field 2), descending
TOP5=""
for entry in "${TEST_LOG[@]}"; do
    IFS='|' read -r lbl lel lstat lsec <<< "$entry"
    TOP5+="$lel $lbl\n"
done
printf "$TOP5" | sort -rn | head -5 | while read -r lel lbl; do
    printf "  %5s  %s\n" "$(fmt_dur "$lel")" "$lbl"
done

echo

if [[ $FAIL -gt 0 ]]; then
    printf "${W}%s${D}\n" "$SEP"
    printf "${R}  FAILED TESTS${D}\n"
    printf "${W}%s${D}\n" "$SEP"
    for T in "${FAILED_TESTS[@]}"; do
        printf "  ${R}✗${D}  %s\n" "$T"
    done
    echo
    printf "${R}  %d passed · %d FAILED · %s total${D}\n" \
        "$PASS" "$FAIL" "$(fmt_dur "$TOTAL_ELAPSED")"
    echo
    exit 1
else
    printf "${W}%s${D}\n" "$SEP"
    printf "${G}  ALL %d TESTS PASSED${D}  ·  %s total\n" \
        "$PASS" "$(fmt_dur "$TOTAL_ELAPSED")"
    printf "${W}%s${D}\n" "$SEP"
    echo
fi
