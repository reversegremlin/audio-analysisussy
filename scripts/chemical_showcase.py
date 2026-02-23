"""
Chemical Renderer Showcase
===========================
Renders a battery of creative option combos for a single audio file,
each exploring a different visual mood of the chromascope-chemical mode.

Usage
-----
    python scripts/chemical_showcase.py tests/assets/shineglowglisten.mp3
    python scripts/chemical_showcase.py tests/assets/shineglowglisten.mp3 --quick
    python scripts/chemical_showcase.py tests/assets/shineglowglisten.mp3 --duration 45 --profile medium
    python scripts/chemical_showcase.py tests/assets/shineglowglisten.mp3 --only iron_plasma copper_midnight

Options
-------
    --quick          30 s clips at 720p/30fps (fast iteration)
    --duration N     Clip length in seconds (default: 60)
    --profile P      low | medium | high (default: low when --quick, else medium)
    --output-dir D   Where to write MP4s (default: ./chemical_renders/)
    --only NAME ...  Render only the named combo(s)
    --list           Print all combo names and exit
    --no-cache       Skip audio cache
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Combo definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Combo:
    name: str
    description: str
    flags: List[str]
    emoji: str = "🧪"


COMBOS: List[Combo] = [
    # ── Iron family ──────────────────────────────────────────────────────────
    Combo(
        name="iron_plasma",
        emoji="🔥",
        description="Molten iron plasma — aggressive red/orange reaction fronts, dense fast nucleation",
        flags=[
            "--chem-style", "plasma_beaker",
            "--chem-palette", "iron",
            "--reaction-gain", "1.9",
            "--crystal-rate", "0.5",
            "--nucleation-threshold", "0.12",
            "--supersaturation", "0.85",
            "--bloom", "1.8",
        ],
    ),
    Combo(
        name="iron_embers",
        emoji="🌋",
        description="Dark forge — slow ember iron with long crystal memory and restrained glow",
        flags=[
            "--chem-style", "midnight_fluor",
            "--chem-palette", "iron",
            "--reaction-gain", "0.8",
            "--crystal-rate", "0.3",
            "--nucleation-threshold", "0.55",
            "--supersaturation", "0.25",
            "--bloom", "0.65",
        ],
    ),
    Combo(
        name="iron_synth",
        emoji="⚡",
        description="Synth iron — red/orange cycling with fast reactions and hue drift",
        flags=[
            "--chem-style", "synth_chem",
            "--chem-palette", "iron",
            "--reaction-gain", "1.5",
            "--crystal-rate", "0.9",
            "--nucleation-threshold", "0.2",
            "--supersaturation", "0.6",
            "--bloom", "1.4",
        ],
    ),

    # ── Copper family ────────────────────────────────────────────────────────
    Combo(
        name="copper_midnight",
        emoji="🌊",
        description="Midnight copper — dark atmosphere, teal/aqua crystals with long phrase memory",
        flags=[
            "--chem-style", "midnight_fluor",
            "--chem-palette", "copper",
            "--reaction-gain", "1.0",
            "--crystal-rate", "0.55",
            "--nucleation-threshold", "0.38",
            "--supersaturation", "0.3",
            "--bloom", "0.75",
        ],
    ),
    Combo(
        name="copper_plasma",
        emoji="💠",
        description="Electric aqua plasma — hyper-saturated cyan reaction fronts, burst nucleation",
        flags=[
            "--chem-style", "plasma_beaker",
            "--chem-palette", "copper",
            "--reaction-gain", "1.7",
            "--crystal-rate", "1.2",
            "--nucleation-threshold", "0.15",
            "--supersaturation", "0.75",
            "--bloom", "1.9",
        ],
    ),
    Combo(
        name="copper_synth",
        emoji="🌀",
        description="Synth teal — copper into lilac hue drift, moderate branching",
        flags=[
            "--chem-style", "synth_chem",
            "--chem-palette", "copper",
            "--reaction-gain", "1.2",
            "--crystal-rate", "1.1",
            "--nucleation-threshold", "0.25",
            "--supersaturation", "0.55",
            "--bloom", "1.3",
        ],
    ),

    # ── Sodium family ────────────────────────────────────────────────────────
    Combo(
        name="sodium_neon",
        emoji="⚗️",
        description="Sodium neon lab — intense amber/gold emission, balanced beat-locked growth",
        flags=[
            "--chem-style", "neon_lab",
            "--chem-palette", "sodium",
            "--reaction-gain", "1.2",
            "--crystal-rate", "1.2",
            "--nucleation-threshold", "0.28",
            "--supersaturation", "0.45",
            "--bloom", "1.45",
        ],
    ),
    Combo(
        name="sodium_plasma",
        emoji="☀️",
        description="Solar sodium — maximum yellow/white intensity, hyper-bloom reaction storm",
        flags=[
            "--chem-style", "plasma_beaker",
            "--chem-palette", "sodium",
            "--reaction-gain", "2.0",
            "--crystal-rate", "0.7",
            "--nucleation-threshold", "0.1",
            "--supersaturation", "0.9",
            "--bloom", "2.0",
        ],
    ),

    # ── Potassium family ─────────────────────────────────────────────────────
    Combo(
        name="potassium_dense",
        emoji="💜",
        description="Potassium fractal — ultra-dense lilac branching, high supersaturation chaos",
        flags=[
            "--chem-style", "neon_lab",
            "--chem-palette", "potassium",
            "--reaction-gain", "1.3",
            "--crystal-rate", "1.8",
            "--nucleation-threshold", "0.08",
            "--supersaturation", "0.95",
            "--bloom", "1.6",
        ],
    ),
    Combo(
        name="potassium_synth",
        emoji="🔮",
        description="Violet synth — lilac/purple cycling, dreamy hue rotation on every beat",
        flags=[
            "--chem-style", "synth_chem",
            "--chem-palette", "potassium",
            "--reaction-gain", "1.1",
            "--crystal-rate", "1.0",
            "--nucleation-threshold", "0.3",
            "--supersaturation", "0.7",
            "--bloom", "1.5",
        ],
    ),

    # ── Mixed / full-spectrum ─────────────────────────────────────────────────
    Combo(
        name="mixed_synth_chaos",
        emoji="🌈",
        description="Full-spectrum chaos — mixed palette cycling, maximum density + bloom",
        flags=[
            "--chem-style", "synth_chem",
            "--chem-palette", "mixed",
            "--reaction-gain", "2.0",
            "--crystal-rate", "1.5",
            "--nucleation-threshold", "0.08",
            "--supersaturation", "1.0",
            "--bloom", "2.0",
        ],
    ),
    Combo(
        name="mixed_neon_balanced",
        emoji="🧬",
        description="Neon lab balanced — mixed palette, default reaction speed, extra bloom",
        flags=[
            "--chem-style", "neon_lab",
            "--chem-palette", "mixed",
            "--reaction-gain", "1.0",
            "--crystal-rate", "1.0",
            "--nucleation-threshold", "0.3",
            "--supersaturation", "0.5",
            "--bloom", "1.5",
        ],
    ),
    Combo(
        name="mixed_midnight_slow",
        emoji="🌌",
        description="Midnight drift — slow mixed crystals, dark atmosphere, long structural memory",
        flags=[
            "--chem-style", "midnight_fluor",
            "--chem-palette", "mixed",
            "--reaction-gain", "0.7",
            "--crystal-rate", "0.4",
            "--nucleation-threshold", "0.45",
            "--supersaturation", "0.35",
            "--bloom", "0.8",
        ],
    ),

    # ── Weird science ─────────────────────────────────────────────────────────
    Combo(
        name="ghost_crystal",
        emoji="👻",
        description="Ghost crystal — near-silent growth, extremely slow dissolution, faint edge shimmer",
        flags=[
            "--chem-style", "midnight_fluor",
            "--chem-palette", "sodium",
            "--reaction-gain", "0.4",
            "--crystal-rate", "2.0",
            "--nucleation-threshold", "0.6",
            "--supersaturation", "0.15",
            "--bloom", "0.5",
            "--no-aberration",
        ],
    ),
    Combo(
        name="superfluid_bloom",
        emoji="🫧",
        description="Superfluid — copper + sodium mixed with maximum branching, no vignette",
        flags=[
            "--chem-style", "plasma_beaker",
            "--chem-palette", "mixed",
            "--reaction-gain", "1.6",
            "--crystal-rate", "1.9",
            "--nucleation-threshold", "0.05",
            "--supersaturation", "0.88",
            "--bloom", "1.7",
            "--no-vignette",
        ],
    ),
    Combo(
        name="raw_reaction",
        emoji="⚛️",
        description="Raw reaction — no glow, no aberration, pure field energy exposed",
        flags=[
            "--chem-style", "neon_lab",
            "--chem-palette", "iron",
            "--reaction-gain", "1.4",
            "--crystal-rate", "1.0",
            "--nucleation-threshold", "0.2",
            "--supersaturation", "0.6",
            "--bloom", "1.0",
            "--no-glow",
            "--no-aberration",
        ],
    ),
]

COMBO_INDEX = {c.name: c for c in COMBOS}


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def _hr(char="─", width=72):
    return char * width


def _render_combo(
    combo: Combo,
    audio: Path,
    output_dir: Path,
    profile: str,
    duration: Optional[float],
    no_cache: bool,
    cli: str,
    index: int,
    total: int,
) -> dict:
    output_path = output_dir / f"{combo.name}.mp4"

    cmd = [
        cli, str(audio),
        "--mode", "chemical",
        "--profile", profile,
        "-o", str(output_path),
        *combo.flags,
    ]
    if duration is not None:
        cmd += ["--max-duration", str(duration)]
    if no_cache:
        cmd += ["--no-cache"]

    print(f"\n{BOLD}{CYAN}[{index}/{total}] {combo.emoji}  {combo.name}{RESET}")
    print(f"  {DIM}{combo.description}{RESET}")
    print(f"  {DIM}→ {output_path.name}{RESET}")
    print(f"  {DIM}$ {' '.join(cmd)}{RESET}\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.perf_counter() - t0

    success = result.returncode == 0 and output_path.exists()
    size_mb = output_path.stat().st_size / 1_048_576 if output_path.exists() else 0

    status = f"{GREEN}✓ done{RESET}" if success else f"{RED}✗ failed (exit {result.returncode}){RESET}"
    print(f"\n  {status}  {elapsed:.1f}s  {size_mb:.1f} MB")

    return {
        "name": combo.name,
        "emoji": combo.emoji,
        "ok": success,
        "elapsed": elapsed,
        "size_mb": size_mb,
        "path": str(output_path) if success else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Render a battery of chemical-mode combos for a single audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=Path, nargs="?", help="Input audio file")
    parser.add_argument("--quick", action="store_true",
                        help="30 s clips at 720p/30fps")
    parser.add_argument("--duration", type=float, default=None,
                        help="Clip length in seconds (default: full track unless --quick)")
    parser.add_argument("--profile", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="Resolution profile (default: low with --quick, else medium)")
    parser.add_argument("--output-dir", type=Path, default=Path("chemical_renders"),
                        help="Output directory (default: ./chemical_renders/)")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="Render only these named combos")
    parser.add_argument("--list", action="store_true",
                        help="Print all combo names and exit")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"\n{BOLD}Available combos ({len(COMBOS)}):{RESET}\n")
        for c in COMBOS:
            print(f"  {c.emoji}  {BOLD}{c.name:<26}{RESET}  {DIM}{c.description}{RESET}")
        print()
        return

    if args.audio is None:
        print(f"{RED}Error: audio file required{RESET}", file=sys.stderr)
        parser.print_usage()
        sys.exit(1)
    if not args.audio.exists():
        print(f"{RED}Error: audio file not found: {args.audio}{RESET}", file=sys.stderr)
        sys.exit(1)

    # Resolve CLI path
    repo_root = Path(__file__).parent.parent
    cli = str(repo_root / ".venv" / "bin" / "chromascope-fractal")
    if not Path(cli).exists():
        cli = "chromascope-fractal"  # fall back to PATH

    # Defaults
    profile = args.profile or ("low" if args.quick else "medium")
    duration = args.duration or (30.0 if args.quick else None)

    # Filter combos
    if args.only:
        bad = [n for n in args.only if n not in COMBO_INDEX]
        if bad:
            print(f"{RED}Unknown combo(s): {bad}  (run --list to see all){RESET}", file=sys.stderr)
            sys.exit(1)
        selected = [COMBO_INDEX[n] for n in args.only]
    else:
        selected = COMBOS

    # Output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Header
    dur_label = f"{duration}s" if duration else "full track"
    print(f"\n{_hr('═')}")
    print(f"{BOLD}  Chromascope · Chemical Showcase{RESET}")
    print(f"{_hr('═')}")
    print(f"  Audio   : {args.audio}")
    print(f"  Profile : {profile}  |  Clip: {dur_label}")
    print(f"  Output  : {args.output_dir.resolve()}")
    print(f"  Combos  : {len(selected)} selected")
    print(_hr('═'))

    results = []
    total_t0 = time.perf_counter()

    for i, combo in enumerate(selected, 1):
        res = _render_combo(
            combo=combo,
            audio=args.audio,
            output_dir=args.output_dir,
            profile=profile,
            duration=duration,
            no_cache=args.no_cache,
            cli=cli,
            index=i,
            total=len(selected),
        )
        results.append(res)

    total_elapsed = time.perf_counter() - total_t0
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]

    # Summary table
    print(f"\n{_hr('═')}")
    print(f"{BOLD}  Summary  —  {len(ok)}/{len(results)} succeeded  ({total_elapsed/60:.1f} min total){RESET}")
    print(_hr('─'))
    for r in results:
        tick = f"{GREEN}✓{RESET}" if r["ok"] else f"{RED}✗{RESET}"
        size = f"{r['size_mb']:5.1f} MB" if r["ok"] else "  failed"
        print(f"  {tick} {r['emoji']} {r['name']:<26}  {r['elapsed']:6.1f}s  {size}")
    print(_hr('═'))

    if ok:
        print(f"\n{BOLD}Output files:{RESET}")
        for r in ok:
            print(f"  {r['emoji']} {r['path']}")

    if fail:
        print(f"\n{RED}{BOLD}Failed:{RESET}")
        for r in fail:
            print(f"  ✗ {r['name']}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All done!{RESET} 🎉\n")


if __name__ == "__main__":
    main()
