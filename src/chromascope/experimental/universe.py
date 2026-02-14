"""Greenfield experimental renderer for cosmic kaleidoscope exports.

This module intentionally leaves the existing rendering flow untouched while
providing a dedicated CLI for high-fidelity experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chromascope.render_video import render_video


UNIVERSE_CONFIG = {
    "style": "universe",
    "mirrors": 20,
    "trailAlpha": 86,
    "baseRadius": 260,
    "orbitRadius": 420,
    "rotationSpeed": 2.9,
    "maxScale": 3.1,
    "minSides": 6,
    "maxSides": 28,
    "baseThickness": 2,
    "maxThickness": 13,
    "shapeSeed": 1337,
    "glassSlices": 64,
    "bgColor": "#010109",
    "bgColor2": "#1d0638",
    "accentColor": "#ffd166",
    "chromaColors": True,
    "saturation": 100,
    "dynamicBg": True,
    "bgReactivity": 96,
    "bgParticles": True,
    "bgPulse": True,
}


RESOLUTION_PRESETS = {
    "preview": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the experimental universe kaleidoscope style.",
    )
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path (default: <audio>_universe.mp4)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(RESOLUTION_PRESETS),
        default="4k",
        help="Resolution preset (default: 4k)",
    )
    parser.add_argument("--fps", type=int, default=60, help="Frame rate (default: 60)")
    parser.add_argument(
        "--quality",
        choices=["high", "medium", "fast"],
        default="high",
        help="Encoding profile (default: high)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional cap in seconds for quick iteration",
    )

    args = parser.parse_args()
    if not args.audio.exists():
        raise SystemExit(f"Audio not found: {args.audio}")

    output = args.output or args.audio.with_name(f"{args.audio.stem}_universe.mp4")
    width, height = RESOLUTION_PRESETS[args.preset]

    print(
        f"Rendering universe experiment: {args.audio} -> {output} "
        f"[{width}x{height} @ {args.fps}fps, quality={args.quality}]"
    )

    render_video(
        audio_path=args.audio,
        output_path=output,
        width=width,
        height=height,
        fps=args.fps,
        max_duration=args.max_duration,
        quality=args.quality,
        config=UNIVERSE_CONFIG.copy(),
    )


if __name__ == "__main__":
    main()
