"""
CLI entry point for the Fractal Kaleidoscope renderer.

Usage:
    chromascope-fractal <audio_file> [options]
    python -m chromascope.experiment <audio_file> [options]
"""

import argparse
import sys
import time
from pathlib import Path

from chromascope.experiment.encoder import encode_video
from chromascope.experiment.renderer import FractalKaleidoscopeRenderer, RenderConfig
from chromascope.pipeline import AudioPipeline


def _progress_bar(current: int, total: int, width: int = 35):
    """Print a progress bar to stdout."""
    pct = current / max(total, 1) * 100
    filled = int(width * current / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    if sys.stdout.isatty():
        sys.stdout.write(f"\r[{bar}] {pct:5.1f}%  frame {current}/{total}")
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")
    else:
        if current % max(1, total // 20) == 0 or current >= total:
            print(f"{pct:5.1f}%  frame {current}/{total}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        prog="chromascope-fractal",
        description="Audio-reactive fractal kaleidoscope video renderer",
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Input audio file (wav, mp3, flac)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output MP4 path (default: <audio>_fractal.mp4)",
    )

    # Resolution & Profile
    parser.add_argument(
        "-p", "--profile", type=str, default="medium",
        choices=["low", "medium", "high"],
        help="Target profile (low: 720p 30fps, medium: 1080p 60fps, high: 4k 60fps)",
    )
    parser.add_argument("--width", type=int, default=None, help="Video width (overrides profile)")
    parser.add_argument("--height", type=int, default=None, help="Video height (overrides profile)")
    parser.add_argument("-f", "--fps", type=int, default=None, help="Frames per second (overrides profile)")

    # Visual
    parser.add_argument(
        "-s", "--segments", type=int, default=8,
        help="Kaleidoscope symmetry segments (default: 8)",
    )
    parser.add_argument(
        "--fractal", type=str, default="blend",
        choices=["julia", "mandelbrot", "blend"],
        help="Fractal type (default: blend)",
    )
    parser.add_argument(
        "--zoom-speed", type=float, default=1.0,
        help="Base zoom speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--rotation-speed", type=float, default=1.0,
        help="Base rotation speed multiplier (default: 1.0)",
    )

    # Post-processing
    parser.add_argument("--no-glow", action="store_true", help="Disable glow")
    parser.add_argument("--no-aberration", action="store_true", help="Disable chromatic aberration")
    parser.add_argument("--no-vignette", action="store_true", help="Disable vignette")

    # Limits
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="Limit output to N seconds",
    )

    # Caching
    parser.add_argument("--no-cache", action="store_true", help="Force re-analysis of audio")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the analysis cache before running")

    # Quality
    parser.add_argument(
        "-q", "--quality", type=str, default=None,
        choices=["high", "medium", "fast"],
        help="Encoding quality (defaults to profile quality)",
    )

    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Map profile to defaults
    PROFILES = {
        "low": {"width": 1280, "height": 720, "fps": 30, "quality": "fast"},
        "medium": {"width": 1920, "height": 1080, "fps": 60, "quality": "medium"},
        "high": {"width": 3840, "height": 2160, "fps": 60, "quality": "high"},
    }
    p_cfg = PROFILES[args.profile]

    width = args.width or p_cfg["width"]
    height = args.height or p_cfg["height"]
    fps = args.fps or p_cfg["fps"]
    quality = args.quality or p_cfg["quality"]

    output = args.output
    if output is None:
        output = args.audio.with_name(f"{args.audio.stem}_fractal.mp4")

    # Step 1: Audio analysis
    print(f"Analyzing audio: {args.audio}")
    t0 = time.time()

    pipeline = AudioPipeline(target_fps=args.fps)

    if args.clear_cache:
        print("Clearing analysis cache...")
        pipeline.clear_cache()

    result = pipeline.process(args.audio, use_cache=not args.no_cache)
    manifest = result["manifest"]

    print(f"  BPM: {result['bpm']:.1f}")
    print(f"  Duration: {result['duration']:.1f}s")
    print(f"  Frames: {result['n_frames']}")
    print(f"  Analysis took {time.time() - t0:.1f}s")

    # Trim if needed
    if args.max_duration is not None:
        max_frames = int(args.max_duration * fps)
        if max_frames < len(manifest["frames"]):
            manifest["frames"] = manifest["frames"][:max_frames]
            print(f"  Limiting to {args.max_duration}s ({max_frames} frames)")

    total_frames = len(manifest["frames"])

    # Step 2: Render
    print(f"\nRendering {total_frames} frames at {width}x{height} @ {fps}fps")
    quality_map = {
        "high": (200, 400),
        "medium": (100, 250),
        "fast": (60, 120),
    }
    base_iter, max_iter = quality_map.get(quality, (100, 250))

    print(f"  Profile: {args.profile}, Fractal: {args.fractal}, Quality: {quality}")

    config = RenderConfig(
        width=width,
        height=height,
        fps=fps,
        num_segments=args.segments,
        fractal_mode=args.fractal,
        base_zoom_speed=args.zoom_speed,
        base_rotation_speed=args.rotation_speed,
        base_max_iter=base_iter,
        max_max_iter=max_iter,
        glow_enabled=not args.no_glow,
        aberration_enabled=not args.no_aberration,
        vignette_strength=0.0 if args.no_vignette else 0.3,
    )

    renderer = FractalKaleidoscopeRenderer(config)
    frame_gen = renderer.render_manifest(manifest, progress_callback=_progress_bar)

    # Step 3: Encode
    t1 = time.time()

    duration = args.max_duration or result["duration"]

    encode_video(
        frame_iterator=frame_gen,
        audio_path=args.audio,
        output_path=output,
        width=width,
        height=height,
        fps=fps,
        quality=quality,
        duration=duration,
        total_frames=total_frames,
    )

    elapsed = time.time() - t1
    file_size_mb = output.stat().st_size / 1024 / 1024

    print(f"\nDone! {file_size_mb:.1f} MB")
    print(f"  Render+encode took {elapsed:.1f}s ({total_frames / max(elapsed, 0.01):.1f} fps)")
    print(f"  Output: {output}")


if __name__ == "__main__":
    main()
