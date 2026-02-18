"""
CLI entry point for the Solar Flare renderer.

Usage:
    chromascope-solar <audio_file> [options]
"""

import argparse
import sys
import time
from pathlib import Path

from chromascope.experiment.encoder import encode_video
from chromascope.experiment.solar import SolarRenderer
from chromascope.pipeline import AudioPipeline

# A simple config class for now
class RenderConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
        prog="chromascope-solar",
        description="Audio-reactive solar flare video renderer",
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
        help="Output MP4 path (default: <audio>_solar.mp4)",
    )

    # Resolution
    parser.add_argument("--width", type=int, default=1920, help="Video width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Video height (default: 1080)")
    parser.add_argument("-f", "--fps", type=int, default=60, help="Frames per second (default: 60)")

    # Visual
    parser.add_argument("--pan-speed-x", type=float, default=0.1, help="Horizontal pan speed (default: 0.1)")
    parser.add_argument("--pan-speed-y", type=float, default=0.05, help="Vertical pan speed (default: 0.05)")
    parser.add_argument("--zoom-speed", type=float, default=0.05, help="Zoom speed (default: 0.05)")
    parser.add_argument("--colormap", type=str, default="default", help="Color map to use (default: default)")


    # Post-processing
    parser.add_argument("--no-glow", action="store_true", help="Disable glow")
    parser.add_argument("--no-aberration", action="store_true", help="Disable chromatic aberration")
    parser.add_argument("--no-vignette", action="store_true", help="Disable vignette")

    # Limits
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="Limit output to N seconds",
    )

    # Quality
    parser.add_argument(
        "-q", "--quality", type=str, default="high",
        choices=["high", "medium", "fast"],
        help="Encoding quality (default: high)",
    )

    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if output is None:
        output = args.audio.with_name(f"{args.audio.stem}_solar.mp4")

    # Step 1: Audio analysis
    print(f"Analyzing audio: {args.audio}")
    t0 = time.time()

    pipeline = AudioPipeline(target_fps=args.fps)
    result = pipeline.process(args.audio)
    manifest = result["manifest"]

    print(f"  BPM: {result['bpm']:.1f}")
    print(f"  Duration: {result['duration']:.1f}s")
    print(f"  Frames: {result['n_frames']}")
    print(f"  Analysis took {time.time() - t0:.1f}s")

    # Trim if needed
    if args.max_duration is not None:
        max_frames = int(args.max_duration * args.fps)
        if max_frames < len(manifest["frames"]):
            manifest["frames"] = manifest["frames"][:max_frames]
            print(f"  Limiting to {args.max_duration}s ({max_frames} frames)")

    total_frames = len(manifest["frames"])

    # Step 2: Render
    print(f"\nRendering {total_frames} frames at {args.width}x{args.height} @ {args.fps}fps")

    config = RenderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        pan_speed_x=args.pan_speed_x,
        pan_speed_y=args.pan_speed_y,
        zoom_speed=args.zoom_speed,
        colormap=args.colormap,
        glow_enabled=not args.no_glow,
        aberration_enabled=not args.no_aberration,
        vignette_strength=0.0 if args.no_vignette else 0.3,
    )

    renderer = SolarRenderer(config)
    frame_gen = renderer.render_manifest(manifest, progress_callback=_progress_bar)

    # Step 3: Encode
    t1 = time.time()

    duration = args.max_duration or result["duration"]

    encode_video(
        frame_iterator=frame_gen,
        audio_path=args.audio,
        output_path=output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        quality=args.quality,
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
