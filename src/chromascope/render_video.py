"""
Video rendering script for kaleidoscope visualization.

Processes audio through the analysis pipeline, then delegates
frame rendering to the Node.js CLI renderer (which uses the same
Canvas 2D render engine as the browser frontend) and produces an MP4.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# Path to the Node.js CLI renderer relative to this file
_RENDER_DIR = Path(__file__).resolve().parent.parent.parent / "render"
_CLI_JS = _RENDER_DIR / "cli.js"


def render_video(
    audio_path: Path,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 60,
    num_mirrors: int = 8,
    trail_alpha: int = 40,
    max_duration: float = None,
    progress_callback: callable = None,
    style: str = "geometric",
    base_radius: float = 150.0,
    max_scale: float = 1.8,
    base_thickness: int = 3,
    max_thickness: int = 12,
    orbit_radius: float = 200.0,
    rotation_speed: float = 2.0,
    min_sides: int = 3,
    max_sides: int = 12,
    config: dict = None,
    quality: str = "high",
):
    """
    Render kaleidoscope video from audio file.

    Analyzes audio via the Python pipeline, then delegates rendering
    to the Node.js CLI renderer which uses the shared Canvas 2D engine.

    Args:
        audio_path: Path to input audio file.
        output_path: Path for output MP4 file.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Frames per second.
        num_mirrors: Number of radial symmetry copies.
        trail_alpha: Trail persistence (0-100).
        max_duration: Maximum duration in seconds (None for full audio).
        progress_callback: Optional callback(progress: int, message: str).
        style: Visualization style name.
        base_radius: Base shape size.
        max_scale: Maximum pulse scale.
        base_thickness: Base line thickness.
        max_thickness: Maximum line thickness.
        orbit_radius: Base orbit distance.
        rotation_speed: Base rotation multiplier.
        min_sides: Minimum polygon sides.
        max_sides: Maximum polygon sides.
        config: Full frontend config dict (overrides individual params).
        quality: Encoding quality profile ("high", "medium", or "fast").
    """
    from chromascope.pipeline import AudioPipeline

    def report_progress(pct: int, msg: str):
        bar_width = 30
        pct_clamped = max(0, min(100, int(pct)))
        filled = int(bar_width * (pct_clamped / 100.0))
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"

        if sys.stdout.isatty():
            sys.stdout.write(f"\r{bar} {pct_clamped:3d}%  {msg:60.60}")
            sys.stdout.flush()
            if pct_clamped >= 100:
                sys.stdout.write("\n")
        else:
            print(f"{pct_clamped:3d}% {msg}", flush=True)

        if progress_callback:
            progress_callback(pct_clamped, msg)

    report_progress(0, f"Processing audio: {audio_path}")

    # Step 1: Analyze audio
    report_progress(5, "Analyzing audio...")
    pipeline = AudioPipeline(target_fps=fps)
    result = pipeline.process(audio_path)
    manifest = result["manifest"]

    report_progress(10, f"Detected BPM: {result['bpm']:.1f}, Duration: {result['duration']:.2f}s")

    # Limit frames if max_duration specified
    if max_duration is not None:
        max_frames = int(max_duration * fps)
        frames = manifest["frames"]
        if max_frames < len(frames):
            manifest["frames"] = frames[:max_frames]
            print(f"Limiting to {max_duration}s ({max_frames} frames)", flush=True)

    # Step 2: Build config for the Node.js renderer
    # If a full config dict was provided, use it; otherwise build from params
    if config is None:
        config = {}

    render_config = {
        "style": config.get("style", style),
        "mirrors": config.get("mirrors", num_mirrors),
        "baseRadius": config.get("baseRadius", base_radius),
        "orbitRadius": config.get("orbitRadius", orbit_radius),
        "rotationSpeed": config.get("rotationSpeed", rotation_speed),
        "maxScale": config.get("maxScale", max_scale),
        "trailAlpha": config.get("trailAlpha", trail_alpha),
        "minSides": config.get("minSides", min_sides),
        "maxSides": config.get("maxSides", max_sides),
        "baseThickness": config.get("baseThickness", base_thickness),
        "maxThickness": config.get("maxThickness", max_thickness),
        "width": width,
        "height": height,
        "fps": fps,
        # Pass through optional frontend config keys
        "shapeSeed": config.get("shapeSeed", 42),
        "glassSlices": config.get("glassSlices", 30),
        "bgColor": config.get("bgColor", "#05050f"),
        "bgColor2": config.get("bgColor2", "#1a0a2e"),
        "accentColor": config.get("accentColor", "#f59e0b"),
        "chromaColors": config.get("chromaColors", True),
        "saturation": config.get("saturation", 85),
        "dynamicBg": config.get("dynamicBg", True),
        "bgReactivity": config.get("bgReactivity", 70),
        "bgParticles": config.get("bgParticles", True),
        "bgPulse": config.get("bgPulse", True),
    }

    # Step 3: Write manifest + config to temp files
    temp_dir = tempfile.mkdtemp(prefix="chromascope_render_")
    manifest_path = os.path.join(temp_dir, "manifest.json")
    config_path = os.path.join(temp_dir, "config.json")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    with open(config_path, "w") as f:
        json.dump(render_config, f)

    report_progress(12, "Starting Node.js renderer...")

    try:
        # Step 4: Spawn the Node.js CLI renderer
        node_cmd = [
            "node", str(_CLI_JS),
            "--manifest", manifest_path,
            "--config", config_path,
            "--audio", str(audio_path),
            "--output", str(output_path),
            "--width", str(width),
            "--height", str(height),
            "--fps", str(fps),
            "--quality", quality,
        ]

        proc = subprocess.Popen(
            node_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Read progress JSON lines from stderr
        for line in proc.stderr:
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get("type") == "progress":
                    # Map renderer progress (0-100) into our 12-95 range
                    renderer_pct = msg.get("percent", 0)
                    overall_pct = 12 + int(renderer_pct * 0.83)
                    report_progress(overall_pct, msg.get("message", "Rendering..."))
                elif msg.get("type") == "error":
                    raise RuntimeError(f"Node.js renderer error: {msg.get('message')}")
            except json.JSONDecodeError:
                # Non-JSON output from ffmpeg or node, ignore
                pass

        proc.wait()
        if proc.returncode != 0:
            # Read any remaining stderr
            remaining = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Node.js renderer exited with code {proc.returncode}: {remaining[-500:]}"
            )

        file_size_mb = output_path.stat().st_size / 1024 / 1024
        report_progress(100, f"Complete! {file_size_mb:.1f} MB")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Render kaleidoscope video from audio"
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
        help="Output MP4 file (default: <audio>_kaleidoscope.mp4)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Video width (default: 1920)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Video height (default: 1080)",
    )

    parser.add_argument(
        "-f", "--fps",
        type=int,
        default=60,
        help="Frames per second (default: 60)",
    )

    parser.add_argument(
        "-m", "--mirrors",
        type=int,
        default=8,
        help="Number of radial mirrors (default: 8)",
    )

    parser.add_argument(
        "-t", "--trail",
        type=int,
        default=40,
        help="Trail persistence 0-100 (default: 40)",
    )

    parser.add_argument(
        "-s", "--style",
        type=str,
        default="geometric",
        help="Visualization style (default: geometric)",
    )

    parser.add_argument(
        "-q", "--quality",
        type=str,
        choices=["high", "medium", "fast"],
        default="high",
        help="Encoding quality: high (yuv444p, slow preset — crispest), "
             "medium (yuv420p, medium preset), fast (quick preview) (default: high)",
    )

    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if output is None:
        output = args.audio.with_name(f"{args.audio.stem}_kaleidoscope.mp4")

    render_video(
        audio_path=args.audio,
        output_path=output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_mirrors=args.mirrors,
        trail_alpha=args.trail,
        style=args.style,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
