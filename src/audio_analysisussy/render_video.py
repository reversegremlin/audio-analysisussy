"""
Video rendering script for kaleidoscope visualization.

Processes audio through the analysis pipeline and renders
an MP4 video with synchronized visuals.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pygame
from PIL import Image

# Initialize pygame without display for headless rendering
os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()


def render_video(
    audio_path: Path,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 60,
    num_mirrors: int = 8,
    trail_alpha: int = 40,
    max_duration: float = None,
):
    """
    Render kaleidoscope video from audio file.

    Renders frames to temp directory then combines with ffmpeg.

    Args:
        audio_path: Path to input audio file.
        output_path: Path for output MP4 file.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Frames per second.
        num_mirrors: Number of radial symmetry copies.
        trail_alpha: Trail persistence (0-255).
        max_duration: Maximum duration in seconds (None for full audio).
    """
    from audio_analysisussy.pipeline import AudioPipeline
    from audio_analysisussy.visualizers.kaleidoscope import (
        KaleidoscopeConfig,
        KaleidoscopeRenderer,
    )

    print(f"Processing audio: {audio_path}", flush=True)

    # Step 1: Analyze audio
    pipeline = AudioPipeline(target_fps=fps)
    result = pipeline.process(audio_path)
    manifest = result["manifest"]

    print(f"Detected BPM: {result['bpm']:.1f}", flush=True)
    print(f"Duration: {result['duration']:.2f}s", flush=True)
    print(f"Total frames: {result['n_frames']}", flush=True)

    # Step 2: Setup renderer
    config = KaleidoscopeConfig(
        width=width,
        height=height,
        fps=fps,
        num_mirrors=num_mirrors,
        trail_alpha=trail_alpha,
    )
    renderer = KaleidoscopeRenderer(config)

    frames = manifest["frames"]
    total_frames = len(frames)

    # Limit frames if max_duration specified
    if max_duration is not None:
        max_frames = int(max_duration * fps)
        if max_frames < total_frames:
            frames = frames[:max_frames]
            total_frames = max_frames
            print(f"Limiting to {max_duration}s ({total_frames} frames)", flush=True)

    # Step 3: Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="kaleidoscope_")
    print(f"Rendering frames to: {temp_dir}", flush=True)

    try:
        # Step 4: Render frames to PNG files
        print("Rendering frames...", flush=True)
        previous_surface = None
        renderer.accumulated_rotation = 0.0

        for i, frame_data in enumerate(frames):
            surface = renderer.render_frame(frame_data, previous_surface)
            arr = renderer.surface_to_array(surface)

            # Save frame as PNG
            img = Image.fromarray(arr)
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            img.save(frame_path, "PNG", compress_level=1)

            # Keep reference for trail effect
            previous_surface = surface.copy()

            if (i + 1) % 500 == 0 or i == total_frames - 1:
                pct = (i + 1) / total_frames * 100
                print(f"  Rendered {i + 1}/{total_frames} frames ({pct:.1f}%)", flush=True)

        # Step 5: Combine frames into video with ffmpeg
        print("Encoding video...", flush=True)

        temp_video = os.path.join(temp_dir, "video.mp4")
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            temp_video,
        ]

        result_encode = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result_encode.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {result_encode.stderr.decode()}")

        # Step 6: Mux with audio
        print("Adding audio track...", flush=True)

        ffmpeg_mux = [
            "ffmpeg",
            "-y",
            "-i", temp_video,
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
        ]

        # Add duration limit if specified
        if max_duration is not None:
            ffmpeg_mux.extend(["-t", str(max_duration)])
        else:
            ffmpeg_mux.append("-shortest")

        ffmpeg_mux.append(str(output_path))

        result_mux = subprocess.run(
            ffmpeg_mux,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result_mux.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed: {result_mux.stderr.decode()}")

        print(f"Done! Output: {output_path}", flush=True)
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB", flush=True)

    finally:
        # Cleanup temp directory
        print("Cleaning up temp files...", flush=True)
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
        help="Trail persistence 0-255 (default: 40)",
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
    )


if __name__ == "__main__":
    main()
