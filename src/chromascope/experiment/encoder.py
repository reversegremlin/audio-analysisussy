"""
FFmpeg video encoder.

Pipes raw RGB frames to ffmpeg via stdin, muxes with original audio.
No intermediate files — frames go straight from numpy arrays to the encoder.
"""

import subprocess
import sys
from pathlib import Path
from typing import Iterator


# Quality presets: (preset, crf, maxrate, bufsize, pix_fmt)
# Maxrate/bufsize targeting YouTube's recommended bitrates
QUALITY_PRESETS = {
    "high": ("slow", "22", "50M", "100M", "yuv420p"),     # 4k profile (target ~45Mbps)
    "medium": ("medium", "24", "15M", "30M", "yuv420p"), # 1080p profile (target ~12Mbps)
    "fast": ("ultrafast", "28", "8M", "16M", "yuv420p"), # 720p profile (target ~7.5Mbps)
}


def encode_video(
    frame_iterator: Iterator,
    audio_path: Path,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 60,
    quality: str = "high",
    duration: float | None = None,
    total_frames: int | None = None,
    progress_callback: callable = None,
) -> Path:
    """
    Encode frames to MP4 with audio.

    Args:
        frame_iterator: Yields (H, W, 3) uint8 numpy arrays.
        audio_path: Path to audio file to mux in.
        output_path: Output MP4 path.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.
        quality: "high", "medium", or "fast".
        duration: Audio duration for progress calculation.
        total_frames: Total frame count for progress reporting.
        progress_callback: Optional callback(current_frame, total_frames).

    Returns:
        Path to the output file.
    """
    preset, crf, maxrate, bufsize, pix_fmt = QUALITY_PRESETS.get(
        quality, QUALITY_PRESETS["medium"]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        # Raw video input from pipe
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        # Audio input
        "-i", str(audio_path),
        # Video encoding
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", crf,
        "-maxrate", maxrate,
        "-bufsize", bufsize,
        "-pix_fmt", pix_fmt,
        # YouTube compatibility
        "-movflags", "+faststart",
        # Audio encoding
        "-c:a", "aac",
        "-b:a", "192k",
        # Trim to shortest stream
        "-shortest",
        # Output
        str(output_path),
    ]

    # Add duration limit if specified
    if duration is not None:
        # Insert -t before the output path
        cmd.insert(-1, "-t")
        cmd.insert(-1, str(duration))

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    frame_count = 0
    try:
        for frame in frame_iterator:
            # Ensure contiguous C-order array
            raw = frame.tobytes()
            proc.stdin.write(raw)
            frame_count += 1

            if progress_callback and total_frames:
                progress_callback(frame_count, total_frames)

    except BrokenPipeError:
        pass
    finally:
        if proc.stdin:
            proc.stdin.close()

    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited with code {proc.returncode}. See above for details."
        )

    return output_path
