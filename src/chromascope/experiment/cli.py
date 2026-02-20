"""
Unified Chromascope CLI.
Modernized for the OPEN UP architecture.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Type

import numpy as np

from chromascope.experiment.base import BaseConfig
from chromascope.experiment.decay import DecayConfig, DecayRenderer
from chromascope.experiment.encoder import encode_video
from chromascope.experiment.fractal import FractalConfig, FractalKaleidoscopeRenderer
from chromascope.experiment.renderer import UniversalMirrorCompositor
from chromascope.experiment.solar import SolarConfig, SolarRenderer
from chromascope.pipeline import AudioPipeline


@dataclass
class MixedConfig(SolarConfig, DecayConfig, FractalConfig):
    """A catch-all config for heterogeneous mixing."""
    pass


def _progress_bar(current: int, total: int, width: int = 35):
    """Print a progress bar."""
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
        prog="chromascope",
        description="Chromascope: Audio-reactive visualizer suite",
    )

    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output MP4 path")

    # Mode
    parser.add_argument(
        "--mode", type=str, default="fractal",
        choices=["fractal", "solar", "decay", "mixed"],
        help="Visualization mode (default: fractal)"
    )
    
    # Mirroring & Interference
    parser.add_argument(
        "--mirror", type=str, default="off",
        choices=["off", "vertical", "horizontal", "diagonal", "circular", "cycle"],
        help="Mirroring mode (default: off)"
    )
    parser.add_argument(
        "--interference", type=str, default="resonance",
        choices=["resonance", "constructive", "destructive", "sweet_spot", "cycle"],
        help="Interference mode (default: resonance)"
    )

    # Resolution & Profile
    parser.add_argument(
        "-p", "--profile", type=str, default="medium",
        choices=["low", "medium", "high"],
        help="Target profile (low: 720p, medium: 1080p, high: 4k)",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("-f", "--fps", type=int, default=None)

    # Post-processing
    parser.add_argument("--no-glow", action="store_true")
    parser.add_argument("--no-aberration", action="store_true")
    parser.add_argument("--no-vignette", action="store_true")
    parser.add_argument("--palette", type=str, default=None, help="Force palette (jewel, solar)")

    # Performance
    parser.add_argument("--no-low-res-mirror", action="store_true", help="Disable low-res mirror scaling")

    # Limits
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    PROFILES = {
        "low": {"width": 1280, "height": 720, "fps": 30, "quality": "fast"},
        "medium": {"width": 1920, "height": 1080, "fps": 60, "quality": "medium"},
        "high": {"width": 3840, "height": 2160, "fps": 60, "quality": "high"},
    }
    p_cfg = PROFILES[args.profile]
    width = args.width or p_cfg["width"]
    height = args.height or p_cfg["height"]
    fps = args.fps or p_cfg["fps"]
    quality = p_cfg["quality"]

    output = args.output or args.audio.with_name(f"{args.audio.stem}_{args.mode}.mp4")

    # Step 1: Audio analysis
    print(f"Analyzing audio: {args.audio}")
    pipeline = AudioPipeline(target_fps=fps)
    result = pipeline.process(args.audio, use_cache=not args.no_cache)
    manifest = result["manifest"]

    # Step 2: Setup Config and Renderer
    viz_cls_b = None
    palette = args.palette
    
    if args.mode == "fractal":
        config_cls = FractalConfig
        viz_cls_a = FractalKaleidoscopeRenderer
    elif args.mode == "solar":
        config_cls = SolarConfig
        viz_cls_a = SolarRenderer
        if not palette: palette = "solar"
    elif args.mode == "decay":
        config_cls = DecayConfig
        viz_cls_a = DecayRenderer
    elif args.mode == "mixed":
        config_cls = MixedConfig 
        viz_cls_a = SolarRenderer
        viz_cls_b = DecayRenderer
        if not palette: palette = "solar"
    else:
        config_cls = BaseConfig
        viz_cls_a = FractalKaleidoscopeRenderer

    config = config_cls(
        width=width,
        height=height,
        fps=fps,
        glow_enabled=not args.no_glow,
        aberration_enabled=not args.no_aberration,
        vignette_strength=0.0 if args.no_vignette else 0.3,
        palette_type=palette or "jewel",
        mirror_mode=args.mirror,
        interference_mode=args.interference,
        low_res_mirror=not args.no_low_res_mirror
    )

    if args.mirror != "off":
        renderer = UniversalMirrorCompositor(viz_cls_a, config, viz_class_b=viz_cls_b)
    else:
        renderer = viz_cls_a(config)

    # Step 3: Render and Encode
    print(f"Rendering mode: {args.mode}, mirror: {args.mirror}, interference: {args.interference}")
    
    # We need a unified render_manifest or just iterate here
    frames = manifest.get("frames", [])
    if args.max_duration:
        frames = frames[:int(args.max_duration * fps)]
        
    def frame_gen():
        for i, f_data in enumerate(frames):
            yield renderer.render_frame(f_data, i)
            _progress_bar(i + 1, len(frames))

    encode_video(
        frame_iterator=frame_gen(),
        audio_path=args.audio,
        output_path=output,
        width=width,
        height=height,
        fps=fps,
        quality=quality,
        duration=args.max_duration or result["duration"],
        total_frames=len(frames),
    )

    print(f"\nDone! Output: {output}")


if __name__ == "__main__":
    main()
