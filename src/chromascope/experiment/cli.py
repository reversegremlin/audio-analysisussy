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

from chromascope.experiment.attractor import AttractorConfig, AttractorRenderer
from chromascope.experiment.base import BaseConfig
from chromascope.experiment.chemical import ChemicalConfig, ChemicalRenderer
from chromascope.experiment.decay import DecayConfig, DecayRenderer
from chromascope.experiment.encoder import encode_video
from chromascope.experiment.fractal import FractalConfig, FractalKaleidoscopeRenderer
from chromascope.experiment.renderer import UniversalMirrorCompositor
from chromascope.experiment.solar import SolarConfig, SolarRenderer
from chromascope.pipeline import AudioPipeline


@dataclass
class MixedConfig(SolarConfig, DecayConfig, FractalConfig, ChemicalConfig, AttractorConfig):
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
        choices=["fractal", "solar", "decay", "mixed", "chemical", "attractor"],
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

    # Fractal-specific
    parser.add_argument(
        "--fractal-mode", type=str, default=None,
        choices=["blend", "mandelbrot"],
        help="Fractal texture mode when --mode fractal (default: blend)",
    )

    # Chemical-specific
    parser.add_argument(
        "--chem-style", type=str, default=None,
        choices=["neon_lab", "plasma_beaker", "midnight_fluor", "synth_chem"],
        help="Visual style preset when --mode chemical (default: neon_lab)",
    )
    parser.add_argument("--reaction-gain", type=float, default=None,
                        help="Reaction front intensity [0–2]")
    parser.add_argument("--crystal-rate", type=float, default=None,
                        help="Crystal growth speed [0–2]")
    parser.add_argument("--nucleation-threshold", type=float, default=None,
                        help="Percussive sensitivity for crystal seeding [0–1]")
    parser.add_argument("--supersaturation", type=float, default=None,
                        help="Baseline branching propensity [0–1]")
    parser.add_argument("--bloom", type=float, default=None,
                        help="Post-glow multiplier [0–2]")
    parser.add_argument(
        "--chem-palette", type=str, default=None,
        choices=["iron", "copper", "sodium", "potassium", "mixed"],
        help="Chemistry-inspired colour palette (default: mixed)",
    )

    # Attractor-specific
    parser.add_argument(
        "--attractor-blend-mode", type=str, default=None,
        choices=["lorenz", "rossler", "dual", "morph"],
        help="Attractor blend mode when --mode attractor (default: dual)",
    )
    parser.add_argument(
        "--num-particles", type=int, default=None,
        help="Number of particles per attractor [default: 3000]",
    )
    parser.add_argument(
        "--trail-decay", type=float, default=None,
        help="Trail fade rate [0,1] (default: 0.96)",
    )
    parser.add_argument(
        "--projection-speed", type=float, default=None,
        help="Base azimuth rotation speed in rad/s (default: 0.2)",
    )
    parser.add_argument(
        "--attractor-palette", type=str, default=None,
        choices=["neon_aurora", "plasma_coil", "void_fire", "quantum_foam"],
        help="Neon colour palette for attractor mode (default: neon_aurora)",
    )

    # Preview
    parser.add_argument(
        "--preview", action="store_true",
        help="Show real-time preview in a pygame window instead of encoding to MP4",
    )

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
    elif args.mode == "chemical":
        config_cls = ChemicalConfig
        viz_cls_a = ChemicalRenderer
    elif args.mode == "attractor":
        config_cls = AttractorConfig
        viz_cls_a = AttractorRenderer
    elif args.mode == "mixed":
        config_cls = MixedConfig
        viz_cls_a = SolarRenderer
        viz_cls_b = DecayRenderer
        if not palette: palette = "solar"
    else:
        config_cls = BaseConfig
        viz_cls_a = FractalKaleidoscopeRenderer

    config_kwargs = dict(
        width=width,
        height=height,
        fps=fps,
        glow_enabled=not args.no_glow,
        aberration_enabled=not args.no_aberration,
        vignette_strength=0.0 if args.no_vignette else 0.3,
        palette_type=palette or "jewel",
        mirror_mode=args.mirror,
        interference_mode=args.interference,
        low_res_mirror=not args.no_low_res_mirror,
    )
    if args.mode == "fractal" and args.fractal_mode is not None:
        config_kwargs["fractal_mode"] = args.fractal_mode
    if args.mode == "attractor":
        if args.attractor_blend_mode is not None:
            config_kwargs["blend_mode"] = args.attractor_blend_mode
        if args.num_particles is not None:
            config_kwargs["num_particles"] = args.num_particles
        if args.trail_decay is not None:
            config_kwargs["trail_decay"] = args.trail_decay
        if args.projection_speed is not None:
            config_kwargs["projection_speed"] = args.projection_speed
        if args.attractor_palette is not None:
            config_kwargs["attractor_palette"] = args.attractor_palette
    if args.mode == "chemical":
        if args.chem_style is not None:
            config_kwargs["style"] = args.chem_style
        if args.reaction_gain is not None:
            config_kwargs["reaction_gain"] = args.reaction_gain
        if args.crystal_rate is not None:
            config_kwargs["crystal_rate"] = args.crystal_rate
        if args.nucleation_threshold is not None:
            config_kwargs["nucleation_threshold"] = args.nucleation_threshold
        if args.supersaturation is not None:
            config_kwargs["supersaturation"] = args.supersaturation
        if args.bloom is not None:
            config_kwargs["bloom"] = args.bloom
        if args.chem_palette is not None:
            config_kwargs["chem_palette"] = args.chem_palette
    config = config_cls(**config_kwargs)

    if args.mirror != "off":
        renderer = UniversalMirrorCompositor(viz_cls_a, config, viz_class_b=viz_cls_b)
    else:
        renderer = viz_cls_a(config)

    # Step 3: Render (preview or encode)
    print(f"Rendering mode: {args.mode}, mirror: {args.mirror}, interference: {args.interference}")

    frames = manifest.get("frames", [])
    if args.max_duration:
        frames = frames[:int(args.max_duration * fps)]

    if args.preview:
        from chromascope.experiment.preview import run_preview
        print(f"Opening preview window ({len(frames)} frames @ {fps} fps)…")
        run_preview(renderer, frames, fps, title=f"Chromascope — {args.mode}")
        print("Preview closed.")
        return

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
