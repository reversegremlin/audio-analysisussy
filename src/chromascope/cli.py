"""
Command-line interface for audio analysis.
"""

import argparse
import json
import sys
from pathlib import Path

from chromascope.pipeline import AudioPipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="audio-analyze",
        description="Extract visual driver features from audio files",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input audio file (wav, mp3, flac)",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output manifest file path (default: <input>_manifest.json)",
    )

    parser.add_argument(
        "-f", "--fps",
        type=int,
        default=60,
        help="Target frames per second (default: 60)",
    )

    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=22050,
        help="Audio sample rate for analysis (default: 22050)",
    )

    parser.add_argument(
        "--format",
        choices=["json", "numpy"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument(
        "--attack",
        type=float,
        default=0.0,
        help="Impact envelope attack time in ms (default: 0)",
    )

    parser.add_argument(
        "--release",
        type=float,
        default=200.0,
        help="Impact envelope release time in ms (default: 200)",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print manifest summary to stdout",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    output_path = args.output
    if output_path is None:
        suffix = ".npz" if args.format == "numpy" else ".json"
        output_path = args.input.with_name(f"{args.input.stem}_manifest{suffix}")

    # Create pipeline
    from chromascope.core.polisher import EnvelopeParams

    pipeline = AudioPipeline(
        target_fps=args.fps,
        sample_rate=args.sample_rate,
        impact_envelope=EnvelopeParams(
            attack_ms=args.attack,
            release_ms=args.release,
        ),
    )

    if not args.quiet:
        print(f"Processing: {args.input}")
        print(f"Target FPS: {args.fps}")

    # Process
    result = pipeline.process(
        args.input,
        output_path=output_path,
        format=args.format,
    )

    if not args.quiet:
        print(f"BPM: {result['bpm']:.1f}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Frames: {result['n_frames']}")
        print(f"Output: {result['output_path']}")

    if args.summary:
        manifest = result["manifest"]
        print("\n--- Manifest Summary ---")
        print(json.dumps(manifest["metadata"], indent=2))

        # Sample frames
        frames = manifest["frames"]
        if len(frames) > 0:
            print(f"\nFirst frame: {json.dumps(frames[0], indent=2)}")
        if len(frames) > 1:
            mid = len(frames) // 2
            print(f"\nMiddle frame ({mid}): {json.dumps(frames[mid], indent=2)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
