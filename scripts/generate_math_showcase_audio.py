#!/usr/bin/env python3
"""Generate mathematically-inspired public-domain classical-style showcase audio."""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 22_050


def _write_wav(path: Path, samples: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        clipped = [max(-1.0, min(1.0, sample)) for sample in samples]
        frames = b"".join(struct.pack("<h", int(sample * 32767)) for sample in clipped)
        wav_file.writeframes(frames)


def fibonacci_etude(seconds: int = 16) -> list[float]:
    fib = [1, 1, 2, 3, 5, 8, 13, 21]
    scale = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 659.25, 783.99]
    total = SAMPLE_RATE * seconds
    output: list[float] = []
    for i in range(total):
        t = i / SAMPLE_RATE
        step = int(t * 6) % len(fib)
        degree = fib[step] % len(scale)
        f = scale[degree]
        phase = (t * 6) % 1
        envelope = 1 - (phase * 0.6)
        sample = 0.35 * math.sin(2 * math.pi * f * t) * envelope
        sample += 0.18 * math.sin(2 * math.pi * (f * 1.5) * t) * envelope
        output.append(sample)
    return output


def golden_ratio_canon(seconds: int = 18) -> list[float]:
    phi = (1 + 5**0.5) / 2
    motif = [261.63, 329.63, 392.0, 523.25, 392.0, 329.63]
    total = SAMPLE_RATE * seconds
    output: list[float] = []
    for i in range(total):
        t = i / SAMPLE_RATE
        beat = t * 2.5
        f1 = motif[int(beat) % len(motif)]
        f2 = motif[int(beat / phi) % len(motif)] * phi / 2
        pulse = (1 - (beat % 1)) ** 1.8
        sample = 0.28 * math.sin(2 * math.pi * f1 * t) * pulse
        sample += 0.22 * math.sin(2 * math.pi * f2 * t) * (0.6 + 0.4 * pulse)
        output.append(sample)
    return output


def prime_pulse_chorale(seconds: int = 15) -> list[float]:
    primes = [2, 3, 5, 7, 11]
    chord = [220.0, 277.18, 329.63, 415.3, 554.37]
    total = SAMPLE_RATE * seconds
    output: list[float] = []
    for i in range(total):
        t = i / SAMPLE_RATE
        sample = 0.15 * math.sin(2 * math.pi * 110 * t)
        for prime, freq in zip(primes, chord):
            gate = 1.0 if int(t * prime) % 2 == 0 else 0.0
            sample += 0.08 * math.sin(2 * math.pi * freq * t) * gate
        output.append(sample)
    return output


def main() -> None:
    output_dir = Path("docs/assets/showcase/audio")
    tracks = {
        "fibonacci_etude.wav": {
            "title": "Fibonacci Etude",
            "description": "Arpeggio index follows Fibonacci sequence.",
            "samples": fibonacci_etude(),
        },
        "golden_ratio_canon.wav": {
            "title": "Golden Ratio Canon",
            "description": "Second voice enters at phi-scaled offset.",
            "samples": golden_ratio_canon(),
        },
        "prime_pulse_chorale.wav": {
            "title": "Prime Pulse Chorale",
            "description": "Voices pulse on prime-number subdivisions.",
            "samples": prime_pulse_chorale(),
        },
    }

    manifest = []
    for filename, info in tracks.items():
        track_path = output_dir / filename
        _write_wav(track_path, info["samples"])
        manifest.append(
            {
                "file": str(track_path),
                "title": info["title"],
                "description": info["description"],
                "sample_rate": SAMPLE_RATE,
                "duration_s": round(len(info["samples"]) / SAMPLE_RATE, 2),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(tracks)} tracks to {output_dir}")


if __name__ == "__main__":
    main()
