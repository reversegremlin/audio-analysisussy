# Fractal Kaleidoscope

Audio-reactive fractal video renderer. Feed it a song, get back a
hypnotic kaleidoscopic fractal video with the audio baked in.

```
chromascope-fractal song.mp3
```

That's it. One command. Out comes `song_fractal.mp4` — a full-length,
1080p 60fps video where Julia sets morph to the harmony, the pattern
breathes with the bass, and beats punch the infinite zoom deeper.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Command Reference](#command-reference)
- [Resolution Presets](#resolution-presets)
- [Fractal Modes](#fractal-modes)
- [Quality Profiles](#quality-profiles)
- [Audio-to-Visual Mapping](#audio-to-visual-mapping)
- [Post-Processing Effects](#post-processing-effects)
- [Performance & Render Times](#performance--render-times)
- [Recipes](#recipes)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

---

## Installation

Requires Python 3.10+ and ffmpeg.

```bash
# Install ffmpeg (if not already present)
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt install ffmpeg
# Windows: choco install ffmpeg

# Install chromascope with the fractal renderer
pip install -e ".[experiment]"

# Verify
chromascope-fractal --help
```

The `experiment` extra installs [Pillow](https://python-pillow.org/),
the only additional dependency beyond the core chromascope audio engine.

---

## Quick Start

### Render a 10-second preview

```bash
chromascope-fractal track.mp3 --max-duration 10 -q fast
```

Low-res, fast encode. Good for checking how the visuals feel
before committing to a full render.

### Render a full HD video

```bash
chromascope-fractal track.mp3 -o output.mp4
```

1920x1080, 60fps, H.264 high quality. The audio is muxed into
the MP4 — ready to upload or play.

### Render 4K

```bash
chromascope-fractal track.mp3 --width 3840 --height 2160
```

### Render a specific section

```bash
chromascope-fractal track.mp3 --max-duration 30
```

Renders only the first 30 seconds. Useful for previewing the
drop, chorus, or any specific section (start from the beginning
of the file — trim your audio beforehand if you need a different
starting point).

---

## How It Works

```
                    audio.mp3
                        |
                        v
    ┌───────────────────────────────────┐
    │         Audio Analysis            │
    │                                   │
    │  librosa HPSS decomposition       │
    │  Beat tracking, onset detection   │
    │  Chroma, spectral brightness      │
    │  Energy bands (low/mid/high)      │
    │  Signal smoothing & normalization │
    │                                   │
    │  Output: per-frame manifest       │
    │  (every audio feature at 60fps)   │
    └───────────────┬───────────────────┘
                    |
                    v
    ┌───────────────────────────────────┐
    │        Frame Renderer             │
    │                                   │
    │  1. Generate fractal texture      │
    │     (Julia set / Mandelbrot)      │
    │                                   │
    │  2. Radial warp (breathing)       │
    │                                   │
    │  3. Kaleidoscope mirror           │
    │     (N-way polar symmetry)        │
    │                                   │
    │  4. Color grade                   │
    │     (jewel-tone HSV palette)      │
    │                                   │
    │  5. Infinite zoom blend           │
    │     (feedback buffer compositing) │
    │                                   │
    │  6. Post-process                  │
    │     (glow, aberration, vignette)  │
    └───────────────┬───────────────────┘
                    |
                    v
    ┌───────────────────────────────────┐
    │        FFmpeg Encoder             │
    │                                   │
    │  Raw RGB frames piped to ffmpeg   │
    │  via stdin (no temp files)        │
    │  Audio muxed in the same pass     │
    │  H.264, AAC 192kbps              │
    └───────────────┬───────────────────┘
                    |
                    v
                output.mp4
```

The audio is analyzed once, producing a manifest with per-frame
values for every musical feature. The renderer reads this manifest
frame by frame, generating fractals whose parameters are driven
entirely by the music. Frames stream directly to ffmpeg — nothing
touches disk until the final MP4 is written.

---

## Command Reference

```
chromascope-fractal <audio> [options]
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `audio` | Input audio file. Accepts `.mp3`, `.wav`, `.flac`, `.ogg`, and anything ffmpeg/librosa can decode. |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `<audio>_fractal.mp4` | Output MP4 file path. Parent directories are created automatically. |

### Resolution & Frame Rate

| Flag | Default | Description |
|------|---------|-------------|
| `-p`, `--profile` | `medium` | Target profile: `low` (720p 30fps), `medium` (1080p 60fps), or `high` (4k 60fps). |
| `--width` | *(profile)* | Video width in pixels (overrides profile). |
| `--height` | *(profile)* | Video height in pixels (overrides profile). |
| `-f`, `--fps` | *(profile)* | Frames per second (overrides profile). |

### Visual Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-s`, `--segments` | `8` | Number of kaleidoscope symmetry segments. More segments = more intricate mandala patterns. Try 6, 8, 12, or 16. |
| `--fractal` | `blend` | Fractal algorithm. `julia`, `mandelbrot`, or `blend`. See [Fractal Modes](#fractal-modes). |
| `--zoom-speed` | `1.0` | Multiplier for the continuous inward zoom. Higher = faster descent into the fractal. `0.5` for slow meditation, `2.0` for an intense dive. |
| `--rotation-speed` | `1.0` | Multiplier for the kaleidoscope rotation. `0.0` freezes rotation. `2.0` doubles it. |

### Post-Processing

| Flag | Default | Description |
|------|---------|-------------|
| `--no-glow` | *(enabled)* | Disable the bloom/glow effect. Glow adds a soft luminous halo around bright fractal edges. |
| `--no-aberration` | *(enabled)* | Disable chromatic aberration. Aberration adds a subtle RGB fringe at high-contrast edges, intensifying on beats. |
| `--no-vignette` | *(enabled)* | Disable edge vignetting. Vignette darkens the corners to focus attention on the center. |

### Duration & Encoding

| Flag | Default | Description |
|------|---------|-------------|
| `--max-duration` | *(full track)* | Limit output to the first N seconds of the audio. |
| `-q`, `--quality` | *(profile)* | Encoding quality preset (`high`, `medium`, `fast`). Defaults to profile's recommended quality. |

---

## Resolution Profiles

Profiles provide a one-flag setup for common targets. They set width, height, FPS, and encoding quality in one go.

| Profile | Resolution | FPS | Quality | Use Case |
|---------|------------|-----|---------|----------|
| `low` | 1280x720 | 30 | `fast` | Quick previews, mobile-friendly. |
| `medium` | 1920x1080 | 60 | `medium` | Standard YouTube HD (default). |
| `high` | 3840x2160 | 60 | `high` | YouTube 4K Ultra HD. |

You can override any profile setting by providing the specific flag afterward:
```bash
# 1080p but with high encoding quality
chromascope-fractal song.mp3 --profile medium --quality high
```

---

## Quality Profiles

Quality profiles control the FFmpeg encoding parameters (CRF, bitrate caps, and presets). All profiles use `yuv420p` for maximum compatibility with YouTube and hardware decoders.

| Quality | Preset | CRF | Max Bitrate | Description |
|---------|--------|-----|-------------|-------------|
| `high` | slow | 22 | 50 Mbps | High fidelity, YouTube 4K compatible. |
| `medium` | medium | 24 | 15 Mbps | Balanced, YouTube 1080p compatible. |
| `fast` | ultrafast | 28 | 8 Mbps | Fast preview, YouTube 720p compatible. |

The render time is dominated by fractal computation, not encoding.
The quality flag mostly affects file size and encoding speed.

```bash
# Preview render (fastest possible)
chromascope-fractal song.mp3 -p low --max-duration 15

# Production render (4K)
chromascope-fractal song.mp3 -p high
```

---

## Audio-to-Visual Mapping

Every visual parameter is driven by a specific audio feature.
Nothing is random — the visuals are a direct translation of the
music's structure.

| Audio Feature | What It Drives |
|---------------|----------------|
| **Percussive impact** | Zoom punch on beats. Glow intensity spike. Fractal mode switching (in blend mode). Chromatic aberration strength. |
| **Harmonic energy** | Rotation speed. Julia `c` parameter drift rate (shape morphing). Color saturation. |
| **Spectral brightness** | Fractal detail level (max iterations). Higher brightness = more intricate patterns. |
| **Global energy** | Base zoom speed. Noise texture blend intensity. Overall visual intensity. |
| **Low-frequency energy** | Radial warp amplitude — the "breathing" effect. The kaleidoscope literally inhales and exhales with the bass. |
| **High-frequency energy** | Segment count modulation. High hats and cymbals subtly shift the kaleidoscope symmetry order. |
| **Dominant chroma** | Base hue of the color palette. The key of the music determines the dominant color family. C = red, E = green, G# = blue, etc. |
| **Beat events** | Instantaneous zoom punch (8% deeper per beat). Hue shift. Glow flash. |
| **Onset events** | Minor hue rotation. Fractal type crossfade (blend mode). |

The audio analysis uses Harmonic-Percussive Source Separation (HPSS)
to cleanly separate drums from melody. This means a loud bass note
won't falsely trigger percussive effects, and a quiet hi-hat won't
be drowned out by a chord — each element drives its own visual layer.

---

## Post-Processing Effects

### Glow (Bloom)

A gaussian-blurred copy of each frame is screen-blended back onto
itself, creating a soft luminous halo around bright fractal edges.
The intensity pulses with percussive impact — beats make the
pattern flash brighter.

Disable with `--no-glow` for a sharper, more clinical look.

### Chromatic Aberration

The red and blue channels are shifted in opposite directions by a
few pixels, creating a prismatic fringe at high-contrast edges.
The offset increases during percussive hits, adding a subtle
"shockwave" feel to the beats.

Disable with `--no-aberration` for clean, un-fringed edges.

### Vignette

A radial darkening gradient applied to the edges and corners of
each frame. Focuses attention on the center of the kaleidoscope
and adds cinematic depth.

Disable with `--no-vignette` for an edge-to-edge uniform image.

### Infinite Zoom (Feedback Buffer)

Not a post-effect you can toggle — it's core to the visual identity.
Each frame is composited over a zoomed-in version of the previous
frame. New fractal detail appears at the edges while the center
continuously falls inward, creating the sensation of falling into
an infinite mandala. The zoom rate is modulated by global energy.

### Radial Warp (Breathing)

A sinusoidal displacement field warps the fractal texture radially
before the kaleidoscope mirror is applied. Driven by low-frequency
energy — the pattern literally breathes with the bass.

---

## Performance & Render Times

All fractal math is vectorized with numpy. No per-pixel Python
loops. Frames stream directly to ffmpeg via stdin pipe — no
intermediate PNGs or temp files on disk.

### Approximate render speeds

| Resolution | FPS | Speed (frames/sec) | 60s clip render time |
|------------|-----|--------------------|-----------------------|
| 640x360 | 30 | ~4–6 fps | ~5 min |
| 1280x720 | 60 | ~2–3 fps | ~20 min |
| 1920x1080 | 60 | ~1–2 fps | ~30–60 min |
| 3840x2160 | 60 | ~0.3–0.5 fps | ~2–4 hours |

*Measured on a standard multi-core CPU. Actual times depend on
fractal complexity (max iterations), post-processing settings,
and system load.*

### Tips for faster renders

- **Drop the frame rate**: `-f 30` halves the frame count. 30fps
  is smooth enough for most music videos.
- **Use fast quality**: `-q fast` makes encoding near-instant
  (the fractal computation is the bottleneck, not encoding).
- **Reduce resolution**: Preview at 640x360, then render final at
  1080p. Fractal computation scales linearly with pixel count.
- **Limit duration**: `--max-duration 30` to render just the
  interesting section.
- **Disable post-processing**: `--no-glow --no-aberration --no-vignette`
  skips three full-frame image operations per frame.

### Memory usage

| Resolution | Working memory |
|------------|---------------|
| 1080p | ~50 MB |
| 4K | ~100 MB |

The renderer uses a generator — only one frame exists in memory
at a time (plus the feedback buffer for infinite zoom).

---

## Recipes

### Slow meditative mandala

```bash
chromascope-fractal ambient-track.mp3 \
  --fractal julia \
  --segments 12 \
  --zoom-speed 0.3 \
  --rotation-speed 0.5 \
  --no-aberration
```

### Aggressive electronic banger

```bash
chromascope-fractal banger.mp3 \
  --fractal blend \
  --segments 6 \
  --zoom-speed 2.0 \
  --rotation-speed 1.5
```

### Clean geometric Mandelbrot

```bash
chromascope-fractal minimal-techno.mp3 \
  --fractal mandelbrot \
  --segments 8 \
  --no-glow \
  --no-aberration \
  --no-vignette
```

### Instagram square reel

```bash
chromascope-fractal track.mp3 \
  --width 1080 --height 1080 \
  --max-duration 30 \
  -q medium
```

### 4K cinematic export

```bash
chromascope-fractal orchestral-piece.mp3 \
  --width 3840 --height 2160 \
  --fractal julia \
  --segments 8 \
  --zoom-speed 0.8 \
  -q high
```

### Fast preview loop (iterate on settings)

```bash
chromascope-fractal track.mp3 \
  --max-duration 10 \
  --width 480 --height 270 \
  -f 24 -q fast \
  -o preview.mp4
```

---

## Architecture

```
src/chromascope/experiment/
├── __init__.py
├── __main__.py        # python -m chromascope.experiment
├── cli.py             # Argument parsing, pipeline orchestration
├── fractal.py         # Julia set, Mandelbrot, noise fractal generators
├── kaleidoscope.py    # Polar mirror, radial warp, infinite zoom blend
├── colorgrade.py      # HSV palette, glow bloom, chromatic aberration, vignette
├── renderer.py        # Frame orchestrator (manifest → RGB arrays)
└── encoder.py         # FFmpeg subprocess pipe (frames + audio → MP4)
```

### Module responsibilities

**`fractal.py`** — Pure numpy fractal generators. Julia sets and
Mandelbrot zooms use vectorized complex-plane arithmetic with
smooth escape-time coloring. Noise fractals use multi-octave
layered sinusoids. All output float32 arrays in [0, 1]. A curated
path of 9 interesting Julia `c` values is provided for smooth
animated morphing.

**`kaleidoscope.py`** — The symmetry engine. Converts Cartesian
coordinates to polar, folds theta into a single segment, reflects
it, and samples the source texture — producing N-way radial
symmetry from a single fractal wedge. Also handles radial warp
(sinusoidal displacement for breathing) and infinite zoom
(crop-resize-blend feedback loop using Pillow).

**`colorgrade.py`** — Maps monochrome escape-time values to RGB
via a jewel-tone HSV palette. Hue is driven by the audio's
dominant chroma (musical key → color), with slow cycling over time.
Post-processing includes gaussian blur screen-blend (glow),
channel-shifted chromatic aberration, and radial vignette.

**`renderer.py`** — The orchestrator. Holds all animation state
(rotation, zoom level, Julia `c` parameter, feedback buffer,
smoothed audio values). Reads one manifest frame at a time,
generates a fractal, warps it, mirrors it, colors it, blends it
with the feedback buffer, and applies post-processing. Yields
frames as a generator for memory efficiency.

**`encoder.py`** — Thin wrapper around an ffmpeg subprocess. Pipes
raw RGB24 bytes to ffmpeg's stdin, muxes with the original audio
file. No temp files, no intermediate PNGs. Three quality presets
control the H.264 encoding parameters.

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | All fractal math, array operations, vectorized transforms |
| `Pillow` | Gaussian blur (glow), image resize (zoom blend) |
| `librosa` | Audio analysis (HPSS, beats, chroma, spectral features) |
| `scipy` | Signal smoothing in the audio pipeline |
| `ffmpeg` | Video encoding (system binary, not a Python package) |

No GPU, no OpenGL, no pygame, no Node.js.

---

## Troubleshooting

### `ffmpeg: command not found`

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

### `ModuleNotFoundError: No module named 'PIL'`

Install with the experiment extra:
```bash
pip install -e ".[experiment]"
```

Or install Pillow directly:
```bash
pip install Pillow
```

### Render is very slow

Fractal computation is CPU-bound and scales with pixel count.
For faster iteration:
```bash
chromascope-fractal song.mp3 \
  --max-duration 10 \
  --width 640 --height 360 \
  -f 30 -q fast
```

### Output video has no audio

Make sure your input file is a valid audio file that ffmpeg can
decode. The audio is muxed directly — if the input is corrupt,
ffmpeg may silently skip it. Test with:
```bash
ffprobe your-audio.mp3
```

### Colors look washed out

All quality profiles now use `yuv420p` by default, which is the standard for web and consumer video. This ensures consistent color across YouTube and all media players.

### Video stutters on playback

If you encounter stuttering, it's likely due to the extreme complexity of fractal motion at 4K.
- **Try the `medium` profile**: 1080p is much easier for hardware to decode.
- **Bitrate Caps**: The renderer now applies bitrate caps (e.g., 50Mbps for 4K) to stay within YouTube's recommended ranges and prevent massive file spikes that choke players.
- **Player**: Use a hardware-accelerated player like VLC or mpv.
- **Framerate**: Consider `-f 30` if 60fps is too heavy for your device.
