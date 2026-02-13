# Chromascope User Guide

Practical guide to using Chromascope as an **audio engineer or musician** – from “drop in a track and get pretty motion” to deep integration in your own visuals.

If you’re comfortable with a DAW but not a programmer, focus on the **Studio** and **Command Line** sections. If you write code or build custom visuals, the **Pipeline**, **Manifest**, and **Integration** sections are for you.

## Table of Contents

1. [What Chromascope Does (Musician’s View)](#what-chromascope-does-musicians-view)
2. [Installing Chromascope](#installing-chromascope)
3. [Fastest Paths to Results](#fastest-paths-to-results)
   - [A. Chromascope Studio (Web UI)](#a-chromascope-studio-web-ui)
   - [B. Command Line (no coding)](#b-command-line-no-coding)
   - [C. Python API (for developers)](#c-python-api-for-developers)
4. [Understanding the Visual Driver Manifest](#understanding-the-visual-driver-manifest)
5. [Working with the Audio Pipeline](#working-with-the-audio-pipeline)
6. [Creating Visuals](#creating-visuals)
   - [Built‑in Kaleidoscope Video](#built-in-kaleidoscope-video)
   - [Driving Your Own Visuals](#driving-your-own-visuals)
7. [Shaping the Feel: Advanced Configuration](#shaping-the-feel-advanced-configuration)
8. [Integrating with Other Tools](#integrating-with-other-tools)
9. [Troubleshooting & Tips](#troubleshooting--tips)

---

## What Chromascope Does (Musician’s View)

Think of Chromascope as a **translator between music and motion**:

- You give it an **audio file** (MP3, WAV, FLAC…).
- It “listens like an audio engineer” – separating **drums vs. harmony**, tracking **beats**, measuring **energy and brightness**, and recognizing **which notes are active**.
- It outputs a **Visual Driver Manifest**: a timeline where every frame knows things like:
  - “Is this a beat?”
  - “How loud is the bass right now?”
  - “Which note is dominating – F#, G, A?”
  - “How bright is the sound?”
- That manifest can then drive:
  - The built‑in **kaleidoscope video renderer**, or
  - Your own visual system in **Blender, TouchDesigner, Unity, p5.js, custom engines**, etc.

Chromascope is built so visuals feel **musical**, not just “EQ bars that bounce.”

---

## Installing Chromascope

### Prerequisites

- **Python 3.10+**
- **FFmpeg** (required for video export)
- **2GB+ RAM** recommended for long or high‑resolution renders

### Step‑by‑Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/chromascope.git
cd chromascope

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate         # Linux/macOS
# or on Windows:
# .venv\Scripts\activate

# 3. Install Chromascope
pip install -e .

# 4. (Optional) Install development extras
pip install -e ".[dev]"

# 5. Make sure the CLI is visible
chromascope --help
```

### Check FFmpeg

```bash
ffmpeg -version
# You should see version information
```

If FFmpeg is missing:
- **Ubuntu / Debian**: `sudo apt install ffmpeg`
- **macOS (Homebrew)**: `brew install ffmpeg`
- **Windows**: download from `https://ffmpeg.org/download.html` and add it to your PATH.

---

## Fastest Paths to Results

### A. Chromascope Studio (Web UI)

**Best for:** producers, musicians, VJs, and visual artists who want **hands‑on knobs and instant feedback**.

#### Start the Studio

```bash
cd chromascope
source .venv/bin/activate
python frontend/server.py
```

Then open `http://localhost:8080` in your browser.

#### The Studio Layout

- **Left panel – Audio & Macro Controls**
  - **Audio Source**: drag & drop files or click to browse.
  - **Style**: pick the visual style (Geometric, Glass, Flower, Spiral, Circuit, Fibonacci, DMT, Sacred Geometry, Mycelial, Fluid, Orrery, Quark).
  - **Geometry & Dynamics**: mirrors (symmetry), base size, orbit distance, rotation speed, punchiness, and motion trails.

- **Center – Live Preview & Waveform**
  - **Canvas**: real‑time visualization locked to your audio.
  - **Waveform**: shows your track with a **playhead** so you can scrub.
  - **Transport**: play / pause / seek / volume.

- **Right panel – Shape, Color & Export**
  - **Shape**: polygon sides, line thickness, orbit patterns.
  - **Color**: main accent color, chroma‑driven hues, saturation.
  - **Background**: gradients, particles, pulse rings, and their reactivity.
  - **Export**: choose resolution (720p / 1080p / 4K) and FPS (30 / 60), then export.

#### Working the Knobs (for audio people)

Knobs behave like studio hardware:
- Click and **drag up/down** to change a value.
- Use the **mouse wheel** for fine adjustments.
- Hover to see **current value** and get a soft highlight.

#### Styles & When to Use Them

All styles share the same musical brain, but “move” differently:

- **Geometric** – clean, orbiting polygons. Great for **EDM, techno, crisp grooves**.
- **Glass** – faceted, gem‑like reflections. Fits **rock, cinematic, orchestral**.
- **Flower** – soft petal shapes, flowing motion. Ideal for **ambient, acoustic, chill**.
- **Spiral** – hypnotic spirals and builds. Works for **trance, progressive, buildups**.
- **Circuit** – hex grids and glowing traces. Perfect for **synthwave, cyberpunk**.
- **Fibonacci** – golden ratio spirals, sacred patterns. Nice for **jazz, classical**.
- **DMT** – intense, fractal, “hyperspace” vibes.
- **Sacred Geometry** – mandalas, symmetry locks, temple‑like visuals.
- **Mycelial** – fungal networks and bioluminescent threads.
- **Fluid** – ferrofluid / liquid metal blobs.
- **Orrery** – orbiting planets and brass rings.
- **Quark** – quantum‑field style with dancing particles.

#### Fullscreen & Performance

- Double‑click the preview or hit the fullscreen button for immersive mode.
- Controls fade out while playing; move the mouse to bring them back.
- For smoother performance while tweaking:
  - Start at **1080p, 30 fps**.
  - Once you like the look, bump it to **60 fps** for the final render.

#### Custom Resolution

The Studio supports custom resolutions beyond the built-in presets (720p / 1080p / 4K):

1. Click the **Custom** button in the Resolution selector.
2. Enter your desired **width** and **height** in pixels.
3. The preview and export will use your custom resolution.

Lower resolutions (e.g. 854x480) run much smoother in the browser. Set up your look at a comfortable resolution, then export at full quality via the CLI.

#### Export Workflow (Studio)

1. Load your audio file.
2. Pick a style that matches the vibe.
3. Tweak knobs while listening – treat it like mixing the light show.
4. Choose resolution and fps.
5. Export using one of these methods:
   - **Export Video** – the server renders a video in-browser.
   - **Copy CLI Command** – generates a ready-to-paste terminal command with all your current settings. Run it on a powerful machine for heavy renders.
   - **Export Config JSON** – downloads your settings as a JSON file for use with `--config`.

#### Copy CLI Command (Studio to Terminal)

When full HD or 4K rendering is too heavy for your browser, use the **Copy CLI Command** button:

1. Configure your visualization in the Studio (style, knobs, colors, resolution).
2. Click **Copy CLI Command** in the Export panel.
3. The generated command appears with all your settings as CLI flags.
4. Click the copy icon, then paste into your terminal.
5. Replace `your_audio.mp3` with your actual audio file path.

This lets you design in the browser and render on a beefy machine or overnight.

> **Tip:** You can also click **Export Config JSON** to download a config file, then use `--config config.json` on the command line. This is cleaner for configs with many custom values.

> **Note:** The Studio uses the rich, style-aware web visualizer. The Python CLI/video renderer currently provides a **geometric kaleidoscope** style; use Studio when you want the full set of 12 styles.

---

### B. Command Line (no coding)

**Best for:** audio people comfortable with a terminal who want reliable, repeatable renders or manifests.

#### Analyze a Track to a Manifest

```bash
# Basic analysis
chromascope song.mp3

# Custom output path and FPS
chromascope song.mp3 -o manifest.json --fps 30

# Show a human‑readable summary
chromascope song.mp3 --summary

# Export as NumPy for faster loading in code
chromascope song.mp3 --format numpy
```

Useful flags (from the CLI in `cli.py`):

- `-o / --output` – where to write the manifest.
- `-f / --fps` – frames per second of the timeline.
- `-s / --sample-rate` – analysis sample rate (usually leave at 22050).
- `--format json|numpy` – file format.
- `--attack` / `--release` – how quickly percussive hits rise/fall.
- `-q / --quiet` – less console noise.
- `--summary` – print manifest metadata and some sample frames.

#### Render a Kaleidoscope Video (Python script)

```bash
# Simple HD video
python -m chromascope.render_video song.mp3 -o video.mp4

# Lower resolution for faster renders
python -m chromascope.render_video song.mp3 \
    --width 1280 --height 720 \
    --fps 30 --quality fast

# With custom visual settings
python -m chromascope.render_video song.mp3 \
    --width 1920 --height 1080 \
    --fps 60 \
    --style geometric \
    --mirrors 12 \
    --trail 60 \
    --base-radius 200 \
    --max-scale 2.5

# With color and background customization
python -m chromascope.render_video song.mp3 \
    --accent-color "#ff6b6b" \
    --bg-color "#0a0014" \
    --saturation 90 \
    --no-particles

# Using a config file exported from the Studio
python -m chromascope.render_video song.mp3 \
    --config chromascope_geometric_config.json
```

Full list of render flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--width` | 1920 | Video width in pixels |
| `--height` | 1080 | Video height in pixels |
| `--fps` | 60 | Frames per second |
| `--style` | geometric | Visualization style |
| `--quality` | high | Encoding quality (high/medium/fast) |
| `--mirrors` | 8 | Radial mirror count |
| `--trail` | 40 | Trail persistence (0-100) |
| `--base-radius` | 150 | Base shape size |
| `--orbit-radius` | 200 | Orbit distance |
| `--rotation-speed` | 2.0 | Spin speed multiplier |
| `--max-scale` | 1.8 | Beat punch intensity |
| `--min-sides` | 3 | Minimum polygon sides |
| `--max-sides` | 12 | Maximum polygon sides |
| `--base-thickness` | 3 | Base line thickness |
| `--max-thickness` | 12 | Max line thickness on beats |
| `--bg-color` | #05050f | Background color 1 |
| `--bg-color2` | #1a0a2e | Background color 2 |
| `--accent-color` | #f59e0b | Accent color |
| `--saturation` | 85 | Color saturation (0-100) |
| `--bg-reactivity` | 70 | Background reactivity (0-100) |
| `--no-chroma-colors` | | Disable chroma-driven colors |
| `--no-dynamic-bg` | | Disable dynamic background |
| `--no-particles` | | Disable background particles |
| `--no-pulse` | | Disable beat pulse effect |
| `--config` | | JSON config file from Studio |

This uses the **geometric** kaleidoscope implemented in Python (`render_video.py` + `visualizers/kaleidoscope.py`), driven by the same manifest you get from the pipeline.

---

### C. Python API (for developers)

**Best for:** people building their own tools, renderers, or research pipelines.

#### One‑liner: audio → manifest file

```python
from chromascope import AudioPipeline

pipeline = AudioPipeline(target_fps=60)

result = pipeline.process(
    "my_song.mp3",
    output_path="my_song_manifest.json",  # or .npz when format="numpy"
    format="json",
)

print(f"BPM: {result['bpm']:.1f}")
print(f"Duration: {result['duration']:.1f}s")
print(f"Frames:  {result['n_frames']}")
```

#### In‑memory manifest (no file)

```python
from chromascope import AudioPipeline

pipeline = AudioPipeline(target_fps=60)
manifest = pipeline.process_to_manifest("my_song.mp3")
```

From there you can feed the manifest directly into your renderer.

---

## Understanding the Visual Driver Manifest

The **Visual Driver Manifest** is a JSON (or NumPy archive) that contains:

- Global **metadata** about the track.
- A list of **frames**, one per visual frame (e.g. 60 per second).

### Manifest Structure

```json
{
  "metadata": {
    "bpm": 124.5,
    "duration": 180.0,
    "fps": 60,
    "n_frames": 10800,
    "version": "1.0"
  },
  "frames": [...]
}
```

Each frame is a musical snapshot at a moment in time.

### Per‑frame Fields (musician‑friendly glossary)

| Field | What it means to you |
|-------|----------------------|
| `frame_index` | Which frame in the sequence this is (0 → n_frames‑1). |
| `time` | Where you are in the song, in seconds. |
| `is_beat` | `true` when this frame is on a beat – great for kicks, flashes, jumps. |
| `is_onset` | `true` on note attacks and transients – more sensitive than `is_beat`. |
| `percussive_impact` | How hard the **drums/transients** are hitting (0.0–1.0). |
| `harmonic_energy` | How strong the **melody/chords** are (0.0–1.0). |
| `global_energy` | Overall loudness / intensity (0.0–1.0). |
| `low_energy` | Bass (roughly 0–200 Hz). |
| `mid_energy` | Body and presence (roughly 200 Hz–4 kHz). |
| `high_energy` | Air and sizzle (roughly 4 kHz+). |
| `spectral_brightness` | Perceived brightness; higher = more high‑frequency content. |
| `dominant_chroma` | The most prominent **note** (e.g. `"G#"`). |
| `chroma_values` | How strong each of the 12 notes (C–B) is at this moment. |

### Reading a Manifest in Python

```python
import json

with open("manifest.json") as f:
    manifest = json.load(f)

frames = manifest["frames"]

# All beat frames
beats = [fr for fr in frames if fr["is_beat"]]
print(f"Total beats: {len(beats)}")

# Average energy over the song
avg_energy = sum(fr["global_energy"] for fr in frames) / len(frames)
print(f"Average energy: {avg_energy:.2f}")

# Loudest moment
loudest = max(frames, key=lambda fr: fr["global_energy"])
print(f"Peak loudness at {loudest['time']:.2f}s")
```

Once you’re comfortable reading these values, designing visuals becomes “mapping musical parameters to shapes/colors.”

---

## Working with the Audio Pipeline

Under the hood, Chromascope runs a 4‑phase pipeline (see `pipeline.py` for the orchestrator):

1. **Decompose** – separate harmonic vs. percussive content.
2. **Analyze** – extract timing, energy, and tonality features.
3. **Polish** – smooth and normalize signals (attack/release envelopes).
4. **Export** – build the manifest (JSON / NumPy).

You can either use the whole thing at once (via `AudioPipeline`) or drive each phase manually.

### Simple Pipeline Usage

```python
from chromascope import AudioPipeline

pipeline = AudioPipeline(target_fps=60)
manifest = pipeline.process_to_manifest("song.mp3")
```

### Phase‑by‑phase Control

```python
from chromascope.core.decomposer import AudioDecomposer
from chromascope.core.analyzer import FeatureAnalyzer
from chromascope.core.polisher import SignalPolisher
from chromascope.io.exporter import ManifestExporter

# Phase A: Decomposition (HPSS)
decomposer = AudioDecomposer()
decomposed = decomposer.decompose_file("song.mp3")
print(f"Duration: {decomposed.duration:.1f}s")

# Phase B: Feature extraction
analyzer = FeatureAnalyzer(target_fps=60)
features = analyzer.analyze(decomposed)
print(f"BPM:   {features.temporal.bpm:.1f}")
print(f"Beats: {len(features.temporal.beat_frames)}")

# Phase C: Polishing / envelopes
polisher = SignalPolisher(fps=60)
polished = polisher.polish(features)

# Phase D: Export to a manifest
exporter = ManifestExporter()
exporter.export_json(
    polished,
    bpm=features.temporal.bpm,
    duration=decomposed.duration,
    output_path="manifest.json",
)
```

### Accessing Raw Arrays

```python
print("RMS shape:", features.energy.rms.shape)
print("Chroma shape:", features.tonality.chroma.shape)

beat_times = features.temporal.beat_times
print("First 5 beats (s):", beat_times[:5])
```

This is useful for debugging or designing custom feature mappings.

---

## Creating Visuals

### Built‑in Kaleidoscope Video

The quickest programmatic way to get a video from audio is the built‑in geometric kaleidoscope:

```python
from pathlib import Path
from chromascope.render_video import render_video

render_video(
    audio_path=Path("song.mp3"),
    output_path=Path("output.mp4"),
    width=1920,      # HD width
    height=1080,     # HD height
    fps=60,          # Frame rate
    num_mirrors=8,   # Radial copies
    trail_alpha=40,  # Motion blur (0–255)
)
```

Internally, this uses `KaleidoscopeRenderer` from `visualizers/kaleidoscope.py` and drives it with the manifest built from your track.

---

### Driving Your Own Visuals

You can treat the manifest as “automation data” for any visual engine.

#### Example: Simple Pygame Visual

```python
import json
import pygame

with open("manifest.json") as f:
    manifest = json.load(f)

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

frames = manifest["frames"]
fps = manifest["metadata"]["fps"]

frame_idx = 0
running = True

while running and frame_idx < len(frames):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    frame = frames[frame_idx]

    energy = frame["global_energy"]
    is_beat = frame["is_beat"]

    screen.fill((0, 0, 0))

    radius = int(50 + energy * 200)
    color = (255, 100, 100) if is_beat else (100, 100, 255)
    pygame.draw.circle(screen, color, (400, 300), radius)

    pygame.display.flip()
    clock.tick(fps)
    frame_idx += 1

pygame.quit()
```

#### Example: p5.js / Processing‑style Sketch

```javascript
// In p5.js
let manifest;
let frameIdx = 0;

function preload() {
  manifest = loadJSON('manifest.json');
}

function setup() {
  createCanvas(800, 600);
  colorMode(HSB, 360, 100, 100);
}

function draw() {
  if (!manifest || frameIdx >= manifest.frames.length) return;

  const frame = manifest.frames[frameIdx];
  background(0);

  const size = 50 + frame.percussive_impact * 200;
  const hue = chromaToHue(frame.dominant_chroma); // your own mapping

  fill(hue, 80, 90);
  noStroke();
  ellipse(width / 2, height / 2, size, size);

  frameIdx++;
}
```

---

## Shaping the Feel: Advanced Configuration

Chromascope exposes a few “macro” controls that change how your visuals feel: **envelopes**, **HPSS strength**, and **sample rate**.

### Envelope Tuning (Attack / Release)

Envelopes answer: *“How quickly should visuals react, and how long should they hang around?”*

```python
from chromascope import AudioPipeline
from chromascope.core.polisher import EnvelopeParams

# Punchy, club‑style
punchy = AudioPipeline(
    target_fps=60,
    impact_envelope=EnvelopeParams(
        attack_ms=0.0,     # hits react instantly
        release_ms=100.0,  # short decay
    ),
    energy_envelope=EnvelopeParams(
        attack_ms=20.0,
        release_ms=150.0,
    ),
)

# Smooth, ambient
smooth = AudioPipeline(
    target_fps=60,
    impact_envelope=EnvelopeParams(
        attack_ms=50.0,    # gentle fade‑in
        release_ms=500.0,  # long tail
    ),
    energy_envelope=EnvelopeParams(
        attack_ms=100.0,
        release_ms=800.0,
    ),
)
```

### HPSS Separation Strength

Controls how aggressively Chromascope separates **drums** from **harmonies**:

```python
# Strong separation: very clear drums vs. music, but might lose some nuance
pipeline = AudioPipeline(hpss_margin=(3.0, 3.0))

# Gentle separation: more blended but retains detail (default)
pipeline = AudioPipeline(hpss_margin=(1.0, 1.0))
```

### Sample Rate vs. Speed

```python
# Higher fidelity (slower, more CPU)
pipeline = AudioPipeline(sample_rate=44100)

# Fast and usually sufficient (default)
pipeline = AudioPipeline(sample_rate=22050)
```

---

## Integrating with Other Tools

Here are some ways to wire Chromascope into common creative pipelines.

### Blender

Use the manifest to keyframe objects in Blender:

```python
# In Blender's Python console
import bpy
import json

with open("/path/to/manifest.json") as f:
    manifest = json.load(f)

obj = bpy.context.active_object
fps = manifest["metadata"]["fps"]

for frame in manifest["frames"]:
    frame_num = frame["frame_index"] + 1
    scale = 1.0 + frame["percussive_impact"]

    obj.scale = (scale, scale, scale)
    obj.keyframe_insert(data_path="scale", frame=frame_num)
```

### TouchDesigner

1. Use a **File In DAT** to load `manifest.json`.
2. Parse with a **JSON DAT**.
3. Convert to a **Table DAT**.
4. Feed that into **DAT to CHOP** to drive parameters with energy/beat data.

### Unity (C#)

```csharp
[System.Serializable]
public class Frame {
    public int frame_index;
    public float time;
    public bool is_beat;
    public float percussive_impact;
    public float harmonic_energy;
}

[System.Serializable]
public class Manifest {
    public Frame[] frames;
}

// Load and parse
var json = File.ReadAllText("manifest.json");
var manifest = JsonUtility.FromJson<Manifest>(json);
```

From there, you can drive camera shakes, light intensity, shader params, etc.

---

## Troubleshooting & Tips

### Common Issues

**“No module named 'chromascope'”**

```bash
source .venv/bin/activate
pip install -e .
```

**“FFmpeg not found”**

```bash
sudo apt install ffmpeg   # Ubuntu / Debian
brew install ffmpeg       # macOS
```

**Out of memory during video render**

```python
from chromascope.render_video import render_video

render_video(
    audio_path="song.mp3",
    output_path="video.mp4",
    width=1280,
    height=720,
)
```

**Video rendering is slow**

- Use **1280×720** instead of 1920×1080.
- Render at **30 fps** instead of 60.
- Close other heavy apps while rendering.

**Beat detection feels “off”**

- Very ambient or free‑time music may not produce stable beats.
- Try different HPSS margins (stronger separation can help for dense mixes).
- Check that the audio has a clear kick/snare pattern for the detector to lock onto.

### Performance Tips

1. **Use 22050 Hz** for quick iteration; switch to 44100 Hz only if needed.
2. **Reuse the same `AudioPipeline`** object when batch‑processing multiple tracks.
3. Prefer **NumPy export (`--format numpy`)** for large manifests.
4. **Render at your target resolution** – avoid heavy upscaling later.

### Getting Help & Going Deeper

- See the detailed **API docs** in `API_REFERENCE.md`.
- For internals and data structures, read `architecture.md`.
- When opening an issue, include:
  - OS + Python version
  - How you installed Chromascope
  - Your command or code snippet
  - Any error messages (and, if possible, a short example audio description or file).

