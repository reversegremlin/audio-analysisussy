# User Guide

Complete guide to using Chromascope for audio analysis and visualization.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Chromascope Studio (Web UI)](#kaleidoscope-studio-web-ui)
4. [Understanding the Output](#understanding-the-output)
5. [Using the Pipeline](#using-the-pipeline)
6. [Creating Visualizations](#creating-visualizations)
7. [Advanced Configuration](#advanced-configuration)
8. [Integrating with Other Tools](#integrating-with-other-tools)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg (for video rendering)
- 2GB+ RAM recommended for long audio files

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/chromascope.git
cd chromascope

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# 3. Install the package
pip install -e .

# 4. (Optional) Install development dependencies
pip install -e ".[dev]"

# 5. Verify installation
chromascope --help
```

### Verify FFmpeg

```bash
ffmpeg -version
# Should show FFmpeg version info
```

If FFmpeg is not installed:
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html

---

## Quick Start

### Analyze an Audio File

```python
from chromascope import AudioPipeline

# Create pipeline with default settings (60 FPS)
pipeline = AudioPipeline()

# Process audio and save manifest
result = pipeline.process(
    "my_song.mp3",
    output_path="my_song_manifest.json"
)

print(f"BPM: {result['bpm']:.1f}")
print(f"Duration: {result['duration']:.1f}s")
print(f"Frames: {result['n_frames']}")
```

### Command Line Usage

```bash
# Basic analysis
chromascope song.mp3

# Custom output path and FPS
chromascope song.mp3 -o output.json --fps 30

# Show summary after processing
chromascope song.mp3 --summary

# Export as NumPy for faster loading
chromascope song.mp3 --format numpy
```

### Render a Video

```bash
# Using the render script
python -m chromascope.render_video song.mp3 -o video.mp4

# With custom settings
python -m chromascope.render_video song.mp3 \
    --width 1920 --height 1080 \
    --fps 60 \
    --mirrors 12 \
    --trail 60
```

---

## Chromascope Studio (Web UI)

The easiest way to create visualizations is through Chromascope Studio, a web-based interface designed for musicians.

### Starting the Studio

```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Start the server
python frontend/server.py

# Open in your browser
# http://localhost:8080
```

### Interface Overview

The Studio is divided into three main areas:

**Left Panel - Audio, Style & Geometry**
- **Audio Source**: Drag & drop or click to load audio files
- **Style**: Choose visualization style (Geometric, Glass, Flower, Spiral)
- **Geometry**: Control mirrors (radial symmetry), size, orbit distance, and rotation speed
- **Dynamics**: Adjust punch intensity, motion trails, and envelope settings

**Center - Preview & Timeline**
- **Canvas**: Real-time visualization preview synced to audio
- **Waveform**: Visual representation of your audio with playhead
- **Transport**: Play/pause, skip, and volume controls

**Right Panel - Shape, Colors & Export**
- **Shape**: Control polygon complexity and stroke thickness
- **Colors**: Accent color, chroma-based coloring, saturation
- **Background**: Two-color gradients, particles, pulse rings, reactivity
- **Export**: Resolution, frame rate, and video export options

### Visualization Styles

The Studio includes six distinct visualization styles, each with its own character:

| Style | Description | Best For |
|-------|-------------|----------|
| **Geometric** | Orbiting polygons with radial symmetry | Electronic, EDM, clean beats |
| **Glass** | Classic broken glass kaleidoscope with triangular reflections | Rock, orchestral, complex audio |
| **Flower** | Organic petal shapes with smooth bezier curves | Ambient, acoustic, soft music |
| **Spiral** | Hypnotic spiraling arms with flowing motion | Trance, progressive, buildups |
| **Circuit** | Hexagonal grid with glowing circuit traces and nodes | Synthwave, cyberpunk, industrial |
| **Fibonacci** | Golden ratio spirals with mathematical harmony | Classical, jazz, contemplative |

Click any style button to switch instantly - changes apply in real-time during playback.

### Fullscreen Mode

Click the fullscreen button (bottom-right of preview) or double-click the canvas to enter immersive fullscreen mode. Controls fade out automatically - move your mouse to reveal them.

### Using Rotary Knobs

The knobs work like real studio hardware:
- **Click and drag up/down** to adjust values
- **Mouse wheel** for fine adjustments
- **Hover** to see the glow effect and current value

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| Click waveform | Seek to position |

### Export Workflow

1. Load your audio file
2. Adjust visualization parameters while previewing
3. Select resolution (720p, 1080p, or 4K)
4. Select frame rate (30 or 60 FPS)
5. Click "Export Video"

The export uses the same rendering engine as the command line, ensuring what you preview is what you get.

---

## Understanding the Output

### The Visual Driver Manifest

The manifest is a JSON file containing frame-by-frame audio features:

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

### Frame Data Fields

Each frame contains these fields:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `frame_index` | int | 0 to n_frames-1 | Sequential frame number |
| `time` | float | 0 to duration | Timestamp in seconds |
| `is_beat` | bool | true/false | True if this frame is on a beat |
| `is_onset` | bool | true/false | True if a note attack occurs |
| `percussive_impact` | float | 0.0 - 1.0 | Drum/transient energy |
| `harmonic_energy` | float | 0.0 - 1.0 | Melody/chord energy |
| `global_energy` | float | 0.0 - 1.0 | Overall volume |
| `low_energy` | float | 0.0 - 1.0 | Bass frequencies (0-200Hz) |
| `mid_energy` | float | 0.0 - 1.0 | Midrange (200Hz-4kHz) |
| `high_energy` | float | 0.0 - 1.0 | Treble (4kHz+) |
| `spectral_brightness` | float | 0.0 - 1.0 | Perceived brightness |
| `dominant_chroma` | string | "C" to "B" | Most prominent musical note |
| `chroma_values` | object | 12 floats | Intensity of each note |

### Example: Reading Frame Data

```python
import json

with open("manifest.json") as f:
    manifest = json.load(f)

# Find all beats
beats = [f for f in manifest["frames"] if f["is_beat"]]
print(f"Total beats: {len(beats)}")

# Get average energy
avg_energy = sum(f["global_energy"] for f in manifest["frames"]) / len(manifest["frames"])
print(f"Average energy: {avg_energy:.2f}")

# Find the loudest moment
loudest = max(manifest["frames"], key=lambda f: f["global_energy"])
print(f"Peak at {loudest['time']:.2f}s")
```

---

## Using the Pipeline

### Basic Pipeline

```python
from chromascope import AudioPipeline

pipeline = AudioPipeline(target_fps=60)
manifest = pipeline.process_to_manifest("song.mp3")
```

### Step-by-Step Processing

For more control, run each phase separately:

```python
from chromascope import (
    AudioDecomposer,
    FeatureAnalyzer,
    SignalPolisher,
    ManifestExporter,
)

# Phase A: Decompose
decomposer = AudioDecomposer()
decomposed = decomposer.decompose_file("song.mp3")
print(f"Duration: {decomposed.duration:.1f}s")

# Phase B: Analyze
analyzer = FeatureAnalyzer(target_fps=60)
features = analyzer.analyze(decomposed)
print(f"BPM: {features.temporal.bpm:.1f}")
print(f"Beats: {len(features.temporal.beat_frames)}")

# Phase C: Polish
polisher = SignalPolisher(fps=60)
polished = polisher.polish(features)

# Phase D: Export
exporter = ManifestExporter()
exporter.export_json(
    polished,
    bpm=features.temporal.bpm,
    duration=decomposed.duration,
    output_path="manifest.json",
)
```

### Accessing Raw Features

```python
# After analysis, access raw numpy arrays
print(f"RMS shape: {features.energy.rms.shape}")
print(f"Chroma shape: {features.tonality.chroma.shape}")

# Get beat times in seconds
beat_times = features.temporal.beat_times
print(f"First 5 beats: {beat_times[:5]}")
```

---

## Creating Visualizations

### Built-in Kaleidoscope

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
    trail_alpha=40,  # Motion blur (0-255)
)
```

### Custom Visualization

Use the manifest data to drive your own graphics:

```python
import json
import pygame

# Load manifest
with open("manifest.json") as f:
    manifest = json.load(f)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Animation loop
frame_idx = 0
running = True

while running and frame_idx < len(manifest["frames"]):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    frame = manifest["frames"][frame_idx]

    # Use frame data for visuals
    energy = frame["global_energy"]
    is_beat = frame["is_beat"]

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw circle that pulses with energy
    radius = int(50 + energy * 200)
    color = (255, 100, 100) if is_beat else (100, 100, 255)
    pygame.draw.circle(screen, color, (400, 300), radius)

    pygame.display.flip()
    clock.tick(manifest["metadata"]["fps"])
    frame_idx += 1

pygame.quit()
```

### Integration with Processing/p5.js

Export manifest and load in JavaScript:

```javascript
// p5.js example
let manifest;
let frameIdx = 0;

function preload() {
  manifest = loadJSON('manifest.json');
}

function draw() {
  if (frameIdx >= manifest.frames.length) return;

  const frame = manifest.frames[frameIdx];

  background(0);

  // Size based on percussive impact
  const size = 50 + frame.percussive_impact * 200;

  // Color based on dominant chroma
  const hue = chromaToHue(frame.dominant_chroma);
  fill(hue, 80, 90);

  ellipse(width/2, height/2, size, size);

  frameIdx++;
}
```

---

## Advanced Configuration

### Envelope Tuning

Control how features are smoothed:

```python
from chromascope import AudioPipeline
from chromascope.core.polisher import EnvelopeParams

# Punchy, responsive settings
punchy = AudioPipeline(
    target_fps=60,
    impact_envelope=EnvelopeParams(
        attack_ms=0.0,     # Instant response
        release_ms=100.0,  # Quick decay
    ),
    energy_envelope=EnvelopeParams(
        attack_ms=20.0,
        release_ms=150.0,
    ),
)

# Smooth, flowing settings
smooth = AudioPipeline(
    target_fps=60,
    impact_envelope=EnvelopeParams(
        attack_ms=50.0,    # Gradual rise
        release_ms=500.0,  # Long tail
    ),
    energy_envelope=EnvelopeParams(
        attack_ms=100.0,
        release_ms=800.0,
    ),
)
```

### HPSS Separation Strength

```python
# Aggressive separation (cleaner but may lose nuance)
pipeline = AudioPipeline(hpss_margin=(3.0, 3.0))

# Gentle separation (preserves detail)
pipeline = AudioPipeline(hpss_margin=(1.0, 1.0))
```

### Sample Rate

```python
# Higher quality analysis (slower)
pipeline = AudioPipeline(sample_rate=44100)

# Faster analysis (usually sufficient)
pipeline = AudioPipeline(sample_rate=22050)
```

---

## Integrating with Other Tools

### Blender

Export manifest and use in Blender Python:

```python
# In Blender's Python console
import bpy
import json

with open("/path/to/manifest.json") as f:
    manifest = json.load(f)

# Keyframe object scale to percussive impact
obj = bpy.context.active_object
fps = manifest["metadata"]["fps"]

for frame in manifest["frames"]:
    frame_num = frame["frame_index"] + 1
    scale = 1.0 + frame["percussive_impact"]

    obj.scale = (scale, scale, scale)
    obj.keyframe_insert(data_path="scale", frame=frame_num)
```

### TouchDesigner

Load JSON and map to CHOPs:

1. Use a **File In DAT** to load `manifest.json`
2. Parse with **JSON DAT**
3. Convert to **Table DAT**
4. Use **DAT to CHOP** for animation data

### Unity

```csharp
// C# example
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

// Load and use
var json = File.ReadAllText("manifest.json");
var manifest = JsonUtility.FromJson<Manifest>(json);
```

---

## Troubleshooting

### Common Issues

**"No module named 'chromascope'"**
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate
pip install -e .
```

**"FFmpeg not found"**
```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu
brew install ffmpeg      # macOS
```

**Out of memory during video render**
```python
# Use lower resolution
render_video(audio_path, output_path, width=1280, height=720)
```

**Video rendering is slow**
- Reduce resolution (1280x720 instead of 1920x1080)
- Lower FPS (30 instead of 60)
- Use faster FFmpeg preset (already set to "fast")

**Beat detection seems off**
- Try different HPSS margins
- Check if audio has clear rhythmic content
- Very ambient or arrhythmic music may not track well

### Performance Tips

1. **Use 22050 Hz sample rate** - Sufficient for most analysis
2. **Process in batches** - For multiple files, reuse the pipeline object
3. **Use NumPy export** - Faster to load than JSON for large manifests
4. **Render at target resolution** - Don't upscale later

### Getting Help

- Check existing issues on GitHub
- Include Python version, OS, and error traceback
- Provide sample audio if possible (or describe genre/tempo)

---

## Next Steps

- Explore the [API Reference](API_REFERENCE.md) for detailed documentation
- Check out [examples/](../examples/) for more use cases
- Read [ARCHITECTURE.md](../architecture.md) to understand the internals
