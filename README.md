# Audio Analysisussy

A high-fidelity audio analysis engine that extracts musical features for reactive generative art. Transform any audio into stunning, music-driven visualizations.

<p align="center">
  <img src="docs/preview.gif" alt="Kaleidoscope Visualization Demo" width="480">
  <br>
  <em>Real-time audio-reactive kaleidoscope visualization</em>
  <br>
  <a href="docs/teaser.mp4">▶️ Watch full demo with audio</a>
</p>

```
Audio File → Analysis Pipeline → Visual Driver Manifest → Kaleidoscope Video
```

## What It Does

This engine doesn't just react to volume—it *understands* the character of music by separating **percussive transients** (drums, attacks) from **harmonic textures** (melodies, chords). This separation is the difference between a generic "bars" visualizer and a professional music video.

### Feature Extraction

| Feature | Description | Visual Application |
|---------|-------------|-------------------|
| **HPSS Separation** | Splits audio into harmonic and percussive components | Independent visual layers |
| **Beat Tracking** | Global BPM and beat timestamps | Sync animations to rhythm |
| **Onset Detection** | Identifies note attacks and transients | Trigger visual events |
| **RMS Energy** | Volume envelope (global + per-component) | Drive scale/intensity |
| **Frequency Bands** | Low (bass), Mid (vocals), High (cymbals) | Multi-band reactivity |
| **Chroma Features** | 12-bin pitch intensity (C, C#, D...) | Color mapping |
| **Spectral Centroid** | "Brightness" of the sound | Shape complexity |

### The Kaleidoscope Visualizer

Built-in geometric visualizer that maps audio features to transformations:

- **Percussive Impact** → Shape pulsing (drums cause shapes to "kick")
- **Harmonic Energy** → Rotation speed (melodies drive smooth motion)
- **Spectral Brightness** → Polygon complexity (3 sides → 12 sides)
- **Dominant Pitch** → Hue colors (musical notes map to spectrum)

**Dynamic Background Effects:**
- Gradient oscillation between two colors based on harmonic energy
- Floating particles that react to beats and energy levels
- Expanding pulse rings on beat hits
- Smooth vignette for cinematic depth
- Fully configurable reactivity and color palette

### Kaleidoscope Studio (Web UI)

A premium web-based interface designed for musicians:

- **Drag & drop** audio files
- **Real-time preview** with live canvas rendering
- **Rotary knob controls** that feel like studio hardware
- **Waveform timeline** with visual playhead
- **One-click export** to HD video

![Kaleidoscope Studio Interface](docs/screenshots/studio-ui.png)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/audio-analysisussy.git
cd audio-analysisussy

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dependencies
pip install -e ".[dev]"
```

### Basic Usage

**Analyze audio and generate manifest:**

```python
from audio_analysisussy import AudioPipeline

pipeline = AudioPipeline(target_fps=60)
manifest = pipeline.process_to_manifest("your_song.mp3")

# Access frame data
for frame in manifest["frames"]:
    if frame["is_beat"]:
        print(f"Beat at {frame['time']:.2f}s - {frame['dominant_chroma']}")
```

**Render a kaleidoscope video:**

```python
from pathlib import Path
from audio_analysisussy.render_video import render_video

render_video(
    audio_path=Path("your_song.mp3"),
    output_path=Path("output.mp4"),
    width=1920,
    height=1080,
    fps=60,
)
```

### Command Line

```bash
# Analyze audio to JSON manifest
audio-analyze song.mp3 -o manifest.json --fps 60 --summary

# Render kaleidoscope video
python -m audio_analysisussy.render_video song.mp3 -o output.mp4
```

### Web Interface (Kaleidoscope Studio)

```bash
# Start the web server
python frontend/server.py

# Open http://localhost:8080 in your browser
```

## Output Format

The Visual Driver Manifest is a JSON file with frame-by-frame data:

```json
{
  "metadata": {
    "bpm": 120.5,
    "duration": 180.0,
    "fps": 60,
    "n_frames": 10800,
    "version": "1.0"
  },
  "frames": [
    {
      "frame_index": 0,
      "time": 0.0,
      "is_beat": true,
      "is_onset": true,
      "percussive_impact": 0.85,
      "harmonic_energy": 0.42,
      "global_energy": 0.65,
      "low_energy": 0.72,
      "mid_energy": 0.58,
      "high_energy": 0.31,
      "spectral_brightness": 0.44,
      "dominant_chroma": "G#",
      "chroma_values": {
        "C": 0.12, "C#": 0.08, "D": 0.15, ...
      }
    }
  ]
}
```

## Architecture

```
src/audio_analysisussy/
├── core/
│   ├── decomposer.py   # HPSS separation (harmonic/percussive)
│   ├── analyzer.py     # Feature extraction (beats, energy, chroma)
│   └── polisher.py     # Envelope smoothing & normalization
├── io/
│   └── exporter.py     # JSON/NumPy manifest export
├── visualizers/
│   └── kaleidoscope.py # Geometric visualization renderer
├── pipeline.py         # Main orchestration
├── render_video.py     # Video export with ffmpeg
└── cli.py              # Command-line interface

frontend/
├── index.html          # Main application
├── styles.css          # Studio-grade dark theme
├── app.js              # Visualization engine & controls
└── server.py           # Development server
```

### Pipeline Phases

1. **Decomposition** - HPSS separates harmonic (melody) from percussive (drums)
2. **Analysis** - Extract temporal, energy, and tonality features
3. **Polishing** - Apply attack/release envelopes for smooth visuals
4. **Export** - Generate FPS-aligned manifest for rendering

## Configuration

### Pipeline Options

```python
from audio_analysisussy import AudioPipeline
from audio_analysisussy.core.polisher import EnvelopeParams

pipeline = AudioPipeline(
    target_fps=60,              # Output frame rate
    sample_rate=22050,          # Analysis sample rate
    hpss_margin=(1.0, 1.0),     # HPSS separation strength
    impact_envelope=EnvelopeParams(
        attack_ms=0.0,          # Instant attack for punchy hits
        release_ms=200.0,       # 200ms decay for "glow" effect
    ),
    energy_envelope=EnvelopeParams(
        attack_ms=50.0,         # Smooth attack
        release_ms=300.0,       # Gradual release
    ),
)
```

### Kaleidoscope Options

```python
from audio_analysisussy.visualizers.kaleidoscope import KaleidoscopeConfig

config = KaleidoscopeConfig(
    width=1920,
    height=1080,
    fps=60,
    num_mirrors=8,        # Radial symmetry count
    trail_alpha=40,       # Motion blur persistence (0-255)
    base_radius=150.0,    # Base shape size
    max_scale=1.8,        # Maximum pulse scale
    min_sides=3,          # Triangle minimum
    max_sides=12,         # Dodecagon maximum
)
```

## Requirements

- Python 3.10+
- FFmpeg (for video export)
- ~500MB disk space per minute of HD video

### Dependencies

- `librosa` - Audio analysis
- `numpy` - Numerical operations
- `scipy` - Signal processing
- `pygame` - Frame rendering
- `Pillow` - Image handling
- `moviepy` - Video composition (optional)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/audio_analysisussy --cov-report=term-missing
```

### Test Coverage

The test suite includes 52 tests covering:
- HPSS decomposition accuracy
- Feature extraction validation
- Envelope smoothing behavior
- Manifest schema compliance
- Full pipeline integration

## Use Cases

- **Music Videos** - Generate reactive visuals for tracks
- **Live Performance** - Real-time audio-reactive graphics
- **Art Installations** - Ambient visualizations
- **Game Development** - Music-driven game mechanics
- **Research** - Music information retrieval

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*Built with the philosophy that high-quality visualization isn't about raw data—it's about **intent**.*
