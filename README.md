# Chromascope: The Geometry of Sound

Chromascope is a mathematical sandbox where music becomes visual form. It is built on the principle that harmony is simply audible geometry, and geometry is frozen music. By decomposing audio into its fundamental harmonic and percussive components, Chromascope extracts the "visual drivers" that animate complex simulations—from fluid dynamics to sacred geometry.

![Chromascope Teaser](docs/assets/preview.gif)

---

## 🎵 The Audio Engine: Decomposing the Waveform

The heart of Chromascope is its high-fidelity audio analysis library. Unlike simple FFT-based visualizers, Chromascope uses **Harmonic-Percussive Source Separation (HPSS)** to isolate the different roles of a soundscape.

- **Percussive Components:** Isolated drum hits and transients drive "impact" visuals, physics impulses, and sudden geometric shifts.
- **Harmonic Components:** Melodic and chordal structures drive "flow" visuals, color palettes, and rotation speeds.
- **Visual Drivers:** The engine extracts a rich feature set used to parameterize the renderers:
    - **Chroma (Pitch):** Maps musical notes directly to the color spectrum.
    - **Spectral Flux:** Detects onset strength for rhythmic synchronization.
    - **Frequency Bands:** Sub-bass, bass, mid-range, and brilliance drive different layers of depth and detail.
    - **Spectral Flatness & Centroid:** Drives the "noisiness" or "brightness" of the visual textures.

```python
# Example of extracting visual drivers in Python
from chromascope.core.decomposer import AudioDecomposer
from chromascope.core.analyzer import FeatureAnalyzer

decomposer = AudioDecomposer()
analyzer = FeatureAnalyzer(target_fps=60)

# Separate harmonic and percussive layers
decomposed = decomposer.decompose_file("audio.wav")
# Extract all visual drivers
features = analyzer.analyze(decomposed)
```

---

## 🌐 Real-Time Exploration: The Web Interface

The `frontend/` provides a high-performance JavaScript render engine for real-time interaction. It translates the extracted audio features into a variety of visual styles through a 2D Canvas pipeline.

| Style | Aesthetic | Driver Mapping |
| :--- | :--- | :--- |
| **Glass** | Prismatic refraction | Chroma → Hue; Brilliance → Reflection density |
| **Circuit** | Hexagonal grid matrix | Flux → Data pulse; Mid-range → Grid stability |
| **Fibonacci** | Sacred phyllotaxis | BPM → Growth rate; Harmonic → Spiral density |
| **Mycelial** | Organic fungal growth | Sub-bass → Spore drift; Percussion → Growth nodes |

### Running the Web Interface
1. Navigate to the `frontend/` directory.
2. Run the server: `python server.py`
3. Open `http://localhost:8000` to explore the styles in real-time.

![Circuit Style](docs/assets/demos/preview_circuit.gif) ![Glass Style](docs/assets/demos/preview_glass.gif)

---

## 🔬 High-Fidelity Rendering: The Experiment Framework

For cinematic output, the `src/chromascope/experiment/` framework offers a Python-based rendering pipeline. This framework leverages the same audio features but applies advanced post-processing and simulation models that exceed real-time capabilities.

- **Unified Visual Polisher:** A post-processing engine that applies audio-reactive **Glow**, **Chromatic Aberration**, and **Soft Tone Mapping**.
- **Specialized Simulations:**
    - `attractor.py`: Chaotic Lorenz and Aizawa systems driven by bass energy.
    - `decay.py`: Radioactive particle decay simulations where half-life is modulated by spectral flux.
    - `solar.py`: Fluid solar flares reacting to high-frequency brilliance.

### Video Generation CLI
```bash
# Generate a cinematic render using the Decay experiment
chromascope-decay my_track.wav --output video.mp4 --style dark_nebula
```

---

## 🎨 Visual Styles Gallery

| ![Fibonacci](docs/assets/demos/preview_fibonacci.png) | ![Flower](docs/assets/demos/preview_flower.png) | ![Geometric](docs/assets/demos/preview_geometric.png) |
| :---: | :---: | :---: |
| **Fibonacci** | **Flower** | **Geometric** |

| ![Spiral](docs/assets/demos/preview_spiral.png) | ![Orrery](docs/assets/demos/preview_orrery.png) | ![Quark](docs/assets/demos/preview_quark.png) |
| :---: | :---: | :---: |
| **Spiral** | **Orrery** | **Quark** |

---

## 🛠 Getting Started

### Installation
```bash
git clone https://github.com/nadiabellamorris/chromascope.git
cd chromascope
pip install -e .
```

### Quick Start
To generate a basic audio-reactive manifest and preview it:
```bash
chromascope path/to/audio.wav --preview
```

---

*Chromascope: Where the math of music meets the geometry of light.*
