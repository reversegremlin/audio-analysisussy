# API Reference

Complete API documentation for Chromascope.

## Table of Contents

- [AudioPipeline](#audiopipeline)
- [AudioDecomposer](#audiodecomposer)
- [FeatureAnalyzer](#featureanalyzer)
- [SignalPolisher](#signalpolisher)
- [ManifestExporter](#manifestexporter)
- [KaleidoscopeRenderer](#kaleidoscoperenderer)
- [Data Classes](#data-classes)

---

## AudioPipeline

Main orchestration class that combines all processing stages.

```python
from chromascope import AudioPipeline
```

### Constructor

```python
AudioPipeline(
    target_fps: int = 60,
    sample_rate: int = 22050,
    hpss_margin: tuple[float, float] = (1.0, 1.0),
    impact_envelope: EnvelopeParams | None = None,
    energy_envelope: EnvelopeParams | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_fps` | int | 60 | Output frames per second |
| `sample_rate` | int | 22050 | Audio analysis sample rate |
| `hpss_margin` | tuple | (1.0, 1.0) | HPSS separation strength |
| `impact_envelope` | EnvelopeParams | None | Envelope for percussive signals |
| `energy_envelope` | EnvelopeParams | None | Envelope for energy signals |

### Methods

#### `process(audio_path, output_path=None, format="json")`

Run complete pipeline from audio file to manifest.

```python
result = pipeline.process(
    "song.mp3",
    output_path="manifest.json",
    format="json"  # or "numpy"
)
```

**Returns:** `dict` with keys:
- `manifest`: Complete manifest dictionary
- `bpm`: Detected tempo
- `duration`: Audio duration in seconds
- `n_frames`: Total frame count
- `fps`: Target FPS
- `output_path`: Path to written file (if output_path provided)

#### `process_to_manifest(audio_path)`

Process audio and return manifest dictionary directly.

```python
manifest = pipeline.process_to_manifest("song.mp3")
```

**Returns:** `dict` - The manifest dictionary

#### `decompose(audio_path)`

Phase A: Load and decompose audio.

```python
decomposed = pipeline.decompose("song.mp3")
```

**Returns:** `DecomposedAudio`

#### `analyze(decomposed)`

Phase B: Extract features from decomposed audio.

```python
features = pipeline.analyze(decomposed)
```

**Returns:** `ExtractedFeatures`

#### `polish(features)`

Phase C: Apply smoothing and normalization.

```python
polished = pipeline.polish(features)
```

**Returns:** `PolishedFeatures`

#### `export(polished, bpm, duration, output_path, format="json")`

Phase D: Export to file.

```python
path = pipeline.export(polished, bpm, duration, "output.json")
```

**Returns:** `Path` to written file

---

## AudioDecomposer

Performs Harmonic-Percussive Source Separation (HPSS).

```python
from chromascope.core.decomposer import AudioDecomposer
```

### Constructor

```python
AudioDecomposer(margin: tuple[float, float] = (1.0, 1.0))
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `margin` | tuple | (1.0, 1.0) | HPSS margins (harmonic, percussive) |

### Methods

#### `load_audio(audio_path, sr=None, mono=True)`

Load audio from file.

```python
y, sr = decomposer.load_audio("song.mp3", sr=22050)
```

**Returns:** `tuple[np.ndarray, int]` - Audio signal and sample rate

#### `separate(y, sr)`

Perform HPSS on audio signal.

```python
decomposed = decomposer.separate(y, sr)
```

**Returns:** `DecomposedAudio`

#### `decompose_file(audio_path, sr=22050)`

Load and decompose in one step.

```python
decomposed = decomposer.decompose_file("song.mp3")
```

**Returns:** `DecomposedAudio`

---

## FeatureAnalyzer

Extracts visual driver features from decomposed audio.

```python
from chromascope.core.analyzer import FeatureAnalyzer
```

### Constructor

```python
FeatureAnalyzer(target_fps: int = 60, n_fft: int = 2048)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_fps` | int | 60 | Target frames per second |
| `n_fft` | int | 2048 | FFT window size |

### Methods

#### `analyze(decomposed)`

Perform complete feature extraction.

```python
features = analyzer.analyze(decomposed)
```

**Returns:** `ExtractedFeatures`

#### `compute_hop_length(sr)`

Calculate hop length for target FPS.

```python
hop = analyzer.compute_hop_length(22050)  # Returns 367
```

**Returns:** `int`

#### `extract_temporal(decomposed, hop_length)`

Extract beat and onset features.

**Returns:** `TemporalFeatures`

#### `extract_energy(decomposed, hop_length)`

Extract RMS, spectral flux, and frequency band energy.

**Returns:** `EnergyFeatures`

#### `extract_tonality(decomposed, hop_length)`

Extract chroma, spectral centroid, flatness, and rolloff.

**Returns:** `TonalityFeatures`

### Class Methods

#### `chroma_index_to_name(index)`

Convert chroma index (0-11) to note name.

```python
FeatureAnalyzer.chroma_index_to_name(9)  # Returns "A"
```

---

## SignalPolisher

Applies aesthetic smoothing to raw features.

```python
from chromascope.core.polisher import SignalPolisher, EnvelopeParams
```

### Constructor

```python
SignalPolisher(
    fps: int = 60,
    impact_envelope: EnvelopeParams | None = None,
    energy_envelope: EnvelopeParams | None = None,
    adaptive_envelopes: bool = False,
)
```

### Methods

#### `polish(features)`

Apply full aesthetic processing.

```python
polished = polisher.polish(features)
```

**Returns:** `PolishedFeatures`

#### `normalize(signal, floor=0.001)`

Normalize signal to [0.0, 1.0] range.

```python
normalized = polisher.normalize(raw_signal)
```

**Returns:** `np.ndarray`

#### `apply_envelope(signal, params)`

Apply attack/release envelope.

```python
smoothed = polisher.apply_envelope(signal, EnvelopeParams(0, 200))
```

**Returns:** `np.ndarray`

---

## ManifestExporter

Exports polished features to various formats.

```python
from chromascope.io.exporter import ManifestExporter
```

### Constructor

```python
ManifestExporter(precision: int = 4)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `precision` | int | 4 | Decimal places for floats |

### Methods

#### `build_manifest(polished, bpm, duration)`

Build complete manifest dictionary.

```python
manifest = exporter.build_manifest(polished, 120.0, 180.0)
```

**Returns:** `dict`

#### `export_json(polished, bpm, duration, output_path, indent=2)`

Export to JSON file.

```python
path = exporter.export_json(polished, 120.0, 180.0, "output.json")
```

**Returns:** `Path`

#### `export_numpy(polished, output_path)`

Export to compressed NumPy archive.

```python
path = exporter.export_numpy(polished, "output.npz")
```

**Returns:** `Path`

#### `to_dict(polished, bpm, duration)`

Return manifest as dictionary.

**Returns:** `dict`

---

## KaleidoscopeRenderer

Renders kaleidoscopic visuals from manifest data.

```python
from chromascope.visualizers.kaleidoscope import (
    KaleidoscopeRenderer,
    KaleidoscopeConfig,
)
```

### KaleidoscopeConfig

```python
@dataclass
class KaleidoscopeConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 60
    num_mirrors: int = 8
    trail_alpha: int = 40
    base_radius: float = 150.0
    max_scale: float = 1.8
    base_thickness: int = 3
    max_thickness: int = 12
    orbit_radius: float = 200.0
    rotation_speed: float = 2.0
    min_sides: int = 3
    max_sides: int = 12
    background_color: tuple[int, int, int] = (5, 5, 15)
```

### Methods

#### `render_frame(frame_data, previous_surface=None)`

Render a single frame.

```python
surface = renderer.render_frame(frame_data, previous_surface)
```

**Returns:** `pygame.Surface`

#### `render_manifest(manifest, progress_callback=None)`

Render all frames from manifest.

```python
surfaces = renderer.render_manifest(manifest)
```

**Returns:** `list[pygame.Surface]`

#### `surface_to_array(surface)`

Convert pygame surface to numpy array.

```python
arr = renderer.surface_to_array(surface)  # Shape: (height, width, 3)
```

**Returns:** `np.ndarray`

---

## Data Classes

### DecomposedAudio

```python
@dataclass
class DecomposedAudio:
    harmonic: np.ndarray      # Harmonic component
    percussive: np.ndarray    # Percussive component
    original: np.ndarray      # Original signal
    sample_rate: int          # Sample rate
    duration: float           # Duration in seconds

    @property
    def n_samples(self) -> int  # Total samples
```

### TemporalFeatures

```python
@dataclass
class TemporalFeatures:
    bpm: float                    # Detected tempo
    beat_frames: np.ndarray       # Frame indices of beats
    beat_times: np.ndarray        # Timestamps of beats
    onset_frames: np.ndarray      # Frame indices of onsets
    onset_times: np.ndarray       # Timestamps of onsets
```

### EnergyFeatures

```python
@dataclass
class EnergyFeatures:
    rms: np.ndarray               # Global RMS energy
    rms_harmonic: np.ndarray      # Harmonic RMS
    rms_percussive: np.ndarray    # Percussive RMS
    spectral_flux: np.ndarray     # Onset strength/change
    frequency_bands: FrequencyBands
```

### FrequencyBands

```python
@dataclass
class FrequencyBands:
    sub_bass: np.ndarray    # 20-60Hz
    bass: np.ndarray        # 60-250Hz
    low_mid: np.ndarray     # 250-500Hz
    mid: np.ndarray         # 500-2000Hz
    high_mid: np.ndarray    # 2000-4000Hz
    presence: np.ndarray    # 4000-6000Hz
    brilliance: np.ndarray  # 6000-20000Hz
    
    # Aggregates
    low: np.ndarray         # 0-200Hz
    mid_aggregate: np.ndarray # 200Hz-4kHz
    high: np.ndarray        # 4kHz+
```

### TonalityFeatures

```python
@dataclass
class TonalityFeatures:
    chroma: np.ndarray                  # Shape: (12, n_frames)
    spectral_centroid: np.ndarray       # Brightness
    spectral_flatness: np.ndarray       # Noisiness
    spectral_rolloff: np.ndarray        # Bandwidth
    zero_crossing_rate: np.ndarray      # Noisiness
    dominant_chroma_indices: np.ndarray # Dominant note per frame
```

### ExtractedFeatures

```python
@dataclass
class ExtractedFeatures:
    temporal: TemporalFeatures
    energy: EnergyFeatures
    tonality: TonalityFeatures
    n_frames: int
    hop_length: int
    sample_rate: int
    frame_times: np.ndarray
```

### PolishedFeatures

```python
@dataclass
class PolishedFeatures:
    is_beat: np.ndarray           # Boolean beat triggers
    is_onset: np.ndarray          # Boolean onset triggers
    percussive_impact: np.ndarray # Smoothed [0, 1]
    harmonic_energy: np.ndarray   # Smoothed [0, 1]
    global_energy: np.ndarray     # Smoothed [0, 1]
    spectral_flux: np.ndarray     # Smoothed [0, 1]
    
    # 7-band system
    sub_bass: np.ndarray
    bass: np.ndarray
    low_mid: np.ndarray
    mid: np.ndarray
    high_mid: np.ndarray
    presence: np.ndarray
    brilliance: np.ndarray
    
    # Legacy bands
    low_energy: np.ndarray        
    mid_energy: np.ndarray        
    high_energy: np.ndarray       
    
    spectral_brightness: np.ndarray
    spectral_flatness: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    
    chroma: np.ndarray            # Shape: (12, n_frames)
    dominant_chroma_indices: np.ndarray
    n_frames: int
    fps: int
    frame_times: np.ndarray
```

### EnvelopeParams

```python
@dataclass
class EnvelopeParams:
    attack_ms: float = 0.0      # Attack time in milliseconds
    release_ms: float = 500.0   # Release time in milliseconds
```

---

## Constants

### Chroma Names

```python
FeatureAnalyzer.CHROMA_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]
```

### Frequency Bands (Extended)

| Band | Frequency Range |
|------|-----------------|
| Sub-Bass | 20 - 60 Hz |
| Bass | 60 - 250 Hz |
| Low-Mid | 250 - 500 Hz |
| Mid | 500 - 2000 Hz |
| High-Mid | 2000 - 4000 Hz |
| Presence | 4000 - 6000 Hz |
| Brilliance | 6000+ Hz |
