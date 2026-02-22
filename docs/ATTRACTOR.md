# Chromascope — Strange Attractor Mode

`--mode attractor` renders thousands of particles orbiting two deterministic
strange attractors in 3D, leaving glowing neon trails that accumulate over time.
Audio reshapes the physics of the system every frame — the bass expands the
Lorenz butterfly, the melody tightens the Rössler spiral, and every beat sends
a brightness flash through the field.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [All CLI Options](#all-cli-options)
  - [Attractor-Specific](#attractor-specific)
  - [Resolution & Output](#resolution--output)
  - [Post-Processing](#post-processing)
  - [Mirroring & Interference](#mirroring--interference)
  - [Universal](#universal)
- [Blend Modes](#blend-modes)
- [Palettes](#palettes)
- [Audio Mapping](#audio-mapping)
- [Physics Parameters](#physics-parameters)
- [Performance Guide](#performance-guide)
- [Preset Recipes](#preset-recipes)

---

## How It Works

Each frame follows this pipeline:

```
1.  Accumulation buffer fades   (× trail_decay)
2.  Lorenz and/or Rössler particles advance via RK4 integration
3.  Audio modulates attractor parameters in real-time
4.  3D positions projected to 2D via rotating view matrix
5.  Each particle splatted as a bilinear blob onto the HDR buffer
6.  Gaussian bloom applied to the float32 HDR buffer
7.  Reinhard tone-mapping compresses highlights → vivid neon
8.  Beat flash, vignette, chromatic aberration → uint8 output
```

**The Lorenz attractor** forms the iconic butterfly — two lobes traced by
particles whipping around a pair of unstable fixed points. Its natural
parameters (σ, ρ, β) are driven by sub-bass and percussion.

**The Rössler attractor** forms a single continuous spiral band that folds
back on itself. It's structurally simpler than Lorenz but musically reactive
in a different register — the harmonic content of the track controls how
tightly it coils.

In **dual** mode both systems are rendered simultaneously in complementary
palette colours, their trails interweaving across the same 3D frame. In
**morph** mode the particle positions physically interpolate between the two
shapes, driven by spectral flatness — noisy audio pulls toward Rössler chaos,
tonal audio holds the Lorenz butterfly.

---

## Quick Start

```bash
# Default settings — dual mode, neon_aurora palette, 1080p 60fps
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor

# Showcase render — dense trails, both attractors, hot palette
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor \
  --attractor-blend-mode dual \
  --attractor-palette plasma_coil \
  --num-particles 5000 \
  --trail-decay 0.97 \
  --projection-speed 0.15 \
  --width 1920 --height 1080 -f 60

# 30-second preview at 720p before committing to a full render
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor \
  --max-duration 30 --width 1280 --height 720 \
  -o /tmp/preview.mp4

# Live preview window (requires pygame)
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor \
  --preview --width 960 --height 540
```

---

## All CLI Options

### Attractor-Specific

These flags only apply when `--mode attractor`.

---

#### `--attractor-blend-mode`

**Choices:** `lorenz` | `rossler` | `dual` | `morph`
**Default:** `dual`

Controls which attractor system(s) are simulated and rendered. See
[Blend Modes](#blend-modes) for a full breakdown.

```bash
--attractor-blend-mode dual     # both systems, complementary colours (default)
--attractor-blend-mode lorenz   # butterfly only
--attractor-blend-mode rossler  # spiral coil only
--attractor-blend-mode morph    # shape physically lerps between the two
```

---

#### `--attractor-palette`

**Choices:** `neon_aurora` | `plasma_coil` | `void_fire` | `quantum_foam`
**Default:** `neon_aurora`

Selects the neon colour gradient used for each attractor type. Particle hue
is derived from Z-depth after 3D projection — particles "in front" and
"behind" land at different points in the gradient — plus a slow global hue
drift tied to spectral centroid. See [Palettes](#palettes) for the full colour
breakdown.

```bash
--attractor-palette neon_aurora   # cyan/magenta + electric lime/ice blue
--attractor-palette plasma_coil   # hot orange/pink + neon yellow/green
--attractor-palette void_fire     # deep blue/cyan + deep purple/gold
--attractor-palette quantum_foam  # electric green/cyan + hot magenta/red
```

---

#### `--num-particles`

**Type:** integer
**Default:** `3000`

Number of particles per attractor. In `dual` mode this means 3000 Lorenz
particles **and** 3000 Rössler particles simultaneously (6000 total splats per
frame). In `lorenz` or `rossler` mode only one system is simulated.

Higher counts produce denser, more opaque trail coverage. Lower counts make
individual particle paths more visible with empty space between them.

```bash
--num-particles 1000   # sparse — individual threads visible
--num-particles 3000   # default — balanced coverage
--num-particles 6000   # dense — butterfly/spiral nearly filled
--num-particles 10000  # very dense — solid neon shapes
```

> **Performance note:** render time scales linearly with particle count. At
> 1080p/60fps, 3000 particles is realtime-capable with Numba; 10000 may be
> slow without it. See [Performance Guide](#performance-guide).

---

#### `--trail-decay`

**Type:** float `[0.0, 1.0]`
**Default:** `0.96`

Multiplicative fade factor applied to the accumulation buffer every frame.
This is the single most impactful parameter for the visual character of the
render.

| Value | Effect |
|---|---|
| `0.80` | Trails vanish in ~5 frames — almost no persistence |
| `0.90` | Short bright streaks — energetic, staccato feel |
| `0.96` | Default — trails last ~25 frames, smooth ribbons |
| `0.97` | Longer trails — ~33 frames, richer depth layering |
| `0.98` | Very long trails — ~50 frames, dense accumulated glow |
| `0.99` | Extremely long — full attractor shape always visible |
| `1.00` | No fade — trails accumulate infinitely, screen burns white |

The `high_energy` audio channel slightly modulates trail_decay each frame
(energetic high frequencies accelerate the fade by up to 4%), so even a
static `trail_decay` value produces organic variation at busy moments.

```bash
--trail-decay 0.90   # sharp, percussive — great for electronic/drum-heavy
--trail-decay 0.97   # cinematic flowing ribbons — great for ambient/orchestral
--trail-decay 0.99   # ghostly accumulated form — great for minimal/drone
```

---

#### `--projection-speed`

**Type:** float (radians per second)
**Default:** `0.2`

Base azimuth rotation speed of the 3D view. The camera continuously orbits
the attractor, gradually revealing all faces of the 3D shape. The actual
speed is further modulated by `global_energy`, so loud passages spin faster.

```
actual_speed = projection_speed × (1 + global_energy × 2)
```

At default energy (0.5), expect about `0.2 × 2.0 = 0.4 rad/s` — roughly
one full rotation every 15 seconds.

```bash
--projection-speed 0.0    # frozen — static viewpoint, no orbit
--projection-speed 0.05   # very slow — geological drift
--projection-speed 0.15   # slow — shapes are readable, good for long renders
--projection-speed 0.2    # default
--projection-speed 0.5    # fast orbit — tumbling effect
--projection-speed 1.0    # rapid spin — only works well at high fps
```

---

### Resolution & Output

#### `-o` / `--output`

Output file path. Defaults to `<audio-stem>_attractor.mp4` next to the audio
file.

```bash
-o /mnt/chromeos/MyFiles/Downloads/my-render.mp4
```

---

#### `-p` / `--profile`

**Choices:** `low` | `medium` | `high`
**Default:** `medium`

Preset resolution/fps/encoder-quality combination:

| Profile | Resolution | FPS | Encoder quality |
|---|---|---|---|
| `low` | 1280 × 720 | 30 | fast |
| `medium` | 1920 × 1080 | 60 | medium |
| `high` | 3840 × 2160 | 60 | high |

Individual `--width`, `--height`, `--fps` flags override the profile values.

```bash
-p low                         # 720p/30 — fast turnaround for testing
-p medium                      # 1080p/60 — default
-p high                        # 4K/60 — archival quality
--width 1920 --height 1080 -f 24  # custom: 1080p at 24fps film rate
--width 3840 --height 2160        # 4K, keep profile fps
```

---

#### `--max-duration`

**Type:** float (seconds)
**Default:** none (full track)

Render only the first N seconds. Essential for previewing.

```bash
--max-duration 30    # first 30 seconds
--max-duration 60    # first minute
```

---

#### `--no-cache`

Force re-analysis of the audio even if a cached manifest exists. Useful after
changing the audio file or debugging audio mapping.

---

### Post-Processing

These apply after the attractor render, implemented as final passes over the
uint8 frame.

---

#### `--no-glow`

Disables the base glow system inherited from `BaseConfig`. Note: the
attractor renderer has its own internal gaussian bloom pass (controlled by
the `glow_radius` config field, not exposed as a CLI flag) that is separate
from this and unaffected by `--no-glow`. This flag disables the outer
VisualPolisher glow layer.

---

#### `--no-aberration`

Disables chromatic aberration — the slight R/B channel horizontal shift that
gives the neon trails a retro CRT fringe. Cleaner output, slightly less
cinematic.

---

#### `--no-vignette`

Disables the radial edge darkening. Without vignette the corners stay fully
lit, which works well for flat, geometric compositions but can feel less
immersive for the attractor's organic forms.

---

#### `--palette`

**Choices:** `jewel` | `solar`
**Default:** `jewel`

Overrides the base palette type passed to the VisualPolisher layer. For
attractor mode this is largely superseded by `--attractor-palette`, but can
affect any VisualPolisher post-processing that runs after the attractor
render pipeline.

---

### Mirroring & Interference

Mirroring works by compositing a flipped copy of the rendered frame, dividing
resolution internally (to 75%) when active to maintain performance.

#### `--mirror`

**Choices:** `off` | `vertical` | `horizontal` | `diagonal` | `circular` | `cycle`
**Default:** `off`

| Value | Effect |
|---|---|
| `off` | No mirroring — raw 3D projection (default) |
| `vertical` | Left-right mirror — bilaterally symmetric butterfly |
| `horizontal` | Top-bottom mirror |
| `diagonal` | Corner fold |
| `circular` | Radial symmetry |
| `cycle` | Automatically cycles through modes over time |

```bash
--mirror vertical    # makes Lorenz butterfly perfectly symmetrical
--mirror circular    # mandala-like attractor
```

---

#### `--interference`

**Choices:** `resonance` | `constructive` | `destructive` | `sweet_spot` | `cycle`
**Default:** `resonance`

Controls how the mirrored copy is composited over the original. Only
meaningful when `--mirror` is not `off`.

---

#### `--no-low-res-mirror`

By default, when mirroring is active the internal simulation runs at 75%
resolution and is upscaled before compositing. This flag disables the
downscale, running at full resolution (slower, sharper).

---

### Universal

#### `--preview`

Opens a real-time pygame window instead of encoding to MP4. Useful for dialling
in parameters before committing to a full render. Controls:

- `SPACE` — pause/resume
- `→` — step one frame (while paused)
- `ESC` — close

```bash
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor \
  --preview --width 960 --height 540
```

> Requires `pygame` — install with `pip install -e ".[experiment]"`.

---

## Blend Modes

### `lorenz` — The Butterfly

Only the Lorenz system is simulated. The classic double-lobe shape — two
wings traced by chaotic orbits around a pair of unstable fixed points.
Coloured with the Lorenz half of the selected palette.

Best for: stark, iconic compositions. The butterfly wing shape is immediately
recognisable, making it ideal when you want the physics to be the subject.

```bash
--attractor-blend-mode lorenz --attractor-palette void_fire
```

---

### `rossler` — The Spiral Coil

Only the Rössler system is simulated. A single continuous spiral that folds
back on itself in 3D. Simpler topology than Lorenz, but with a distinctive
tilted coil silhouette.

Best for: ambient / drone music where the spiral slowly unfurls. The Rössler
is more sensitive to its `a` parameter, so harmonic-rich music produces dramatic
tightening and loosening of the coil.

```bash
--attractor-blend-mode rossler --attractor-palette quantum_foam
```

---

### `dual` — Interweaving Systems (default)

Both attractors are simulated simultaneously and rendered into the same
accumulation buffer. The Lorenz particles use the first colour gradient of
the palette, Rössler uses the second. Their trails overlay and overlap,
creating depth through colour separation and Z-ordering.

Best for: most music. The two systems respond to different frequency bands
(Lorenz to sub-bass/percussion, Rössler to harmonics), so the visual tension
between them maps naturally to musical structure.

```bash
--attractor-blend-mode dual --attractor-palette plasma_coil
```

---

### `morph` — Physics Lerp

Both systems are integrated every frame, but instead of rendering them
separately, the particle positions are blended:

```
rendered_pos = (1 - w) × lorenz_pos + w × rossler_pos
```

where `w` is driven by `spectral_flatness` (0 = tonal = pure Lorenz
butterfly, 1 = noisy = pure Rössler coil). Colour blends smoothly between
the two palette halves using the same weight.

The result is a shape that physically morphs between the butterfly and the
spiral coil as the music shifts registers. Electronic music with alternating
noisy/tonal sections is particularly effective.

```bash
--attractor-blend-mode morph --attractor-palette neon_aurora
```

---

## Palettes

Each palette defines two colour gradients — one for the Lorenz attractor, one
for Rössler. Within each gradient, particle hue interpolates from `h0`
(particles at Z-depth = 0, furthest back in the rotating 3D view) to `h1`
(Z-depth = 1, closest to camera). A slow global hue drift tied to spectral
centroid cycles the entire gradient over time.

All palettes use full HSV saturation (S = 1.0) to maintain pure neon
intensity. The HDR accumulation + Reinhard tone-mapping means overlapping
particles bloom to bright white at their cores.

---

### `neon_aurora` (default)

| System | Gradient | Notes |
|---|---|---|
| Lorenz | Cyan → Magenta | Blue-to-pink sweep across the butterfly wings |
| Rössler | Electric Lime → Ice Blue | The coil glows yellow-green at depth, fading to cold blue in front |

Cool-spectrum palette. Works well for most music styles.

---

### `plasma_coil`

| System | Gradient | Notes |
|---|---|---|
| Lorenz | Hot Orange → Hot Pink | Warm fire-spectrum butterfly |
| Rössler | Neon Yellow → Neon Green | Acid-bright coil |

High-energy warm palette. Excellent for electronic, techno, or anything
percussive. The Lorenz orange and Rössler yellow create a vivid contrast in
dual mode.

---

### `void_fire`

| System | Gradient | Notes |
|---|---|---|
| Lorenz | Deep Blue → Cyan | Cold to electric transition |
| Rössler | Deep Purple → Gold | Dark to warm transition |

Cinematic high-contrast palette. The dark start colours let the glow bloom
feel dramatic — trails emerge from near-black and flare toward bright gold
and cyan. Best with long trail_decay values (0.97+).

---

### `quantum_foam`

| System | Gradient | Notes |
|---|---|---|
| Lorenz | Electric Green → Cyan | Matrix-green to cold blue |
| Rössler | Hot Magenta → Neon Red | Pink-to-red sweep |

Maximum contrast palette — complementary hues that vibrate against each other
in dual mode. Aggressive and bright. Works well for noise, industrial, or
heavily distorted audio.

---

## Audio Mapping

The renderer extracts these signals from the audio manifest each frame and
applies them every frame:

| Signal | Effect | Details |
|---|---|---|
| `sub_bass` | Lorenz σ ±40% | σ ranges from `lorenz_sigma × 0.6` (quiet) to `× 1.4` (loud sub-bass). Higher σ makes the butterfly lobes expand and pulse outward. |
| `percussive_impact` | Lorenz ρ ±30% | ρ ranges from `lorenz_rho × 0.7` to `× 1.3`. Higher ρ stretches the lobes vertically — distinct "kicks" create visible bulges in the trail. |
| `harmonic_energy` | Rössler `a` ±150% | `a` ranges from `rossler_a × 0.5` to `× 3.0`. Low harmonic energy = tight compact coil. High = loose spreading spiral that nearly escapes the frame. |
| `brilliance` | Particle re-seed pulse | When brilliance > 0.4, up to 5% of particles are rebirthed near the attractor centre. Produces a burst of fresh bright trails from the core. |
| `is_beat` | Brightness flash + 10% reseed | Beat frames multiply exposure by 1.8 in the tone-mapping step, creating a visible flash. Also reseeds 10% of all particles. |
| `global_energy` | Projection rotation speed | `actual_speed = projection_speed × (1 + energy × 2)`. Quiet passages orbit slowly; loud passages spin faster. |
| `spectral_flux` | Elevation wobble | Camera elevation oscillates proportionally — rapid spectral changes create gentle up/down tilts that reveal different cross-sections of the attractor. |
| `spectral_flatness` | Morph blend weight | Only active in `morph` mode. Tonal audio (flatness → 0) holds the Lorenz butterfly. White-noise-like audio (flatness → 1) pulls toward the Rössler coil. |
| `spectral_centroid` | Global hue drift | The entire palette hue offset drifts by `centroid × 0.003` per frame. High-frequency heavy audio slowly rotates the colour wheel. |
| `high_energy` | Trail decay modulation | Higher high-frequency energy accelerates trail fade by up to 4%, making busy passages slightly crisper. |

All audio signals are smoothed with exponential moving averages before
application. Beats use a faster smoothing factor to preserve transient punch.

---

## Physics Parameters

These are `AttractorConfig` dataclass fields — not exposed as CLI flags but
accessible when using the Python API directly.

### Lorenz System

The Lorenz system: `dx = σ(y−x)`, `dy = x(ρ−z)−y`, `dz = xy−βz`

| Parameter | Default | Role |
|---|---|---|
| `lorenz_sigma` | `10.0` | Rate of heat transfer. Audio-modulated via sub_bass ±40%. |
| `lorenz_rho` | `28.0` | Temperature differential. Audio-modulated via percussive ±30%. Must stay > ~13 for chaos. |
| `lorenz_beta` | `2.6667` | Physical geometry parameter (2⁸⁄₃). Fixed — rarely needs changing. |

The classic chaotic regime requires ρ > ~24.74 (Hopf bifurcation). The
renderer clamps σ to [2, 25] and ρ to [10, 45] to prevent escape to infinity
even under extreme audio modulation.

### Rössler System

The Rössler system: `dx = −y−z`, `dy = x+ay`, `dz = b+z(x−c)`

| Parameter | Default | Role |
|---|---|---|
| `rossler_a` | `0.2` | Controls spiral tightness. Audio-modulated via harmonic ±150%. Values > ~0.3 produce a broader, slower spiral. |
| `rossler_b` | `0.2` | Fixed offset. Small values (near 0.2) give the characteristic single-lobed attractor. |
| `rossler_c` | `5.7` | Funnel depth. c < 4 gives a stable limit cycle; c > ~5.7 is fully chaotic. Fixed. |

### Integration

| Parameter | Default | Role |
|---|---|---|
| `substeps` | `6` | RK4 sub-steps per frame. More sub-steps = more accurate integration at the cost of compute time. 6 is a good balance for dt = 0.01/6 ≈ 1.67ms per step. |

### Visual

| Parameter | Default | Role |
|---|---|---|
| `glow_radius` | `1.5` | Sigma (pixels) for the gaussian bloom pass applied to the HDR accumulation buffer before tone-mapping. Larger values create a wider soft halo. |
| `particle_brightness` | `1.2` | HDR exposure multiplier in the Reinhard tone-mapping step: `mapped = (accum × exposure) / (1 + accum × exposure)`. Values > 1 push mid-tones toward bright neon. Beat frames use `brightness × 1.8`. |

---

## Performance Guide

### With Numba (recommended)

If `numba` is installed (it is in `[experiment]` extras), the RK4 integration
kernels compile on first run and execute in parallel across all CPU cores.
Expect a 10–30 second JIT warmup on the first frame, then near-realtime speed
for most configs at 1080p.

```bash
pip install -e ".[experiment]"   # includes numba>=0.57
```

### Without Numba

Pure NumPy fallbacks are used. Vectorized over the particle array (N particles
handled as a matrix, not a Python loop), so they're fast but single-threaded.

### Resolution vs Particle Count Trade-offs

| Target | Recommended settings |
|---|---|
| Quick preview | `--width 640 --height 360 --num-particles 500 --max-duration 30` |
| Testing params | `--width 1280 --height 720 --num-particles 1000 -f 30` |
| Standard render | `--width 1920 --height 1080 --num-particles 3000 -f 60` |
| Dense showcase | `--width 1920 --height 1080 --num-particles 6000 -f 60` |
| Archive 4K | `--width 3840 --height 2160 --num-particles 5000 -f 60` |

The accumulation buffer is `H × W × 3 × float32` — a 1080p buffer is 24 MB,
a 4K buffer is 96 MB. The per-frame bloom pass (gaussian_filter) is the
heaviest operation per frame at high resolution.

---

## Preset Recipes

### Ambient / Drone

Long trails, slow rotation, morph mode so the shape drifts between systems
as the harmonic texture shifts:

```bash
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor \
  --attractor-blend-mode morph \
  --attractor-palette void_fire \
  --trail-decay 0.985 \
  --projection-speed 0.05 \
  --num-particles 4000 \
  --width 1920 --height 1080 -f 60
```

---

### Electronic / Techno

Short snappy trails that respond to every kick, dual mode with a high-energy
palette:

```bash
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor \
  --attractor-blend-mode dual \
  --attractor-palette plasma_coil \
  --trail-decay 0.90 \
  --projection-speed 0.3 \
  --num-particles 3000 \
  --width 1920 --height 1080 -f 60
```

---

### Orchestral / Cinematic

Dense particle count, long trails, cold palette, slow orbit — the butterfly
and coil shapes become readable as persistent structures:

```bash
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor \
  --attractor-blend-mode dual \
  --attractor-palette neon_aurora \
  --trail-decay 0.975 \
  --projection-speed 0.12 \
  --num-particles 6000 \
  --width 1920 --height 1080 -f 60
```

---

### Lorenz Butterfly Solo — Pure Chaos

Just the butterfly, vivid warm palette, medium trails:

```bash
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor \
  --attractor-blend-mode lorenz \
  --attractor-palette quantum_foam \
  --trail-decay 0.96 \
  --projection-speed 0.2 \
  --num-particles 5000 \
  --width 1920 --height 1080 -f 60
```

---

### Mirror Symmetry — Mandala Mode

Dual attractor rendered into a circular mirror composite:

```bash
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor \
  --attractor-blend-mode dual \
  --attractor-palette neon_aurora \
  --mirror circular \
  --interference resonance \
  --trail-decay 0.97 \
  --num-particles 3000 \
  --width 1920 --height 1080 -f 60
```

---

### Fast Preview

Check your settings at low cost before a long render:

```bash
.venv/bin/python -m chromascope.experiment song.mp3 --mode attractor \
  --attractor-blend-mode dual \
  --attractor-palette plasma_coil \
  --trail-decay 0.97 \
  --projection-speed 0.15 \
  --num-particles 1000 \
  --max-duration 30 \
  --width 640 --height 360 -f 30 \
  -o /tmp/attractor-preview.mp4
```
