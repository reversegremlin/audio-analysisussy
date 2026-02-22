# Chromascope Chemical Renderer

Audio-reactive chemical reactions and crystallization — music as a catalyst for luminous lab phenomena.

Beats inject reagents. Snares seed crystals. Harmony drives branching. Silence lets structures dissolve. The result is a living chemistry simulation that is frame-accurately synchronized to any audio track.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [CLI Reference](#2-cli-reference)
3. [Styles](#3-styles)
4. [Palettes](#4-palettes)
5. [Audio Mapping](#5-audio-mapping)
6. [How the Simulation Works](#6-how-the-simulation-works)
7. [Post-Processing](#7-post-processing)
8. [Performance](#8-performance)
9. [Showcase Script](#9-showcase-script)
10. [Tuning Guide & Recipes](#10-tuning-guide--recipes)
11. [Determinism](#11-determinism)

---

## 1. Quick Start

```bash
# Install with experiment extras
pip install -e ".[experiment]"

# Interactive preview (pygame window)
chromascope-fractal my_track.mp3 --mode chemical --preview

# Render a full 1080p60 MP4
chromascope-fractal my_track.mp3 --mode chemical -o output.mp4

# Quick 30-second proof at 720p
chromascope-fractal my_track.mp3 --mode chemical --max-duration 30 --profile low -o test.mp4
```

The `chromascope-fractal` binary is the unified experiment CLI. The `--mode chemical` flag activates the chemical renderer. All other flags are identical across modes unless otherwise noted.

---

## 2. CLI Reference

### Standard flags (all modes)

| Flag | Default | Description |
|---|---|---|
| `--profile low\|medium\|high` | `medium` | 720p/30fps · 1080p/60fps · 4K/60fps |
| `--width N` | from profile | Override output width in pixels |
| `--height N` | from profile | Override output height in pixels |
| `-f / --fps N` | from profile | Override frame rate |
| `-o / --output PATH` | `<stem>_chemical.mp4` | Output MP4 path |
| `--max-duration N` | full track | Render only the first N seconds |
| `--mirror MODE` | `off` | `vertical` · `horizontal` · `diagonal` · `circular` · `cycle` |
| `--interference MODE` | `resonance` | `resonance` · `constructive` · `destructive` · `sweet_spot` · `cycle` |
| `--no-glow` | glow on | Disable bloom post-processing |
| `--no-aberration` | aberration on | Disable chromatic fringe |
| `--no-vignette` | vignette on | Disable edge darkening |
| `--palette` | `jewel` | Override base palette (`jewel` · `solar`) — chemical mode uses its own palette on top |
| `--preview` | off | Open a real-time pygame preview instead of encoding |
| `--no-cache` | cache on | Force re-analysis of the audio file |
| `--no-low-res-mirror` | scaled | Disable internal downscaling when mirroring |

### Chemical-specific flags

| Flag | Range | Default | Description |
|---|---|---|---|
| `--chem-style` | see §3 | `neon_lab` | Visual style preset |
| `--chem-palette` | see §4 | `mixed` | Chemistry-inspired colour palette |
| `--reaction-gain` | `0.0–2.0` | `1.0` | Scales the intensity and speed of reaction fronts |
| `--crystal-rate` | `0.0–2.0` | `1.0` | Baseline crystal growth speed |
| `--nucleation-threshold` | `0.0–1.0` | `0.3` | Minimum percussive energy needed to seed new crystals |
| `--supersaturation` | `0.0–1.0` | `0.5` | Baseline branching propensity of growing crystals |
| `--bloom` | `0.0–2.0` | `1.0` | Post-glow intensity multiplier |

---

## 3. Styles

A style sets the *character* of the simulation — how aggressively the Gray-Scott reaction runs, how fast heat dissipates, and the base glow intensity. Think of styles as the lab environment.

### `neon_lab` (default)
The reference style. Balanced reaction speed, moderate crystal lifetime, full emissive glow. Reads well at all energy levels. Good starting point for any track.

```
GS feed: 0.037  |  kill: 0.060  |  heat decay: 0.88  |  crystal decay: 0.004
```

### `plasma_beaker`
Aggressive. Higher feed rate pushes the Gray-Scott toward denser, faster-propagating patterns. Heat dissipates quickly so the reaction front reads as punchy bursts rather than lingering pools. Crystal decay is faster, keeping the field dynamic on busy passages.

```
GS feed: 0.055  |  kill: 0.062  |  heat decay: 0.80  |  crystal decay: 0.006
```
Best paired with: high `reaction-gain`, bright palettes (`iron`, `sodium`, `copper`).

### `midnight_fluor`
Dark and patient. The slowest Gray-Scott feed produces fine-grained, persistent patterns. Heat lingers longest, giving crystals room to accumulate across full musical phrases. Crystal decay is minimal — structures built during a build section will still be visible well into the drop.

```
GS feed: 0.025  |  kill: 0.057  |  heat decay: 0.93  |  crystal decay: 0.002
```
Best paired with: low `reaction-gain` (0.4–0.9), high `crystal-rate`, `potassium` or `copper` palettes.

### `synth_chem`
Similar kinetics to `neon_lab` but activates a per-frame **hue rotation** pass. The entire RGB output is continuously rotated through the colour wheel, driven by the audio's spectral centroid and a slow time drift. The starting palette (`--chem-palette`) sets the base colour that gets rotated away from. Every style flag other than `synth_chem` treats colour as static per-palette; this is the only style that cycles.

```
GS feed: 0.042  |  kill: 0.059  |  heat decay: 0.85  |  crystal decay: 0.003
Hue rotation: time × 0.05 + spectral_centroid × 0.3  (radians normalized to [0,1])
```
Best paired with: `mixed` palette or any single-element palette for maximum contrast.

---

## 4. Palettes

A palette defines three neon-pushed colour anchors — one for the **reaction heat field**, one for the **crystal body**, and one for the **crystal edges** (sparkle/facets). All colours are drawn from real-world emission spectroscopy, then saturated into vivid neon territory.

### `iron`
| Layer | Colour | Notes |
|---|---|---|
| Heat | Hot red → ember orange `(1.00, 0.27, 0.00)` | Iron plasma / molten metal emission |
| Crystal | Dark red-brown `(0.55, 0.10, 0.00)` | Cooled slag / magnetite body |
| Edge | White-hot `(1.00, 0.95, 0.80)` | Incandescent rim light |

Mood: industrial, volcanic, aggressive. Pairs best with `plasma_beaker` and high `reaction-gain`.

### `copper`
| Layer | Colour | Notes |
|---|---|---|
| Heat | Cyan-aqua `(0.00, 1.00, 0.80)` | Copper(II) flame — characteristic turquoise |
| Crystal | Dark teal `(0.00, 0.40, 0.40)` | Malachite / verdigris body |
| Edge | Electric aqua `(0.50, 1.00, 1.00)` | Oxidised copper rim sparkle |

Mood: oceanic, cool, luminous. Very legible at low brightness — great for `midnight_fluor`.

### `sodium`
| Layer | Colour | Notes |
|---|---|---|
| Heat | Intense yellow `(1.00, 0.87, 0.00)` | Sodium D-line — the brightest single-element emission |
| Crystal | Amber-gold `(0.60, 0.40, 0.00)` | Cooled NaCl crystal body |
| Edge | White-gold `(1.00, 1.00, 0.80)` | Facet highlights |

Mood: radiant, solar, celebratory. `sodium_plasma` at full `reaction-gain` 2.0 and `bloom` 2.0 approaches total saturation — use `tone_map_soft` (always on) to pull highlights back from pure white.

### `potassium`
| Layer | Colour | Notes |
|---|---|---|
| Heat | Violet `(0.61, 0.15, 0.80)` | Potassium flame — deep violet/lilac |
| Crystal | Deep purple `(0.29, 0.05, 0.40)` | Dense crystal body |
| Edge | Soft lilac `(0.90, 0.70, 1.00)` | Delicate facet shimmer |

Mood: psychedelic, dreamy, complex. High `supersaturation` values produce richly branched purple lattices that read beautifully on dark backgrounds.

### `mixed` (default)
Not a fixed palette — `mixed` blends continuously between all four element palettes based on **spectral centroid** and a slow **time drift**. The blend index steps through `iron → copper → sodium → potassium → iron` as the centroid moves from low to high and time accumulates.

```
blend_index = (spectral_centroid × 3.0 + time × 0.03) mod 4.0
```

Tonal, harmonically-rich passages (high centroid) pull toward sodium/potassium. Bass-heavy, low-centroid passages tend toward iron/copper. The slow time drift ensures colour keeps shifting even on monotone audio.

`mixed` is the safest default for an unknown track; it tends to produce complementary colour transitions that feel musically motivated.

---

## 5. Audio Mapping

Every simulation parameter is driven by smoothed audio features extracted by the chromascope audio pipeline. Here is the complete mapping.

### Feature smoothing

All incoming per-frame values are exponentially smoothed before use. Fast-decaying features track transients; slow-decaying features track phrase-level energy.

| Feature | Smoothing speed | Notes |
|---|---|---|
| `percussive_impact` | fast (α=0.35 on beat, 0.15 off) | Snare/kick energy |
| `sub_bass` | fast | Sub-bass RMS |
| `spectral_flux` | fast | Rate of spectral change |
| `brilliance` | fast | Very high-frequency energy |
| `spectral_flatness` | medium (α=0.12) | Tonal vs. noisy character |
| `sharpness` | medium | Psychoacoustic sharpness |
| `global_energy` | slow (α=0.06) | Overall RMS |
| `harmonic_energy` | slow | Tonal harmonic content |
| `low_energy` | slow | Low-mid band |
| `high_energy` | slow | High-mid band |
| `spectral_centroid` | slow | Brightness centre of mass |

### Kick / sub-bass → Reagent injection

When `is_beat` is true **or** `sub_bass > 0.4`, the renderer injects reagent B at random positions in the simulation grid. Injection count and patch radius scale with `sub_bass × reaction_gain`. This is the primary driver of reaction front activity — every kick creates a new zone of chemical interaction.

```
injections = 1 + smooth_sub_bass × reaction_gain × 3
patch_radius = 1 + smooth_sub_bass × 3 px (sim-grid units)
pulse_strength = 0.4 + smooth_sub_bass × 0.6
```

### Snare / transient → Nucleation bursts

When `percussive_impact` exceeds `nucleation_threshold`, crystal seed points are placed at random positions. The number of seeds scales with how far above the threshold the impact is (relative burst density), multiplied by `crystal_rate`.

```
burst_density = (percussive - nucleation_threshold) / (1 - nucleation_threshold)
seeds = int(burst_density × 6 × crystal_rate) + 1
```

Lowering `--nucleation-threshold` makes the renderer sensitive to softer hits. Setting it above 0.5 restricts nucleation to only the loudest impacts.

### RMS energy → Reaction front width and speed

`global_energy` and `low_energy` control two things:

1. **Gray-Scott integration speed** — higher energy runs more reaction steps per frame, widening fronts: `speed = 1 + smooth_energy × 2 × reaction_gain`
2. **Heat diffusion radius** — `heat_sigma = 1 + smooth_low × 3`, spreading the glow of active zones over larger areas on bass-heavy frames.

### Harmonic energy → Supersaturation / branching

`harmonic_energy` continuously nudges the internal supersaturation state upward. During harmonically rich passages (lush chords, pads) crystals branch more aggressively. During sparse or percussive sections the branching propensity relaxes back toward the `--supersaturation` baseline.

```
target_ss = supersaturation + smooth_harmonic × 0.5
smooth_supersat = lerp(smooth_supersat, target_ss, α=0.04)
```

### Spectral flatness → Crystal orderliness

`spectral_flatness` (0 = tonal, 1 = noise-like) controls the blend between two crystal growth kernels:

- **Isotropic (high flatness)** — equal σ in x and y, producing rounded, coherent lattice growth
- **Anisotropic (low flatness)** — different σ in x vs y (driven by supersaturation), producing elongated dendritic branching

Tonal, stable passages produce large, regular crystal bodies. Dissonant or noisy passages produce chaotic, spiky branching.

### High band / brilliance → Edge sharpness and scintillation

The crystal edge field (Sobel gradient magnitude of the crystal body) is scaled by high-frequency content:

```
edge_boost = 2.0 + smooth_brilliance × 4.0 + smooth_high × 2.0
```

Bright, treble-heavy moments cause crystal facets to sparkle intensely. This gives the sense of light catching crystal edges on cymbal hits or high synth notes.

### Section dynamics → Phase behaviour

The energy-gated heat decay is the primary mechanism for large-scale section changes:

```
heat_persist = max(0.5, style_heat_decay - (1 - smooth_energy) × 0.30)
```

- **Build sections** (rising energy): heat persists longer, supersaturation rises with harmonics, crystal structures accumulate — the field becomes progressively more complex
- **Drop** (peak energy, strong beats): maximum reagent injection, nucleation storms, heat at full intensity, edge field blazing
- **Outro / quiet passages**: heat collapses rapidly (persist ≈ 0.5 → zero within ~10 frames), growth threshold rises above available heat, dissolution takes over — structures gracefully erode

---

## 6. How the Simulation Works

The renderer operates on an internal **reduced simulation grid** at one-quarter of the output resolution (e.g. 480×270 for 1920×1080 output). All fields are `float32` in `[0, 1]`. At render time each field is bilinearly upscaled to the full output size.

### Five simulation fields

```
field_a       Reagent A concentration  (Gray-Scott, starts near 1.0 — "empty solution")
field_b       Reagent B concentration  (Gray-Scott, starts near 0.0 — seeded at init)
field_heat    Reaction heat / activation front  (A × B product, diffused and decayed)
field_crystal Crystal mass accumulation  (nucleated, grown, dissolved)
field_edge    Crystal facet energy  (Sobel gradient magnitude of field_crystal)
field_noise   Impurity texture  (4-octave vectorised fBm, slow time drift)
```

### Gray-Scott reaction-diffusion

The reaction-diffusion backbone is the Gray-Scott model, a classic two-species system that produces an enormous variety of patterns (spots, stripes, spirals, labyrinths) depending on feed `f` and kill `k` rates:

```
∂A/∂t = Da·∇²A  −  A·B²  +  f·(1 − A)
∂B/∂t = Db·∇²B  +  A·B²  −  (f + k)·B
```

Where:
- `Da = 0.16`, `Db = 0.08` (fixed diffusion constants — coral/finger-like regime)
- Laplacians are approximated as `gaussian_filter(field, σ) − field`
- `f` and `k` come from the active style preset, then `f` is modulated upward by `sub_bass × reaction_gain`
- The full integration step is scaled by `speed = 1 + smooth_energy × 2 × reaction_gain`

The GS system self-sustains once seeded — it doesn't need constant audio input to keep evolving — but its character and pace are continuously sculpted by the audio.

### Crystal growth

Crystals cannot appear spontaneously; they require nucleation events (beat-driven seed injection). Once seeded, they grow by a diffusion-expansion rule: each frame, the existing crystal field is convolved with a Gaussian kernel and the result `grown` is taken as the new field where:
- heat is above the current threshold
- `grown × growth_amount > existing_crystal`

The kernel is anisotropic — x and y sigmas differ based on supersaturation and spectral flatness — producing directionally-biased (dendritic) or isotropic growth depending on the music.

### Dissolution

Every frame a dissolution amount is subtracted from the entire crystal field:

```
dissolution = crystal_decay × (2.0 + (1 − smooth_energy) × 5.0)
```

At peak energy the baseline rate is `crystal_decay × 2`. In silence it rises to `crystal_decay × 7`, clearing accumulated structures within tens of seconds. This ensures the field reads differently in quiet and loud passages and prevents indefinite accumulation.

### Field composition (get_raw_field)

The raw energy map returned by `get_raw_field()` is a weighted sum used by the `UniversalMirrorCompositor` when running in mirror mode:

```
energy = heat × 0.45  +  crystal × 0.30  +  edge × 0.20  +  noise × 0.05
```

In normal (non-mirror) mode, `render_frame()` bypasses this and applies the chemistry palette directly to the individual fields, giving each layer its own colour.

### Colour compositing

The chemical palette layer order:

```
1. Crystal body    × crystal_col × 0.70    (muted emissive base)
2. Reaction heat   × heat_col × heat_boost  (bright emissive core, heat_boost = 1 + high × 0.6)
3. Crystal edges   × edge_col × edge_gain   (sharp sparkle, edge_gain = 1 + brilliance × 2.5 + sharpness × 1.5)
4. Impurity noise  × tiny RGB offset        (subtle colour texture, prevents uniform black zones)
```

In `synth_chem` style a vectorised HSV hue-rotation pass runs over the final composited RGB before post-processing.

---

## 7. Post-Processing

The chemical renderer uses the same post-processing stack as all other modes.

### Glow / bloom

Gaussian-blurred screen-blend. Intensity is audio-reactive:

```
glow_intensity = base_intensity × bloom × (1 + percussive × 0.4 + flux × 0.5)
```

`--bloom` is a direct multiplier on the base intensity — set it to `2.0` for a heavily overexposed neon look, `0.5` for a crisp, minimal style.

Disable entirely with `--no-glow`.

### Chromatic aberration

Horizontal R/B channel offset, scaled by transient energy:

```
offset = base_offset × (1 + percussive × 1.5 + sharpness × 2.0)
```

Creates lens fringe on beat hits. Disable with `--no-aberration`.

### Vignette

Radial edge darkening, scaled by low bass:

```
strength = base_strength × (1 + low × 0.4 + sub_bass × 0.8)
```

Deepens on bass-heavy frames, grounding the composition. Disable with `--no-vignette`.

### Tone mapping

Soft-knee Reinhard compression (shoulder = 78% of 255) is always applied last, preventing flat white clipping on very bright sodium/iron passages while preserving saturated midtones.

---

## 8. Performance

### Simulation grid scaling

The simulation always runs at `width÷4 × height÷4` (minimum 16px). This means:

| Output resolution | Sim grid | Grid pixels |
|---|---|---|
| 1280 × 720 | 320 × 180 | 57,600 |
| 1920 × 1080 | 480 × 270 | 129,600 |
| 3840 × 2160 | 960 × 540 | 518,400 |

All heavy operations (Gray-Scott convolutions, crystal growth kernels, Sobel gradients, fBm) run on the small grid. Upscaling to output resolution is a single bilinear resize per field per frame.

### Render time estimates

On a modern CPU (single-threaded scipy):

| Profile | Typical fps | Notes |
|---|---|---|
| `low` (720p/30) | ~8–12 fps | Fast for iteration |
| `medium` (1080p/60) | ~4–6 fps | Standard encode |
| `high` (4K/60) | ~1–2 fps | Long encode times |

The simulation is the bottleneck, not the upscale. The two most expensive operations per frame are the two `gaussian_filter` calls for Gray-Scott Laplacians and the two for crystal growth.

### Tips for faster renders

- Use `--profile low` or `--max-duration 30` to iterate on settings before committing to a full render.
- The `--preview` flag opens a pygame window at real-time speed — useful to feel how settings respond to a specific track before encoding.
- Mirror modes (`--mirror circular` etc.) run two independent simulation instances; expect roughly 1.5–1.8× the base render time.

---

## 9. Showcase Script

`scripts/chemical_showcase.py` renders a curated battery of 16 named combos in sequence, each exploring a different facet of the parameter space.

```bash
# List all 16 combos with descriptions
python scripts/chemical_showcase.py --list

# Run all combos as quick 30s clips at 720p
python scripts/chemical_showcase.py my_track.mp3 --quick

# Run all combos as 60s clips at 1080p
python scripts/chemical_showcase.py my_track.mp3 --duration 60 --profile medium

# Run only specific combos
python scripts/chemical_showcase.py my_track.mp3 --quick \
  --only iron_plasma copper_plasma potassium_dense mixed_synth_chaos

# Custom output directory
python scripts/chemical_showcase.py my_track.mp3 --quick --output-dir ~/renders/showcase
```

### Combo index

| Name | Style | Palette | Character |
|---|---|---|---|
| `iron_plasma` | plasma_beaker | iron | Molten red/orange, aggressive, dense nucleation |
| `iron_embers` | midnight_fluor | iron | Dark forge, slow ember glow, long memory |
| `iron_synth` | synth_chem | iron | Red/orange hue cycling, fast reactions |
| `copper_midnight` | midnight_fluor | copper | Dark teal aqua, long phrase memory |
| `copper_plasma` | plasma_beaker | copper | Electric aqua, hyper-saturated burst |
| `copper_synth` | synth_chem | copper | Teal into lilac hue drift |
| `sodium_neon` | neon_lab | sodium | Amber/gold balanced lab |
| `sodium_plasma` | plasma_beaker | sodium | Maximum yellow/white, near-total bloom |
| `potassium_dense` | neon_lab | potassium | Ultra-dense lilac branching, high chaos |
| `potassium_synth` | synth_chem | potassium | Dreamy violet hue rotation |
| `mixed_synth_chaos` | synth_chem | mixed | Full-spectrum maximum everything |
| `mixed_neon_balanced` | neon_lab | mixed | Mixed palette, default speed, clean |
| `mixed_midnight_slow` | midnight_fluor | mixed | Slow dark drift, long crystal memory |
| `ghost_crystal` | midnight_fluor | sodium | Near-silent growth, faint shimmer, no aberration |
| `superfluid_bloom` | plasma_beaker | mixed | Max branching, no vignette, edge-to-edge |
| `raw_reaction` | neon_lab | iron | No glow, no aberration, pure field energy |

---

## 10. Tuning Guide & Recipes

### Understanding the parameter interactions

The six chemical-specific parameters interact in layered ways. Before tuning, it helps to know what each one *primarily* affects:

| Parameter | What it controls |
|---|---|
| `reaction-gain` | How strongly bass/beats activate the GS reaction — the pace and density of glowing fronts |
| `crystal-rate` | How quickly crystals spread from a nucleation seed, and how many seeds spawn per transient |
| `nucleation-threshold` | The "sensitivity" of crystal seeding — lower = reacts to softer hits, higher = only loud impacts |
| `supersaturation` | The baseline branching propensity — lower = rounder blobs, higher = spiky dendritic arms |
| `bloom` | A direct multiplier on the glow pass intensity — affects perceived brightness more than structure |
| `chem-style` | Sets the GS kinetics and heat/crystal decay rates — the fundamental character of the simulation |

### Recipe: "Musical patience" — slow crystalline structures that span phrases

Emphasise the long-timescale crystal accumulation. Set a high threshold so only strong beats nucleate, and a slow crystal rate so growth unfolds over many seconds.

```bash
chromascope-fractal track.mp3 --mode chemical \
  --chem-style midnight_fluor \
  --chem-palette copper \
  --reaction-gain 0.7 \
  --crystal-rate 0.4 \
  --nucleation-threshold 0.5 \
  --supersaturation 0.3 \
  --bloom 0.8
```

### Recipe: "Reaction storm" — non-stop catalytic chaos

Maximise reagent injection and front intensity. Lower the nucleation threshold so even quiet snares trigger seeding. Keep crystal-rate moderate so structures don't overwhelm.

```bash
chromascope-fractal track.mp3 --mode chemical \
  --chem-style plasma_beaker \
  --chem-palette iron \
  --reaction-gain 2.0 \
  --crystal-rate 0.8 \
  --nucleation-threshold 0.1 \
  --supersaturation 0.8 \
  --bloom 1.8
```

### Recipe: "Crystallography lab" — geometry-forward, minimal glow

Suppress the glow to expose raw crystal structure. Works best with high supersaturation (sharp branches) and a palette with high-contrast edges.

```bash
chromascope-fractal track.mp3 --mode chemical \
  --chem-style neon_lab \
  --chem-palette potassium \
  --reaction-gain 1.1 \
  --crystal-rate 1.5 \
  --nucleation-threshold 0.2 \
  --supersaturation 0.9 \
  --bloom 0.6 \
  --no-glow
```

### Recipe: "Full spectrum live" — hue-cycling palette for DJ sets

`synth_chem` + `mixed` produces the widest colour range. Leave all other values at defaults for a clean, balanced result.

```bash
chromascope-fractal track.mp3 --mode chemical \
  --chem-style synth_chem \
  --chem-palette mixed \
  --bloom 1.4
```

### Recipe: "Barely there" — near-invisible micro-motion for ambient music

Use a very high nucleation threshold and low reaction/growth rates. The simulation will produce faint drifting textures with only the occasional faint crystal needle on the loudest moments.

```bash
chromascope-fractal track.mp3 --mode chemical \
  --chem-style midnight_fluor \
  --chem-palette sodium \
  --reaction-gain 0.3 \
  --crystal-rate 2.0 \
  --nucleation-threshold 0.7 \
  --supersaturation 0.1 \
  --bloom 0.4 \
  --no-aberration
```

Note: `crystal-rate 2.0` here actually helps keep faint crystals visible — they grow fast when seeds *do* appear, creating brief bright needles that then dissolve. The high threshold means those moments are rare.

### Common mistakes

**Everything is too bright / washed out**
Lower `--bloom` and check the palette. `sodium` at full `plasma_beaker` genuinely saturates. Try `--bloom 0.7` or switch to `copper`/`potassium`.

**Nothing is happening / screen is mostly black**
The nucleation threshold may be too high for the track's dynamic range. Lower `--nucleation-threshold` to 0.1–0.15 and raise `--reaction-gain` to 1.5.

**Crystals look like circular blobs, not branches**
Increase `--supersaturation` to 0.7+ and pair with a track section that has high harmonic energy. The `spectral_flatness` of a tonal pad will push growth toward isotropic; more dissonant or percussive content produces better branching.

**Crystals fill the screen and never clear**
The dissolution rate depends on energy. If the track is consistently loud, crystals accumulate. Increase base `crystal_decay` by switching to `plasma_beaker` (which has the highest decay rate), or reduce `--crystal-rate`.

**The simulation looks the same throughout the track**
Check that the audio pipeline analysed the track correctly (look for per-frame `is_beat`, `percussive_impact`, `sub_bass` values in the manifest). If those values are flat the audio analysis may have failed. Re-run with `--no-cache` to force fresh analysis.

---

## 11. Determinism

With a fixed seed the chemical renderer produces bit-identical output across runs. The seed controls the initial Gray-Scott perturbation pattern, all nucleation site positions, and reagent injection locations.

The seed is set at the Python API level when constructing `ChemicalRenderer(config, seed=N)`. There is no `--seed` flag exposed in the CLI yet; the default is `None` (random per-run).

For reproducible renders in the current CLI, the workaround is to render once, keep the output you like, and note the flags used. Determinism testing is covered in `tests/experiment/test_chemical.py::TestDeterminism`.
