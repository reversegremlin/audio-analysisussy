# Chromascope Experiment PRD
## `chromascope-chemical` — Audio-Reactive Chemical Reactions & Crystallization

## 1. Vision
Create a jaw-dropping experiment where music appears to **catalyze reactions in a luminous lab**: fluids collide, ions ignite, precipitation blooms, and crystalline structures grow organically in sync with rhythm and harmony.

The visual identity should balance:
- **Reality cues** (chemistry-informed color/material language: e.g., iron reds/oranges, copper cyan/teal, sodium yellow, potassium lilac).
- **Neon stylization** (high-brightness emissive rendering, glowing volumetric bloom, saturated highlights).
- **Organic delight** (slow, branching, believable crystal growth that rewards prolonged viewing).

This mode should feel like *watching a scientific miracle happen to the beat*.

---

## 2. Product Summary

**Feature Name:** `chromascope-chemical`  
**Type:** New experiment renderer mode (Energy First architecture)  
**Primary Mood:** Neon laboratory, reactive fluids, crystalline emergence  
**Output:** Full-length synchronized MP4 via existing pipeline

---

## 3. Problem Statement
Current modes emphasize geometry, fields, and particle dynamics. We do not yet have a mode centered on **material transformation**:
- reagents meeting,
- reaction fronts propagating,
- nucleation events,
- dendritic crystal growth.

Artists need a mode that expresses both **impact** (chemical reaction bursts) and **patience** (crystal accretion), mapped musically with frame-accurate synchronization.

---

## 4. Goals and Non-Goals

### 4.1 Goals
1. Add a CLI mode equivalent to existing experiment commands.
2. Simulate a stylized reaction-diffusion + crystallization field in Energy First form (`float32 [0..1]`).
3. Map rhythm/transients to reaction kinetics and harmonic content to crystal morphology.
4. Provide bright neon color grading while preserving chemistry-inspired palette anchors.
5. Keep deterministic output with fixed `--seed`.

### 4.2 Non-Goals
1. Scientifically exact molecular dynamics.
2. Full physically accurate thermodynamics/phase diagrams.
3. Real-time UI editor in v1.

---

## 5. Experience Principles
1. **Catalytic drama:** Beats should feel like reagent injections and activation pulses.
2. **Organic patience:** Crystal growth should emerge, branch, and thicken over musical phrases.
3. **Spectral truth + stylized glow:** Colors should nod to real chemistry, then push into vivid neon.
4. **No dead frames:** Even quiet passages retain low-level micro-motion and glimmer.
5. **Harmonic synchronization:** Visual morphology changes must feel musically intentional, not random.

---

## 6. Functional Requirements

### 6.1 CLI & Config Surface
Provide `chromascope-chemical <audio> [options]` with standard options plus:
- `--style {neon_lab,plasma_beaker,midnight_fluor,synth_chem}`
- `--reaction-gain` (0.0–2.0): scales reaction front intensity
- `--crystal-rate` (0.0–2.0): baseline growth speed
- `--nucleation-threshold` (0.0–1.0): sensitivity for crystal seed creation
- `--supersaturation` (0.0–1.0): controls branching propensity
- `--bloom` (0.0–2.0): post-glow multiplier
- `--chem-palette {iron,copper,sodium,potassium,mixed}`

### 6.2 Core Simulation Layers (Energy First)
Renderer internally maintains separate scalar fields:
1. **Reagent Field A/B** — two interacting fluid densities.
2. **Reaction Heat Field** — activation/intensity front.
3. **Crystal Mass Field** — accumulated crystal body.
4. **Crystal Edge Field** — high-frequency sparkle/rim energy.
5. **Impurity Noise Field** — stochastic imperfections for natural growth.

`get_raw_field()` returns a merged normalized energy map, while style/color pass maps field semantics to palette.

### 6.3 Audio-to-Visual Mapping
1. **Kick/Sub Bass → Injection Pulses**  
   Increase reagent inflow and local pressure at attractor zones.
2. **Snare/Transient Onsets → Nucleation Bursts**  
   Spawn seed points; higher transient confidence = denser seed clusters.
3. **RMS + Low/Mid Energy → Reaction Front Width**  
   Drives thickness and propagation speed of glowing fronts.
4. **Spectral Centroid/High Band → Spark & Facet Sharpness**  
   Higher brightness = sharper crystal edges and scintillation.
5. **Chroma Stability → Crystal Orderliness**  
   Tonal stability encourages larger coherent lattice growth; dissonance increases branching irregularity.
6. **Section Dynamics (build/drop) → Phase Mode**  
   Build: supersaturation rises, branching accelerates.  
   Drop: controlled bloom + accelerated crystallization capture.

### 6.4 Visual Language & Color System
Color policy combines realism anchors with neon exaggeration:
- **Iron-inspired:** hot red → ember orange with white-hot highlights.
- **Copper-inspired:** cyan/teal cores with electric aqua bloom.
- **Sodium-inspired:** intense yellow/gold emission.
- **Potassium-inspired:** violet/lilac plasma accents.

Rules:
- Keep black background deep for contrast.
- Clamp only extreme peaks; preserve saturated mids.
- Use emissive edge treatment for crystal contours.
- Ensure reaction cores never become uniformly white for long durations.

### 6.5 Crystallization Behavior Requirements
1. Growth must be **anisotropic and branching** (no circular blob-only expansion).
2. Crystal structures should show:
   - seed birth,
   - branch extension,
   - occasional merging/competition,
   - partial dissolution in low-energy passages.
3. Growth memory should span phrases (seconds), not just individual frames.
4. Quiet sections should maintain subtle twinkling edge activity.

---

## 7. Architecture & Implementation (following `EXPERIMENT_GUIDE.md`)

### 7.1 New Config Dataclass
Create `ChemicalConfig(BaseConfig)` in `src/chromascope/experiment/chemical.py` with parameters above and defaults tuned for musical readability.

### 7.2 New Visualizer
Create `ChemicalRenderer(BaseVisualizer)` implementing:
- `update(frame_data)`:
  - call `_smooth_audio(frame_data)`
  - advance reaction-diffusion field
  - update nucleation map
  - evolve crystal mass/edges
- `get_raw_field()`:
  - compose fields into normalized energy map
  - return `np.float32` array in `[0,1]`

### 7.3 Determinism & Performance
- Use `self.rng` / seeded numpy generator only.
- Avoid per-pixel Python loops; prefer vectorized operations.
- Profile 1080p60 path and cap expensive morphology passes.

### 7.4 Mixed Mode Compatibility
Register in `src/chromascope/experiment/cli.py`:
1. Add `ChemicalConfig` to mixed config inheritance.
2. Add mode to CLI choices.
3. Wire renderer selection in dispatch logic.

---

## 8. Presets
1. **`neon_lab` (default):** balanced realism + bright emissive edges.
2. **`plasma_beaker`:** aggressive glow, fast reactions, high event density.
3. **`midnight_fluor`:** darker atmosphere, selective highlights, long crystal memory.
4. **`synth_chem`:** stylized color cycling keyed to harmonic shifts.

---

## 9. Acceptance Criteria
1. A 3-minute track renders successfully with muxed audio at 1080p60.
2. Beat-aligned events visually coincide with transients (target ±1 frame at 60fps).
3. Crystal growth is visibly organic over time (branching + persistence).
4. Color output is bright, neon, and chemistry-referential (iron/copper/etc. palette validation).
5. With fixed seed and identical input, output is deterministic.

---

## 10. QA Plan (Sync-Check)
1. **Temporal sync test:** compare onset timestamps vs nucleation burst frames.
2. **Low-energy test:** ambient/quiet audio still yields subtle micro-motion.
3. **High-density test:** busy passages avoid full-screen clipping and preserve structure.
4. **Determinism test:** hash comparison of rendered frame subsets under fixed seed.
5. **Palette sanity test:** verify style presets keep expected color anchors.

---

## 11. Milestones
1. **M1 — Prototype fields:** reaction-diffusion + basic crystal mask.
2. **M2 — Audio mapping:** bind all primary features with smoothing and limiter curves.
3. **M3 — Color engine:** chemistry-informed neon palette system + style presets.
4. **M4 — CLI integration:** mode registration, flags, mixed-mode support.
5. **M5 — Validation:** sync/perf/determinism tests and tuning.

---

## 12. Open Questions
1. Should crystal dissolution be user-controllable or purely audio-driven?
2. Do we expose per-element palette blending weights in v1 or keep preset-only?
3. Should mixed mode allow chemical field interference with fractal/solar in first release?
4. Is optional GPU acceleration needed for 4K `best` quality in this mode?
