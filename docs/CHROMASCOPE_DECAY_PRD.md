# PRD: chromascope-decay

## 1. Product Summary

**Feature Name:** `chromascope-decay`  
**Type:** Audio-reactive visualizer preset + renderer mode  
**Inspiration:** Cloud chamber tracks from uranium ore (Uraninite), where alpha and beta ionization events leave short/thick and long/thin vapor trails.

`chromascope-decay` is a new cinematic visual mode that translates music into a live cloud-chamber-like field of radioactive trails. The effect should feel scientific, uncanny, and beautiful: dense particle events, glowing condensation streaks, soft diffusion, and stochastic bursts that remain rhythmically tied to the source audio.

---

## 2. Problem Statement

Current visual modes (e.g., fractal/kaleidoscope) deliver rich geometry and symmetry, but we do not yet have a mode centered on **probabilistic micro-events** and **ephemeral trails**. Users who want visuals that feel like invisible physics made visible (rather than geometric kaleidoscopes) need a specialized renderer and mapping model.

---

## 3. Goals and Non-Goals

### 3.1 Goals

1. Ship a CLI mode equivalent in usability to `chromascope-fractal`:
   - `chromascope-decay <audio> [options]`
2. Render a cloud chamber aesthetic with three core trail personalities:
   - **Alpha-like:** short, thick, bright, high diffusion
   - **Beta-like:** long, thin, fast, lower diffusion
   - **Gamma-like/background ionization proxy:** sparse flashes/speck events
3. Synchronize particle dynamics to audio features at frame-level precision.
4. Preserve beautiful motion under low-energy passages (ambient) and high-density passages (drum-heavy).
5. Produce full-length 1080p60 output with muxed source audio using existing encode pipeline.

### 3.2 Non-Goals

1. Scientific simulation of exact radiation transport physics.
2. Medical/scientific accuracy guarantees for decay chain modeling.
3. Real-time interactive editing UI in this phase (CLI-first implementation).

---

## 4. Target Users

1. **Creative coders / VJs** needing an "organic physics" look.
2. **Music video creators** wanting a monochrome-to-neon atmospheric style.
3. **Experimental visual artists** looking for event-driven motion beyond mandalas/fractals.

---

## 5. User Stories

1. As a creator, I can run one command on a song and receive a finished decay-style MP4 with synchronized audio.
2. As a power user, I can tune trail density, persistence, and glow without modifying source code.
3. As an experimental artist, I can choose style presets (e.g., `lab`, `aurora`, `noir`) to quickly explore moods.
4. As a QA user, I can verify beat-linked bursts align with audible transients.

---

## 6. Experience Principles

1. **Invisible forces made visible:** events should appear to "materialize" from a latent vapor field.
2. **Temporal authenticity:** trails should have believable lifetimes (birth, condensation, fade).
3. **Music first:** event rate, thickness, and glow should feel locked to rhythm and timbre.
4. **Controlled chaos:** stochastic behavior should feel alive but never noisy or random for randomness' sake.

---

## 7. Functional Requirements

### 7.1 CLI and Configuration

1. Add a new entry point:
   - `chromascope-decay`
2. Core flags:
   - `audio` (input)
   - `-o/--output`
   - `-p/--profile` (`low`, `medium`, `high`)
   - `--width`, `--height`, `--fps`
   - `--max-duration`
   - `-q/--quality` (`fast`, `balanced`, `best`)
3. Decay-specific flags:
   - `--base-cpm` (default simulated baseline event rate, e.g., 6000)
   - `--trail-persistence` (`0.0-1.0`)
   - `--diffusion` (`0.0-1.0`)
   - `--ionization-gain` (`0.0-2.0`)
   - `--style` (`lab`, `noir`, `uranium`, `neon`) 

### 7.2 Audio-to-Decay Mapping

At target FPS, derive visual control channels from existing manifest features. Each mapping must be implemented with explicit behavioral intent, including anti-patterns to avoid.

1. **Event Rate Driver**
   - Source: onset strength + transient confidence + high-band energy.
   - **Wrong implementation:** event rate linearly tracks RMS, which reduces the effect to a loudness meter with particles.
   - **Right implementation:** event rate follows a nonlinear response curve over onset/transient activity with:
     - a **soft floor** (background ionization never reaches zero), and
     - a **hard ceiling** (high-energy sections never collapse into visual noise).
   - Expected behavior: loud-but-sustained passages produce moderate, steady trail density; snare-like attacks create sharper local density spikes.

2. **Alpha/Beta Mix Driver**
   - Source: low/mid/high band ratios + transient profile.
   - **Wrong implementation:** alpha and beta are only size variants with otherwise identical motion.
   - **Right implementation:** alpha and beta use distinct kinematic models:
     - **Alpha:** slow, thicker, slight curvature, faster energy loss.
     - **Beta:** fast, thin, near-straight trajectories, longer apparent travel.
   - Expected behavior: low-frequency dominance (e.g., bass/808) weights toward alpha-like tracks; high-frequency transients (e.g., cymbals/hats) weight toward beta-like threads.

3. **Trail Energy Driver**
   - Source: RMS + spectral flux.
   - Output: trail core luminance, condensation halo intensity, and overexposure threshold behavior.
   - Expected behavior: sustained energy increases halo fullness and core persistence without causing uniform white clipping.

4. **Directional Bias Driver**
   - Source: chroma entropy + spectral centroid drift + melodic contour proxy.
   - Output: mean trail orientation drift + wobble amplitude.
   - Expected behavior: melodic phrases should gently "lean" the field (subtle angular drift) rather than causing abrupt directional jumps.

5. **Beat Burst Driver**
   - Source: beat/onset trigger.
   - **Wrong implementation:** boolean switch that doubles spawn count for one frame.
   - **Right implementation:** short envelope with approximately 1-frame attack and 8–12-frame decay, simultaneously modulating spawn multiplier and bloom radius.
   - Expected behavior: a weighted shockwave sensation that decays naturally instead of a single-frame pop.

### 7.3 Visual Simulation and Rendering

1. Maintain a particle/trail pool with per-event lifecycle state:
   - `spawn_frame`, `position`, `direction`, `velocity`, `width`, `intensity`, `life`
2. Simulate trail deposition into accumulation buffers:
   - **Core trail buffer** (sharp line energy)
   - **Condensation buffer** (blurred vapor halo)
   - **Spark buffer** (occasional point events)
3. Fade and diffuse buffers each frame:
   - Exponential decay controlled by `trail_persistence`
   - Spatial blur controlled by `diffusion`
4. Composite with post stack:
   - Tone map
   - Bloom
   - Chromatic tint profile (style-driven)
   - Film grain / vignette (optional)

### 7.4 Presets

Provide tuning packs:

1. **lab:** near-monochrome, realistic haze, restrained bloom
2. **uranium:** green-white luminescence, moderate glow, balanced density
3. **noir:** high-contrast black/white with strong trail edges
4. **neon:** stylized cyan/magenta spectral glow

### 7.5 Export

1. Use existing frame-to-ffmpeg pipeline.
2. Preserve source audio track in output MP4.
3. Ensure deterministic renders when `--seed` is fixed.

---

## 8. Technical Design Requirements

### 8.1 Architecture

Leverage existing pattern used by `chromascope-fractal`:

1. **Analysis pass** (manifest generation at FPS)
2. **Renderer pass** (frame synthesis from manifest)
3. **Encoder pass** (stdin pipe to ffmpeg with audio mux)

### 8.2 Performance Targets

1. 1080p60 medium profile should process a 3-minute track within practical offline bounds (target: <= 1.5x realtime on reference dev machine).
2. Memory should remain bounded by fixed-size buffers + particle pool (no unbounded list growth).
3. Fast profile should permit short-iteration previews (`--max-duration 10`).

### 8.3 Determinism and Stability

1. Given same audio + same seed + same parameters, frame output is reproducible.
2. Guardrails for extreme settings (clamp event counts, clamp blur kernel radius).
3. No frame drops in offline render path.

---

## 9. Data Contract Additions (Manifest Usage)

No mandatory schema break required; `chromascope-decay` consumes existing normalized features. Optional renderer-private derived channels may be computed at load time:

- `decay_event_rate`
- `decay_alpha_weight`
- `decay_beta_weight`
- `decay_burst_gain`
- `decay_drift`

If persisted, these should be additive, optional fields.

---

## 10. Quality and Validation Plan

### 10.1 Synchronization Checks

1. Validate beat burst alignment against onset/beat frames.
2. Spot-check frame indices at major transients (intro drop, chorus entry).

### 10.2 Visual QA

1. Confirm alpha trails read as short/thick and beta as long/thin.
2. Confirm trails fade naturally without stepping artifacts.
3. Confirm low-energy songs still produce subtle living background activity.

### 10.3 Regression and Performance

1. Add renderer unit tests for:
   - deterministic seed behavior
   - parameter clamping
   - frame buffer shape/type invariants
2. Add smoke tests for:
   - CLI invocation
   - 3–10 second preview render

---

## 11. Milestones

### M1: Spec + Skeleton

- CLI command scaffold
- renderer module skeleton
- parameter schema and defaults

### M2: Core Visual Engine

- particle spawn model
- trail buffers + decay/diffusion
- baseline grayscale output

### M3: Audio Reactivity

- mapping channels from manifest
- beat burst behavior
- alpha/beta balancing

### M4: Art Direction + Presets

- style presets
- glow/tone mapping polish
- documentation and examples

### M5: Validation + Release

- tests and perf checks
- sample outputs
- rollout in docs and README

---

## 12. Risks and Mitigations

1. **Risk:** Visual clutter at high-energy songs.  
   **Mitigation:** Hard cap concurrent events + adaptive thinning.
2. **Risk:** Over-randomized output feels disconnected from music.  
   **Mitigation:** Tie stochastic sampling strictly to feature-derived lambda and beat envelopes.
3. **Risk:** Performance degradation from heavy blur/bloom at 4k.  
   **Mitigation:** multi-resolution buffers and quality-dependent kernel sizes.

---

## 13. Success Metrics

1. Users can generate a finished decay video with a single command.
2. Internal QA rates synchronization quality as "good or better" on curated test tracks.
3. Render stability: no crashes across supported profiles on reference audio set.
4. Aesthetic acceptance: at least one preset considered "production ready" by project maintainers.

---

## 14. Open Questions

1. Should `chromascope-decay` default to monochrome (`lab`) or stylized (`uranium`) palette?
2. Do we expose direct alpha/beta/gamma weights in CLI or keep them implicit from audio mapping?
3. Is GPU shader path required in v1, or is optimized CPU/Numpy renderer acceptable initially?
4. Should decay overlays be composable with existing fractal/kaleidoscope modes in a future hybrid preset?



---

## 15. Visual Experience Reference — "What the Music Feels Like as Physics"

This section defines phenomenology over parameters. A developer should be able to close their eyes, hear a passage, and predict what the chamber is doing.

1. **The resting field**
   - Even near silence, the chamber is never dead.
   - Maintain low-rate background ionization: sparse single-pixel specks and occasional wandering beta threads crossing diagonally.
   - Feel target: visual room tone (like vinyl hiss), indicating the universe is always active at low intensity.

2. **A sustained bass note or pad**
   - Increase alpha dominance near center-mass.
   - Spawn dense, slow, thick tracks with broad, soft condensation halos.
   - Feel target: gravitational pressure in the medium, not speed.

3. **A hi-hat or high transient**
   - Emit brief sprays of thin beta threads at acute angles.
   - Keep lifetime short (few frames), with faint afterimage that dissipates in under ~0.5s.
   - Feel target: scattered, high-velocity energy that cannot hold shape.

4. **A snare or kick on the beat**
   - Trigger beat-burst behavior: clustered outward alpha spawns with stochastic angular spread.
   - Bloom should spike for roughly 2–3 frames, then roll off exponentially.
   - Feel target: vapor shockwave (inhalation/release), never a fireworks-style explosion.

5. **A melodic line or vocal**
   - Use tonal/chroma contour to bias orientation drift (approximately 10–15° across phrases).
   - Modulate trail width subtly with pitch confidence.
   - Feel target: melody causes the chamber to "lean" in musically legible but understated ways.

6. **A build or riser**
   - As spectral flux rises, shift mix toward beta-dominant kinematics (longer, thinner, faster trails).
   - Allow partial persistence in accumulation buffers to create cross-hatching and long-exposure character.
   - Feel target: structured chaos where field-level luminosity increases without one trail dominating.

7. **The drop**
   - Permit one (or two at most) near-maximum event-density frames plus peak bloom.
   - Immediately begin controlled pullback on the following frames.
   - Feel target: pressure release through the whole medium, not a white flash or hard scene cut.

8. **A quiet outro**
   - Taper event rate, shorten and thicken alpha behavior slightly, and gradually drain accumulation.
   - End state should return to sparse beta crossings over mostly clear vapor.
   - Feel target: the chamber going back to sleep.

### 15.1 Self-Evaluation Heuristic

Implementations should be judged against this question during listening tests:

> Does the chamber feel like it is inside the music, or does it merely react to numeric thresholds?

The bar for acceptance is the former.
