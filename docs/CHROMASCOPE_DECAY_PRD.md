# Chromascope Decay: PRD & Specification
## "Radioactive Cloud Chamber" Audio-Reactive Renderer

### 1. Vision
An organic, high-energy particle simulation that translates music into a radioactive decay field. It mimics the aesthetics of a Wilson Cloud Chamber but introduces a "Matter-Antimatter" sliding mirror architecture where two independent worlds collide and annihilate in the center.
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

### 2. Core Physics: "The Smokey Engine"
*   **Dual-Buffer Vapor:**
    *   **Track Buffer:** Sharp, high-intensity lines representing the ionization paths.
    *   **Vapor Buffer:** Diffuse, persistent smoke that drifts and distorts over time.
*   **Particle Types:**
    *   **Alpha:** Short, thick, slow. Starts with a high-velocity kick and suffers heavy drag (0.88).
    *   **Beta:** Long, thin, fast. Maintains speed (0.98 drag) and creates sweeping arcs.
    *   **Gamma:** Sparse flashes and point-events (speckles).
*   **Drag Dynamics:** All particles obey a friction-based decay law, causing them to "smoke out" as they lose energy.

---

### 3. Mirror Architecture: "Sliding Planes"
*   **Symmetrical Opposites:** The visual is split into two independent simulations (Identity A and Identity B) that move in exactly opposite directions along a locked axis (Vertical, Horizontal, or Diagonal).
*   **Centered Intersection:** All panning is centered on the screen's midpoint, ensuring that the "collision" always happens in the center.
*   **Cinematic Motion:** Slow, sweeping pans (0.15 - 0.55 phase speed) driven by music energy.
*   **Interference Modes:**
    *   **Resonance:** Multiplicative overlap with a 12.0x intensity boost.
    *   **Constructive:** Additive merging.
    *   **Destructive:** Difference-based annihilation.
    *   **Sweet Spot:** Max-based merging with resonance "sparking" in the center.

---

### 4. Coloring: "Matter-Antimatter Annihilation"
*   **Dynamic Harmony:** Uses triadic offsets to shift trail "tips" based on their distance to the nearest ore.
*   **Annihilation Palette:** When Simulation A and B overlap, the color engine detects the collision mask and triggers a high-contrast shift to **Cyan / Magenta / White-Hot**.
*   **White-Hot Core:** Particle tracks in the overlap zone are boosted to brilliant white, simulating an annihilation reaction.

---

### 5. CLI & Controls
*   `--style`: `uranium` (green/yellow), `neon` (blue/pink), `noir` (B&W).
*   `--mirror`: `vertical`, `horizontal`, `diagonal`, `circular`, `cycle`.
*   `--interference`: `resonance`, `constructive`, `destructive`, `sweet_spot`, `cycle`.
*   `--vapor-persistence`: Controls how long the smoke lingers (0.90 - 0.99).
*   `--distortion`: Controls the intensity of the "heat shimmer" vapor warp.

---

### 6. Technical Implementation
*   **Sub-Pixel Shifting:** Uses Bilinear Interpolation (`map_coordinates`) to ensure 100% jitter-free motion.
*   **Independent Seeds:** Every mirrored instance uses its own local `Random` and `np.random.Generator` states (Seeds: 42 and 1337).
*   **Coordinate Wrapping:** Motion utilizes "Wrap" mode, ensuring the screen is always filled even during large sweeps.
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
