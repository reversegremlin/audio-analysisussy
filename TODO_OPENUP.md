# TODO: The "OPEN UP" Architectural Refactor
## Goal: Global Mirroring & Cross-Visualizer Interference

We have successfully built the "Blueprint" in `chromascope-decay`. The next step is to refactor the entire engine so that **Fractal**, **Solar**, and future modes can all utilize independent multi-instance simulations and spatial interference.

---

### 1. The "Energy First" Refactor (Core Principle)
To make interference patterns (like `resonance` or `destructive`) look good, we must interfere **before** we color.
- [ ] **Refactor `FractalKaleidoscopeRenderer`**: 
    - Move fractal logic out of `render_frame`.
    - Create `get_raw_field()`: Returns the raw escape-time float array (0.0 - 1.0).
- [ ] **Refactor `SolarRenderer`**: 
    - Create `get_raw_field()`: Returns the raw particle density map/accumulation buffer.
- [ ] **Universal Styling**: Move `apply_palette` and post-processing into a shared `VisualPolisher` that can take *any* raw energy field and turn it into a finished frame.

### 2. The Unified `BaseVisualizer` Interface
We need to standardize how renderers behave so one "Compositor" can manage any of them.
- [ ] **Implement `BaseVisualizer` Abstract Class**:
    - `__init__(seed, center_pos, config)`: Mandatory local random state and spatial panning.
    - `update(frame_data)`: Advances the simulation state.
    - `get_energy_field()`: Returns the raw float32 buffer.
- [ ] **Decouple Randomness**: Audit all renderers to ensure NO `random.seed()` or global `np.random` calls. Every instance must use its own `self.rng`.

### 3. The "Master Compositor" (Mirror 2.0)
A single class that can handle mirroring for *any* visualizer type.
- [ ] **Implement `UniversalMirrorCompositor(VisualizerClass, config)`**:
    - Should take a class reference (e.g., `DecayRenderer` or `SolarRenderer`).
    - Spawns two instances with the panned centers and independent seeds we perfected today.
    - Handles the smooth `cycle` logic for modes and interference.

### 4. Cross-Visualizer Interference (The "End Game")
- [ ] **Enable Heterogeneous Interference**:
    - Architect the compositor to allow `Instance A` to be **Solar** and `Instance B` to be **Decay**.
    - The interference zone will show where Solar gravity meets Radioactive decay.

### 5. Performance & CLI 
- [ ] **Memory Management**: Running two 4K fractals is heavy. Implement a `MultiProfile` that lowers internal simulation resolution during mirrored renders if needed.
- [ ] **Global CLI Flags**: Add `--mirror` and `--interference` to the main `chromascope` entry point so they work for all styles.

---

### Context for next session (Back up to speed):
*   **What works now:** `DecayRenderer` is fully modernized. It has local `self.rng`, a `center_pos` pan parameter, and a `get_raw_buffers()` method.
*   **The Secret Sauce:** We found that the "Sweet Spot" in the middle comes from `(a * b) * 5.0` (Resonance) math applied to the raw energy buffers *before* styling.
*   **Spatial Logic:** We are using a 200px gradient mask for the split, which creates a wide, beautiful overlap zone.
*   **Cycling:** We have a `change_potential` accumulator that cross-fades modes over ~2 seconds based on `global_energy`.

**Next Session Priority:** Refactor `Solar` first, as its particle-based nature will look incredible when two suns interfere with each other.
