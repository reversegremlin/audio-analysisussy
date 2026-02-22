This specification moves away from the rigid symmetry of geometry and the neon chaos of DMT, focusing instead on **stochastic beauty** and **particle decay physics**. In a cloud chamber, radiation is made visible by "contrails" formed as ionized particles pass through a supersaturated vapor.

The goal is to map audio transients to the "emission events" of uranium ore, using a kaleidoscope to turn random decay into a mesmerizing, rhythmic dance.

---

## 1. The Core Simulation: Cloud Chamber Physics

Before the kaleidoscope, the visualizer must simulate the "source" (the Uranium Ore) at the center of the frame.

* **Alpha Particles ():** Broad, thick, straight tracks. These are mapped to **low-frequency transients (kick drums, deep bass)**. They have high "ionizing power," meaning they should glow the brightest but travel the shortest distance.
* **Beta Particles ():** Thin, wispy, erratic "zig-zag" tracks. These are mapped to **high-frequency elements (snares, hi-hats, glitch textures)**. They move faster and cover more screen area.
* **Gamma Rays ():** Represented as faint, secondary "Compton scattering" effects—tiny bursts of light that appear away from the source. These are mapped to **reverb tails and ambient pads**.

---

## 2. The Audio-Emission Trigger

The visualizer functions as a **Geiger counter for sound**.

* **Threshold Detection:** When the audio signal exceeds a specific decibel threshold in a frequency band, a "decay event" is triggered.
* **Trajectory Logic:** The angle of the emission should be randomized but influenced by the stereo panning of the sound. A sound panned hard-left triggers a particle emission on the left side of the "ore."
* **Persistence (The Vapor Trail):** The tracks should not vanish instantly. They should linger and slowly dissipate/drift, mimicking the behavior of alcohol mist in a real chamber. Use a **Perlin Noise** field to simulate subtle air currents moving the trails.

---

## 3. The Radioactive Kaleidoscope

The "raw" cloud chamber simulation is then fed through a specialized kaleidoscope filter to create order from the random decay.

### The "Mirror" Mechanism

* **Segment Count:** Use a high segment count (12, 24, or 36) to create a "snowflake" effect from the chaotic alpha/beta lines.
* **Rotational Inertia:** Instead of a fixed rotation, the kaleidoscope should have "momentum." Every time a heavy bass hit (Alpha particle) occurs, the kaleidoscope's rotation should accelerate, then slowly friction-brake back to a crawl.
* **Temporal Mirroring:** The kaleidoscope doesn't just mirror space; it mirrors **time**. One segment might show the "current" decay, while the mirrored segment shows the decay from 500ms ago, creating a sense of "echoed history."

---

## 4. Visual Aesthetics & Shaders

To maintain the "Uranium" theme, the palette remains haunting and scientific.

* **The "Cherenkov" Glow:** High-energy events should emit a pale, ghostly blue light.
* **The "Pitchblende" Core:** The center of the kaleidoscope (the source) should be a dark, craggy, crystalline mass that pulses with internal heat (orange/red) when the RMS volume is high.
* **Ionization Bloom:** Use a heavy bloom shader on the "heads" of the tracks to simulate the intense energy of the particle, with the "tail" being a muted, semi-transparent grey-white.

---

## 5. Mathematical Function: The Decay Curve

The life of every visual element is governed by the law of radioactive decay:


*  is the brightness/opacity of the track at time .
*  is the "decay constant" mapped to the **tempo (BPM)** of the track—faster songs have a higher  (shorter half-life), causing the visuals to clear faster to prevent clutter.

---

## 6. Execution Strategy: The "Critical Mass" Mode

As the song reaches its climax (maximum energy), the visualizer enters **Critical Mass**:

1. **Chain Reaction:** Each particle track has a 10% chance to "split" and create secondary tracks, filling the kaleidoscope with a web of white lines.
2. **Inversion:** The background flips from black to a blinding radioactive white, with the particle tracks turning into pitch-black "shadows."

