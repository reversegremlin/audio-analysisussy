This specification focuses on **viscous metal simulation** and **specular distortion**. The goal is to create a visualizer that feels heavy, expensive, and physically reactive—a digital "Ferrofluid" that obeys the laws of a distorted, kaleidoscopic physics engine.

---

## 1. The Core Primitive: Ray-Marched Metaballs

Unlike flat shapes, these are 3D "blobs" defined by mathematical fields. When two blobs get close, they don’t just touch; their surfaces stretch and fuse like molten solder.

* **The Fusion Formula:** The visualizer calculates an equipotential surface where:



*Where  is the radius (mapped to volume),  is the pixel position, and  is the surface threshold.*
* **Audio-Induced Surface Tension:**
* **High Volume:** Lowers the  threshold, causing the mercury to "shatter" into a cloud of tiny, independent mercury beads.
* **Silence:** High threshold causes all beads to pull back into one giant, shivering central mass.



---

## 2. Viscosity & Fluid Dynamics

The "thickness" of the liquid reacts to the complexity of the audio spectrum.

* **The Slime Factor (Low Frequency):** Bass frequencies increase the "viscosity" variable in the fluid solver. The motion becomes heavy and deliberate, creating deep, slow-moving ripples that catch the light.
* **Splash & Spray (High Frequency):** High-end transients (snares, hi-hats) act as kinetic impacts on the fluid surface. They create sharp, needle-like spikes and microscopic droplets that spray outward toward the kaleidoscope boundaries.
* **The Submerged Speaker:** Imagine a 3D displacement map under the fluid. The "floor" of the simulation vibrates based on the waveform, kicking the mercury upward in 3D space toward the camera.

---

## 3. The Metallic Shader (The "Chrome" Look)

The liquid mercury effect relies entirely on **Environment Mapping** and **Specular Highlights**.

* **High-Dynamic Range (HDR) Reflections:** The chrome shouldn't just reflect grey; it should reflect a hidden, high-contrast "nebula" or "industrial cityscape" texture that provides deep blues, sharp oranges, and blinding whites.
* **Anisotropic Filtering:** As the metal stretches, the reflections should blur along the direction of the stretch, mimicking the microscopic grain of real brushed liquid metals.
* **Fresnel Effect:** The edges of the blobs should be significantly brighter and more reflective than the center, giving them a distinct sense of roundness and 3D depth.

---

## 4. Kaleidoscope Logic: The Gravity Wells

In this mode, the kaleidoscope lines aren't just mirrors; they are **physical boundaries and gravitational attractors.**

* **The Pooling Effect:** As the liquid is "flung" outward by the music, it hits the invisible seams of the kaleidoscope segments. The fluid should "stack up" against these mirrors, creating a thick rim of mercury around the edges of the screen.
* **Radial Suction:** During "drops" or quiet moments, the center of the kaleidoscope acts as a black hole, sucking the liquid back from the edges toward a singular, wobbling sphere in the middle.
* **Mirror Reflection Phase:** To increase the "trippiness," the reflection angle of the kaleidoscope can oscillate slightly out of phase with the beat, making the liquid appear to flow "around the corner" into the next segment.

---

## 5. Technical Execution: The "Phase Shift"

To prevent the visual from being too "clean," we introduce **Phase Transitions**:

1. **Solid State (Quiet):** The mercury is dark, sluggish, and moves like heavy oil.
2. **Liquid State (Mid-Energy):** Classic "Terminator 2" flow; highly reflective and constantly merging/splitting.
3. **Gaseous/Vapor State (Peak Energy):** The mercury atomizes. The kaleidoscope is filled with millions of microscopic chrome "dust" particles that move in a turbulent vortex, creating a shimmering, silver mist.
