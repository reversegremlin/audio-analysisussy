This spec outlines a high-performance, real-time music visualizer designed to replicate the "hyperspace" aesthetics of a DMT experience. The focus is on **non-Euclidean geometry**, **recursive self-similarity**, and **saturated color palettes**, all processed through a dynamic kaleidoscope engine.

---

## 1. Visual Primitive Library

Instead of standard bars or waves, the visualizer uses "entities" and "architectures" as the base units.

* **The Hyper-Grid:** A 3D wireframe mesh that curves toward a central singularity.
* **The Chrysanthemum:** A dense, interlocking floral pattern made of overlapping circles and cycloids.
* **Aperiodic Tiling:** Non-repeating patterns (like Penrose tiles) that shift their internal angles based on frequency.
* **Impossible Solids:** 3D shapes where vertices connect in ways that defy standard perspective (e.g., hyper-cubes or Klein bottles).

---

## 2. The Kaleidoscope Engine

The engine doesn’t just mirror an image; it performs **radial recursive folding**.

### Fold Parameters

* **Segment Count ():** The number of pie-slices in the circle. This should oscillate between 6 and 24 based on the complexity of the music.
* **Reflection Depth:** How many times a shape is mirrored within its own segment before being projected to the next.
* **The "Breathing" Offset:** The center point of the kaleidoscope (the "eye") should drift in a Lissajous curve, preventing the visual from feeling static.

### Mathematical Deformation

To capture the "oozing" nature of the hallucination, apply a spatial warp before the kaleidoscope fold:



*Where  represents the pixel coordinate and  is a constant mapped to the bass amplitude.*

---

## 3. Audio-to-Visual Mapping

The visuals must react to the "energy" of the track rather than just the volume.

| Audio Feature | Visual Manifestation |
| --- | --- |
| **Sub-Bass (<100Hz)** | **Pulse & Scale:** Controls the "breathing" of the entire kaleidoscope and the thickness of the lines. |
| **High Mids (2kHz - 4kHz)** | **Complexity:** Triggers the subdivision of shapes (e.g., a triangle becomes a fractal Sierpinski gasket). |
| **High-End Percussion** | **Color Shifting:** Sharp transients (snare/hats) trigger instantaneous hue rotations across the palette. |
| **Stereo Width** | **Rotation Speed:** Wide stereo imaging increases the angular velocity of the kaleidoscope segments. |

---

## 4. Color Theory & Shaders

The palette should avoid "Earth tones," focusing instead on high-saturation neon and "impossible" gradients.

* **Iridescence Shader:** Simulates the oily, rainbow-slick look on the edges of primitives.
* **Feedback Loops:** 10% of the previous frame is retained, blurred, and slightly scaled up, creating a "tracer" effect.
* **Chromatic Aberration:** Fringes the edges of the kaleidoscope segments with red and blue offsets, increasing in intensity during high-energy musical crescendos.

---

## 5. Movement Logic: The "Breakthrough"

The visualizer should have two distinct states:

1. **Waiting Room (Low Intensity):** Slower rotations, 2D primitives, predictable symmetry.
2. **The Breakthrough (High Intensity):** The kaleidoscope segments begin to overlap and "z-fight," creating a sense of 3D depth. The camera zooms infinitely into the center of the kaleidoscope (the "Zeno’s Paradox" zoom).

> **Technical Note:** To prevent viewer fatigue, the number of kaleidoscope facets () should inversely correlate with the "chaos" of the texture to maintain visual legibility.

