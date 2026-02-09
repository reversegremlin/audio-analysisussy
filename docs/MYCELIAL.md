This specification focuses on **Biomorphic Recursive Growth**. The goal is to simulate a living, breathing organism that consumes sound and excretes light. Unlike the sharp edges of Sacred Geometry, this is soft, translucent, and pulse-driven.

---

## 1. The Growth Algorithm: L-Systems & Space Colonization

The mycelium isn't just a static image; it is a "search" for nutrients (audio energy).

* **Hyphae Branching:** Use **L-Systems** (string-rewriting math) to determine how threads split.
* **The "Nutrient" Field:** The screen is mapped with invisible "food" particles. High-frequency melodies act as light sources that "pull" the growth of the hyphae toward specific coordinates.
* **Apical Dominance:** The "tips" of the mycelium glow with intense bioluminescence. When a new note is struck, the growth speed at the tips increases:



*Where  is velocity,  is a constant, and  is the amplitude of the specific frequency.*

---

## 2. Psilocybin Aesthetics: Subtle Fruiting Bodies

To integrate the "shroom" element without being kitschy, the mushrooms should appear as **emergent structures** rather than icons.

* **Fruiting Events:** When the mycelium density in a specific segment hits a "critical mass" (triggered by a sustained chorus or crescendo), a **primordia** (baby mushroom) begins to pin.
* **Cap Morphology:** The caps should be semi-transparent, using a **Sub-Surface Scattering (SSS) shader** so they look like they are filled with glowing nectar.
* **Spore Release:** During percussive hits, the mushrooms release "spores"—tiny, glowing particles that drift into the wind of the Perlin noise field, eventually landing to start new mycelial growth elsewhere.

---

## 3. Breathing Colors: The "Ooze" Palette

The colors should not be static; they must cycle through a **Circadian Rhythm** of hues.

* **Bioluminescent Pulse:** The base color is a deep "Forest Floor" (dark browns, moss greens). The mycelium itself pulses with "Electric Cyan" and "Amethyst Purple."
* **Vein-Staining:** When a heavy bass note hits, imagine a "dye" being injected into the center of the network. This color (e.g., a deep saffron or crimson) flows outward through the existing threads, slowly staining the entire network before fading.
* **Chromatic Shift:** Using a **Hue-Rotation Shader**, the entire visualizer slowly rotates 360° over the course of 5 minutes, ensuring no two moments look identical.

---

## 4. The Mycelial Kaleidoscope (The Petri Dish)

The kaleidoscope acts as the "containment field" for the growth.

* **The Wrap-Around Effect:** This is the "secret sauce." When a hypha reach the edge of a kaleidoscope segment (the "mirror line"), it shouldn't just reflect. It should **teleport** to the opposite mirror line of that segment, creating the illusion of a single thread weaving in and out of 4-dimensional space.
* **Membrane Tension:** The mirror lines should slightly "bow" or curve outward when the bass is heavy, as if the living organism inside is trying to push through the glass.
* **Symmetry Breaks:** Occasionally, for a "trippy" effect, one segment of the kaleidoscope should "desync" and rotate 1 degree faster than the others, breaking the perfect symmetry before snapping back into place.

---

## 5. Technical Shader Stack

To achieve the "eerie/alive" feel, use these specific post-processing effects:

* **Soft Focus / Bokeh:** The background mycelium is blurred, while the "active" growth in the foreground is pin-sharp.
* **Caustics:** A light overlay that looks like sunlight filtering through moving water, projected onto the mycelium.
* **Vein Pulse Shader:** A sine-wave displacement applied to the thickness of the lines:



*This creates the "pumping" sensation of an organism breathing.*

