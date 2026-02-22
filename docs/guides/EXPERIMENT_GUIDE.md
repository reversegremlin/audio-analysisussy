# Chromascope: Developer Experiment Guide

This guide explains how to create new visual experiments within the **OPEN UP** architecture. This architecture is "Energy First," meaning all simulations produce raw float32 fields that are interfered with and styled *after* the simulation logic.

---

## 1. The Core Architecture

Every experiment consists of two parts:
1.  **A Config Dataclass**: Inherits from `BaseConfig`.
2.  **A Visualizer Class**: Inherits from `BaseVisualizer`.

### Why "Energy First"?
By returning a raw energy field (`0.0` to `1.0`) instead of a colored image, your visualizer automatically gains:
- **Mirroring**: Vertical, Horizontal, Diagonal, and Circular symmetry.
- **Interference**: `Resonance`, `Constructive`, and `Destructive` mixing with other instances.
- **Mixed Mode**: The ability to "clash" your visualizer against any other (e.g., Solar vs. your new mode).

---

## 2. Step-by-Step: Creating a New Visualizer

### Step A: Define your Config
Create a new file (e.g., `src/chromascope/experiment/my_viz.py`) and define your settings.

```python
from dataclasses import dataclass
from chromascope.experiment.base import BaseConfig

@dataclass
class MyVizConfig(BaseConfig):
    speed_factor: float = 1.0
    chaos_amount: float = 0.5
```

### Step B: Implement the Visualizer
Inherit from `BaseVisualizer`. You must implement `update` and `get_raw_field`.

```python
import numpy as np
from chromascope.experiment.base import BaseVisualizer

class MyNewRenderer(BaseVisualizer):
    def __init__(self, config=None, seed=None, center_pos=None):
        super().__init__(config or MyVizConfig(), seed, center_pos)
        self.cfg: MyVizConfig = self.cfg # For type hinting
        
        # Initialize your simulation state here
        self.particles = []

    def update(self, frame_data: dict):
        """Advance simulation based on audio."""
        self.time += 0.1
        self._smooth_audio(frame_data) # Populates self._smooth_energy, etc.
        
        # Use smoothed audio for reactivity
        power = self._smooth_sub_bass * self.cfg.speed_factor
        # ... update particles ...

    def get_raw_field(self) -> np.ndarray:
        """Return a (H, W) float32 array in range [0, 1]."""
        # Create your 'energy' field here
        field = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        # ... draw particles to field ...
        return np.clip(field, 0, 1)
```

---

## 3. Leveraging the Master Compositor

You don't need to write any mirroring or blending code. Once your visualizer returns a raw field, the `UniversalMirrorCompositor` handles everything.

- **Mirroring**: To test symmetry, use `--mirror vertical` in the CLI.
- **Interference**: To test how your visualizer "fights" itself, use `--mirror vertical --interference resonance`.

---

## 4. Registering for "Mixed Mode"

To allow your new visualizer to be mixed with others (like Solar or Decay), follow these steps in `src/chromascope/experiment/cli.py`:

1.  **Update `MixedConfig`**: Add your new config to the inheritance list.
    ```python
    @dataclass
    class MixedConfig(SolarConfig, DecayConfig, FractalConfig, MyVizConfig):
        pass
    ```
2.  **Add to CLI choices**: Add your mode to the `argparse` choices and the `if/elif` logic in `main()`.

---

## 5. Best Practices

- **Local Randomness**: NEVER use `random.random()` or `np.random.rand()`. Always use `self.rng.random()`. This ensures that mirrored instances (Instance A and Instance B) have deterministic but different behaviors.
- **Coordinate Independence**: Use `self.center_pos` instead of hardcoding `width/2`. This allows the compositor to "slide" your visualizer around for symmetrical clashing.
- **Performance**: Use vectorized `numpy` operations where possible. For 4K rendering, Python loops will be too slow.
