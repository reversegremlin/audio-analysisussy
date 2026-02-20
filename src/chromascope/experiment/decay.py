"""
Audio-reactive cloud chamber decay renderer.
Modernized for the OPEN UP architecture.

Particle physics are JIT-compiled via Numba when available, yielding
a large speed-up for the per-particle update loop (typically 50-200×
faster than the equivalent CPython loop at max_particles=6000).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from chromascope.experiment.base import BaseConfig, BaseVisualizer

# ---------------------------------------------------------------------------
# Optional Numba JIT acceleration
# ---------------------------------------------------------------------------
try:
    import numba as _numba

    @_numba.njit(cache=True, fastmath=True)
    def _step_particles(
        alive: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        life: np.ndarray,
        decay_rate: np.ndarray,
        drag: np.ndarray,
        last_x: np.ndarray,
        last_y: np.ndarray,
        h_wobble: float,
    ) -> None:
        """In-place particle physics step (JIT-compiled)."""
        for i in range(alive.shape[0]):
            if not alive[i]:
                continue
            life[i] -= decay_rate[i]
            if life[i] <= 0.0:
                alive[i] = False
                continue
            last_x[i] = x[i]
            last_y[i] = y[i]
            x[i] += vx[i]
            y[i] += vy[i]
            vx[i] *= drag[i]
            vy[i] *= drag[i]
            if h_wobble > 0.0:
                vx[i] += np.random.uniform(-h_wobble, h_wobble)
                vy[i] += np.random.uniform(-h_wobble, h_wobble)

    _NUMBA_OK = True

except ImportError:  # pragma: no cover

    def _step_particles(  # type: ignore[misc]
        alive, x, y, vx, vy, life, decay_rate, drag, last_x, last_y, h_wobble
    ) -> None:
        """Pure-Python fallback (no Numba)."""
        rng = np.random.default_rng()
        for i in range(len(alive)):
            if not alive[i]:
                continue
            life[i] -= decay_rate[i]
            if life[i] <= 0.0:
                alive[i] = False
                continue
            last_x[i] = x[i]
            last_y[i] = y[i]
            x[i] += vx[i]
            y[i] += vy[i]
            vx[i] *= drag[i]
            vy[i] *= drag[i]
            if h_wobble > 0.0:
                vx[i] += rng.uniform(-h_wobble, h_wobble)
                vy[i] += rng.uniform(-h_wobble, h_wobble)

    _NUMBA_OK = False


# ---------------------------------------------------------------------------
# Particle type constants
# ---------------------------------------------------------------------------
_TYPE_ALPHA = np.int8(0)
_TYPE_BETA = np.int8(1)
_TYPE_GAMMA = np.int8(2)
_TYPE_NAMES = {0: "alpha", 1: "beta", 2: "gamma"}


@dataclass
class DecayConfig(BaseConfig):
    """Configuration for the decay renderer."""
    base_cpm: int = 12000
    trail_persistence: float = 0.95
    vapor_persistence: float = 0.98
    base_diffusion: float = 0.08
    ionization_gain: float = 1.2
    max_particles: int = 6000
    palette_type: str = "jewel"  # Can use jewel or custom


class DecayRenderer(BaseVisualizer):
    """
    Renders organic, smokey decay trails.

    Internally stores all particle state in Structure-of-Arrays (SoA)
    numpy arrays for Numba-accelerated physics updates.
    """

    def __init__(
        self,
        config: Optional[DecayConfig] = None,
        seed: Optional[int] = None,
        center_pos: Optional[Tuple[float, float]] = None,
    ):
        super().__init__(config or DecayConfig(), seed, center_pos)
        self.cfg: DecayConfig = self.cfg  # type: ignore[assignment]

        M = self.cfg.max_particles
        self._max_particles = M

        # --- Structure-of-Arrays particle state ---
        self._p_alive = np.zeros(M, dtype=np.bool_)
        self._p_x = np.zeros(M, dtype=np.float32)
        self._p_y = np.zeros(M, dtype=np.float32)
        self._p_last_x = np.zeros(M, dtype=np.float32)
        self._p_last_y = np.zeros(M, dtype=np.float32)
        self._p_vx = np.zeros(M, dtype=np.float32)
        self._p_vy = np.zeros(M, dtype=np.float32)
        self._p_life = np.zeros(M, dtype=np.float32)
        self._p_decay_rate = np.zeros(M, dtype=np.float32)
        self._p_thickness = np.zeros(M, dtype=np.float32)
        self._p_intensity = np.zeros(M, dtype=np.float32)
        self._p_drag = np.zeros(M, dtype=np.float32)
        self._p_type = np.zeros(M, dtype=np.int8)      # 0=alpha,1=beta,2=gamma
        self._p_generation = np.zeros(M, dtype=np.int8)

        # High-water mark: slots 0.._n_slots-1 have ever been used
        self._n_slots: int = 0

        # Render buffers
        self.track_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.vapor_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)

        # Ore animation state
        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0

    # ------------------------------------------------------------------
    # Particle management
    # ------------------------------------------------------------------

    @property
    def particle_count(self) -> int:
        """Number of currently live particles."""
        return int(self._p_alive[:self._n_slots].sum())

    @property
    def particles(self):
        """
        Compatibility view: returns a list of lightweight objects for each
        live particle.  Access is O(max_particles) — use particle_count for
        cheap size checks.
        """
        from collections import namedtuple
        P = namedtuple(
            "Particle",
            ["x", "y", "vx", "vy", "life", "decay_rate", "thickness",
             "intensity", "type", "last_x", "last_y", "drag", "generation"],
        )
        out = []
        for i in range(self._n_slots):
            if self._p_alive[i]:
                out.append(P(
                    x=float(self._p_x[i]),
                    y=float(self._p_y[i]),
                    vx=float(self._p_vx[i]),
                    vy=float(self._p_vy[i]),
                    life=float(self._p_life[i]),
                    decay_rate=float(self._p_decay_rate[i]),
                    thickness=float(self._p_thickness[i]),
                    intensity=float(self._p_intensity[i]),
                    type=_TYPE_NAMES.get(int(self._p_type[i]), "gamma"),
                    last_x=float(self._p_last_x[i]),
                    last_y=float(self._p_last_y[i]),
                    drag=float(self._p_drag[i]),
                    generation=int(self._p_generation[i]),
                ))
        return out

    def _alloc_slot(self) -> int:
        """Return index of a free slot, or -1 if at capacity."""
        # Fast path: extend the high-water mark
        if self._n_slots < self._max_particles:
            slot = self._n_slots
            self._n_slots += 1
            return slot
        # Scan for a dead slot
        for i in range(self._max_particles):
            if not self._p_alive[i]:
                return i
        return -1  # truly full

    def spawn_particle(
        self,
        p_type: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        vx: Optional[float] = None,
        vy: Optional[float] = None,
        gen: int = 0,
    ) -> None:
        if self.particle_count >= self._max_particles:
            return

        cx, cy = self.center_pos
        if x is None:
            base_radius = 50.0 * self.ore_scale
            spawn_angle = self.rng.uniform(0, 2 * math.pi)
            r = base_radius * self.rng.uniform(0.7, 1.1)
            x = cx + math.cos(spawn_angle) * r
            y = cy + math.sin(spawn_angle) * r
            angle = spawn_angle + self.rng.uniform(-0.4, 0.4) + self.drift_angle
        else:
            angle = math.atan2(vy or 0.0, vx or 0.0) + self.rng.uniform(-0.3, 0.3)

        kick = 1.0 + self._smooth_percussive * 1.5
        if p_type == "alpha":
            speed = self.rng.uniform(5, 15) * kick
            decay = self.rng.uniform(0.02, 0.05)
            thick = self.rng.uniform(8, 16)
            pdrag = 0.88
            ptype = _TYPE_ALPHA
        elif p_type == "beta":
            speed = self.rng.uniform(30, 60) * kick
            decay = self.rng.uniform(0.005, 0.015)
            thick = self.rng.uniform(2, 4)
            pdrag = 0.98
            ptype = _TYPE_BETA
        else:
            speed = self.rng.uniform(0, 10)
            decay = 0.1
            thick = self.rng.uniform(4, 10)
            pdrag = 0.95
            ptype = _TYPE_GAMMA

        slot = self._alloc_slot()
        if slot < 0:
            return

        self._p_alive[slot] = True
        self._p_x[slot] = x
        self._p_y[slot] = y
        self._p_last_x[slot] = x
        self._p_last_y[slot] = y
        self._p_vx[slot] = math.cos(angle) * speed
        self._p_vy[slot] = math.sin(angle) * speed
        self._p_life[slot] = 1.0
        self._p_decay_rate[slot] = decay
        self._p_thickness[slot] = thick
        self._p_intensity[slot] = self.rng.uniform(0.7, 1.5)
        self._p_drag[slot] = pdrag
        self._p_type[slot] = ptype
        self._p_generation[slot] = gen

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def update(self, frame_data: Dict[str, Any]) -> None:
        """Advance the decay simulation."""
        dt = 1.0 / self.cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)

        self.ore_rotation += 0.04 + self._smooth_harmonic * 0.3
        self.ore_scale = self._lerp(
            self.ore_scale, 1.0 + self._smooth_sub_bass * 1.2, 0.4
        )
        self.drift_angle += (self._smooth_centroid - 0.5) * dt * 6.0
        self.view_zoom = self._lerp(
            self.view_zoom, 1.0 + self._smooth_sub_bass * 0.3, 0.1
        )

        # --- Spawn ---
        cpm = self.cfg.base_cpm * (
            1.0 + self._smooth_energy * 4.0 + self._smooth_flux * 3.0
        )
        num_spawns = self.rng.poisson((cpm / 60.0) * dt)
        if frame_data.get("is_beat", False):
            num_spawns += int(
                60 * self._smooth_percussive * self.cfg.ionization_gain
            )

        alpha_prob = self._smooth_low / (self._smooth_low + self._smooth_high + 1e-6)
        for _ in range(num_spawns):
            r = self.rng.random()
            if r < alpha_prob * 0.5:
                self.spawn_particle("alpha")
            elif r < 0.92:
                self.spawn_particle("beta")
            else:
                self.spawn_particle("gamma")

        # --- Physics update (JIT-compiled) ---
        n = self._n_slots
        h_wobble = float(self._smooth_harmonic * 4.0)
        _step_particles(
            self._p_alive[:n],
            self._p_x[:n],
            self._p_y[:n],
            self._p_vx[:n],
            self._p_vy[:n],
            self._p_life[:n],
            self._p_decay_rate[:n],
            self._p_drag[:n],
            self._p_last_x[:n],
            self._p_last_y[:n],
            h_wobble,
        )

        # --- Branching (sparse, stays in Python) ---
        branch_prob = 0.01 * self._smooth_flux
        if branch_prob > 0:
            alive_idx = np.nonzero(self._p_alive[:n])[0]
            for i in alive_idx:
                if (
                    self._p_generation[i] < 1
                    and self.rng.random() < branch_prob
                ):
                    self.spawn_particle(
                        _TYPE_NAMES[int(self._p_type[i])],
                        float(self._p_x[i]),
                        float(self._p_y[i]),
                        float(self._p_vx[i]),
                        float(self._p_vy[i]),
                        int(self._p_generation[i]) + 1,
                    )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def get_raw_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (track_buffer, vapor_buffer)."""
        cfg = self.cfg

        # Persistence and diffusion
        self.track_buffer *= cfg.trail_persistence
        self.vapor_buffer *= cfg.vapor_persistence
        self.vapor_buffer = gaussian_filter(
            self.vapor_buffer,
            sigma=cfg.base_diffusion * 6 * (0.5 + self._smooth_harmonic),
        )

        # Draw to buffers
        track_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        vapor_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        draw_t = ImageDraw.Draw(track_img)
        draw_v = ImageDraw.Draw(vapor_img)

        self._draw_ore(draw_t, self.center_pos, self.ore_scale)

        v_cx, v_cy = cfg.width / 2, cfg.height / 2
        n = self._n_slots
        alive_idx = np.nonzero(self._p_alive[:n])[0]

        for i in alive_idx:
            life = float(self._p_life[i])
            intensity = float(self._p_intensity[i])
            thickness = int(float(self._p_thickness[i]) * (0.7 + life * 0.3))
            color = float(intensity * life)

            zoom = self.view_zoom
            tx = v_cx + (float(self._p_x[i]) - v_cx) * zoom
            ty = v_cy + (float(self._p_y[i]) - v_cy) * zoom
            tlx = v_cx + (float(self._p_last_x[i]) - v_cx) * zoom
            tly = v_cy + (float(self._p_last_y[i]) - v_cy) * zoom

            if self._p_type[i] == _TYPE_GAMMA:
                draw_t.ellipse(
                    [tx - thickness, ty - thickness, tx + thickness, ty + thickness],
                    fill=color,
                )
            else:
                draw_t.line(
                    [(tlx, tly), (tx, ty)],
                    fill=color,
                    width=max(1, thickness // 2),
                )
                draw_v.line(
                    [(tlx, tly), (tx, ty)],
                    fill=color * 0.6,
                    width=thickness,
                )

        self.track_buffer = np.maximum(self.track_buffer, np.array(track_img))
        self.vapor_buffer = np.maximum(self.vapor_buffer, np.array(vapor_img))

        return self.track_buffer, self.vapor_buffer

    def _draw_ore(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        scale: float,
    ) -> None:
        cx, cy = center
        num_pts = 16
        pts = []
        for i in range(num_pts):
            angle = i * (2 * math.pi / num_pts) + self.ore_rotation
            r = (50.0 + self.rng.uniform(-10, 10) * self._smooth_energy) * scale
            pts.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))

        draw.polygon(pts, fill=0.6 + self._smooth_harmonic * 0.4)
        for _ in range(3):
            hr = self.rng.uniform(5, 15) * scale
            hx = cx + self.rng.uniform(-10, 10) * scale
            hy = cy + self.rng.uniform(-10, 10) * scale
            draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=1.0)
