"""
Audio-reactive strange attractor renderer.

Simulates thousands of particles orbiting two deterministic strange attractors
(Lorenz + Rössler) in 3D. Particles leave neon-colored trails in a persistent
accumulation buffer that fades over time, creating cinematic glowing trail-ribbons.
Audio drives attractor parameters, projection rotation speed, hue cycling, and
particle re-seeding.

Audio mapping:
  - sub_bass         → Lorenz σ ±40% — expansion/contraction pulse
  - percussive       → Lorenz ρ ±30% — lobing distortion
  - harmonic         → Rössler a ±150% — spiral tightness
  - brilliance       → particle re-seed pulse (5% of particles reborn)
  - spectral_centroid → global hue offset drift
  - global_energy    → projection rotation speed
  - spectral_flux    → elevation wobble amplitude
  - spectral_flatness → morph blend weight (noisy audio = more Rössler chaos)
  - is_beat          → brightness flash × 1.8 + reseed 10% of particles
  - high_energy      → trail decay rate modulation
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from chromascope.experiment.base import BaseConfig, BaseVisualizer
from chromascope.experiment.colorgrade import chromatic_aberration, vignette

# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------
try:
    import numba as _numba

    _NUMBA_OK: bool = True
except ImportError:  # pragma: no cover
    _numba = None  # type: ignore[assignment]
    _NUMBA_OK = False

# ---------------------------------------------------------------------------
# Palette anchors: (h0, h1, s, v) per attractor type
# Hue linearly interpolated with z_depth: hue = h0 + (h1-h0) * z_depth
# ---------------------------------------------------------------------------
_ATTRACTOR_PALETTES: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
    "neon_aurora": {
        "lorenz":  (0.50, 0.83, 1.0, 1.0),   # cyan → magenta
        "rossler": (0.25, 0.55, 1.0, 0.95),  # electric lime → ice blue
    },
    "plasma_coil": {
        "lorenz":  (0.06, 0.92, 1.0, 1.0),   # hot orange → hot pink
        "rossler": (0.17, 0.38, 1.0, 1.0),   # neon yellow → neon green
    },
    "void_fire": {
        "lorenz":  (0.63, 0.53, 1.0, 1.0),   # deep blue → cyan
        "rossler": (0.75, 0.14, 1.0, 1.0),   # deep purple → gold
    },
    "quantum_foam": {
        "lorenz":  (0.33, 0.53, 1.0, 1.0),   # electric green → cyan
        "rossler": (0.92, 0.03, 1.0, 1.0),   # hot magenta → neon red
    },
}

# ---------------------------------------------------------------------------
# Numba-accelerated kernels (compiled on first call)
# ---------------------------------------------------------------------------
if _NUMBA_OK:

    @_numba.njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
    def _rk4_lorenz_kernel(
        pts: np.ndarray,
        sigma: float,
        rho: float,
        beta: float,
        dt: float,
        substeps: int,
    ) -> None:
        """In-place RK4 integration of N Lorenz particles. Thread-safe: each
        particle is independent (prange over rows)."""
        N = pts.shape[0]
        for i in _numba.prange(N):  # type: ignore[attr-defined]
            x = pts[i, 0]
            y = pts[i, 1]
            z = pts[i, 2]
            for _ in range(substeps):
                k1x = sigma * (y - x)
                k1y = x * (rho - z) - y
                k1z = x * y - beta * z

                x2 = x + 0.5 * dt * k1x
                y2 = y + 0.5 * dt * k1y
                z2 = z + 0.5 * dt * k1z
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z2) - y2
                k2z = x2 * y2 - beta * z2

                x3 = x + 0.5 * dt * k2x
                y3 = y + 0.5 * dt * k2y
                z3 = z + 0.5 * dt * k2z
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z3) - y3
                k3z = x3 * y3 - beta * z3

                x4 = x + dt * k3x
                y4 = y + dt * k3y
                z4 = z + dt * k3z
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z4) - y4
                k4z = x4 * y4 - beta * z4

                x = x + dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
                y = y + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
                z = z + dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
            pts[i, 0] = x
            pts[i, 1] = y
            pts[i, 2] = z

    @_numba.njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
    def _rk4_rossler_kernel(
        pts: np.ndarray,
        a: float,
        b: float,
        c: float,
        dt: float,
        substeps: int,
    ) -> None:
        """In-place RK4 integration of N Rössler particles."""
        N = pts.shape[0]
        for i in _numba.prange(N):  # type: ignore[attr-defined]
            x = pts[i, 0]
            y = pts[i, 1]
            z = pts[i, 2]
            for _ in range(substeps):
                k1x = -y - z
                k1y = x + a * y
                k1z = b + z * (x - c)

                x2 = x + 0.5 * dt * k1x
                y2 = y + 0.5 * dt * k1y
                z2 = z + 0.5 * dt * k1z
                k2x = -y2 - z2
                k2y = x2 + a * y2
                k2z = b + z2 * (x2 - c)

                x3 = x + 0.5 * dt * k2x
                y3 = y + 0.5 * dt * k2y
                z3 = z + 0.5 * dt * k2z
                k3x = -y3 - z3
                k3y = x3 + a * y3
                k3z = b + z3 * (x3 - c)

                x4 = x + dt * k3x
                y4 = y + dt * k3y
                z4 = z + dt * k3z
                k4x = -y4 - z4
                k4y = x4 + a * y4
                k4z = b + z4 * (x4 - c)

                x = x + dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
                y = y + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
                z = z + dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
            pts[i, 0] = x
            pts[i, 1] = y
            pts[i, 2] = z

    @_numba.njit(cache=True)  # type: ignore[misc]
    def _splat_glow_kernel(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        r_arr: np.ndarray,
        g_arr: np.ndarray,
        b_arr: np.ndarray,
        weight_arr: np.ndarray,
        accum: np.ndarray,
        H: int,
        W: int,
    ) -> None:
        """Bilinear scatter of N particles onto RGB accum buffer (serial — safe)."""
        N = x_arr.shape[0]
        for i in range(N):
            px = x_arr[i]
            py = y_arr[i]
            x0 = int(math.floor(px))
            y0 = int(math.floor(py))
            fx = float(px - x0)
            fy = float(py - y0)
            ri = r_arr[i] * weight_arr[i]
            gi = g_arr[i] * weight_arr[i]
            bi = b_arr[i] * weight_arr[i]
            for dy in range(2):
                wy = fy if dy else (1.0 - fy)
                yi = y0 + dy
                if yi < 0 or yi >= H:
                    continue
                for dx in range(2):
                    wx = fx if dx else (1.0 - fx)
                    xi = x0 + dx
                    if xi < 0 or xi >= W:
                        continue
                    w = wx * wy
                    accum[yi, xi, 0] += ri * w
                    accum[yi, xi, 1] += gi * w
                    accum[yi, xi, 2] += bi * w


# ---------------------------------------------------------------------------
# NumPy fallbacks (always defined, used when Numba is unavailable)
# ---------------------------------------------------------------------------

def _rk4_lorenz_numpy(
    pts: np.ndarray,
    sigma: float,
    rho: float,
    beta: float,
    dt: float,
    substeps: int,
) -> None:
    """Vectorized NumPy RK4 for Lorenz system (in-place)."""
    for _ in range(substeps):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        k1x = sigma * (y - x)
        k1y = x * (rho - z) - y
        k1z = x * y - beta * z

        x2 = x + 0.5 * dt * k1x
        y2 = y + 0.5 * dt * k1y
        z2 = z + 0.5 * dt * k1z
        k2x = sigma * (y2 - x2)
        k2y = x2 * (rho - z2) - y2
        k2z = x2 * y2 - beta * z2

        x3 = x + 0.5 * dt * k2x
        y3 = y + 0.5 * dt * k2y
        z3 = z + 0.5 * dt * k2z
        k3x = sigma * (y3 - x3)
        k3y = x3 * (rho - z3) - y3
        k3z = x3 * y3 - beta * z3

        x4 = x + dt * k3x
        y4 = y + dt * k3y
        z4 = z + dt * k3z
        k4x = sigma * (y4 - x4)
        k4y = x4 * (rho - z4) - y4
        k4z = x4 * y4 - beta * z4

        pts[:, 0] += dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        pts[:, 1] += dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        pts[:, 2] += dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0


def _rk4_rossler_numpy(
    pts: np.ndarray,
    a: float,
    b: float,
    c: float,
    dt: float,
    substeps: int,
) -> None:
    """Vectorized NumPy RK4 for Rössler system (in-place)."""
    for _ in range(substeps):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        k1x = -y - z
        k1y = x + a * y
        k1z = b + z * (x - c)

        x2 = x + 0.5 * dt * k1x
        y2 = y + 0.5 * dt * k1y
        z2 = z + 0.5 * dt * k1z
        k2x = -y2 - z2
        k2y = x2 + a * y2
        k2z = b + z2 * (x2 - c)

        x3 = x + 0.5 * dt * k2x
        y3 = y + 0.5 * dt * k2y
        z3 = z + 0.5 * dt * k2z
        k3x = -y3 - z3
        k3y = x3 + a * y3
        k3z = b + z3 * (x3 - c)

        x4 = x + dt * k3x
        y4 = y + dt * k3y
        z4 = z + dt * k3z
        k4x = -y4 - z4
        k4y = x4 + a * y4
        k4z = b + z4 * (x4 - c)

        pts[:, 0] += dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        pts[:, 1] += dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        pts[:, 2] += dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0


def _splat_glow_numpy(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    r_arr: np.ndarray,
    g_arr: np.ndarray,
    b_arr: np.ndarray,
    weight_arr: np.ndarray,
    accum: np.ndarray,
    H: int,
    W: int,
) -> None:
    """Bilinear scatter via np.add.at (NumPy fallback)."""
    x0 = np.floor(x_arr).astype(np.int64)
    y0 = np.floor(y_arr).astype(np.int64)
    fx = (x_arr - x0).astype(np.float32)
    fy = (y_arr - y0).astype(np.float32)
    rgb = np.stack(
        [r_arr * weight_arr, g_arr * weight_arr, b_arr * weight_arr], axis=1
    ).astype(np.float32)

    for dy in range(2):
        wy = fy if dy else (1.0 - fy)
        for dx in range(2):
            wx = fx if dx else (1.0 - fx)
            w = (wx * wy).astype(np.float32)
            xi = x0 + dx
            yi = y0 + dy
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            if not np.any(valid):
                continue
            np.add.at(
                accum,
                (yi[valid], xi[valid]),
                rgb[valid] * w[valid, np.newaxis],
            )


# ---------------------------------------------------------------------------
# Module-level dispatch: call Numba kernels if available, else NumPy
# ---------------------------------------------------------------------------
if _NUMBA_OK:

    def rk4_lorenz(
        pts: np.ndarray,
        sigma: float,
        rho: float,
        beta: float,
        dt: float,
        substeps: int,
    ) -> None:
        _rk4_lorenz_kernel(pts, sigma, rho, beta, dt, substeps)

    def rk4_rossler(
        pts: np.ndarray,
        a: float,
        b: float,
        c: float,
        dt: float,
        substeps: int,
    ) -> None:
        _rk4_rossler_kernel(pts, a, b, c, dt, substeps)

    def splat_glow(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        r_arr: np.ndarray,
        g_arr: np.ndarray,
        b_arr: np.ndarray,
        weight_arr: np.ndarray,
        accum: np.ndarray,
        H: int,
        W: int,
    ) -> None:
        _splat_glow_kernel(x_arr, y_arr, r_arr, g_arr, b_arr, weight_arr, accum, H, W)

else:  # pragma: no cover

    def rk4_lorenz(pts, sigma, rho, beta, dt, substeps):  # type: ignore[misc]
        _rk4_lorenz_numpy(pts, sigma, rho, beta, dt, substeps)

    def rk4_rossler(pts, a, b, c, dt, substeps):  # type: ignore[misc]
        _rk4_rossler_numpy(pts, a, b, c, dt, substeps)

    def splat_glow(x_arr, y_arr, r_arr, g_arr, b_arr, weight_arr, accum, H, W):  # type: ignore[misc]
        _splat_glow_numpy(x_arr, y_arr, r_arr, g_arr, b_arr, weight_arr, accum, H, W)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AttractorConfig(BaseConfig):
    """Configuration for the strange attractor renderer."""

    # Particle simulation
    num_particles: int = 3000
    lorenz_sigma: float = 10.0
    lorenz_rho: float = 28.0
    lorenz_beta: float = 2.6667
    rossler_a: float = 0.2
    rossler_b: float = 0.2
    rossler_c: float = 5.7

    # Rendering behaviour
    blend_mode: str = "dual"       # lorenz | rossler | dual | morph
    trail_decay: float = 0.96      # [0, 1] — how fast trails fade
    substeps: int = 6              # RK4 sub-steps per frame
    projection_speed: float = 0.2  # base azimuth rotation speed (rad/s)

    # Visuals
    attractor_palette: str = "neon_aurora"
    glow_radius: float = 1.5       # sigma for gaussian bloom pass (overrides BaseConfig int)
    particle_brightness: float = 1.2  # HDR exposure multiplier


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _rotation_matrix(az: float, el: float) -> np.ndarray:
    """3×3 rotation matrix: azimuth (around Y) then elevation (around X)."""
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    Ry = np.array(
        [[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float64
    )
    Rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]], dtype=np.float64
    )
    return Rx @ Ry


def _hsv_to_rgb_batch(
    h: np.ndarray, s: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Vectorized HSV → RGB for 1D arrays of N particles. Returns (N,3) float32."""
    h6 = (h * 6.0) % 6.0
    idx = h6.astype(np.int32)
    f = (h6 - idx).astype(np.float32)
    p = (v * (1.0 - s)).astype(np.float32)
    q = (v * (1.0 - s * f)).astype(np.float32)
    t = (v * (1.0 - s * (1.0 - f))).astype(np.float32)
    v = v.astype(np.float32)

    N = len(h)
    rgb = np.zeros((N, 3), dtype=np.float32)
    for mask_val, r_src, g_src, b_src in [
        (0, v, t, p),
        (1, q, v, p),
        (2, p, v, t),
        (3, p, q, v),
        (4, t, p, v),
        (5, v, p, q),
    ]:
        m = idx == mask_val
        if np.any(m):
            rgb[m, 0] = r_src[m]
            rgb[m, 1] = g_src[m]
            rgb[m, 2] = b_src[m]
    return rgb


class AttractorRenderer(BaseVisualizer):
    """
    Renders audio-reactive Lorenz and Rössler strange attractors.

    Particles orbit the attractors in 3D; their 3D positions are projected onto
    a 2D screen and splatted as colored blobs onto a persistent float32
    accumulation buffer. The buffer fades each frame (trail_decay) producing
    glowing trail-ribbons. HDR Reinhard tone-mapping and a gaussian bloom pass
    create the neon cinematic look.
    """

    _DT_SIM: float = 0.01  # simulation timestep per frame

    def __init__(
        self,
        config: Optional[AttractorConfig] = None,
        seed: Optional[int] = None,
        center_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(config or AttractorConfig(), seed, center_pos)
        self.cfg: AttractorConfig = self.cfg  # type annotation

        H, W = self.cfg.height, self.cfg.width
        N = self.cfg.num_particles

        # Particle positions — float64 for numerical stability
        self._lorenz_pts: np.ndarray = np.empty((N, 3), dtype=np.float64)
        self._rossler_pts: np.ndarray = np.empty((N, 3), dtype=np.float64)

        # Persistent HDR accumulation buffer (H, W, 3) float32
        self._accum: np.ndarray = np.zeros((H, W, 3), dtype=np.float32)

        # Projection state
        self._proj_az: float = 0.0
        self._proj_el: float = 0.3

        # Hue drift
        self._hue_offset: float = 0.0

        # Normalization constants (filled by _warmup_and_normalize)
        self._lorenz_norm: Tuple[np.ndarray, float] = (np.zeros(3), 1.0)
        self._rossler_norm: Tuple[np.ndarray, float] = (np.zeros(3), 1.0)

        # Initialise particles and compute normalization
        self._warmup_and_normalize()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _warmup_and_normalize(self) -> None:
        """Seed particles, run 2000 integration steps, compute normalization."""
        N = self.cfg.num_particles
        dt = self._DT_SIM

        # Lorenz: start near centre with small perturbations
        self._lorenz_pts[:] = self.rng.normal(0.0, 0.5, (N, 3))
        self._lorenz_pts[:, 2] += 25.0  # shift toward typical Z centre

        # Rössler: start near attractor entrance
        self._rossler_pts[:] = self.rng.normal(0.0, 0.5, (N, 3))
        self._rossler_pts[:, 2] += 3.0

        # Warm up — many steps to settle onto the attractor
        n_warmup = 2000
        rk4_lorenz(
            self._lorenz_pts,
            self.cfg.lorenz_sigma,
            self.cfg.lorenz_rho,
            self.cfg.lorenz_beta,
            dt,
            n_warmup,
        )
        rk4_rossler(
            self._rossler_pts,
            self.cfg.rossler_a,
            self.cfg.rossler_b,
            self.cfg.rossler_c,
            dt,
            n_warmup,
        )

        # Safety: reset any escaped particles
        self._fix_nans(self._lorenz_pts, np.array([0.0, 0.0, 25.0]))
        self._fix_nans(self._rossler_pts, np.array([0.0, 0.0, 3.0]))

        # Record normalization from current cloud bounds
        self._lorenz_norm = self._compute_norm(self._lorenz_pts)
        self._rossler_norm = self._compute_norm(self._rossler_pts)

    def _compute_norm(
        self, pts: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Return (center, scale) such that (pts - center)/scale ≈ [-1, 1]."""
        center = pts.mean(axis=0)
        scale = float(np.abs(pts - center).max()) * 1.1
        return center, max(scale, 1e-6)

    def _fix_nans(self, pts: np.ndarray, fallback: np.ndarray) -> None:
        """Reset any NaN/Inf rows to the fallback position."""
        bad = ~np.isfinite(pts).all(axis=1)
        n_bad = int(bad.sum())
        if n_bad:
            pts[bad] = fallback + self.rng.normal(0.0, 0.1, (n_bad, 3))

    def _reseed_fraction(
        self,
        pts: np.ndarray,
        frac: float,
        norm_center: np.ndarray,
        norm_scale: float,
    ) -> None:
        """Re-scatter `frac` fraction of particles near the attractor centre."""
        N = len(pts)
        n_reseed = max(1, int(N * frac))
        indices = self.rng.integers(0, N, n_reseed)
        pts[indices] = norm_center + self.rng.normal(
            0.0, norm_scale * 0.1, (n_reseed, 3)
        )

    # ------------------------------------------------------------------
    # Audio-driven parameter modulation
    # ------------------------------------------------------------------

    def _modulate_params(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Return audio-modulated (sigma, rho, beta, a, b, c)."""
        cfg = self.cfg

        # Lorenz: sub_bass → σ ±40%, percussive → ρ ±30%
        sigma = cfg.lorenz_sigma * (0.6 + self._smooth_sub_bass * 0.8)
        rho = cfg.lorenz_rho * (0.7 + self._smooth_percussive * 0.6)
        beta = cfg.lorenz_beta  # keep stable

        # Rössler: harmonic → a ±150% (clamped positive)
        a = cfg.rossler_a * (0.5 + self._smooth_harmonic * 2.5)
        b = cfg.rossler_b
        c = cfg.rossler_c

        # Clamp to numerically safe chaotic regimes
        sigma = float(np.clip(sigma, 2.0, 25.0))
        rho = float(np.clip(rho, 10.0, 45.0))
        beta = float(np.clip(beta, 0.5, 5.0))
        a = float(np.clip(a, 0.05, 0.45))
        b = float(np.clip(b, 0.05, 0.4))
        c = float(np.clip(c, 2.0, 10.0))

        return sigma, rho, beta, a, b, c

    # ------------------------------------------------------------------
    # 3-D projection and colour
    # ------------------------------------------------------------------

    def _project_3d_2d(
        self,
        pts: np.ndarray,
        norm: Tuple[np.ndarray, float],
        az: float,
        el: float,
        W: int,
        H: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalise → rotate → map to pixel coords.

        Returns:
            x_px, y_px : float64 pixel coordinates
            z_depth    : float32 depth in [0, 1]
        """
        center, scale = norm
        pts_n = (pts - center) / scale  # (N, 3), roughly in [-1, 1]

        rot = _rotation_matrix(az, el)
        pts_rot = pts_n @ rot.T  # (N, 3)

        # Scale to maintain aspect ratio (fit within shorter screen dimension)
        half_side = min(W, H) / 2.0 * 0.85
        cx = W / 2.0
        cy = H / 2.0

        x_px = cx + pts_rot[:, 0] * half_side
        y_px = cy - pts_rot[:, 1] * half_side  # flip Y for screen coords
        z_depth = np.clip((pts_rot[:, 2] + 1.0) / 2.0, 0.0, 1.0).astype(
            np.float32
        )
        return x_px, y_px, z_depth

    def _compute_colors(
        self,
        z_depth: np.ndarray,
        attractor_type: str,
        palette_name: str,
    ) -> np.ndarray:
        """Map z_depth → neon RGB via palette gradient + hue offset.

        Returns (N, 3) float32 RGB in [0, 1].
        """
        pal = _ATTRACTOR_PALETTES.get(
            palette_name, _ATTRACTOR_PALETTES["neon_aurora"]
        )[attractor_type]
        h0, h1, s_val, v_val = pal
        N = len(z_depth)
        hue = ((h0 + (h1 - h0) * z_depth + self._hue_offset) % 1.0).astype(
            np.float32
        )
        s = np.full(N, s_val, dtype=np.float32)
        v = np.full(N, v_val, dtype=np.float32)
        return _hsv_to_rgb_batch(hue, s, v)

    # ------------------------------------------------------------------
    # Splat helper
    # ------------------------------------------------------------------

    def _project_and_splat(
        self,
        pts: np.ndarray,
        norm: Tuple[np.ndarray, float],
        attractor_type: str,
        palette_name: str,
        H: int,
        W: int,
    ) -> None:
        """Project particles to 2D, compute colours, splat onto _accum."""
        x_px, y_px, z_depth = self._project_3d_2d(
            pts, norm, self._proj_az, self._proj_el, W, H
        )
        rgb = self._compute_colors(z_depth, attractor_type, palette_name)
        # Depth-based brightness: front particles (z→1) are brighter
        weights = (
            self.cfg.particle_brightness * (0.3 + 0.7 * z_depth)
        ).astype(np.float32)

        splat_glow(
            x_px.astype(np.float64),
            y_px.astype(np.float64),
            rgb[:, 0],
            rgb[:, 1],
            rgb[:, 2],
            weights,
            self._accum,
            H,
            W,
        )

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def update(self, frame_data: Dict[str, Any]) -> None:
        """Advance simulation by one frame: integrate, project, splat."""
        dt = 1.0 / self.cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)

        is_beat = frame_data.get("is_beat", False)
        H, W = self.cfg.height, self.cfg.width
        palette = self.cfg.attractor_palette
        mode = self.cfg.blend_mode
        dt_sim = self._DT_SIM

        # 1. Fade accumulation buffer
        # Modulate trail decay: higher high_energy → slightly faster fade
        decay = self.cfg.trail_decay * (1.0 - self._smooth_high * 0.04)
        self._accum *= float(np.clip(decay, 0.8, 0.999))

        # 2. Modulate attractor parameters
        sigma, rho, beta, a, b, c = self._modulate_params()

        # 3. Integrate particles
        if mode in ("lorenz", "dual", "morph"):
            rk4_lorenz(
                self._lorenz_pts,
                sigma,
                rho,
                beta,
                dt_sim / self.cfg.substeps,
                self.cfg.substeps,
            )
            self._fix_nans(
                self._lorenz_pts, self._lorenz_norm[0]
            )

        if mode in ("rossler", "dual", "morph"):
            rk4_rossler(
                self._rossler_pts,
                a,
                b,
                c,
                dt_sim / self.cfg.substeps,
                self.cfg.substeps,
            )
            self._fix_nans(
                self._rossler_pts, self._rossler_norm[0]
            )

        # 4. Re-seed particles on beat or high brilliance
        reseed_frac = 0.0
        if is_beat:
            reseed_frac = 0.10
        elif self._smooth_brilliance > 0.4:
            reseed_frac = self._smooth_brilliance * 0.05
        if reseed_frac > 0:
            if mode in ("lorenz", "dual", "morph"):
                self._reseed_fraction(
                    self._lorenz_pts, reseed_frac, *self._lorenz_norm
                )
            if mode in ("rossler", "dual", "morph"):
                self._reseed_fraction(
                    self._rossler_pts, reseed_frac, *self._rossler_norm
                )

        # 5. Update projection angles
        rot_speed = self.cfg.projection_speed * (
            1.0 + self._smooth_energy * 2.0
        )
        self._proj_az += rot_speed * dt
        el_wobble = self._smooth_flux * 0.3
        el_target = 0.3 + el_wobble
        self._proj_el = self._lerp(self._proj_el, el_target, 0.05)

        # 6. Hue drift from spectral centroid
        centroid = frame_data.get("spectral_centroid", 0.5)
        self._hue_offset = (self._hue_offset + centroid * 0.003) % 1.0

        # 7. Project and splat
        if mode == "lorenz":
            self._project_and_splat(
                self._lorenz_pts, self._lorenz_norm, "lorenz", palette, H, W
            )
        elif mode == "rossler":
            self._project_and_splat(
                self._rossler_pts, self._rossler_norm, "rossler", palette, H, W
            )
        elif mode == "dual":
            self._project_and_splat(
                self._lorenz_pts, self._lorenz_norm, "lorenz", palette, H, W
            )
            self._project_and_splat(
                self._rossler_pts, self._rossler_norm, "rossler", palette, H, W
            )
        elif mode == "morph":
            blend_weight = float(np.clip(self._smooth_flatness, 0.0, 1.0))
            lc, ls = self._lorenz_norm
            rc, rs = self._rossler_norm
            lorenz_n = (self._lorenz_pts - lc) / ls
            rossler_n = (self._rossler_pts - rc) / rs
            N_pts = min(len(lorenz_n), len(rossler_n))
            morph_n = (
                (1.0 - blend_weight) * lorenz_n[:N_pts]
                + blend_weight * rossler_n[:N_pts]
            )
            # Project the already-normalized morph points directly
            rot = _rotation_matrix(self._proj_az, self._proj_el)
            pts_rot = morph_n @ rot.T
            half_side = min(W, H) / 2.0 * 0.85
            x_px = W / 2.0 + pts_rot[:, 0] * half_side
            y_px = H / 2.0 - pts_rot[:, 1] * half_side
            z_depth = np.clip(
                (pts_rot[:, 2] + 1.0) / 2.0, 0.0, 1.0
            ).astype(np.float32)

            # Blend colours between both attractor palettes
            rgb_l = self._compute_colors(z_depth, "lorenz", palette)
            rgb_r = self._compute_colors(z_depth, "rossler", palette)
            rgb = (
                (1.0 - blend_weight) * rgb_l + blend_weight * rgb_r
            ).astype(np.float32)
            weights = (
                self.cfg.particle_brightness * (0.3 + 0.7 * z_depth)
            ).astype(np.float32)
            splat_glow(
                x_px.astype(np.float64),
                y_px.astype(np.float64),
                rgb[:, 0],
                rgb[:, 1],
                rgb[:, 2],
                weights,
                self._accum,
                H,
                W,
            )

    def get_raw_field(self) -> np.ndarray:
        """Return luminance of accumulation buffer (for BaseVisualizer compat)."""
        return np.clip(self._accum.mean(axis=2), 0.0, 1.0).astype(np.float32)

    def render_frame(
        self, frame_data: Dict[str, Any], frame_index: int
    ) -> np.ndarray:
        """Override: HDR bloom + Reinhard tone-map → uint8."""
        self.update(frame_data)
        is_beat = frame_data.get("is_beat", False)

        # Bloom pass: gaussian blur of float32 HDR accum
        glow_sigma = float(self.cfg.glow_radius)
        bloom = gaussian_filter(
            self._accum, sigma=[glow_sigma, glow_sigma, 0]
        )

        # Composite: base + bloom
        composite = self._accum + bloom

        # Reinhard HDR tone mapping with exposure
        exposure = self.cfg.particle_brightness
        if is_beat:
            exposure *= 1.8
        mapped = (composite * exposure) / (1.0 + composite * exposure)

        # uint8 conversion
        out = np.clip(mapped * 255.0, 0, 255).astype(np.uint8)

        # Post-processing
        if self.cfg.vignette_strength > 0:
            out = vignette(out, strength=self.cfg.vignette_strength)
        if self.cfg.aberration_enabled and self.cfg.aberration_offset > 0:
            out = chromatic_aberration(out, offset=self.cfg.aberration_offset)

        return out
