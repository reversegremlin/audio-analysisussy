"""
Audio-reactive chemical reaction and crystallization renderer.
Implements the chromascope-chemical experiment mode (CHEMICAL_CRYSTALLIZATION_PRD).

Simulation layers (all float32 [0,1], reduced-grid for performance):
  - Reagent fields A, B  — Gray-Scott reaction-diffusion
  - Reaction heat field  — activation/intensity front (A × B)
  - Crystal mass field   — nucleated + grown from heat zones
  - Crystal edge field   — gradient magnitude of crystal body (sparkle)
  - Impurity noise field — slow vectorised fBm imperfections

Audio mapping:
  - sub_bass / beat → reagent injection pulses + nucleation seeds
  - percussive_impact → nucleation burst density
  - energy / low      → reaction front width and propagation speed
  - high / brilliance → crystal edge sharpness and scintillation
  - spectral_flatness → orderliness (high flatness = coherent lattice)
  - harmonic          → supersaturation / branching propensity
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from chromascope.experiment.base import BaseConfig, BaseVisualizer
from chromascope.experiment.colorgrade import (
    add_glow,
    chromatic_aberration,
    tone_map_soft,
    vignette,
)

# ---------------------------------------------------------------------------
# Palette anchors (normalised RGB, neon-pushed)
# ---------------------------------------------------------------------------
_PALETTES: Dict[str, Tuple[Tuple, Tuple, Tuple]] = {
    # (heat_rgb, crystal_rgb, edge_rgb)
    "iron":      ((1.00, 0.27, 0.00), (0.55, 0.10, 0.00), (1.00, 0.95, 0.80)),
    "copper":    ((0.00, 1.00, 0.80), (0.00, 0.40, 0.40), (0.50, 1.00, 1.00)),
    "sodium":    ((1.00, 0.87, 0.00), (0.60, 0.40, 0.00), (1.00, 1.00, 0.80)),
    "potassium": ((0.61, 0.15, 0.80), (0.29, 0.05, 0.40), (0.90, 0.70, 1.00)),
    # "mixed" is resolved dynamically per-frame from audio centroid
}


@dataclass
class ChemicalConfig(BaseConfig):
    """Configuration for the chemical reaction / crystallization renderer."""

    # Visual style preset
    style: str = "neon_lab"          # neon_lab | plasma_beaker | midnight_fluor | synth_chem

    # Simulation knobs
    reaction_gain: float = 1.0       # [0,2] scales reaction front intensity
    crystal_rate: float = 1.0        # [0,2] baseline crystal growth speed
    nucleation_threshold: float = 0.3  # [0,1] percussive sensitivity for seed creation
    supersaturation: float = 0.5     # [0,1] baseline branching propensity
    bloom: float = 1.0               # [0,2] post-glow multiplier

    # Palette
    chem_palette: str = "mixed"      # iron | copper | sodium | potassium | mixed


class ChemicalRenderer(BaseVisualizer):
    """
    Renders audio-reactive chemical reactions and crystallization.

    Internally operates on a *reduced* simulation grid
    (sim_w × sim_h ≈ width/4 × height/4) for performance,
    then upscales to output resolution in render_frame().
    """

    # Gray-Scott diffusion constants (default: coral/finger-like patterns)
    _Da: float = 0.16   # Reagent A diffusivity
    _Db: float = 0.08   # Reagent B diffusivity

    def __init__(
        self,
        config: Optional[ChemicalConfig] = None,
        seed: Optional[int] = None,
        center_pos: Optional[Tuple[float, float]] = None,
    ):
        super().__init__(config or ChemicalConfig(), seed, center_pos)
        self.cfg: ChemicalConfig = self.cfg  # type annotation

        # Reduced simulation grid (¼ of output, minimum 16 px)
        self._sim_w = max(16, self.cfg.width // 4)
        self._sim_h = max(16, self.cfg.height // 4)
        sw, sh = self._sim_w, self._sim_h

        # Gray-Scott reagent fields: A starts near 1 (empty solution), B near 0
        self._field_a = np.ones((sh, sw), dtype=np.float32)
        self._field_b = np.zeros((sh, sw), dtype=np.float32)

        # Reaction heat, crystal body, crystal edges, impurity noise
        self._field_heat = np.zeros((sh, sw), dtype=np.float32)
        self._field_crystal = np.zeros((sh, sw), dtype=np.float32)
        self._field_crystal_sharp = np.zeros((sh, sw), dtype=np.float32)
        self._field_edge = np.zeros((sh, sw), dtype=np.float32)
        self._field_noise = np.zeros((sh, sw), dtype=np.float32)

        # Noise animation state
        self._noise_offsets = [float(self.rng.uniform(0, 1000)) for _ in range(6)]

        # Hex lattice phase (random start per seed for variety)
        self._hex_phase = float(self.rng.uniform(0, 10))

        # Style-specific parameters resolved once
        self._style_params = self._resolve_style(self.cfg.style)

        # Supersaturation smooth state
        self._smooth_supersat = float(self.cfg.supersaturation)

        # Seed the Gray-Scott field with a few perturbation patches
        self._init_gs_seed()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_style(self, style: str) -> Dict[str, Any]:
        """Return style-dependent parameter overrides."""
        presets = {
            "neon_lab": {
                "gs_feed": 0.037, "gs_kill": 0.060,
                "heat_decay": 0.88, "crystal_decay": 0.004,
                "glow_boost": 1.0,
            },
            "plasma_beaker": {
                "gs_feed": 0.055, "gs_kill": 0.062,
                "heat_decay": 0.80, "crystal_decay": 0.006,
                "glow_boost": 1.5,
            },
            "midnight_fluor": {
                "gs_feed": 0.025, "gs_kill": 0.057,
                "heat_decay": 0.93, "crystal_decay": 0.002,
                "glow_boost": 0.7,
            },
            "synth_chem": {
                "gs_feed": 0.042, "gs_kill": 0.059,
                "heat_decay": 0.85, "crystal_decay": 0.003,
                "glow_boost": 1.2,
            },
        }
        return presets.get(style, presets["neon_lab"])

    def _init_gs_seed(self) -> None:
        """Seed Gray-Scott field with small square perturbations."""
        sw, sh = self._sim_w, self._sim_h
        n_seeds = max(3, sw * sh // 800)
        for _ in range(n_seeds):
            cx = int(self.rng.integers(5, sw - 5))
            cy = int(self.rng.integers(5, sh - 5))
            r = int(self.rng.integers(2, 5))
            self._field_b[
                max(0, cy - r):cy + r,
                max(0, cx - r):cx + r,
            ] = self.rng.uniform(0.4, 0.8)
            self._field_a[
                max(0, cy - r):cy + r,
                max(0, cx - r):cx + r,
            ] = 1.0 - self._field_b[
                max(0, cy - r):cy + r,
                max(0, cx - r):cx + r,
            ]

    def _laplacian(self, field: np.ndarray, sigma: float) -> np.ndarray:
        """Approximate Laplacian via Gaussian blur difference."""
        return gaussian_filter(field, sigma=sigma) - field

    def _fBm_noise(self, scale: float, offset_idx: int) -> np.ndarray:
        """Vectorised 2-D fractional Brownian motion on the sim grid."""
        sh, sw = self._sim_h, self._sim_w
        y_n = np.linspace(0, scale, sh, dtype=np.float32)
        x_n = np.linspace(0, scale, sw, dtype=np.float32)
        xg, yg = np.meshgrid(x_n, y_n)

        layer = np.zeros((sh, sw), dtype=np.float32)
        amp = 1.0
        freq = 1.0
        off = self._noise_offsets
        for i in range(4):
            ph = off[(offset_idx + i) % len(off)]
            t = self.time * freq * 0.08
            layer += (
                np.sin(xg * freq * math.tau + ph + t) * amp
                + np.cos(yg * freq * math.tau + ph * 1.618 + t * 0.7) * amp
            )
            amp *= 0.5
            freq *= 2.0

        # Normalise to [0, 1]
        layer = (layer + 4.0) / 8.0
        return np.clip(layer, 0.0, 1.0).astype(np.float32)

    def _hex_lattice_full(self, scale: float = 10.0) -> np.ndarray:
        """Hexagonal lattice at full output resolution for crystal interior texture.

        Three cosine waves at 0°, 60°, 120° tile a true hexagonal grid.
        Returns float32 [0, 1] — peaks mark facet centres, troughs mark crystal planes.
        """
        h, w = self.cfg.height, self.cfg.width
        x = np.linspace(0, scale, w, dtype=np.float32) + self._hex_phase
        y = np.linspace(0, scale * h / w, h, dtype=np.float32) + self._hex_phase * 0.618
        xg, yg = np.meshgrid(x, y)

        tau = math.tau
        a1 = np.cos(tau * xg)
        a2 = np.cos(tau * (0.5 * xg + 0.866 * yg))
        a3 = np.cos(tau * (-0.5 * xg + 0.866 * yg))
        lattice = (a1 + a2 + a3) / 3.0  # [-1, 1]
        # Sharpen planes so facet centres are clearly bright, planes clearly dark
        lattice = np.tanh(lattice * 2.5)
        return ((lattice + 1.0) * 0.5).astype(np.float32)

    def _palette_colors(
        self, smoothed: Dict[str, float]
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """Return (heat_rgb, crystal_rgb, edge_rgb) for the current palette."""
        p = self.cfg.chem_palette
        if p != "mixed":
            return _PALETTES.get(p, _PALETTES["iron"])

        # "mixed": blend palettes based on spectral centroid + time
        names = ["iron", "copper", "sodium", "potassium"]
        idx_f = (smoothed["centroid"] * 3.0 + self.time * 0.03) % 4.0
        i0 = int(idx_f) % 4
        i1 = (i0 + 1) % 4
        frac = idx_f - int(idx_f)
        h0, c0, e0 = _PALETTES[names[i0]]
        h1, c1, e1 = _PALETTES[names[i1]]
        blend = lambda a, b: tuple(a[k] * (1 - frac) + b[k] * frac for k in range(3))
        return blend(h0, h1), blend(c0, c1), blend(e0, e1)

    # ------------------------------------------------------------------
    # Simulation update
    # ------------------------------------------------------------------

    def update(self, frame_data: Dict[str, Any]) -> None:
        """Advance the chemical simulation by one frame."""
        dt = 1.0 / self.cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)

        sp = self._style_params
        cfg = self.cfg
        is_beat = frame_data.get("is_beat", False)

        # --- Supersaturation state (drives branching) ---
        target_ss = cfg.supersaturation + self._smooth_harmonic * 0.5
        self._smooth_supersat = self._lerp(self._smooth_supersat, target_ss, 0.04)
        ss = np.clip(self._smooth_supersat, 0.0, 1.0)

        # --- Audio-modulated Gray-Scott parameters ---
        feed = sp["gs_feed"] * (1.0 + self._smooth_sub_bass * cfg.reaction_gain * 0.8)
        kill = sp["gs_kill"] * (1.0 - self._smooth_harmonic * 0.15)

        # --- Reagent injection pulses (kick / beat) ---
        if is_beat or self._smooth_sub_bass > 0.4:
            n_injections = int(1 + self._smooth_sub_bass * cfg.reaction_gain * 3)
            for _ in range(n_injections):
                ix = int(self.rng.integers(2, self._sim_w - 2))
                iy = int(self.rng.integers(2, self._sim_h - 2))
                r = int(1 + self._smooth_sub_bass * 3)
                pulse = float(0.4 + self._smooth_sub_bass * 0.6)
                self._field_b[
                    max(0, iy - r):iy + r,
                    max(0, ix - r):ix + r,
                ] = np.clip(
                    self._field_b[
                        max(0, iy - r):iy + r,
                        max(0, ix - r):ix + r,
                    ] + pulse, 0, 1
                )
                self._field_a[
                    max(0, iy - r):iy + r,
                    max(0, ix - r):ix + r,
                ] = 1.0 - self._field_b[
                    max(0, iy - r):iy + r,
                    max(0, ix - r):ix + r,
                ]

        # --- Gray-Scott reaction-diffusion step ---
        lap_a = self._laplacian(self._field_a, sigma=1.5)
        lap_b = self._laplacian(self._field_b, sigma=1.0)
        reaction = self._field_a * self._field_b ** 2

        da = self._Da * lap_a - reaction + feed * (1.0 - self._field_a)
        db = self._Db * lap_b + reaction - (feed + kill) * self._field_b

        # Reaction front width controlled by RMS energy
        speed = 1.0 + self._smooth_energy * 2.0 * cfg.reaction_gain
        self._field_a = np.clip(self._field_a + da * speed, 0.0, 1.0)
        self._field_b = np.clip(self._field_b + db * speed, 0.0, 1.0)

        # --- Reaction heat: where A and B co-exist ---
        raw_heat = self._field_a * self._field_b * (2.0 + self._smooth_energy * 2.0)
        raw_heat = np.clip(raw_heat, 0.0, 1.0)
        # Energy-gated persistence: heat dies faster in quiet passages
        heat_persist = max(0.5, sp["heat_decay"] - (1.0 - self._smooth_energy) * 0.30)
        self._field_heat = self._field_heat * heat_persist + raw_heat * (1.0 - heat_persist)
        heat_sigma = 1.0 + self._smooth_low * 3.0
        self._field_heat = gaussian_filter(self._field_heat, sigma=heat_sigma)
        self._field_heat = np.clip(self._field_heat, 0.0, 1.0)

        # --- Nucleation bursts (snare / transient onsets) ---
        if self._smooth_percussive > cfg.nucleation_threshold:
            burst_density = (self._smooth_percussive - cfg.nucleation_threshold) / (
                1.0 - cfg.nucleation_threshold + 1e-6
            )
            n_seeds = int(burst_density * 6 * cfg.crystal_rate + 1)
            for _ in range(n_seeds):
                sx = int(self.rng.integers(1, self._sim_w - 1))
                sy = int(self.rng.integers(1, self._sim_h - 1))
                seed_r = max(1, int(1 + ss * 2))
                self._field_crystal[
                    max(0, sy - seed_r):sy + seed_r,
                    max(0, sx - seed_r):sx + seed_r,
                ] = 1.0

        # --- Crystal growth: 3-axis directional for faceted star/dendrite shape ---
        # Taking the MAX of three directional blurs (X, Y, diagonal) creates
        # a star/diamond growth pattern instead of an isotropic blob.
        flatness = self._smooth_flatness
        sigma_long = cfg.crystal_rate * (0.6 + ss * 2.0)
        sigma_short = max(0.15, cfg.crystal_rate * 0.15)

        grown_x = gaussian_filter(self._field_crystal, sigma=[sigma_short, sigma_long])
        grown_y = gaussian_filter(self._field_crystal, sigma=[sigma_long, sigma_short])
        # Diagonal arm (approximates the 45°/135° crystal axes)
        sigma_d = sigma_long * 0.75
        grown_d = gaussian_filter(self._field_crystal, sigma=[sigma_d, sigma_d])
        grown_dir = np.maximum(np.maximum(grown_x, grown_y), grown_d)

        # Blend toward isotropic on tonal passages (coherent lattice body)
        grown_iso = gaussian_filter(self._field_crystal, sigma=sigma_long * 0.5)
        grown = grown_dir * (1.0 - flatness * 0.5) + grown_iso * (flatness * 0.5)

        # Only grow where heat is above threshold (raised in quiet passages)
        heat_thresh = 0.10 + (1.0 - self._smooth_energy) * 0.25
        growth_mask = self._field_heat > heat_thresh
        growth_amount = grown * (0.6 + cfg.crystal_rate * 0.4)
        self._field_crystal = np.where(
            growth_mask,
            np.maximum(self._field_crystal, growth_amount),
            self._field_crystal,
        )

        # Partial dissolution — much stronger in quiet passages
        dissolution = sp["crystal_decay"] * (2.0 + (1.0 - self._smooth_energy) * 5.0)
        self._field_crystal = np.clip(self._field_crystal - dissolution, 0.0, 1.0)

        # --- Sharpen crystal field → hard faceted boundaries ---
        # Logistic curve pushes mid-range values toward 0 or 1.
        # Sharpness scales with spectral flatness: tonal music = harder crystal faces.
        crystal_sharpness = 5.0 + flatness * 8.0
        self._field_crystal_sharp = (
            1.0 / (1.0 + np.exp(-crystal_sharpness * (self._field_crystal - 0.30)))
        ).astype(np.float32)

        # --- Crystal edge field on the SHARPENED crystal for crisp facet lines ---
        from scipy.ndimage import sobel as _sobel
        sx_grad = _sobel(self._field_crystal_sharp, axis=1)
        sy_grad = _sobel(self._field_crystal_sharp, axis=0)
        edge_mag = np.sqrt(sx_grad ** 2 + sy_grad ** 2)

        # Facet alignment: cos(6θ) peaks every 30° → matches hexagonal symmetry.
        # Edges aligned to crystallographic planes receive a brightness bonus.
        angle_map = np.arctan2(sy_grad, sx_grad + 1e-8)
        hex_align = (np.cos(6.0 * angle_map) + 1.0) * 0.5  # [0, 1]
        edge_boost = 1.5 + self._smooth_brilliance * 4.0 + self._smooth_high * 2.0
        self._field_edge = np.clip(
            edge_mag * (1.5 + hex_align * 3.0) * edge_boost, 0.0, 1.0
        ).astype(np.float32)

        # --- Advance hex lattice phase (slow drift animates crystal planes) ---
        self._hex_phase += dt * 0.25 * (1.0 + self._smooth_harmonic * 0.4)

        # --- Impurity noise field (slow fBm drift) ---
        self._field_noise = self._fBm_noise(scale=3.0 + ss, offset_idx=0) * 0.3

    # ------------------------------------------------------------------
    # Field composition
    # ------------------------------------------------------------------

    def get_raw_field(self) -> np.ndarray:
        """Return merged normalised energy map (float32, sim-grid size)."""
        combined = (
            self._field_heat * 0.40
            + self._field_crystal_sharp * 0.30
            + self._field_edge * 0.25
            + self._field_noise * 0.05
        )
        return np.clip(combined, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Custom render path (chemistry-informed neon palette)
    # ------------------------------------------------------------------

    def render_frame(self, frame_data: Dict[str, Any], frame_index: int) -> np.ndarray:
        """Override to apply chemistry-informed neon color grading."""
        self.update(frame_data)
        smoothed = self._smooth_audio_dict()

        # Upscale sim fields to output resolution
        heat = self._upscale(self._field_heat)
        edge = self._upscale(self._field_edge)
        noise = self._upscale(self._field_noise)

        # Sharpen crystal at full output resolution for hard facet boundaries.
        # (Re-applying logistic at full res gives cleaner results than upscaling
        # the sim-grid sharpened field, since boundary aliasing is avoided.)
        crystal_smooth = self._upscale(self._field_crystal)
        cs = 5.0 + self._smooth_flatness * 8.0
        crystal_sharp = (
            1.0 / (1.0 + np.exp(-cs * (crystal_smooth - 0.30)))
        ).astype(np.float32)

        # Apply chemical palette
        frame_rgb = self._apply_chemical_palette(heat, crystal_sharp, edge, noise, smoothed)

        # Post-processing (standard pipeline)
        cfg = self.cfg
        if cfg.glow_enabled:
            glow_int = cfg.glow_intensity * cfg.bloom * (
                1.0 + smoothed["percussive"] * 0.4 + smoothed["flux"] * 0.5
            )
            glow_int = min(glow_int, 1.2)
            frame_rgb = add_glow(frame_rgb, intensity=glow_int, radius=cfg.glow_radius)

        if cfg.aberration_enabled:
            ab_offset = int(
                cfg.aberration_offset
                * (1.0 + smoothed["percussive"] * 1.5 + smoothed["sharpness"] * 2.0)
            )
            frame_rgb = chromatic_aberration(frame_rgb, offset=ab_offset)

        if cfg.vignette_strength > 0:
            vign_str = cfg.vignette_strength * (
                1.0 + smoothed["low"] * 0.4 + smoothed["sub_bass"] * 0.8
            )
            frame_rgb = vignette(frame_rgb, strength=vign_str)

        return tone_map_soft(frame_rgb)

    def _upscale(self, field: np.ndarray) -> np.ndarray:
        """Upscale sim-grid field to output resolution using PIL BILINEAR."""
        if field.shape == (self.cfg.height, self.cfg.width):
            return field
        from PIL import Image
        img = Image.fromarray(field, mode="F")
        img = img.resize((self.cfg.width, self.cfg.height), Image.BILINEAR)
        return np.array(img, dtype=np.float32)

    def _apply_chemical_palette(
        self,
        heat: np.ndarray,
        crystal_sharp: np.ndarray,
        edge: np.ndarray,
        noise: np.ndarray,
        smoothed: Dict[str, float],
    ) -> np.ndarray:
        """Composite field layers into a chemistry-inspired neon RGB frame.

        crystal_sharp is the logistically-sharpened crystal field (hard facet
        boundaries). The hexagonal lattice is composited over it to give the
        crystal body visible internal crystallographic plane structure.
        """
        heat_col, crystal_col, edge_col = self._palette_colors(smoothed)

        h, w = heat.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        # --- Hexagonal lattice (full output resolution) ---
        # Scale varies with harmonics (more tonal = finer lattice planes visible)
        # and supersaturation (higher branching = broader hex cells)
        lattice_scale = 9.0 + smoothed["harmonic"] * 6.0 + self._smooth_supersat * 4.0
        lattice = self._hex_lattice_full(scale=lattice_scale)

        # --- Layer 1: Crystal interior with hex lattice planes ---
        # Dark at crystal-plane boundaries (troughs), bright at facet centres (peaks).
        # This gives the translucent depth of a real mineral specimen.
        crystal_interior = crystal_sharp * (0.15 + 0.85 * lattice)
        for ch, val in enumerate(crystal_col):
            rgb[:, :, ch] += crystal_interior * val * 0.9

        # --- Layer 2: Reaction heat — brighter where it's inside a crystal ---
        # Heat trapped inside a crystal structure glows more intensely.
        heat_inside_boost = 1.0 + crystal_sharp * 0.5
        heat_boost = (1.0 + smoothed["high"] * 0.6) * heat_inside_boost
        for ch, val in enumerate(heat_col):
            rgb[:, :, ch] += heat * val * heat_boost

        # --- Layer 3: Crystal facet edges (very bright) ---
        # These are the light-catching rims where two crystal planes meet.
        # Much higher gain than before so facets truly blaze.
        edge_gain = 3.0 + smoothed["brilliance"] * 6.0 + smoothed["sharpness"] * 3.0
        for ch, val in enumerate(edge_col):
            rgb[:, :, ch] += edge * val * edge_gain

        # --- Layer 4: Specular vertex sparks ---
        # Tiny near-white flashes at crystal vertices (where multiple facets meet).
        # These exist at the intersection of high edge AND high crystal_sharp.
        # Specular highlights are always near-white regardless of palette.
        spark_raw = edge * crystal_sharp
        spark_intensity = (
            smoothed["brilliance"] * 0.8 + smoothed["sharpness"] * 0.5 + 0.1
        )
        sparks = np.clip(spark_raw * spark_intensity * 4.0, 0.0, 1.0)
        rgb[:, :, 0] += sparks * 0.95
        rgb[:, :, 1] += sparks * 0.98
        rgb[:, :, 2] += sparks * 1.00

        # --- Layer 5: Impurity noise (subtle colour texture, prevents dead zones) ---
        rgb[:, :, 0] += noise * 0.04
        rgb[:, :, 1] += noise * 0.02
        rgb[:, :, 2] += noise * 0.06

        # synth_chem: rotate hue continuously with centroid
        if self.cfg.style == "synth_chem":
            rgb = self._hue_rotate(rgb, shift=self.time * 0.05 + smoothed["centroid"] * 0.3)

        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255).astype(np.uint8)

    @staticmethod
    def _hue_rotate(rgb: np.ndarray, shift: float) -> np.ndarray:
        """Rotate hue of an RGB [0,1] array by `shift` (0–1 full circle)."""
        # Convert to HSV, shift H, convert back — all vectorised
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin + 1e-8

        # Hue
        h = np.where(
            cmax == r, (g - b) / delta % 6.0,
            np.where(cmax == g, (b - r) / delta + 2.0, (r - g) / delta + 4.0),
        ) / 6.0
        h = (h + shift) % 1.0

        # Saturation
        s = np.where(cmax < 1e-8, 0.0, delta / (cmax + 1e-8))
        v = cmax

        # HSV → RGB
        h6 = h * 6.0
        i = h6.astype(np.int32)
        f = h6 - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        out = np.zeros_like(rgb)
        for mi, rv, gv, bv in [
            (i == 0, v, t, p), (i == 1, q, v, p), (i == 2, p, v, t),
            (i == 3, p, q, v), (i == 4, t, p, v), (i == 5, v, p, q),
        ]:
            out[:, :, 0] = np.where(mi, rv, out[:, :, 0])
            out[:, :, 1] = np.where(mi, gv, out[:, :, 1])
            out[:, :, 2] = np.where(mi, bv, out[:, :, 2])

        return np.clip(out, 0.0, 1.0)
