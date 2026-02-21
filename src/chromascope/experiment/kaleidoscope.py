"""
Kaleidoscope symmetry and infinite zoom engine.

Polar coordinate remapping for N-way radial mirrors,
plus feedback buffer blending for the infinite zoom effect.

Performance notes:
  • _build_polar_remap caches (r, base_theta) keyed by (width, height) —
    eliminates sqrt + arctan2 over every pixel on every frame.
  • flow_field_warp uses a Numba fused kernel when available — both fBm
    fields are computed in one parallel pass, halving memory traffic.
"""

import math
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Optional Numba acceleration (same try-import pattern as fractal.py)
# ---------------------------------------------------------------------------
try:
    import numba as _numba

    _NUMBA_OK: bool = True
except ImportError:  # pragma: no cover
    _numba = None  # type: ignore[assignment]
    _NUMBA_OK = False


if _NUMBA_OK:
    @_numba.njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
    def _fbm_2fields_kernel(
        dx_out: np.ndarray,
        dy_out: np.ndarray,
        scale: float,
        time: float,
        octaves: int,
        h: int,
        w: int,
    ) -> None:
        """Fused parallel fBm kernel filling dx and dy displacement fields.

        Computes both fields in a single pass over all pixels, eliminating
        6 temporary H×W allocations and the Python octave-loop overhead.
        """
        _2pi = 2.0 * math.pi
        for i in _numba.prange(h):  # type: ignore[attr-defined]
            y = i / (h - 1) * scale if h > 1 else 0.0
            for j in range(w):
                x = j / (w - 1) * scale if w > 1 else 0.0
                dx = 0.0
                dy = 0.0
                amp = 1.0
                for k in range(octaves):
                    freq = 2.0 ** k
                    t_x = time * (k + 1) * 0.30
                    t_y = time * (k + 1) * 0.30 + 1.7
                    # dx field: sin in x + cos in y (perpendicular pair)
                    dx += math.sin(x * freq * _2pi + t_x) * amp
                    dx += math.cos(y * freq * _2pi + t_x * 0.71) * (amp * 0.7)
                    # dy field: cos in x + sin in y
                    dy += math.cos(x * freq * _2pi + t_y) * (amp * 0.7)
                    dy += math.sin(y * freq * _2pi + t_y * 1.31) * amp
                    amp *= 0.5
                dx_out[i, j] = dx
                dy_out[i, j] = dy


def _warmup_flow_jit() -> None:
    """Trigger JIT compilation on a 4×4 dummy render at import time."""
    if not _NUMBA_OK:
        return
    _dx = np.zeros((4, 4), dtype=np.float64)
    _dy = np.zeros((4, 4), dtype=np.float64)
    _fbm_2fields_kernel(_dx, _dy, 3.0, 0.0, 2, 4, 4)


# ---------------------------------------------------------------------------
# Polar remap base cache — keyed by (width, height)
# Stores (r, base_theta) as float32 arrays.
# arctan2 + sqrt only run once per resolution; subsequent frames just
# add the rotation offset and fold.
# ---------------------------------------------------------------------------
_REMAP_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


def _get_polar_base(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (r, base_theta) arrays for the given resolution, computing once."""
    key = (width, height)
    if key not in _REMAP_CACHE:
        cx, cy = width / 2.0, height / 2.0
        y_coords = np.arange(height, dtype=np.float32) - cy
        x_coords = np.arange(width, dtype=np.float32) - cx
        xg, yg = np.meshgrid(x_coords, y_coords)
        r = np.sqrt(xg ** 2 + yg ** 2)          # float32, (H, W)
        base_theta = np.arctan2(yg, xg)         # float32, (H, W)
        _REMAP_CACHE[key] = (r, base_theta)
    return _REMAP_CACHE[key]


def _build_polar_remap(
    width: int,
    height: int,
    num_segments: int,
    rotation: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build coordinate remap arrays for kaleidoscope mirror.

    Converts each pixel to polar coordinates, folds theta into
    a single segment, reflects it, then converts back to Cartesian
    to get source sampling coordinates.

    Args:
        width: Image width.
        height: Image height.
        num_segments: Number of radial symmetry segments.
        rotation: Rotation offset in radians.

    Returns:
        (src_y, src_x) integer index arrays for remapping.
    """
    cx, cy = width / 2.0, height / 2.0

    # r and arctan2 are cached per resolution — only rotation differs per frame
    r, base_theta = _get_polar_base(width, height)
    theta = base_theta - rotation

    # Fold into single segment
    segment_angle = 2 * np.pi / num_segments
    theta_folded = np.mod(theta, segment_angle)

    # Reflect within segment for mirror symmetry
    half_segment = segment_angle / 2.0
    past_half = theta_folded > half_segment
    theta_folded[past_half] = segment_angle - theta_folded[past_half]

    # Back to Cartesian (source coordinates)
    src_x = r * np.cos(theta_folded) + cx
    src_y = r * np.sin(theta_folded) + cy

    # Clamp to valid range
    src_x = np.clip(src_x, 0, width - 1).astype(np.intp)
    src_y = np.clip(src_y, 0, height - 1).astype(np.intp)

    return src_y, src_x


def polar_mirror(
    texture: np.ndarray,
    num_segments: int = 8,
    rotation: float = 0.0,
) -> np.ndarray:
    """
    Apply radial kaleidoscope mirror to a texture.

    Args:
        texture: Input array — either (H, W) float or (H, W, 3) RGB.
        num_segments: Number of symmetry segments (6, 8, 12, etc.).
        rotation: Rotation angle in radians.

    Returns:
        Mirrored array with same shape and dtype as input.
    """
    if texture.ndim == 2:
        h, w = texture.shape
    else:
        h, w = texture.shape[:2]

    src_y, src_x = _build_polar_remap(w, h, num_segments, rotation)

    return texture[src_y, src_x]


def radial_warp(
    texture: np.ndarray,
    amplitude: float = 0.05,
    frequency: float = 3.0,
    time: float = 0.0,
) -> np.ndarray:
    """
    Apply sinusoidal radial warp for organic breathing effect.

    Displaces pixels radially based on sin(r * freq + time).

    Args:
        texture: Input (H, W) or (H, W, 3) array.
        amplitude: Warp strength as fraction of image size.
        frequency: Spatial frequency of the warp.
        time: Time offset for animation.

    Returns:
        Warped array.
    """
    if texture.ndim == 2:
        h, w = texture.shape
    else:
        h, w = texture.shape[:2]

    cx, cy = w / 2.0, h / 2.0
    y_coords = np.arange(h, dtype=np.float32) - cy
    x_coords = np.arange(w, dtype=np.float32) - cx
    xg, yg = np.meshgrid(x_coords, y_coords)

    r = np.sqrt(xg ** 2 + yg ** 2)
    max_dim = max(w, h)

    # Radial displacement
    displacement = np.sin(r / max_dim * frequency * 2 * np.pi + time) * amplitude * max_dim
    theta = np.arctan2(yg, xg)

    src_x = (xg + displacement * np.cos(theta) + cx).astype(np.float32)
    src_y = (yg + displacement * np.sin(theta) + cy).astype(np.float32)

    src_x = np.clip(src_x, 0, w - 1).astype(np.intp)
    src_y = np.clip(src_y, 0, h - 1).astype(np.intp)

    return texture[src_y, src_x]


def flow_field_warp(
    texture: np.ndarray,
    amplitude: float = 0.05,
    scale: float = 3.0,
    time: float = 0.0,
    octaves: int = 3,
) -> np.ndarray:
    """
    Flow-field warp using two independent multi-octave fBm displacement fields.

    Unlike :func:`radial_warp` (which applies a single radially-symmetric
    sine displacement), this function drives *x* and *y* offsets from two
    separate, orthogonal noise fields built from angled sine fBm.  The
    result is an organic, swirling, non-symmetric distortion reminiscent
    of Perlin flow fields — no centre-locked breathing pattern.

    Uses a Numba fused parallel kernel when available for ~3-5× speedup
    over the NumPy fallback path.

    Args:
        texture: Input (H, W) or (H, W, 3) array.
        amplitude: Maximum pixel displacement as a fraction of image size.
        scale: Spatial frequency of the underlying noise field.
        time: Animation time offset.
        octaves: Number of fBm octaves (more = richer detail, slightly slower).

    Returns:
        Warped array with the same shape and dtype as *texture*.
    """
    if texture.ndim == 2:
        h, w = texture.shape
    else:
        h, w = texture.shape[:2]

    max_disp = max(w, h) * amplitude

    dx_field = np.empty((h, w), dtype=np.float64)
    dy_field = np.empty((h, w), dtype=np.float64)

    # Numba parallel wins at 1080p+ (≥1M pixels); below that, numpy SIMD is faster.
    if _NUMBA_OK and h * w >= 921_600:  # 1280×720
        # Fused parallel kernel: both fields in one pass
        _fbm_2fields_kernel(dx_field, dy_field, float(scale), float(time), int(octaves), h, w)
    else:
        # NumPy fallback — identical results
        x_norm = np.linspace(0.0, scale, w, dtype=np.float32)
        y_norm = np.linspace(0.0, scale, h, dtype=np.float32)
        xg, yg = np.meshgrid(x_norm, y_norm)

        dx_field[:] = 0.0
        dy_field[:] = 0.0
        amp = 1.0
        for k in range(octaves):
            freq = 2.0 ** k
            t_x = time * (k + 1) * 0.30
            t_y = time * (k + 1) * 0.30 + 1.7

            dx_field += np.sin(xg * freq * 2.0 * np.pi + t_x) * amp
            dx_field += np.cos(yg * freq * 2.0 * np.pi + t_x * 0.71) * (amp * 0.7)
            dy_field += np.cos(xg * freq * 2.0 * np.pi + t_y) * (amp * 0.7)
            dy_field += np.sin(yg * freq * 2.0 * np.pi + t_y * 1.31) * amp
            amp *= 0.5

    # Normalise each field independently to [-1, 1]
    m_dx = np.abs(dx_field).max()
    dx_field /= m_dx + 1e-8
    m_dy = np.abs(dy_field).max()
    dy_field /= m_dy + 1e-8

    # Apply as pixel offsets
    y_px = np.arange(h, dtype=np.float32)
    x_px = np.arange(w, dtype=np.float32)
    xg_px, yg_px = np.meshgrid(x_px, y_px)

    src_x = np.clip(xg_px + dx_field * max_disp, 0.0, w - 1).astype(np.intp)
    src_y = np.clip(yg_px + dy_field * max_disp, 0.0, h - 1).astype(np.intp)

    return texture[src_y, src_x]


def infinite_zoom_blend(
    current_frame: np.ndarray,
    feedback_buffer: np.ndarray | None,
    zoom_factor: float = 1.02,
    feedback_alpha: float = 0.85,
) -> np.ndarray:
    """
    Blend zoomed-in previous frame with new frame for infinite zoom.

    The previous frame is scaled inward (zoomed) and blended behind
    the new frame, creating the illusion of falling into the pattern.

    Args:
        current_frame: New frame, (H, W, 3) uint8.
        feedback_buffer: Previous output frame, or None for first frame.
        zoom_factor: Scale factor per frame (>1 = zoom in).
        feedback_alpha: Opacity of the previous frame (0-1).

    Returns:
        Blended frame (H, W, 3) uint8.
    """
    if feedback_buffer is None:
        return current_frame

    h, w = current_frame.shape[:2]

    # Scale the feedback buffer inward using Pillow affine
    fb_img = Image.fromarray(feedback_buffer)

    # Crop center region (simulates zoom in)
    crop_margin_x = int(w * (1 - 1.0 / zoom_factor) / 2)
    crop_margin_y = int(h * (1 - 1.0 / zoom_factor) / 2)
    crop_margin_x = max(1, crop_margin_x)
    crop_margin_y = max(1, crop_margin_y)

    cropped = fb_img.crop((
        crop_margin_x,
        crop_margin_y,
        w - crop_margin_x,
        h - crop_margin_y,
    ))
    zoomed = cropped.resize((w, h), Image.LANCZOS)
    # Counteract resize blur so feedback retains edge detail
    # across many iterations instead of cumulative smoothing.
    zoomed = zoomed.filter(
        ImageFilter.UnsharpMask(radius=2, percent=70, threshold=0)
    )
    zoomed_arr = np.asarray(zoomed, dtype=np.float32)

    # Energy-conserving blend: feedback takes a fraction of the dark
    # areas, and the current frame gets ALL remaining weight.  This
    # prevents brightness loss from weights not summing to 1.0.
    current_f = current_frame.astype(np.float32)
    current_brightness = current_f.mean(axis=2, keepdims=True) / 255.0
    # feedback_share: how much of the frame goes to the zoom tunnel.
    # Bright areas → 0 (fresh fractal only). Dark areas → up to
    # feedback_alpha (subtle tunnel echo behind the fractal).
    feedback_share = feedback_alpha * np.clip(1.0 - current_brightness * 3.0, 0, 0.20)

    blended = (
        current_f * (1.0 - feedback_share)
        + zoomed_arr * feedback_share
    )

    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Trigger JIT compilation at import time
# ---------------------------------------------------------------------------
_warmup_flow_jit()
