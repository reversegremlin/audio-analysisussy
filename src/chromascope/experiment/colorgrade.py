"""
Color grading and post-processing.

Maps fractal escape-time values to jewel-tone palettes,
adds glow bloom and chromatic aberration.
"""

import numpy as np
from PIL import Image, ImageFilter


def apply_palette(
    escape_values: np.ndarray,
    hue_base: float = 0.0,
    time: float = 0.0,
    saturation: float = 0.85,
    contrast: float = 1.5,
) -> np.ndarray:
    """
    Map escape-time scalar field to RGB via jewel-tone HSV palette.

    Args:
        escape_values: (H, W) float32 array in [0, 1].
        hue_base: Base hue from audio chroma (0-1).
        time: Time parameter for color cycling.
        saturation: HSV saturation (0-1).
        contrast: Gamma-style contrast applied before mapping.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    h, w = escape_values.shape

    # Apply contrast curve (gamma)
    vals = np.clip(escape_values, 0, 1)
    vals = vals ** (1.0 / contrast)  # <1 darkens, >1 brightens mid-tones

    # Hue: base + cycling + variation from escape value
    hue_cycle = time * 0.02  # slow drift
    hue = (hue_base + hue_cycle + vals * 0.6) % 1.0

    # Saturation: high for jewel tones, slightly reduced in bright areas
    sat = np.full_like(vals, saturation)
    sat = sat * (0.7 + 0.3 * vals)  # desaturate near-zero areas slightly

    # Value: driven by escape-time — dark core, bright edges.
    # Soft curve naturally reaches 0 at interior (pure black) which
    # creates sharp boundary transitions.  Glow bloom fills dark areas
    # with soft light from nearby boundary pixels.
    value = (1.0 - (1.0 - vals) ** 1.3)

    # HSV to RGB (vectorized)
    rgb = _hsv_to_rgb_array(hue, sat, value)

    return (rgb * 255).astype(np.uint8)


def _hsv_to_rgb_array(
    h: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Vectorized HSV to RGB conversion.

    Args:
        h, s, v: Arrays of same shape, values in [0, 1].

    Returns:
        (H, W, 3) float array in [0, 1].
    """
    h6 = (h * 6.0) % 6.0
    i = h6.astype(np.int32)
    f = h6 - i

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    # Build RGB based on sector
    shape = h.shape + (3,)
    rgb = np.zeros(shape, dtype=np.float32)

    mask0 = i == 0
    mask1 = i == 1
    mask2 = i == 2
    mask3 = i == 3
    mask4 = i == 4
    mask5 = i == 5

    rgb[mask0, 0] = v[mask0]; rgb[mask0, 1] = t[mask0]; rgb[mask0, 2] = p[mask0]
    rgb[mask1, 0] = q[mask1]; rgb[mask1, 1] = v[mask1]; rgb[mask1, 2] = p[mask1]
    rgb[mask2, 0] = p[mask2]; rgb[mask2, 1] = v[mask2]; rgb[mask2, 2] = t[mask2]
    rgb[mask3, 0] = p[mask3]; rgb[mask3, 1] = q[mask3]; rgb[mask3, 2] = v[mask3]
    rgb[mask4, 0] = t[mask4]; rgb[mask4, 1] = p[mask4]; rgb[mask4, 2] = v[mask4]
    rgb[mask5, 0] = v[mask5]; rgb[mask5, 1] = p[mask5]; rgb[mask5, 2] = q[mask5]

    return rgb


def tone_map_soft(
    frame: np.ndarray,
    shoulder: float = 0.78,
) -> np.ndarray:
    """
    Soft-knee tone mapping to compress highlights without hard clipping.

    Pixels below ``shoulder`` (as a fraction of 255) pass through unchanged.
    Highlights above the shoulder are compressed via a Reinhard-style curve
    so they asymptotically approach 255 instead of slamming to it.

    Args:
        frame: (H, W, 3) uint8 RGB array.
        shoulder: Brightness fraction (0-1) where compression begins.

    Returns:
        (H, W, 3) uint8 RGB array with compressed highlights.
    """
    threshold = shoulder * 255.0
    headroom = 255.0 - threshold

    f = frame.astype(np.float32)
    above = np.maximum(f - threshold, 0.0)
    compressed = threshold + above * headroom / (above + headroom)

    # Only apply to pixels that exceeded the threshold
    mask = f > threshold
    result = np.where(mask, compressed, f)

    return result.astype(np.uint8)


def add_glow(
    frame: np.ndarray,
    intensity: float = 0.3,
    radius: int = 15,
) -> np.ndarray:
    """
    Screen-blend a gaussian-blurred copy for bloom/glow.

    Args:
        frame: (H, W, 3) uint8 RGB array.
        intensity: Glow opacity (0-1).
        radius: Blur radius in pixels.

    Returns:
        (H, W, 3) uint8 RGB array with glow applied.
    """
    if intensity <= 0:
        return frame

    img = Image.fromarray(frame)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred_arr = np.asarray(blurred, dtype=np.float32)
    frame_f = frame.astype(np.float32)

    # Screen blend: result = 1 - (1-a)(1-b)
    a = frame_f / 255.0
    b = blurred_arr / 255.0 * intensity
    screen = 1.0 - (1.0 - a) * (1.0 - b)

    return (screen * 255).astype(np.uint8)


def chromatic_aberration(
    frame: np.ndarray,
    offset: int = 3,
) -> np.ndarray:
    """
    Shift R and B channels for chromatic fringe effect.

    Args:
        frame: (H, W, 3) uint8 RGB array.
        offset: Pixel shift amount for R and B channels.

    Returns:
        (H, W, 3) uint8 RGB array with aberration.
    """
    if offset <= 0:
        return frame

    result = frame.copy()
    h, w = frame.shape[:2]

    # Shift red channel outward from center
    # Horizontal shift only for subtle effect
    result[:, offset:, 0] = frame[:, :w - offset, 0]
    result[:, :w - offset, 2] = frame[:, offset:, 2]

    return result


def vignette(
    frame: np.ndarray,
    strength: float = 0.4,
) -> np.ndarray:
    """
    Apply radial vignette darkening.

    Args:
        frame: (H, W, 3) uint8.
        strength: Vignette darkness (0 = none, 1 = full black at edges).

    Returns:
        (H, W, 3) uint8.
    """
    if strength <= 0:
        return frame

    h, w = frame.shape[:2]
    cy, cx = h / 2, w / 2
    max_r = np.sqrt(cx ** 2 + cy ** 2)

    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    xg, yg = np.meshgrid(x, y)
    r = np.sqrt(xg ** 2 + yg ** 2) / max_r

    # Smooth vignette curve
    vign = 1.0 - np.clip(r * strength, 0, 1) ** 2
    vign = vign[:, :, np.newaxis]

    return (frame.astype(np.float32) * vign).astype(np.uint8)
