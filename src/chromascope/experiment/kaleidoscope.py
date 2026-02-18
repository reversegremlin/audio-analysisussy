"""
Kaleidoscope symmetry and infinite zoom engine.

Polar coordinate remapping for N-way radial mirrors,
plus feedback buffer blending for the infinite zoom effect.
"""

import numpy as np
from PIL import Image, ImageFilter


def _build_polar_remap(
    width: int,
    height: int,
    num_segments: int,
    rotation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
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

    # Build coordinate grid
    y_coords = np.arange(height, dtype=np.float32) - cy
    x_coords = np.arange(width, dtype=np.float32) - cx
    xg, yg = np.meshgrid(x_coords, y_coords)

    # To polar
    r = np.sqrt(xg ** 2 + yg ** 2)
    theta = np.arctan2(yg, xg) - rotation

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
