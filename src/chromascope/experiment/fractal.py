"""
Fractal texture generators.

Vectorized with numpy — no per-pixel Python loops.
All outputs are float32 arrays in [0, 1] representing escape-time
or intensity values ready for palette mapping.
"""

import numpy as np


def julia_set(
    width: int,
    height: int,
    c: complex,
    center: complex = 0 + 0j,
    zoom: float = 1.0,
    max_iter: int = 256,
) -> np.ndarray:
    """
    Render Julia set escape-time values.

    Args:
        width: Output width in pixels.
        height: Output height in pixels.
        c: Julia constant (determines shape).
        center: Viewport center in the complex plane.
        zoom: Zoom level (higher = more zoomed in).
        max_iter: Maximum iterations (controls detail).

    Returns:
        float32 array of shape (height, width) with values in [0, 1].
    """
    # Build complex coordinate grid
    aspect = width / height
    r_span = 3.0 / zoom
    i_span = r_span / aspect

    re = np.linspace(
        center.real - r_span / 2,
        center.real + r_span / 2,
        width,
        dtype=np.float32,
    )
    im = np.linspace(
        center.imag - i_span / 2,
        center.imag + i_span / 2,
        height,
        dtype=np.float32,
    )

    re_grid, im_grid = np.meshgrid(re, im)
    z = re_grid + 1j * im_grid

    # Escape-time iteration
    output = np.zeros((height, width), dtype=np.float32)
    mask = np.ones((height, width), dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + c
        escaped = mask & (np.abs(z) > 2.0)
        # Smooth coloring: fractional escape count
        if np.any(escaped):
            # log2(log2(|z|)) smoothing
            abs_z = np.abs(z[escaped])
            smooth_val = i + 1 - np.log2(np.log2(np.maximum(abs_z, 1.001)))
            output[escaped] = smooth_val
        mask &= ~escaped

    # Normalize escape-time values to [0, 1]
    max_val = output.max()
    if max_val > 0:
        output /= max_val

    # Color interior points by their final orbit magnitude.
    # This gives the Julia set interior natural fractal-derived
    # structure instead of uniform black, keeping it dimmer than
    # the bright escape-time boundary.
    if np.any(mask):
        interior_z = np.abs(z[mask])
        iz_max = interior_z.max()
        if iz_max > 0:
            output[mask] = (interior_z / iz_max) * 0.35

    return output


def mandelbrot_zoom(
    width: int,
    height: int,
    center: complex = -0.75 + 0.1j,
    zoom: float = 1.0,
    max_iter: int = 256,
) -> np.ndarray:
    """
    Render Mandelbrot set at a given zoom and center.

    Args:
        width: Output width in pixels.
        height: Output height in pixels.
        center: Zoom target in the complex plane.
        zoom: Zoom level.
        max_iter: Maximum iterations.

    Returns:
        float32 array of shape (height, width) with values in [0, 1].
    """
    aspect = width / height
    r_span = 3.5 / zoom
    i_span = r_span / aspect

    re = np.linspace(
        center.real - r_span / 2,
        center.real + r_span / 2,
        width,
        dtype=np.float32,
    )
    im = np.linspace(
        center.imag - i_span / 2,
        center.imag + i_span / 2,
        height,
        dtype=np.float32,
    )

    re_grid, im_grid = np.meshgrid(re, im)
    c = re_grid + 1j * im_grid
    z = np.zeros_like(c)

    output = np.zeros((height, width), dtype=np.float32)
    mask = np.ones((height, width), dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + c[mask]
        escaped = mask & (np.abs(z) > 2.0)
        if np.any(escaped):
            abs_z = np.abs(z[escaped])
            smooth_val = i + 1 - np.log2(np.log2(np.maximum(abs_z, 1.001)))
            output[escaped] = smooth_val
        mask &= ~escaped

    max_val = output.max()
    if max_val > 0:
        output /= max_val

    # Interior coloring via final orbit magnitude
    if np.any(mask):
        interior_z = np.abs(z[mask])
        iz_max = interior_z.max()
        if iz_max > 0:
            output[mask] = (interior_z / iz_max) * 0.35

    return output


def noise_fractal(
    width: int,
    height: int,
    time: float = 0.0,
    octaves: int = 4,
    scale: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Multi-octave sine-based fractal noise field.

    Uses layered sinusoidal patterns with varying frequencies
    to approximate organic noise without external dependencies.

    Args:
        width: Output width.
        height: Output height.
        time: Time parameter for animation.
        octaves: Number of frequency layers.
        scale: Base spatial frequency.
        seed: Random seed for phase offsets.

    Returns:
        float32 array of shape (height, width) with values in [0, 1].
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(0, scale, width, dtype=np.float32)
    y = np.linspace(0, scale, height, dtype=np.float32)
    xg, yg = np.meshgrid(x, y)

    output = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0

    for octave in range(octaves):
        freq = 2.0 ** octave
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        angle = rng.uniform(0, np.pi)

        # Rotated coordinates for varied direction per octave
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xr = xg * cos_a - yg * sin_a
        yr = xg * sin_a + yg * cos_a

        layer = np.sin(xr * freq * 2 * np.pi + phase_x + time * (octave + 1) * 0.5)
        layer += np.sin(yr * freq * 2 * np.pi + phase_y + time * (octave + 1) * 0.3)
        layer *= 0.5  # normalize sum-of-two-sines to [-1, 1]

        output += layer * amplitude
        amplitude *= 0.5

    # Normalize to [0, 1]
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)
    return output


# Curated path of interesting Julia c-values for smooth animation
JULIA_C_PATH = [
    -0.7269 + 0.1889j,
    -0.8 + 0.156j,
    -0.4 + 0.6j,
    0.285 + 0.01j,
    0.285 + 0.0j,
    -0.70176 - 0.3842j,
    -0.835 - 0.2321j,
    -0.1 + 0.651j,
    0.0 + 0.8j,
    -0.7269 + 0.1889j,  # loop back to start
]


def interpolate_c(t: float) -> complex:
    """
    Interpolate along the curated Julia c-value path.

    Args:
        t: Parameter in [0, 1] cycling through the path.

    Returns:
        Interpolated complex c value.
    """
    t = t % 1.0
    n = len(JULIA_C_PATH) - 1  # last == first for looping
    segment = t * n
    idx = int(segment)
    frac = segment - idx
    idx = min(idx, n - 1)

    c0 = JULIA_C_PATH[idx]
    c1 = JULIA_C_PATH[idx + 1]

    # Smooth interpolation (smoothstep)
    frac = frac * frac * (3 - 2 * frac)

    return complex(
        c0.real + (c1.real - c0.real) * frac,
        c0.imag + (c1.imag - c0.imag) * frac,
    )
