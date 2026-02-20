"""
Real-time preview window for Chromascope renderers.

Opens a pygame window and renders frames on-the-fly from any
BaseVisualizer (or UniversalMirrorCompositor), matching the
analysis manifest.  Useful for quick local iteration without
encoding a full MP4.

Usage (via CLI):
    chromascope audio.wav --preview
    chromascope audio.wav --preview --mode decay

Keyboard controls while previewing:
    ESC / Q      — quit
    SPACE        — pause / resume
    RIGHT arrow  — step one frame (while paused)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def run_preview(
    renderer: Any,
    frames: List[Dict[str, Any]],
    fps: int,
    title: str = "Chromascope Preview",
) -> None:
    """
    Display rendered frames in a pygame window at the given *fps*.

    Args:
        renderer: Any object with a ``render_frame(frame_data, index)``
                  method (BaseVisualizer or UniversalMirrorCompositor).
        frames:   List of per-frame dicts from the audio manifest.
        fps:      Target playback frame-rate.
        title:    Window title string.
    """
    try:
        import pygame
    except ImportError:
        print(
            "Real-time preview requires pygame.\n"
            "Install it with:  pip install pygame"
        )
        return

    if not frames:
        print("No frames to preview.")
        return

    pygame.init()

    # Render first frame to learn output dimensions
    first_frame: np.ndarray = renderer.render_frame(frames[0], 0)
    h, w = first_frame.shape[:2]

    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    def _blit(arr: np.ndarray) -> None:
        # pygame expects (W, H, 3) for surfarray, numpy gives (H, W, 3)
        surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        screen.blit(surface, (0, 0))

    def _overlay(text: str) -> None:
        label = font.render(text, True, (200, 200, 200))
        screen.blit(label, (8, 8))

    paused = False
    frame_idx = 0
    n_frames = len(frames)

    # Draw the first frame (already rendered)
    _blit(first_frame)
    _overlay(f"Frame 1 / {n_frames}  [SPACE=pause  ESC=quit]")
    pygame.display.flip()
    frame_idx = 1

    running = True
    while running and frame_idx < n_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    # Single-step
                    frame = renderer.render_frame(frames[frame_idx], frame_idx)
                    _blit(frame)
                    _overlay(f"Frame {frame_idx + 1} / {n_frames}  [PAUSED]")
                    pygame.display.flip()
                    frame_idx += 1

        if not running:
            break

        if not paused:
            frame = renderer.render_frame(frames[frame_idx], frame_idx)
            _blit(frame)
            _overlay(f"Frame {frame_idx + 1} / {n_frames}  [SPACE=pause  ESC=quit]")
            pygame.display.flip()
            frame_idx += 1

        clock.tick(fps)

    pygame.quit()
