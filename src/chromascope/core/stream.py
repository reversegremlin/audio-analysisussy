"""
Real-time audio analysis stream for live visualization (Phase 3 — Run stub).

Architecture Overview
---------------------
::

    Audio Device
        │
        ▼  (ring buffer, e.g. 2 048 samples @ 44 100 Hz)
    RealtimeAnalyzer.process_chunk(chunk)
        │
        ├─► FeatureAnalyzer (windowed STFT, hop=chunk_size)
        │        └─► TonalityFeatures, EnergyFeatures (no beat tracking)
        │
        ├─► Online beat estimator  (Phase 3 — to be implemented)
        │        └─► estimated beat_position, BPM
        │
        ├─► SignalPolisher  (ring-buffer smoothing)
        │        └─► PolishedFeatures snapshot (single frame)
        │
        └─► LiveFeatures  (returned to the caller for rendering)

Design Goals
------------
* **Low latency**: target < 20 ms end-to-end at 512-sample chunks / 44 100 Hz.
* **Thread-safe**: process_chunk() is safe to call from a PyAudio callback.
* **Graceful degradation**: features that cannot be computed within the time
  budget are returned as None and the renderer falls back to simpler visuals.

Implementation Status
---------------------
This module is a Phase 3 stub.  The class interface is defined and
documented; the bodies will be fleshed out in a future iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LiveFeatures:
    """
    Single-frame snapshot of audio intelligence features for live rendering.

    All fields default to None so callers can check availability before use.
    """

    # Timestamp of this snapshot
    chunk_index: int = 0
    time_sec: float = 0.0

    # Energy
    global_energy: Optional[float] = None
    percussive_impact: Optional[float] = None
    harmonic_energy: Optional[float] = None
    spectral_flux: Optional[float] = None

    # Frequency bands [0,1]
    sub_bass: Optional[float] = None
    bass: Optional[float] = None
    mid: Optional[float] = None
    brilliance: Optional[float] = None

    # Tonality
    spectral_brightness: Optional[float] = None
    spectral_flatness: Optional[float] = None
    dominant_chroma_index: Optional[int] = None
    pitch_hue: Optional[float] = None

    # Rhythm
    is_beat: bool = False
    beat_position: Optional[float] = None
    bpm: Optional[float] = None

    # Pitch
    f0_hz: Optional[float] = None
    f0_voiced: bool = False

    # Key
    key_root_index: Optional[int] = None
    key_mode: Optional[str] = None


class RealtimeAnalyzer:
    """
    Real-time audio analysis pipeline for live visualization.

    Processes audio chunks as they arrive from the sound device and
    returns a :class:`LiveFeatures` snapshot for each chunk.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz (default: 44 100).
    chunk_size:
        Number of samples per chunk / hop length (default: 512).
    history_seconds:
        Length of the internal audio ring buffer in seconds (default: 2.0).
        Must be long enough for beat tracking.
    target_fps:
        Desired output frame rate.  Chunks smaller than
        ``sample_rate / target_fps`` are accumulated before emitting a frame.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_size: int = 512,
        history_seconds: float = 2.0,
        target_fps: int = 60,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.history_seconds = history_seconds
        self.target_fps = target_fps

        self._chunk_index: int = 0
        # Ring buffer — populated in process_chunk()
        self._buffer: np.ndarray = np.zeros(
            int(sample_rate * history_seconds), dtype=np.float32
        )

        # Phase 3: internal state for beat tracking, key tracking, etc.
        # (to be implemented)

    def process_chunk(self, chunk: np.ndarray) -> Optional[LiveFeatures]:
        """
        Process one audio chunk and return a LiveFeatures snapshot.

        This method is designed to be called from a real-time audio callback.
        It must complete within the chunk duration to avoid buffer under-runs.

        Parameters
        ----------
        chunk:
            1-D float32 audio samples, length == chunk_size.

        Returns
        -------
        LiveFeatures | None
            A feature snapshot if a full output frame is ready, otherwise None
            (when chunks are being accumulated to reach the target FPS).

        Notes
        -----
        Phase 3 stub — the body is not yet implemented.  The method
        currently returns a minimal LiveFeatures with just the timestamp.
        """
        self._chunk_index += 1
        time_sec = self._chunk_index * self.chunk_size / self.sample_rate

        # TODO (Phase 3): update ring buffer, run windowed analysis,
        #  call FeatureAnalyzer, run online beat estimator, poll polisher.

        return LiveFeatures(
            chunk_index=self._chunk_index,
            time_sec=time_sec,
        )
