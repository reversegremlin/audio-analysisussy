"""
Harmonic-Percussive Source Separation (HPSS) module.

Separates audio signals into harmonic (melodic/chordal) and
percussive (transient/drum) components for independent visual mapping.

Phase 2 W1 stubs: SeparatedAudio dataclass and SourceSeparator class are
included here.  SourceSeparator requires the optional ``demucs`` package;
a clear ImportError is raised if it is absent.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np


@dataclass
class DecomposedAudio:
    """Container for separated audio components."""

    harmonic: np.ndarray
    percussive: np.ndarray
    original: np.ndarray
    sample_rate: int
    duration: float

    @property
    def n_samples(self) -> int:
        """Total number of samples in the original signal."""
        return len(self.original)


class AudioDecomposer:
    """
    Performs Harmonic-Percussive Source Separation on audio files.

    The separation allows "impact" visuals to be driven by percussion
    while "flow" visuals respond to harmonic content.
    """

    def __init__(self, margin: tuple[float, float] = (1.0, 1.0)):
        """
        Initialize the decomposer.

        Args:
            margin: HPSS margin parameters (harmonic_margin, percussive_margin).
                    Higher values create more aggressive separation.
        """
        self.margin = margin

    def load_audio(
        self,
        audio_path: Union[str, Path],
        sr: int | None = None,
        mono: bool = True,
    ) -> tuple[np.ndarray, int]:
        """
        Load audio from file.

        Args:
            audio_path: Path to audio file (wav, mp3, flac).
            sr: Target sample rate. None preserves original.
            mono: Convert to mono if True.

        Returns:
            Tuple of (audio_signal, sample_rate).
        """
        y, sr_out = librosa.load(audio_path, sr=sr, mono=mono)
        return y, sr_out

    def separate(
        self,
        y: np.ndarray,
        sr: int,
    ) -> DecomposedAudio:
        """
        Perform HPSS on an audio signal.

        Args:
            y: Audio time series.
            sr: Sample rate.

        Returns:
            DecomposedAudio containing harmonic and percussive components.
        """
        harmonic, percussive = librosa.effects.hpss(y, margin=self.margin)

        duration = librosa.get_duration(y=y, sr=sr)

        return DecomposedAudio(
            harmonic=harmonic,
            percussive=percussive,
            original=y,
            sample_rate=sr,
            duration=duration,
        )

    def decompose_file(
        self,
        audio_path: Union[str, Path],
        sr: int | None = 22050,
    ) -> DecomposedAudio:
        """
        Load and decompose an audio file in one step.

        Args:
            audio_path: Path to audio file.
            sr: Target sample rate (default 22050 for efficiency).

        Returns:
            DecomposedAudio with separated components.
        """
        y, sr_out = self.load_audio(audio_path, sr=sr)
        return self.separate(y, sr_out)


# ---------------------------------------------------------------------------
# Phase 2 W1 — Demucs source separation (stub)
# ---------------------------------------------------------------------------

@dataclass
class SeparatedAudio:
    """
    Four-stem source separation result (Phase 2 W1).

    Populated by :class:`SourceSeparator` when the optional ``demucs``
    package is installed.
    """

    drums: np.ndarray
    bass: np.ndarray
    vocals: np.ndarray
    other: np.ndarray
    sample_rate: int


class SourceSeparator:
    """
    Phase 2 W1 stub — four-stem source separation using Demucs.

    Install the optional extra to use this class::

        pip install "chromascope[separation]"

    Raises:
        ImportError: At construction time if ``demucs`` is not installed.
    """

    def __init__(self, model_name: str = "htdemucs"):
        """
        Initialize the separator and load the Demucs model.

        Args:
            model_name: Demucs model identifier.

        Raises:
            ImportError: If demucs is not installed.
        """
        try:
            from demucs.pretrained import get_model  # type: ignore[import]
            self._model = get_model(model_name)
        except ModuleNotFoundError as exc:
            raise ImportError(
                "The 'demucs' package is required for source separation.\n"
                "Install it with:  pip install 'chromascope[separation]'"
            ) from exc

    def separate(
        self,
        y: np.ndarray,
        sr: int,
    ) -> SeparatedAudio:
        """
        Separate an audio signal into four stems.

        Args:
            y: Mono audio signal.
            sr: Sample rate.

        Returns:
            SeparatedAudio with drums, bass, vocals, and other stems.
        """
        raise NotImplementedError(
            "SourceSeparator.separate() is a Phase 2 W1 stub and is not yet implemented."
        )
