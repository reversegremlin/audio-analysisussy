"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

# Default sample rate for test audio
TEST_SR = 22050


@pytest.fixture
def sample_rate() -> int:
    """Default sample rate for tests."""
    return TEST_SR


@pytest.fixture
def pure_sine(sample_rate: int) -> tuple[np.ndarray, int]:
    """
    Generate a pure 440Hz sine wave (A4 note).

    Returns:
        Tuple of (audio_signal, sample_rate).
    """
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    frequency = 440.0  # A4
    y = 0.5 * np.sin(2 * np.pi * frequency * t)
    return y.astype(np.float32), sample_rate


@pytest.fixture
def click_track(sample_rate: int) -> tuple[np.ndarray, int]:
    """
    Generate a simple click track at 120 BPM.

    Returns:
        Tuple of (audio_signal, sample_rate).
    """
    duration = 2.0
    bpm = 120
    samples_per_beat = int(sample_rate * 60 / bpm)
    total_samples = int(sample_rate * duration)

    y = np.zeros(total_samples, dtype=np.float32)

    # Add clicks (short impulses) at each beat
    click_duration = int(sample_rate * 0.01)  # 10ms click
    for beat_start in range(0, total_samples, samples_per_beat):
        click_end = min(beat_start + click_duration, total_samples)
        # Exponential decay click
        click_samples = click_end - beat_start
        decay = np.exp(-np.linspace(0, 5, click_samples))
        y[beat_start:click_end] = 0.8 * decay

    return y, sample_rate


@pytest.fixture
def white_noise(sample_rate: int) -> tuple[np.ndarray, int]:
    """
    Generate white noise.

    Returns:
        Tuple of (audio_signal, sample_rate).
    """
    np.random.seed(42)  # Reproducible
    duration = 2.0
    samples = int(sample_rate * duration)
    y = np.random.randn(samples).astype(np.float32) * 0.3
    return y, sample_rate


@pytest.fixture
def mixed_signal(sample_rate: int) -> tuple[np.ndarray, int]:
    """
    Generate a signal with both harmonic and percussive content.

    Returns:
        Tuple of (audio_signal, sample_rate).
    """
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Harmonic: chord (C major: C4, E4, G4)
    harmonic = (
        0.2 * np.sin(2 * np.pi * 261.63 * t) +  # C4
        0.2 * np.sin(2 * np.pi * 329.63 * t) +  # E4
        0.2 * np.sin(2 * np.pi * 392.00 * t)    # G4
    )

    # Percussive: clicks at 120 BPM
    bpm = 120
    samples_per_beat = int(sample_rate * 60 / bpm)
    total_samples = len(t)
    percussive = np.zeros(total_samples)

    click_duration = int(sample_rate * 0.01)
    for beat_start in range(0, total_samples, samples_per_beat):
        click_end = min(beat_start + click_duration, total_samples)
        click_samples = click_end - beat_start
        decay = np.exp(-np.linspace(0, 5, click_samples))
        percussive[beat_start:click_end] = 0.5 * decay

    y = (harmonic + percussive).astype(np.float32)
    return y, sample_rate


@pytest.fixture
def temp_audio_file(tmp_path, mixed_signal):
    """Create a temporary audio file for testing file I/O."""
    import soundfile as sf

    y, sr = mixed_signal
    audio_path = tmp_path / "test_audio.wav"
    sf.write(audio_path, y, sr)
    return audio_path
