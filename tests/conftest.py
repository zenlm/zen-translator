"""Pytest configuration and fixtures."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_audio():
    """Generate sample audio data for testing."""
    # 3 seconds of audio at 16kHz
    duration_seconds = 3.0
    sample_rate = 16000
    samples = int(duration_seconds * sample_rate)

    # Generate a simple sine wave
    t = np.linspace(0, duration_seconds, samples)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def sample_video_frame():
    """Generate sample video frame for testing."""
    # RGB frame 256x256
    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def temp_audio_file(tmp_path, sample_audio):
    """Create a temporary audio file."""
    import soundfile as sf

    audio, sr = sample_audio
    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), audio, sr)

    return audio_path


@pytest.fixture
def translator_config():
    """Create test translator configuration."""
    from zen_translator.config import TranslatorConfig

    return TranslatorConfig(
        device="cpu",
        dtype="float32",
        enable_lip_sync=False,  # Disable for faster tests
        use_flash_attention=False,
    )


@pytest.fixture
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / "data"
