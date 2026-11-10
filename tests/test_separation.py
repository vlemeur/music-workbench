"""Tests for the automatic non-WAV to WAV conversion guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_analysis.separation import CONVERTED_DIR_NAME, ensure_wav

SAMPLE_RATE = 44100


def _write_sine(path: Path, frequency_hz: float = 440.0, duration_s: float = 1.0) -> None:
    """Write a mono sine wave; the format is inferred from the extension."""
    samples = np.sin(
        2 * np.pi * frequency_hz * np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    ).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE)


def test_ensure_wav_leaves_wav_untouched(tmp_path: Path) -> None:
    """A .wav input must be returned as-is, without any conversion."""
    wav_path = tmp_path / "track.wav"
    _write_sine(wav_path)

    result = ensure_wav(wav_path, output_dir=tmp_path / "outputs")

    assert result == wav_path
    assert not (tmp_path / "outputs" / CONVERTED_DIR_NAME).exists()


def test_ensure_wav_converts_mp3_to_wav(tmp_path: Path) -> None:
    """An .mp3 input must be decoded to a valid WAV inside output_dir/converted/."""
    mp3_path = tmp_path / "track.mp3"
    _write_sine(mp3_path)

    result = ensure_wav(mp3_path, output_dir=tmp_path / "outputs")

    assert result.suffix == ".wav"
    assert result == tmp_path / "outputs" / CONVERTED_DIR_NAME / "track.wav"
    assert result.exists()

    data, sample_rate = sf.read(result, dtype="float32")
    assert sample_rate == SAMPLE_RATE
    assert len(data) > 0
    # The converted clip must keep roughly the same duration (mp3 is lossy).
    assert len(data) == pytest.approx(SAMPLE_RATE, abs=SAMPLE_RATE // 10)


def test_ensure_wav_reuses_existing_conversion(tmp_path: Path) -> None:
    """An already-converted file must be reused, never re-encoded."""
    output_dir = tmp_path / "outputs"
    mp3_path = tmp_path / "track.mp3"
    _write_sine(mp3_path, duration_s=0.5)

    result = ensure_wav(mp3_path, output_dir=output_dir)
    # Tamper with the conversion: a re-encode would restore ~0.5 s of audio.
    result.unlink()
    sf.write(result, np.zeros(128, dtype="float32"), SAMPLE_RATE)

    reused = ensure_wav(mp3_path, output_dir=output_dir)

    assert reused == result
    data, _ = sf.read(reused, dtype="float32")
    assert len(data) == 128


def test_ensure_wav_rejects_undecodable_file(tmp_path: Path) -> None:
    """A file with a supported container but garbage content must raise."""
    bad_path = tmp_path / "broken.mp3"
    bad_path.write_bytes(b"this is not audio data")

    with pytest.raises((OSError, ValueError, RuntimeError)):
        ensure_wav(bad_path, output_dir=tmp_path / "outputs")
