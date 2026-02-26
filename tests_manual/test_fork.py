"""Manual test script for the cjm-demucs-v4 fork.

Run from the repo root:
    python tests_manual/test_fork.py

Tests:
1. Import demucs and verify version
2. List available models
3. Create Separator with htdemucs
4. Separate test.mp3
5. Save vocals in WAV, FLAC, and MP3 formats
6. Verify output files exist and have non-zero size
"""

import sys
import tempfile
from pathlib import Path

# Add repo root to path for development
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

TEST_FILE = REPO_ROOT / "test_files" / "segment_000.mp3"


def test_import():
    """Test that demucs imports correctly."""
    print("=" * 60)
    print("Test 1: Import demucs")
    print("=" * 60)
    import demucs
    print(f"  Version: {demucs.__version__}")
    assert demucs.__version__ == "0.0.1"

    from demucs.api import Separator, save_audio, list_models
    from demucs.audio import AudioFile
    print("  All imports successful")
    print()


def test_list_models():
    """Test listing available models."""
    print("=" * 60)
    print("Test 2: List models")
    print("=" * 60)
    from demucs.api import list_models
    models = list_models()
    print(f"  Single models: {list(models['single'].keys())}")
    print(f"  Bag models: {list(models['bag'].keys())}")
    assert len(models["bag"]) > 0, "Expected at least one bag model"
    assert "htdemucs" in models["bag"], "Expected htdemucs in bag models"
    print()


def test_separator():
    """Test creating a Separator and running separation."""
    print("=" * 60)
    print("Test 3: Separator + separation")
    print("=" * 60)
    import torch
    from demucs.api import Separator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("  Loading model...")
    separator = Separator(model="htdemucs", device=device)
    print(f"  Model loaded: samplerate={separator.samplerate}, "
          f"channels={separator.audio_channels}")

    assert TEST_FILE.exists(), f"Test file not found: {TEST_FILE}"
    print(f"  Separating: {TEST_FILE}")
    origin, separated = separator.separate_audio_file(TEST_FILE)

    stems = list(separated.keys())
    print(f"  Stems: {stems}")
    assert "vocals" in stems, "Expected 'vocals' stem"
    assert "drums" in stems, "Expected 'drums' stem"
    assert "bass" in stems, "Expected 'bass' stem"
    assert "other" in stems, "Expected 'other' stem"

    for name, tensor in separated.items():
        print(f"    {name}: shape={tuple(tensor.shape)}, "
              f"dtype={tensor.dtype}, "
              f"range=[{tensor.min():.3f}, {tensor.max():.3f}]")
    print()
    return separator, separated


def test_save_audio(separator, separated):
    """Test saving audio in WAV, FLAC, and MP3 formats."""
    print("=" * 60)
    print("Test 4: Save audio (WAV, FLAC, MP3)")
    print("=" * 60)
    from demucs.audio import save_audio

    vocals = separated["vocals"]

    out_dir = REPO_ROOT / "test_output"
    out_dir.mkdir(exist_ok=True)
    print(f"  Output directory: {out_dir}")

    formats = {
        "vocals.wav": {},
        "vocals.flac": {},
        "vocals.mp3": {"bitrate": 320},
        "vocals_float32.wav": {"as_float": True},
        "vocals_24bit.wav": {"bits_per_sample": 24},
    }

    for filename, extra_kwargs in formats.items():
        out_path = out_dir / filename
        save_audio(vocals, str(out_path),
                   samplerate=separator.samplerate, **extra_kwargs)
        size = out_path.stat().st_size
        print(f"  {filename}: {size:,} bytes")
        assert out_path.exists(), f"Output file not created: {out_path}"
        assert size > 0, f"Output file is empty: {out_path}"

    print()


def test_audio_file_read():
    """Test AudioFile reading via ffmpeg."""
    print("=" * 60)
    print("Test 5: AudioFile read")
    print("=" * 60)
    from demucs.audio import AudioFile

    af = AudioFile(TEST_FILE)
    print(f"  Duration: {af.duration:.2f}s")
    print(f"  Channels: {af.channels()}")
    print(f"  Samplerate: {af.samplerate()}")

    wav = af.read(streams=0, samplerate=44100, channels=2)
    print(f"  Read tensor: shape={tuple(wav.shape)}, dtype={wav.dtype}")
    assert wav.shape[0] == 2, "Expected 2 channels"
    assert wav.shape[1] > 0, "Expected non-zero length"
    print()


def main():
    print()
    print("cjm-demucs-v4 Fork Test Suite")
    print("=" * 60)
    print()

    test_import()
    test_list_models()
    test_audio_file_read()
    separator, separated = test_separator()
    test_save_audio(separator, separated)

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
