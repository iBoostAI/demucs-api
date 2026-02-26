# cjm-demucs-v4

Inference-only fork of [Demucs v4](https://github.com/adefossez/demucs) (Hybrid Transformer) for Python 3.12+ with [TorchCodec](https://github.com/meta-pytorch/torchcodec) replacing torchaudio I/O.

## What Changed

- **TorchCodec migration:** Replaced `torchaudio.save()` and `torchaudio.load()` with TorchCodec's `AudioEncoder` and `AudioDecoder`
- **Removed `audio_legacy.py`:** torchaudio 2.1 compatibility shim no longer needed
- **Stripped training code:** Removed training, evaluation, augmentation, distributed training, and experiment grid files
- **Updated dependencies:** `torchcodec` replaces `torchaudio`; requires `torch>=2.0`, `python>=3.12`

## Installation

```bash
pip install cjm-demucs-v4
```

Or from source:

```bash
pip install git+https://github.com/cj-mills/cjm-demucs-v4.git
```

### Requirements

- Python >= 3.12
- PyTorch >= 2.0
- FFmpeg (for audio reading)
- CUDA GPU recommended (CPU inference works but is ~10-20x slower)

## Usage

### Python API

```python
from demucs.api import Separator, save_audio

separator = Separator(model="htdemucs", device="cuda")
origin, separated = separator.separate_audio_file("input.mp3")

# Save just the vocals
save_audio(separated["vocals"], "vocals.wav", samplerate=separator.samplerate)
```

### CLI

```bash
demucs input.mp3 -o output_dir
demucs input.mp3 --two-stems vocals  # vocals + accompaniment only
```

## Available Models

| Model | Description |
|-------|-------------|
| `htdemucs` | Hybrid Transformer Demucs v4 (default) |
| `htdemucs_ft` | Fine-tuned version (better quality, 4x slower) |
| `htdemucs_6s` | 6-source variant (adds piano, guitar) |


All 4-stem models separate into: **drums**, **bass**, **vocals**, **other**.

## License

MIT License (same as upstream Demucs).

## Credits

Based on [Demucs](https://github.com/adefossez/demucs) by Alexandre Defossez / Meta Platforms, Inc.
