# Demucs Music Source Separation

# Replicate API Deployment Guide

This project is based on [cjm-demucs-v4](https://github.com/cj-mills/cjm-demucs-v4) (inference-only fork of Demucs v4), using TorchCodec to replace torchaudio.

### File Overview

| File | Description |
|------|-------------|
| `cog.yaml` | Cog build config (GPU, Python 3.12, system deps) |
| `requirements-api.txt` | Python dependencies for API runtime |
| `predict.py` | API entry point, defines `Predictor` class (setup/predict) |
| `demucs/` | Demucs inference engine (cjm-demucs-v4 fork) |

### Key Differences from Original Demucs

- **No torchaudio**: Audio I/O uses TorchCodec + ffmpeg
- **No dora/diffq**: Training-related dependencies removed
- **Inference only**: No training, evaluation, or data augmentation code

### API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | File | Required | Input audio file (WAV/MP3/FLAC, etc.) |
| `model` | string | `htdemucs_ft` | Model: htdemucs / htdemucs_ft / htdemucs_6s |
| `stem` | string | `vocals` | Target stem: vocals / drums / bass / other / all |
| `shifts` | int | `1` | Number of random shifts (higher = better quality, slower) |

### Output

- `vocals.wav` — Vocals
- `no_vocals.wav` — Accompaniment (when stem=vocals or all)

**The officially maintained Demucs** is at [Demucs](https://github.com/adefossez/demucs).

This is the 4th release of Demucs (v4), featuring Hybrid Transformer based source separation.

Demucs is a state-of-the-art music source separation model, currently capable of separating
drums, bass, and vocals from the rest of the accompaniment.

# Demucs v4 API Deployment Guide

## VPS Environment Setup (Debian)

### 1. Install Docker

```bash
# Update system
sudo apt update && apt upgrade -y

# Install Docker
sudo apt install -y docker.io
sudo systemctl enable --now docker

# Verify
docker info
```

### 2. Install Cog

```bash
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_Linux_x86_64
chmod +x /usr/local/bin/cog

# Verify
cog --version
```

### 3. Login to Replicate

```bash
# Set API Token (get from https://replicate.com/account/api-tokens)
Optional: export REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxxxxxxxxxxxxx

# Login
cog login

If the token variable is not set, you will be prompted to press Enter and visit a webpage to obtain the token.
Since you cannot open webpages on the VPS, you can open https://replicate.com/auth/token on your development machine,
copy the token displayed on the webpage, and paste it into the console.
```

---

## Project Setup

### 4. Clone the Repository

```bash
git clone https://github.com/iBoostAI/demucs-api
```

## Build and Push

### 5. Build Docker Image

```bash
cd ~/demucs-api
cog build
```

### 6. Push to Replicate

```bash
# First create a model page at https://replicate.com/create:
# - Model name: demucs-api
# - Visibility: Public or Private

# Then push
cog push r8.im/yourname/demucs-api
```

---

## Using the API

### Python Example

```python
import replicate
import requests

output = replicate.run(
    "iboostai/demucs-api",
    input={
        "audio": open("audio.wav", "rb"),
        "model": "htdemucs_ft",
        "stem": "vocals",
        "shifts": 1
    }
)

# Download results
for name, url in output.items():
    response = requests.get(str(url))
    with open(f"{name}.wav", "wb") as f:
        f.write(response.content)
    print(f"Saved: {name}.wav")
```

---

## Quick Start (VPS One-Liner)

```bash
# === 1. Environment Setup ===
apt update && apt install -y docker.io
systemctl enable --now docker
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_Linux_x86_64
chmod +x /usr/local/bin/cog

# === 2. Login ===
export REPLICATE_API_TOKEN=r8_xxxxxxxx
cog login

# === 3. Clone Code ===
git clone https://github.com/iBoostAI/demucs-api

# === 4. Build and Push ===
cd ~/demucs-api
cog build
cog push r8.im/yourname/demucs-api
```

---

## Notes

1. **First build time**: ~10 minutes (downloading dependencies and building image)
2. **Image size**: ~5-10GB (includes PyTorch and models)
3. **Cost**: Replicate T4 GPU ~$0.02/run
4. **Cold start**: ~30-60 seconds on first call (model loading)
