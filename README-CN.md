# Demucs Music Source Separation

# Replicate API 部署说明

本项目基于 [cjm-demucs-v4](https://github.com/cj-mills/cjm-demucs-v4)（Demucs v4 推理版 fork），使用 TorchCodec 替代 torchaudio。

### 文件说明

| 文件 | 说明 |
|------|------|
| `cog.yaml` | Cog 构建配置（GPU、Python 3.12、系统依赖） |
| `requirements-api.txt` | API 运行时 Python 依赖 |
| `predict.py` | API 入口，定义 `Predictor` 类（setup/predict） |
| `demucs/` | Demucs 推理引擎（从 cjm-demucs-v4 fork） |

### 与原版 demucs 的主要区别

- **无 torchaudio**：音频 I/O 使用 TorchCodec + ffmpeg
- **无 dora/diffq**：移除了训练相关依赖
- **仅推理**：不包含训练、评估、数据增强代码

### API 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `audio` | File | 必填 | 输入音频（WAV/MP3/FLAC 等） |
| `model` | string | `htdemucs_ft` | 模型：htdemucs / htdemucs_ft / htdemucs_6s |
| `stem` | string | `vocals` | 提取目标：vocals / drums / bass / other / all |
| `shifts` | int | `1` | 随机偏移次数（越高质量越好，速度越慢） |

### 返回

- `vocals.wav` — 人声
- `no_vocals.wav` — 伴奏（当 stem=vocals 或 all 时）

**The officially maintained Demucs** is at [Demucs](https://github.com/adefossez/demucs).

This is the 4th release of Demucs (v4), featuring Hybrid Transformer based source separation.

Demucs is a state-of-the-art music source separation model, currently capable of separating
drums, bass, and vocals from the rest of the accompaniment.

# Demucs v4 API 服务部署指南

## VPS 环境准备 (Debian)

### 1. 安装 Docker

```bash
# 更新系统
sudo apt update && apt upgrade -y

# 安装 Docker
sudo apt install -y docker.io
sudo systemctl enable --now docker

# 验证
docker info
```

### 2. 安装 Cog

```bash
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_Linux_x86_64
chmod +x /usr/local/bin/cog

# 验证
cog --version
```

### 3. 登录 Replicate

```bash
# 设置 API Token (从 https://replicate.com/account/api-tokens 获取)
可选: export REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxxxxxxxxxxxxx

# 登录
cog login

如果没有设置 token 变量会提示回车后访问网页获取 token
在VPS上不能打开网页，可在开发机器上打开 https://replicate.com/auth/token
拷贝网页上显示的 token 粘贴到控制台
```

---

## 项目准备

### 4. 下载项目源码

```bash
git clone https://github.com/iBoostAI/demucs-api
```

## 构建和推送

### 5. 构建 Docker 镜像

```bash
cd ~/demucs-api
cog build
```

### 6. 推送到 Replicate

```bash
# 首先在 https://replicate.com/create 创建模型页面:
# - 模型名称: demucs-api
# - 可见性: Public 或 Private

# 然后推送
cog push r8.im/yourname/demucs-api
```

---

## 使用 API

### Python 调用

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

# 下载结果
for name, url in output.items():
    response = requests.get(str(url))
    with open(f"{name}.wav", "wb") as f:
        f.write(response.content)
    print(f"Saved: {name}.wav")
```

---

## 完整命令摘要 (VPS 一键执行)

```bash
# === 1. 环境准备 ===
apt update && apt install -y docker.io
systemctl enable --now docker
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_Linux_x86_64
chmod +x /usr/local/bin/cog

# === 2. 登录 ===
export REPLICATE_API_TOKEN=r8_xxxxxxxx
cog login

# === 3. 下载代码 ===
git clone https://github.com/iBoostAI/demucs-api

# === 4. 构建和推送 ===
cd ~/demucs-api
cog build
cog push r8.im/yourname/demucs-api
```

---

## 注意事项

1. **首次构建时间**: 约 10 分钟（下载依赖和构建镜像）
2. **镜像大小**: 约 5-10GB（包含 PyTorch 和模型）
3. **成本**: Replicate T4 GPU ~$0.02/次
4. **冷启动**: 首次调用约 30-60 秒（加载模型）

