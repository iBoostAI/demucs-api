# Demucs Music Source Separation

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
export REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxxxxxxxxxxxxx

# 登录
cog login
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

