# cubic

## 环境配置（uv + Python 3.12 + CUDA 13.1）

本项目使用 `uv` 管理 Python 环境与依赖。

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 创建并激活 Python 3.12 环境

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 3. 安装依赖

PyTorch 使用 CUDA 13.1 轮子源：

```bash
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu131
```

安装 diffusers 及常用依赖：

```bash
uv add diffusers transformers accelerate safetensors huggingface_hub datasets peft scipy einops opencv-python
```

如果需要 `pip` 命令：

```bash
uv pip install pip
```

### 4. 验证安装

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.version.cuda); print('cuda_available:', torch.cuda.is_available())"
python -c "import diffusers; print(diffusers.__version__)"
```

### 5. 复现安装（新机器）

仓库内已有 `pyproject.toml` 和 `uv.lock`，可直接：

```bash
uv sync
```
