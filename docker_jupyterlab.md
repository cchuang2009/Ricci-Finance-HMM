# Docker Deployment Guide
# Docker 部署指南

**Ricci Finance V16, Jupyterlab**

Dynamic Multi-Sector Financial Network Analysis using

- Ricci Curvature
- Ricci Flow
- Dynamic Graph Neural Networks
- Graph Attention Networks
- Streamlit

---

# Contents / 目錄

1. Why Docker?
2. Project Structure
3. Docker Architecture
4. Dockerfile
5. .dockerignore
6. pyproject.toml + uv
7. Build Image
8. Multi-platform Build
9. Apple Silicon
10. GPU Version
11. Docker Compose
12. Persistent Data
13. Publish Image
14. Image Size Optimization
15. Best Practices

---

# 1. Why Docker?
# 為何使用 Docker？

Docker provides a reproducible execution environment.

Docker 提供一致且可重現的執行環境。

Advantages

- identical environment
- easy installation
- dependency isolation
- cloud deployment
- reproducible research
- simplified collaboration

---

# 2. Project Structure

```text
RicciFinanceV16/

│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── .dockerignore
├── README.md
├── DOCKER.md
│
├── cache/
├── data/
├── figures/
│
├── ricci_finance/
│
├── requirements-dev.txt
│
└── tests/
```

---

# 3. Docker Architecture

Instead of installing everything directly into one image,

不要將所有套件直接安裝到同一層。

Recommended architecture

```text
                Builder Stage

                install uv
                     │
                     │
             uv sync --frozen
                     │
             build dependencies
                     │
                     ▼

                Runtime Stage

            python:3.12-slim
                     │
             copy packages
                     │
             copy source code
                     │
                Streamlit
```

Advantages

- smaller image
- cleaner image
- faster build
- production ready

---

## 3.1 uv Brief Introduction

```shell
# Initialize project
> uv init v16
> cd v16
# add dependencies, i.e. convert requirements.txt to pyproject.toml
> uv add ipykernel jupyterlab matplotlib networkx numpy pandas plotly scikit-learn scipy torch umap-learn yfinance
> uv sync 
> uv lock
> docker build -f Dockerfile.notebook  -t ricci-finance-jupyter:v16 .
# chech which docker made
> docker images
> docker run -p 8501:8501 ricci-finance:v16
> docker run -p 8888:8888 ricci-finance-jupyter:v16

# if failed, enter to check
# > docker run --rm -it --entrypoint /bin/bash ricci-finance:v16
> docker run -p 8800:8800 ricci-finance-jupyter:v16
# sync and rebuild
> uv sync 
>
```
The the pyproject.toml generated is like:

```text
[project]
name = "ricci-finance-v16"
version = "16.0.0"
requires-python = ">=3.12"

dependencies = [
    "hmmlearn>=0.3.3",
    "ipython>=9.16.0",
    "nbformat>=5.10.4",
    "networkx>=3.6.1",
    "numpy>=2.5.1",
    "pandas>=3.0.5",
    "plotly>=6.9.0",
    "pyecharts>=2.1.0",
    "pytest>=9.1.1",
    "scikit-learn>=1.9.0",
    "scipy>=1.18.0",
    "streamlit>=1.60.0",
    "streamlit-echarts>=0.7.0",
    "torch>=2.13.0",
    "yfinance>=1.5.2",
    "umap-learn",
    "jupyterlab",
    "ipykernel",
    "graphriccicurvature>=0.5.3.1",
    "matplotlib>=3.11.1",
]
```

<b>Note:</b> build Python-3.13 version due to hmmlearn.

# 4. Dockerfile

```dockerfile
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

# System libraries needed by scientific Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Install dependencies first
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

# Copy V16 source
COPY ricci_finance/ ./ricci_finance/

# Copy notebooks
#COPY notebooks/ ./notebooks/
COPY ricci_finance_v16_lecture.ipynb ./notebooks/
# Verify important packages during BUILD
RUN .venv/bin/python -c \
    "import numpy, pandas, scipy, networkx, torch, matplotlib; \
     print('V16 scientific environment OK')"

RUN .venv/bin/python -c \
    "import jupyter; print('Jupyter OK')"

EXPOSE 8888
# ,"--no-browser"
CMD [".venv/bin/jupyter", "lab","--ip=0.0.0.0","--port=8888","--no-browser", "--allow-root"]
```

---

# 5. .dockerignore

```
.git
.github

__pycache__/

*.pyc

.ipynb_checkpoints/

*.ipynb

.cache/

.vscode/

.idea/

figures/

cache/

tests/

docs/

*.gif
*.png
*.jpg
*.pdf

*.mp4

data/*.csv
data/*.parquet
```

---

# 6. Dependency Management

Ricci Finance V16 uses

```
uv
```

instead of

```
pip
```

Advantages

- much faster
- reproducible
- deterministic lockfile
- smaller cache
- modern Python workflow

Install

```bash
uv sync
```

Update

```bash
uv lock
```

---

# 7. Build Docker Image

```bash
docker build \
-t ricci-finance:v16 .
```

Verify

```bash
docker images
```

---

# 8. Run

```bash
docker run \
-p 8501:8501 \
ricci-finance:v16
```

Open

```
http://localhost:8501
```

---

# 9. Apple Silicon (M1/M2/M3/M4)

Ricci Finance fully supports

- Mac mini
- Mac Studio
- MacBook Air
- MacBook Pro
- iMac

Apple Silicon build

```bash
docker build \
--platform linux/arm64 \
-t ricci-finance:v16-arm64 .
```

Intel build

```bash
docker build \
--platform linux/amd64 \
-t ricci-finance:v16-amd64 .
```

Universal image

```bash
docker buildx build \
--platform linux/amd64,linux/arm64 \
-t ricci-finance:v16 \
--push .
```

Docker automatically downloads the correct architecture.

---

# 10. Apple GPU (MPS)

For GCN/GAT training,

```python
import torch

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model.to(device)
```

No CUDA installation is required.

---

# 11. NVIDIA GPU Version

Optional Dockerfile

```
Dockerfile.cuda
```

Base image

```dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04
```

Run

```bash
docker run \
--gpus all \
-p 8501:8501 \
ricci-finance:v16-cuda
```

---

# 12. Docker Compose

```yaml
services:

  ricci-finance:

    build: .

    container_name: ricci-finance-v16

    restart: unless-stopped

    ports:

      - "8501:8501"

    volumes:

      - ./data:/app/data

      - ./cache:/app/cache

      - ./figures:/app/figures
```

Run

```bash
docker compose up --build
```

Stop

```bash
docker compose down
```

---

# 13. Persistent Data

Recommended directories

```
data/
cache/
figures/
```

Mount

```bash
-v ./data:/app/data
```

so downloaded market data and generated figures persist outside the container.

---

# 14. Publish Docker Image

Docker Hub

```bash
docker tag ricci-finance:v16 USER/ricci-finance:v16

docker push USER/ricci-finance:v16
```

GitHub Container Registry

```bash
docker tag ricci-finance:v16 \
ghcr.io/USER/ricci-finance:v16

docker push \
ghcr.io/USER/ricci-finance:v16
```

---

# 15. Image Size Optimization

Recommended optimizations

| Method | Benefit |
|---------|---------|
| Multi-stage build | Remove build tools |
| python:3.12-slim-bookworm | Smaller base image |
| uv | Faster install |
| uv.lock | Reproducible dependencies |
| .dockerignore | Exclude notebooks, figures, datasets |
| Remove Jupyter from production | Smaller runtime |
| Separate development dependencies | Cleaner image |
| --no-cache | Remove package cache |
| PYTHONDONTWRITEBYTECODE | Avoid unnecessary files |
| Clean apt cache | Reduce layer size |

Expected image size

| Version | Size |
|----------|------|
| Original | 2.5–4 GB |
| Optimized | 700–900 MB |
| ARM64 | 650–850 MB |

---

# Best Practices / 最佳實務

Recommended files

```
Dockerfile
Dockerfile.cuda
docker-compose.yml
pyproject.toml
uv.lock
.dockerignore
DOCKER.md
```

Recommended GitHub Actions

```
.github/workflows/docker.yml
```

Automatically build

- AMD64 image
- ARM64 image
- Multi-platform image

and publish to

- Docker Hub
- GitHub Container Registry (GHCR)

---

# Conclusion / 結論

Ricci Finance V16 adopts a modern Docker workflow based on:

- Multi-stage builds
- `uv` dependency management
- Multi-platform (AMD64 + ARM64) support
- Streamlit deployment
- Optional NVIDIA CUDA and Apple Metal (MPS) acceleration
- Optimized image size (~700–900 MB)
- Reproducible and portable research environments

This deployment strategy makes Ricci Finance V16 suitable for personal development, academic collaboration, cloud deployment, and production research workflows.


