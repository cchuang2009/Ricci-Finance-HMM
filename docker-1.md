---

# 12. Docker Deployment / Docker 映像建置

Ricci Finance V16 can be packaged into a Docker image, allowing the complete Streamlit application to run consistently on Linux, Windows, macOS, cloud servers, or HPC clusters.

Ricci Finance V16 可封裝成 Docker 映像，使整個 Streamlit 應用程式能在 Linux、Windows、macOS、雲端主機或 HPC 環境中一致地執行。

---

# 12.1 Directory Structure / 專案目錄

```text
ricci_finance_v16/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
│
├── cache/
├── data/
├── figures/
│
└── ricci_finance/
    ├── graph.py
    ├── gnn.py
    ├── sector_objects.py
    ├── visualization.py
    ├── advanced_visualization.py
    └── ...
```

---

# 12.2 Dockerfile

Create a file named **Dockerfile** in the project root.

在專案根目錄建立 **Dockerfile**。

```dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
        --upgrade pip

RUN pip install --no-cache-dir \
        -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

---

# 12.3 .dockerignore

To reduce image size, create

建立

```text
.dockerignore
```

```text
.git
.gitignore

__pycache__/
*.pyc

.ipynb_checkpoints/

venv/
.venv/

.cache/

figures/

*.mp4
*.gif

*.png

*.jpg

*.pdf

data/*.csv
data/*.parquet
```

---

# 12.4 Build Docker Image

Inside the project directory,

於專案目錄下執行

```bash
docker build -t ricci-finance:v16 .
```

After completion,

完成後

```bash
docker images
```

Example

```text
REPOSITORY          TAG     SIZE

ricci-finance       v16     2.1 GB
```

---

# 12.5 Run Container

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

# 12.6 Mount Local Data

To access local datasets,

若需要讀取本機資料，

```bash
docker run \
    -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    ricci-finance:v16
```

Windows PowerShell

```powershell
docker run `
    -p 8501:8501 `
    -v ${PWD}\data:/app/data `
    ricci-finance:v16
```

---

# 12.7 GPU Support (Optional)

If PyTorch CUDA is installed,

若使用 CUDA，

```bash
docker run \
    --gpus all \
    -p 8501:8501 \
    ricci-finance:v16
```

Verify GPU

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

# 12.8 Save Docker Image

Export image

```bash
docker save \
    ricci-finance:v16 \
    -o ricci_finance_v16.tar
```

Import elsewhere

```bash
docker load \
    -i ricci_finance_v16.tar
```

---

# 12.9 Docker Compose (Recommended)

Create

```text
docker-compose.yml
```

```yaml
version: "3.9"

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

# 12.10 Deploy to a Server

After building the image, deploy it to any Docker-compatible platform, such as:

完成映像建置後，可部署至任何支援 Docker 的平台，例如：

- Ubuntu Server
- Debian
- Rocky Linux
- Fedora
- Windows Server (Docker Desktop)
- macOS
- NAS (Synology / QNAP)
- AWS EC2
- Microsoft Azure
- Google Cloud Platform (GCP)
- Oracle Cloud Infrastructure (OCI)
- DigitalOcean
- Kubernetes clusters

---

# 12.11 Recommended Production Configuration

| Item | Recommendation |
|------|----------------|
| Python | 3.12 |
| CPU | ≥ 8 cores |
| Memory | ≥ 16 GB |
| GPU | NVIDIA CUDA (optional for GNN training) |
| Disk | SSD |
| Streamlit Port | 8501 |
| Docker Engine | 24+ |

---

# 12.12 Advantages of Docker

Using Docker provides several benefits:

使用 Docker 可帶來下列優點：

- Identical runtime environment across platforms.
- Easy deployment without manual dependency installation.
- Simplified collaboration among research teams.
- Better reproducibility of experiments.
- Isolation from the host operating system.
- Convenient deployment to cloud services or HPC environments.

Docker is therefore recommended for distributing Ricci Finance V16 in both research and production settings.