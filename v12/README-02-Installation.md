# Installation

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
git clone <repo>
cd Ricci-Finance-v12
uv venv
source .venv/bin/activate
uv sync
uv run streamlit run app_v12.py
```

## Jupyter

```bash
uv sync --extra lecture
uv run jupyter lab
```
