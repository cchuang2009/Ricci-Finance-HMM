# Installation

## Recommended environment

```bash
uv venv
source .venv/bin/activate
uv sync
```

If the project does not yet contain `pyproject.toml` and `uv.lock`, use:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app_v12.py
```

## Test the installation

```bash
python smoke_test.py
```

A successful test prints:

```text
v12 Phase 3 HMM story smoke test passed
```

---

**Documentation:** [Documentation Home](README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Modules](README-05-Modules.md) · [Lecture Guide](README-06-Lecture.md) · [Developer Guide](README-07-Developer.md) · [Future Development](README-08-Future.md)

← [Introduction](README-01-Introduction.md) | [Documentation Home](README.md) | [Architecture](README-03-Architecture.md) →
