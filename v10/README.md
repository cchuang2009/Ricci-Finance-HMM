# Ricci Finance v10.6 — Real-market 3D Ricci-Capital Manifold + HMM

Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Includes:
- real Yahoo Close + Volume via `download_market_data(auto_adjust=False)`
- dollar-volume node mass and capital-weighted edge transport
- 2D rolling animation
- dynamic 3D Ricci-capital animation
- HMM hidden regimes from Ricci + capital-flow features
- surgery-risk direction score, not actual graph cutting

Notebook:

```bash
jupyter notebook ricci_finance_v10_lecture.ipynb
```
