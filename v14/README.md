# Ricci Finance v14 Streamlit App

Bilingual yfinance-only Ricci-network and HMM/Viterbi research app.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app downloads market data, builds rolling Ricci-capital graphs, fits a diagonal Gaussian HMM, displays Viterbi and posterior regimes, renders a 3-D regime-colored snapshot, and exports the feature table.
