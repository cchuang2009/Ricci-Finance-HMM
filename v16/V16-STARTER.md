# Ricci Finance V16 Starter

V16 is built directly on the V15 codebase. Its first objective is to replace the single-sector assumption with normalized multi-sector profiles without breaking the existing V15 visualizations.

## New V16 objects

### `SectorProfile`

Defined in `ricci_finance/sector_objects.py`.

Each ticker has:

- official Yahoo sector
- official Yahoo industry
- business description
- normalized multi-sector memberships
- primary sector
- source: automatic, manual, or fallback

Example:

```python
SectorProfile(
    ticker="NVDA",
    official_sector="Technology",
    official_industry="Semiconductors",
    memberships={
        "Technology": 0.45,
        "Semiconductors": 0.25,
        "AI": 0.12,
        "DataCenter": 0.10,
        "Networking": 0.08,
    },
    source="automatic",
)
```

All memberships are normalized per ticker:

```text
sum(profile.memberships.values()) == 1.0
```

### Sector modes in `app.py`

- **Automatic**: Yahoo sector + industry + keyword themes from the business summary.
- **Hybrid**: automatic profiles, with complete manual overrides for tickers listed in the text area.
- **Manual**: all tickers must be supplied by the user.

Manual weighted syntax:

```text
NVDA=Semiconductors:6|AI:3|DataCenter:1
ANET=Networking:7|AI:2|DataCenter:1
MU=Memory:7|Semiconductors:3
```

The values do not need to sum to one. V16 normalizes them automatically.

## Backward compatibility

Existing V15 network and Galaxy functions still receive one primary sector per ticker. The primary sector is the membership with the largest normalized weight.

The following components use the full multi-sector vector:

- GNN weighted multi-hot node features
- sector momentum
- sector capital-flow matrix

Ricci curvature and HMM calculations remain unchanged.

## Current automatic detector

The starter detector is intentionally transparent:

1. Read Yahoo `sector`, `industry`, and `longBusinessSummary`.
2. Detect themes using an editable keyword taxonomy.
3. Blend official sector, official industry, and inferred themes.
4. Normalize all positive weights to one.
5. Fall back to `Other: 1.0` if metadata cannot be retrieved.

This is a V16 foundation, not a claim that keyword inference is final. Later V16 stages can add sentence embeddings, SEC segment revenue, and graph-community memberships.


## Main V16 notebook

Open `ricci_finance_v16_lecture.ipynb` for the complete bilingual V16 workflow.
