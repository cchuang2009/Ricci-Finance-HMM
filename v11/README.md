# Ricci Finance v11

Graduate lecture demo for dynamic Ricci-capital financial networks.

## Main fixes from v10

1. `app.py` imports `compute_components` from `helper.py`, fixing the missing import in the 3D after-flow view.
2. All network figures now include a fixed Ricci-curvature colorbar.
3. Edge color has consistent meaning across static 2D, rolling 2D animation, static 3D, dynamic 3D animation, and Ricci-flow before/after views.
4. Ricci-flow before/after comparison uses the defined `visualize_network_3d()` function.
5. Notebook text is upgraded to v11 and explains the visual encoding rule.

## Run Streamlit

```bash
pip install streamlit yfinance pandas numpy networkx plotly matplotlib scikit-learn hmmlearn
pip install GraphRicciCurvature pot networkit   # optional, for true Ollivier-Ricci
streamlit run app_v11.py
```

## Color meaning

- Red edge: negative Ricci curvature, fragile bridge / stress route.
- White edge: near-zero curvature.
- Blue edge: positive Ricci curvature, coherent local capital basin.
- Edge width: estimated dollar-volume capital transport.
- Node size: dollar-volume market mass.
- Node color: graph cluster/component.
