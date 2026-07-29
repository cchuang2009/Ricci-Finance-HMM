# Migration from V15.2

1. Create or activate the V15 virtual environment.
2. Run `./uninstall_forbidden.sh`, or manually uninstall TensorFlow, Keras, POT and GraphRicciCurvature.
3. Install `requirements.txt`.
4. Run `python check_install.py`.
5. Start with `streamlit run app.py`.

The old `method="OTD"` and `proc=1` arguments no longer exist. Use:

```python
build_rolling_frames(..., curvature_engine="forman")
build_rolling_frames(..., curvature_engine="ollivier_lp", alpha=0.5)
```
