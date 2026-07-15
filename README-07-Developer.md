# Developer Guide

New functionality should be added to a focused module instead of expanding
`helper.py`. Keep `helper.py` only for backward compatibility.

## Development checks

```bash
python -m compileall ricci_finance app_v12.py
python smoke_test.py
streamlit run app_v12.py
```

## Streamlit API

Use the current width syntax:

```python
st.dataframe(df, width="stretch")
st.plotly_chart(fig, width="stretch")
```

Do not add new `use_container_width=True` calls.

## HMM compatibility

Use:

```python
model.fit(X_scaled)
states = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
```

Do not assume that the installed `hmmlearn` version implements `fit_predict()`.

---

**Documentation:** [Documentation Home](README.md) · [Introduction](README-01-Introduction.md) · [Installation](README-02-Installation.md) · [Architecture](README-03-Architecture.md) · [Mathematics](README-04-Mathematics.md) · [Modules](README-05-Modules.md) · [Lecture Guide](README-06-Lecture.md) · [Developer Guide](README-07-Developer.md) · [Future Development](README-08-Future.md)

← [Lecture Guide](README-06-Lecture.md) | [Documentation Home](README.md) | [Future Development](README-08-Future.md) →
