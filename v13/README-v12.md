# Ricci Finance v12 — Phase 3 HMM Story Manifold

Phase 3 turns the rolling graph animation into a synchronized HMM market story.

## New features

- `GaussianHMM.fit()` followed by `predict()` for compatibility with hmmlearn
  releases that do not provide `fit_predict()`.
- posterior state probabilities from `predict_proba()`;
- HMM probability attached to every `FrameData` object;
- automatic comparison between each frame and the previous frame;
- evidence-based frame headline and narrative;
- strongest negative-curvature edge;
- largest capital-transport edge;
- animation title updated with date, regime, and HMM confidence;
- animation-side story annotation;
- selected-frame story dashboard;
- HMM posterior-probability history;
- all Streamlit `use_container_width` calls replaced by `width="stretch"`.

## New module

```text
ricci_finance/story.py
```

Important functions:

```python
stories = build_frame_stories(frames)
story_table = frame_story_table(stories)
```

## Run

```bash
pip install -r requirements.txt
streamlit run app_v12.py
```

## Interpretation

The story is descriptive, not causal. It reports changes that are supported by
the calculated graph statistics and HMM posterior probabilities. It does not
claim that a stress regime guarantees a future market decline.
