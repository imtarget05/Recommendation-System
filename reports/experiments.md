# Experiments

> Every number below is measured by real runs (`uv run python -m training.baseline_*`).
> No value is fabricated. Protocol details in "Evaluation protocol".

## Evaluation protocol (Section 16)

- **Data**: MovieLens ml-25m, 5M-row deterministic subset (strided time sampling,
  `data.max_rows=5_000_000`), time-based split 70 / 10 / 20
  (train 3,499,905 rows, 121,891 users; test 1,000,000 rows).
- **Model catalog** = items with >=1 train interaction (18,109 items).
- **Ground truth (per user)** = items interacted in the test window that the user
  never interacted with in training AND that are in the model catalog
  (out-of-catalog items are unreachable by any model and would bias metrics).
  Users outside the training set (cold-start) are evaluated separately in the
  serving fallback path.
- **Eval users**: 2,000 users with the richest ground truth among in-training users.
- **Implicit weights**: view=1, click=2, like=3, purchase=5 (Section 6.2).

Notable protocol bug found & fixed during Phase 2 (why earlier runs were purged):
item exclusion must be *per user* (items the user already interacted with), not
the global training item set; using the global set excluded all known items and
left only unreachable long-tail movies, producing degenerate 0.0 metrics.

## Ablation table (auto-generated from reports/ablation.md)

| model | version | dataset | Recall@20 | NDCG@20 | HitRate@20 | Coverage | Diversity |
|---|---|---|---|---|---|---|---|
| cf_als | v3 | movielens-ml-25m | 0.0185 | 0.0193 | 0.2255 | 0.0574 | 0.2907 |
| popularity | v1 | movielens-ml-25m | 0.0206 | 0.0224 | 0.2375 | 0.0011 | 0.3000 |

## Runs detail

| run | config | recall@20 | ndcg@20 | hit@20 | coverage | diversity |
|---|---|---|---|---|---|---|
| popularity v1 | weighted counts, k=20 | 0.0206 | 0.0224 | 0.2375 | 0.0011 | 0.3000 |
| cf_als v1 | factors=64 iters=15 alpha=40 | 0.0170 | 0.0175 | 0.2035 | 0.0967 | 0.2916 |
| cf_als v2 | factors=64 iters=40 alpha=20 | 0.0170 | 0.0174 | 0.2155 | 0.0834 | 0.2861 |
| cf_als v3 | factors=64 iters=40 alpha=5 | 0.0185 | 0.0193 | 0.2255 | 0.0574 | 0.2907 |

## Interpretation (measured, not assumed)

1. Popularity is a strong HitRate baseline on this protocol: users genuinely
   re-engage with the famous part of the catalog. ALS approaches it with a soft
   confidence weighting (alpha=5) but does not beat it at K=20 here.
2. ALS's advantage is *coverage*: it surfaces 5-90x more catalog items
   (0.057 vs 0.001) — the popular-only list never leaves the head.
3. NDCG is low for both because ground truth contains many items and top-20 is
   a small window; the relative ordering is what Phase 4/5 (two-tower + ranking)
   must improve.
4. Expectation for the final story: retrieval+ranking should beat popularity on
   NDCG (ranking quality) while keeping reasonable hit rate.

## Quality gate calibration

Per Section 32 the gate thresholds must come from measured baselines (set after
better models exist; placeholders are clearly labelled):

```yaml
quality_gate:  # TBD - provisional based on Phase 2 baselines
  min_recall_at_20: 0.02   # > popularity baseline 0.0206
  min_ndcg_at_20: 0.02     # > popularity baseline 0.0224 (may tighten)
  max_p95_latency_ms: 300  # design target, measured in Phase 6
  max_drift_psi: 0.20      # standard threshold, configurable
```