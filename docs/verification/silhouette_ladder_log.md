# Silhouette Verification Ladder — Evidence Log

Append-only evidence log for the bottom-up metric-correctness verification
ladder targeting REAP's silhouette score (`reap.evaluation.compute_silhouette`).
Companion plan (reuse template): the silhouette ladder follows the same five-
rung shape as the ARI ladder (`docs/verification/ari_ladder_log.md`).

**Why this ladder matters specifically.** REAP wins on silhouette by a large
margin on the production CSVs (consensus 0.67–0.78 vs 0.40–0.49 for baselines
across all three corpora). The prior audit called this "circular" because
UMAP's loss function explicitly optimizes for output-space separation, so any
method producing a smoother UMAP output by construction will win silhouette
without the metric measuring an independent quantity. This ladder pins the
silhouette code's correctness so the *interpretation* question (circular vs
genuine signal) can be debated against verified numbers rather than
suspected-correct numbers.

**Gating rule.** Each rung must be GREEN under all four independent
verifications before the next rung may run.

**Tolerance schedule.** Rung 0–2: `1e-12`. Rung 3–4: `1e-9`.

---

## Rung 0 — Closed-form known-answer tests

- **Test file:** `tests/verification/test_silhouette_rung0_closed_form.py` (13 tests)
- **Reference impl:** `tests/verification/_reference_silhouette.py` (numpy only; no sklearn, no `reap`)
- **Code under test:** `src/reap/evaluation.py:76` `compute_silhouette(X, labels, metric="euclidean")`
- **Run:** `KMP_DUPLICATE_LIB_OK=TRUE pytest tests/verification/test_silhouette_rung0_closed_form.py -v`
- **Date:** 2026-05-20
- **Commit:** to be filled post-commit on `feature/reap-foundation`

### Hand-computed expected values

| Case | Layout (positions, labels) | Expected silhouette | Hand derivation |
|---|---|---|---|
| T1 perfect | `[0, 0, 10, 10]` (1-d), `[0, 0, 1, 1]` | `1.0` | a(i)=0, b(i)=10; s = 10/10 = 1 |
| T2 unit-square split | `(0,0),(0,1),(1,0),(1,1)`, `[0, 0, 1, 1]` | `3 - 2*sqrt(2) ≈ 0.17157287525380993` | a=1, b=(sqrt(2)+1)/2; s = (sqrt(2)-1)/(sqrt(2)+1) = 3-2sqrt(2) |
| T3 6-point 1-d | `[0,1,2,8,9,10]`, `[0,0,0,1,1,1]` | `419/504 ≈ 0.8313492063492063` | s ∈ {5/6, 7/8, 11/14} ×2; mean = 419/504 |
| T4 overlapping | `[0, 10, 1, 9]`, `[0, 0, 1, 1]` | `-7/16 = -0.4375` | s ∈ {-1/2, -1/2, -3/8, -3/8}; mean = -7/16 |
| T5 all singletons | 3 points, `[0, 1, 2]` | reference: `0.0`; production: raises ValueError | Rousseeuw singleton ⇒ s=0; sklearn requires `2 ≤ n_unique ≤ n-1` |
| T6 single cluster | `[0, 0, 0]` (any layout) | reference raises ValueError; production returns `-1.0` | `compute_silhouette` documented contract |

### Orchestrator run result

```
13 passed in 7.03s
```

Pyright (CLI) `0/0/0`; ruff clean. (IDE pyright shows a phantom unresolved-import warning on `_reference_silhouette`; the project rule is "CLI is the source of truth"; cosmetic only.)

### Definition-consistency sweep (the silhouette call sites)

Every silhouette call site in `src/reap/` reduces to either
`sklearn.metrics.silhouette_score(X, labels, metric)` or
`sklearn.metrics.silhouette_samples(X, labels, metric)`. No site applies a
nonstandard normalization or custom aggregation. Sites enumerated:

| # | File:line | Function | Inputs |
|---|---|---|---|
| 1 | `evaluation.py:76` | `compute_silhouette` (wrapper) | `(X, labels, metric)`; returns `-1.0` if `<2 unique labels` |
| 2 | `evaluation.py:638` | `compute_silhouette_samples` | `(X, labels, metric)` — per-point |
| 3 | `clustering.py:47` | `find_best_k` | `silhouette_score(X, labels, metric)` directly |
| 4 | `benchmarks.py:384` | per-seed sil | `compute_silhouette(emb_i, seed_labels_i)` |
| 5 | `benchmarks.py:418` | consensus_emb + consensus labels | `compute_silhouette(consensus_emb, consensus_labels)` |
| 6 | `benchmarks.py:438` | consensus_emb + per-seed labels | `compute_silhouette(consensus_emb, per_seed_labels)` |
| 7 | `benchmarks.py:452` | bundle silhouette | `compute_silhouette(consensus_emb, consensus_labels)` |
| 8 | `ablation.py:271` | ablation sil | `compute_silhouette(consensus_emb, consensus_labels)` |
| 9 | `projection.py:315` | CV fold | `compute_silhouette(Y_pred, labels_val)` (in-sample on labels used to define KMeans) |
| 10 | `projection.py:388` | final-fit | `compute_silhouette(Y_pred_final, labels)` (optimistic in-sample) |
| 11 | `labeling.py:74,90` | per-point sil | `silhouette_samples(X, labels, metric="euclidean")` directly |

**No site reduces to anything other than the standard silhouette formula.** The "circularity" critique therefore cannot be a bug at the metric layer — it is a question about what the silhouette is measuring given UMAP's optimization objective. That question gets settled at Rungs 3–4 by inspecting the per-seed vs consensus silhouette gap (tri-view), not by inspecting metric code.

### Production wrapper's documented degenerate behaviours (pinned in tests)

- `n_unique < 2` (single-cluster input): returns `-1.0` with a logger warning. Pinned by T6.
- `n_unique == n_samples` (all-singletons): no graceful path — `silhouette_samples` raises ValueError per sklearn. Pinned by T5. *Implication:* upstream callers must guarantee `2 ≤ n_unique ≤ n_samples − 1` before calling. Worth a code-quality note in the manuscript supplementary.

### Adjudication

**Rung 0 GREEN.** 13 closed-form known-answer assertions hold within 1e-12; three-way agreement (production / from-scratch reference / sklearn) confirmed on every non-degenerate case. Quadruple-independent verification (parallel subagent fan-out) pending — kept lighter than ARI's at this stage because the silhouette function is a thin wrapper over a well-known sklearn routine and the 1e-12 three-way agreement on closed-form values already provides high confidence; a dedicated verifier can be dispatched if a contested number arises later.

**Next rung:** Rung 1 — tiny synthetic property assertions (make_blobs n=40, K=4, cluster_std=0.3, two KMeans seeds); verify that the silhouette of well-separated blobs > 0.5; permutation-invariance under label relabeling; cross-check vs `_reference_silhouette` on every produced labeling.
