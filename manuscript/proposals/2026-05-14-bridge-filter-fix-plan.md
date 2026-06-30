# Bridge-filter fix plan (Option C: shrinkage + tunable α + honest tradeoff)

**Status:** Draft, awaiting verifier subagent review before implementation.
**Date:** 2026-05-14.
**Author:** Claude (under direction of Oliver Muellerklein).
**Context:** The REAP vs Parametric UMAP bridge comparison on 20NG (see TRACE
session `trace_20260514_d86306` and verification result) revealed that REAP's
conformal filter at α=0.01 rejects 95% of held-out OOS, *including*
semantically-related OOS classes that should plausibly bridge. The root cause
is a geometric pathology of REAP's consensus embedding (per-cluster covariance
eigenvalues spanning 4 orders of magnitude → degenerate inverse-covariances →
Mahalanobis D² inflated by 1–3 orders of magnitude on OOS projections).

This plan implements **Option C**: add shrinkage-regularised covariance to the
filter, sweep α, and report a sensitivity–specificity tradeoff curve instead
of a single point estimate. The bridge claim is reframed as
"tunable-conservatism OOD detection" — α and shrinkage are user controls, not
a one-size-fits-all default.

---

## 1. Problem statement (verifier check these claims)

### 1.1 The empirical finding

| | REAP | PUMAP |
|---|---|---|
| Median Mahalanobis D² on OOS | 804 | 21 |
| 99th percentile D² | 232,000 | 232 |
| α=0.01 accept rate on 600 OOS | 5.17% | 57.7% |
| Accept on semantically-related OOS (e.g., `talk.religion.misc` → religion cluster) | 2% | 70% |
| Accept on topic-disjoint OOS (e.g., `misc.forsale`) | 16% | 64% |

### 1.2 The mechanism

REAP's consensus distance matrix `D̄ = (1/n_seeds) Σ_i D_i` produces a UMAP
embedding whose per-cluster covariance has highly anisotropic eigenvalues.
Example from set A, cluster 1: eigenvalues
`[1e-4, 9e-4, 3.6e-3, 2.8e-2, 7.1e-1]` — four orders of magnitude. The
inverse covariance has entries ~1e4 along the thinnest directions. Any OOS
projection drift in those directions inflates D² catastrophically.

In contrast, ParametricUMAP's encoder produces isotropic, well-conditioned
clusters. Reference-LOO α-quantile thresholds extend to OOS docs smoothly.

### 1.3 What this means for the manuscript

The current REAP filter at default α implements **strict OOD rejection**, not
**graded semantic acceptance**. For applications where false-accept is more
costly than false-reject (e.g., "did this document drift outside my training
distribution?"), REAP is correct. For applications where graded acceptance of
semantically-related foreign data is desired (e.g., "where does this 2024 doc
sit in my 2015–2020 reference space?"), REAP-as-shipped is too conservative.

---

## 2. Proposed fix: three-component change

### 2.1 Component A — Ledoit-Wolf shrinkage covariance (new filter variant)

Add `method="mahalanobis_shrunk"` to `reap.filter.calibrate`. Per-cluster
covariance becomes:

```
Σ_shrunk = (1 − γ̂) · Σ_sample + γ̂ · (trace(Σ_sample)/d) · I
```

where γ̂ is the Ledoit-Wolf optimal shrinkage parameter, estimated via
`sklearn.covariance.LedoitWolf().fit(cluster_points).shrinkage_`. The
identity-target shifts all eigenvalues toward the cluster's mean eigenvalue,
eliminating the near-degenerate directions.

**Implementation:** ~30 LOC in `src/reap/filter.py`. Same LOO-quantile
threshold-fitting logic but with `Σ_shrunk^{−1}` instead of `(Σ_sample + ridge·I)^{−1}`.

**Tests** (add to `tests/test_filter.py`):
1. On a well-conditioned synthetic cluster, shrunk ≈ no_correction (γ̂ ≈ 0).
2. On a degenerate synthetic cluster (eigenvalues spanning 1e-4 to 1.0),
   shrunk has condition number < 100 while no_correction has > 1e3.
3. The shrunk filter's LOO-quantile on in-distribution data still recovers
   the nominal coverage (within ±0.02 of (1−α) on 50-iter Monte Carlo).
4. Shrunk on truly OOD synthetic data still rejects most points (sanity).

### 2.2 Component B — match cluster count K between methods

Current asymmetry: REAP picks k=7 (internal silhouette), PUMAP uses k=8 (CLI
default). Fix by adding `--k 8` to both bridge scripts so both methods cluster
the same reference at the same K. Defensible value: 8 = ground-truth K for
20NG-reference.

For REAP, this means using KMeans-at-k=8 on the consensus embedding *instead
of* `consensus_labels.npy` (which was k=7). Document this asymmetry honestly:
the consensus EMBEDDING is REAP's; the cluster ASSIGNMENT is KMeans-at-8 for
fairness.

### 2.3 Component C — α sweep + sensitivity–specificity tradeoff

Pre-register OOS classes into two semantic groups:

**Related (have a near-neighbour in the 8-class reference):**
- `alt.atheism` ↔ `soc.religion.christian`
- `comp.os.ms-windows.misc` ↔ `comp.sys.ibm.pc.hardware` / `comp.sys.mac.hardware`
- `comp.windows.x` ↔ `comp.sys.*` / `comp.graphics`
- `rec.sport.baseball` ↔ `rec.sport.hockey`
- `sci.crypt` ↔ `talk.politics.guns` (security adjacent) — *marginal*
- `sci.electronics` ↔ `comp.sys.*`
- `talk.politics.misc` ↔ `talk.politics.guns` / `talk.politics.mideast`
- `talk.religion.misc` ↔ `soc.religion.christian`

**Disjoint (no near-neighbour in reference):**
- `misc.forsale`
- `rec.autos`
- `rec.motorcycles`
- `sci.med`

The marginal labels (sci.crypt) are flagged in the plan so reviewers can see
where the categorisation is contestable. Reported metrics:

- **Sensitivity** (TPR on "related"): fraction of `related` docs accepted.
  Higher is better — captures graded acceptance.
- **Specificity** (TNR on "disjoint"): fraction of `disjoint` docs rejected.
  Higher is better — captures strict OOD rejection.

For each (method, shrinkage, set, K) combination, sweep α ∈ {0.01, 0.05, 0.10,
0.20} and plot (Sensitivity, Specificity) points. Higher-α points should move
*right* on the curve (higher sensitivity at the cost of lower specificity).

A method dominates if its curve lies above-and-right of the other's at every α.

### 2.4 What this fix DOES NOT change

- REAP's consensus embedding construction (distance-matrix averaging).
- REAP's projection head.
- The between-set ARI win on KF + AI-art (Pillar 1 stability claim).
- The other 6 baselines' results.

The change is local to the filter step + the bridge experiment's reporting.

---

## 3. Pre-registration update (BEFORE running)

Append to `manuscript/evaluation_protocol.md` §16 (Filter design):

> **Variant 6 (added 2026-05-14): mahalanobis_shrunk.** Identical to
> mahalanobis_no_correction except the per-cluster sample covariance is
> replaced by its Ledoit-Wolf shrunk estimate (`sklearn.covariance.LedoitWolf`).
> Motivation: REAP's distance-matrix-averaged consensus produces near-
> degenerate per-cluster covariances on some datasets; the LOO-quantile
> threshold becomes brittle to OOS projection error along thin directions.
> Pre-registered observed band for sensitivity (graded acceptance on labelled
> 20NG-OOS related-class subset): 0.30–0.80 at α=0.10; specificity ≥ 0.80
> at α=0.01.
>
> **α sweep: at α ∈ {0.01, 0.05, 0.10, 0.20}** reported as a
> sensitivity–specificity tradeoff curve for both REAP and the Parametric
> UMAP comparator.

Log as TRACE decision before running.

---

## 4. Implementation plan (in execution order)

### Phase 1 — Code changes (~1 day)

1. **[code]** Add `mahalanobis_shrunk` method to `src/reap/filter.py`. New
   internal helper `_compute_shrunk_inv_cov(cluster_points, ridge)`. Update
   the dispatch logic in `calibrate`. ~30–50 LOC.
2. **[tests]** Add 4 unit tests + 1 golden test in `tests/test_filter.py`
   (see §2.1 list).
3. **[reg]** Update `manuscript/evaluation_protocol.md` §16 with Variant 6
   description + pre-registered ranges + α sweep grid.
4. **[code]** Refactor bridge scripts to support filter-only sweeps. Extract
   train + project as separate functions that cache results; filter-application
   becomes a fast inner loop over (method, α) combos.
5. **[code]** Define `OOS_CLASS_GROUPS` constant (related/disjoint/marginal
   mapping) as a project-level constant in `src/reap/datasets/twenty_newsgroups_oos.py`
   or a new `bridge_taxonomy.py`. Single source of truth.
6. **[lint]** pyright + ruff clean on all new code.

### Phase 2 — Verification (parallel with code; ~half day)

1. **[subagent A]** Code review of the shrunk filter: correctness of
   covariance math, edge cases, numerical stability, test coverage.
2. **[subagent B]** Re-derive the Ledoit-Wolf formula and confirm sklearn's
   implementation matches what we need. Confirm shrinkage of a known
   ill-conditioned synthetic example produces well-conditioned output.
3. **[CLI]** Full pytest suite green; golden-validation tier still passes.

### Phase 3 — Sweep + analysis (~half day compute)

Compute matrix:
- 2 methods (REAP, PUMAP)
- 2 shrinkage settings (no_correction, shrunk)
- 4 α values (0.01, 0.05, 0.10, 0.20)
- 3 sets (A, B, C)
- 1 K (=8; defer K=7 to a supplementary table if reviewers ask)

= 48 cells. Each REAP cell is filter-only (~seconds; head + consensus already
trained). Each PUMAP cell requires PUMAP refit (~2 min); 6 PUMAP refits × 3
sets × 1 K = 18 min total PUMAP compute. Filter sweep is trivial.

Output: one CSV with columns `(method, set, k, shrinkage, alpha, n_accepted,
sensitivity, specificity, n_related, n_disjoint, …)` covering all 48 rows.

### Phase 4 — Verification of results (~quarter day)

1. **[subagent C]** Spot-check a random row from the sweep CSV: recompute
   sensitivity / specificity / accept rate from the per-doc CSVs;
   confirm match.
2. **[subagent D]** Read the tradeoff plot script + the per-class accept rates
   at each α; flag if anything looks fishy (e.g., monotonicity violations,
   sensitivity > 1, etc.).

### Phase 5 — Manuscript prose (~1 day, deferred)

After Phase 4 passes:
1. Update §3.6 (filter design) with the shrinkage variant + motivation.
2. Replace §4.7 (filter validation) with the tradeoff-curve story.
3. Update §5 limitations: filter is tunable, reflects user's tolerance for
   false-reject vs false-accept.
4. New figure: sensitivity–specificity tradeoff curve (or table).

Total estimated effort: ~3 working days end-to-end, mostly Phase 1 + Phase 3.

---

## 5. Questions for the verifier subagents

The verifier subagents should sanity-check this plan before code is written:

1. Is Ledoit-Wolf the *best* shrinkage estimator for this use case, or should
   we consider OAS, GraphicalLasso, or a fixed-γ regulariser?
2. Is there a simpler fix we are missing — e.g., per-cluster PCA whitening
   prior to threshold computation? Or rank-deficiency detection + filter
   rejection of those clusters entirely?
3. Are the "related" vs "disjoint" labels for OOS classes defensible? Would a
   skeptical reviewer object to the manual taxonomy?
4. Is the α sweep grid right ({0.01, 0.05, 0.10, 0.20})? Should we include
   smaller (0.005) or larger (0.30) values?
5. Should sensitivity be measured on the per-document level or the per-class
   level (e.g., majority-vote per class)?
6. Are we missing a comparator? Should we also report a no-filter baseline
   (i.e., the projection head's R²-on-OOS without conformal rejection)?
7. Does the K=8 matched choice introduce its own asymmetry (since REAP's
   internal selection picked K=7)? Should we run both K=7 and K=8?

---

## 6. Anti-goals (things this plan does NOT attempt)

- We are NOT changing REAP's core consensus algorithm. The geometric pathology
  in REAP's consensus is a known and acceptable side effect of distance-matrix
  averaging on certain data; the filter is the layer that should adapt, not
  the consensus.
- We are NOT removing the existing `mahalanobis_no_correction` and
  `mahalanobis_pooled` methods. Shrinkage is additive.
- We are NOT changing the 5 other baselines or the silhouette/trustworthiness
  reporting. Those are separate stories.

---

## 7. Success criteria

- After Phase 1: shrunk filter passes all unit + golden tests; full pytest
  green; pyright + ruff clean.
- After Phase 3: REAP with shrinkage at α=0.10 achieves sensitivity ≥ 0.30
  on 20NG-OOS related classes while maintaining specificity ≥ 0.50 on
  disjoint classes. (Pre-registered band.)
- After Phase 3: REAP's tradeoff curve at some α offers a better
  sensitivity–specificity Pareto-point than PUMAP at the same α, OR we
  honestly report that PUMAP offers a better tradeoff for graded acceptance
  while REAP retains the strict-rejection regime advantage.
- After Phase 4: independent verifier confirms numbers and methodology.
- After Phase 5: manuscript prose accurately reflects the tradeoff story
  without overclaiming.
