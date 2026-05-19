# Key validated results (historical reference)

This file captures the headline metrics from prior sibling-project runs
that motivated REAP's design and pre-registered evaluation protocol.
**These are not the manuscript's final numbers** — those will be
produced by the protocol §7 30-seed × 3-seed-set sweep on the locked
snapshots under `~/.cache/reap/datasets/`. Treat this file as a
reference baseline, useful for sanity-checking that newer runs land in
the same neighborhood.

When the full benchmark sweep produces final numbers, results land
under `results/<dataset>/<method>/<seed_set>/metrics.csv` per
evaluation_protocol.md §7, with sibling `bundle.json` files per §12.
Those committed CSVs — not this file — are what the manuscript cites.

---

## Korean forest policy

Source: `green-narrative/hye_in/data/consensus_analysis/`
(`analysis_metadata.json` "winner" config), 30 seeds, random_state 42.

| Param | Value |
|---|---|
| n_components (d) | 18 |
| n_neighbors (nn) | 19 |
| min_dist (md) | 0.005 |
| Selected K | 8 |

| Metric | Value |
|---|---|
| Trustworthiness | 0.892 |
| Silhouette | 0.841 |
| Distance-matrix ARI | 0.75 (+32% vs Procrustes 0.56) |

REAP cache equivalent: `load_korean_forest()` — 905 × 384 multilingual MiniLM.

**Smoke check (5 seeds, this repo, 2026-04-14):** Trustworthiness 0.890.
Within 0.002 of the validated 30-seed reference, as expected for a
reduced-seed run with otherwise-identical params.

---

## AI-art discourse

Source: `When-Algorithms-Meet-Artists/figures/config_comparison/all_results.json`
"Best combined" config, 30 seeds.

| Param | Value |
|---|---|
| n_components (d) | 5 |
| n_neighbors (nn) | 53 |
| min_dist (md) | 0.01 |
| K | 20 |

| Metric | Value |
|---|---|
| Trustworthiness | 0.916 |
| Silhouette | 0.657 |
| Projection R² | 0.904 |

REAP cache equivalent: `load_ai_art()` — 1,736 × 1,024 e5-large-v2 (`query: ` prefix).

---

## Out-of-Sample Conformal Filter — Korean forest validation (2026-05-09)

Source:
`green-narrative/hye_in/for_hyein/park_moon_results_2026-05-07_v2/`.
Validated 2026-05-09 against the headline filter design that ships in
`reap.filter` (manuscript §3.6, §4.7). Decision provenance and rejected
alternatives in `manuscript/supplementary/oos_filter_design_decisions.md`;
cross-project TRACE sessions `trace_20260509_305aaf` (decisions and
rejected alternatives) and `trace_20260509_c4f4ac` (Hye In handoff
contributions).

**Setup.** Reference: 905 Korean forest planning sentences (the same
corpus exposed by `load_korean_forest()`), reduced to the 18-d REAP
consensus space (the validated configuration above). Out-of-sample:
1,662 district-level political pledge sentences from three South
Korean presidential election cycles (Lee = 480, Park = 633, Moon = 549),
embedded with the same MiniLM model and projected via REAP's projection
head (§3.4). Both corpora live in `green-narrative/hye_in/`; the OOS
corpus is targeted for inclusion in REAP's dataset cache as
`load_korean_forest_oos()` (currently sibling-project artifact only).

| Filter design choice | Value |
|---|---|
| Distance metric | Mahalanobis (per-cluster reference covariance Σ_c) |
| Location correction | Pooled (μ̂_c = mean of OOS points in cluster c, pooled across presidents) |
| Threshold | Empirical leave-one-out at α = 0.01 (reference LOO Mahalanobis quantile per cluster) |
| Reference n | 905 |
| OOS n | 1,662 |
| K | 8 |

| Filter retention | Value |
|---|---|
| Lee (n=480) | 238 (49.6%) |
| Park (n=633) | 320 (50.6%) |
| Moon (n=549) | 288 (52.5%) |
| **Overall (n=1,662)** | **846 (50.9%)** |

**Five-variant head-to-head:** Mahalanobis vs Euclidean (both with
pooled correction) disagree on 27.5% of points — shape adaptation is
doing real work. Pooled vs per-president location correction disagree
on only 5.2% — pooled is preferred for simplicity. Source CSVs:
`five_variant_retention.csv`, `five_variant_disagreement_matrix.csv`,
`five_variant_per_cluster.csv`.

**Per-cluster retention** (Mahalanobis + pooled correction, α = 0.01)
ranges from 30.2% (Cluster 3 Urban Forestry) to 87.9% (Cluster 2
Forest Welfare). Variability reflects real differences in OOS-vs-reference
topic alignment, confirmed by qualitative review (`text_analysis_preliminary.md`).

**Status of these numbers in the REAP manuscript.** This is a
sibling-project validation of the filter design. For the manuscript's
final filter numbers, the same harness must be run against a locked
snapshot of the OOS corpus committed under `~/.cache/reap/datasets/`,
with a `bundle.json` capturing the seed-set, commit hash, and
environment. Domain-expert (Hye In Kim) review of the per-pledge
KEEP/REMOVE decisions is in progress; her judgments inform but do not
replace the harness re-run.

---

## Notes

- Numbers above are reference checkpoints from the sibling projects, not REAP-internal benchmarks.
- A fresh REAP run on the same data should land near these numbers; large drift would indicate a regression in the consensus pipeline (or, less commonly, in the loader).
- The pre-registered Tier-2 ranges in `evaluation_protocol.md` §6.c apply to the 20newsgroups golden fixture, not to these real datasets. Real-dataset ranges are not pre-registered and are reported as observed across the full 30 × 3 sweep, with bootstrap CIs.
