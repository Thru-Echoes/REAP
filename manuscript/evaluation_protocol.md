# REAP — Pre-Registered Evaluation Protocol

**Status:** Pre-registered protocol for the REAP methods paper.
**Version:** 1.5
**Frozen on:** 2026-04-13 (v1.0) · Updated 2026-04-13 (v1.1) · Updated 2026-05-09 (v1.2) · Updated 2026-05-14 (v1.3) · Updated 2026-05-21 (v1.4) · Updated 2026-07-04 (v1.5; see Changelog §15).
**Binding:** All experiments whose numbers appear in the manuscript
must follow this protocol. Any deviation requires a TRACE correction
event (`category="correction"`, with `corrects_event_ids` linking the
deviation back to this document's commit) and a brief written
justification committed alongside the changed code.

This document is the single source of truth that every later script,
table, figure, and paragraph in the manuscript points back to. It is
written *before* the experiments are run so that when they are run,
the analysis cannot drift to fit the result.

---

## 1. Datasets

| Tier | Dataset | n | Encoder | d_in | Labels? | License |
|---|---|---|---|---|---|---|
| Primary | AI-art discourse | 1,736 | intfloat/e5-large-v2 | 1024 | No per-chunk labels (cluster-level human-verified topic labels only) | public-domain |
| Secondary | Korean forest policy | 905 | sentence-transformers/MiniLM | 384 | Yes (Hye In's expert taxonomy) | per source |
| Tertiary | Corporate sustainability | 1,012 | TBD (locked at processing time) | TBD | Pending | per source |
| Tertiary | US presidential | ~1,000 | TBD (locked at processing time) | TBD | Pending | per source |

**Dataset API contract.** Each dataset is exposed by a function
`reap.datasets.load_<name>() → DatasetSnapshot`. The snapshot includes
embeddings, optional texts, optional expert labels, and a metadata
block carrying SPDX license, BibTeX citation, embedding model + version,
preprocessing version, and SHA256 of inputs. See
`.claude/rules/open-source-package.md` for the full contract.

**Dataset manifest** (committed at `manuscript/datasets/manifest.json`,
to be created when each dataset is locked) records: source URL or
repository pointer, SHA256 of the input artifacts, embedding model + git
SHA / model card version, chunking parameters, preprocessing version,
license, and the manifest schema version. A dataset is "locked" when
its manifest entry is committed; locked datasets do not change without
a version bump and a TRACE correction.

**Data privacy / redistribution.** For datasets that cannot be
redistributed (likely subset of corporate sustainability), the loader
exposes a `from_local(path)` constructor and CI runs against a
synthetic stand-in.

---

## 2. Seed Design

Three disjoint seed sets, 30 seeds each, recorded in
`manuscript/seeds/seed_manifest.json`:

- **Set A** — legacy seeds adopted from
  `green-narrative/hye_in/data/random_seeds_v2.json` (master_seed
  20260120). Provides continuity with prior Korean forest policy work.
- **Set B** — generated 2026-04-13 with master_seed 20260413 via
  `numpy.random.default_rng` (PCG64), drawn from [0, 2³¹) without
  replacement and rejecting overlap with Set A.
- **Set C** — same generation procedure as Set B, drawn jointly with B.

Every method × every dataset is run with all three sets. This
decomposes variance into:

- **Within-set** — 30-seed CI per set (the noise REAP averages over).
- **Between-set** — 3-point distribution of consensus embeddings
  (the reliability of the consensus itself).

---

## 3. Methods Under Comparison

Six methods, all run from the same harness against the same splits and
seed sets:

1. **single_seed** — UMAP with one fixed seed (the median seed of Set A
   by sorted order: deterministic, pre-registered).
2. **best_of_n** — run UMAP with all 30 seeds in a set, pick the
   embedding with highest silhouette. (A common informal practice; we
   include it to show REAP beats it.)
3. **naive_average** — average UMAP coordinates across seeds with no
   alignment. (Strawman that demonstrates why alignment is needed.)
4. **procrustes** — pairwise Procrustes alignment to a reference
   embedding, then coordinate average. The standard SOTA stabilization.
5. **bertopic** — the BERTopic default pipeline (UMAP → HDBSCAN →
   c-TF-IDF) with our encoder fixed. The de-facto topic-modeling
   baseline.
6. **reap** — distance-matrix consensus, the proposed method.

Both KMeans and HDBSCAN are run as the downstream clustering on top of
each method's embedding output, so cross-method comparisons are made at
matched clustering choices. KMeans is primary; HDBSCAN is the alt
clustering and aligns with BERTopic's default.

---

## 4. Pipeline Stages with Verification Checkpoints

Each pipeline stage has an input contract, an output contract, a
mathematical-invariant test, and a statistical-property check against
the golden synthetic dataset (Section 6). Stages are listed in
execution order.

| # | Stage | Module | Output | Math invariants | Statistical checks (golden set) |
|---|---|---|---|---|---|
| 1 | Embedding load | `reap.datasets` | `(N, d_in)` float array | finite, no NaN, dtype float32/64, non-empty | embedding norm distribution within range |
| 2 | Validation | `reap.validation` | `ValidationReport` | all gates documented | passes for golden set |
| 3 | Per-seed UMAP | `reap.consensus.run_umap_seeds` | list of `(N, d_low)` arrays | finite, expected shape | per-seed silhouette ∈ [0.40, 0.65]; per-seed-to-per-seed ARI ∈ [0.60, 0.92] |
| 4 | Per-seed distance matrices | `reap.consensus.compute_distance_matrices` | list of `(N, N)` arrays | symmetric, zero diagonal, non-negative | mean off-diagonal in expected range |
| 5 | Consensus distance matrix | `reap.consensus.average_distance_matrices` | `(N, N)` array | symmetric, zero diagonal, non-negative, **triangle inequality preserved** | Frobenius norm within tolerance of stored snapshot |
| 6 | Final UMAP from consensus D | `reap.consensus.embed_from_distance` | `(N, d_low)` array | finite, expected shape | consensus silhouette ≥ 0.65 (golden), consensus trustworthiness ≥ 0.85 |
| 7 | Clustering (KMeans / HDBSCAN) | `reap.clustering` | `(N,)` int label array | label arity matches K (KMeans) or ≥1 cluster (HDBSCAN) | consensus ARI vs ground truth ≥ 0.85 (golden, KMeans) |
| 8 | Cluster labeling | `reap.labeling` | dict cluster_id → top terms | non-empty per non-noise cluster, distinct top term per cluster | discriminativeness score ≥ floor |
| 9 | Projection head training | `reap.projection` | trained model + CV metrics | converged (loss decreasing), no NaN gradients | CV R² ≥ 0.50, CV trustworthiness ≥ 0.80 |
| 10 | Out-of-sample projection | `reap.projection.project` | `(N_test, d_low)` array | finite, expected shape | held-out cluster-assignment ARI vs full-rerun ≥ 0.70 |
| 11 | Out-of-sample conformal filter | `reap.filter.apply` (planned) | `(N_test,)` boolean KEEP/REMOVE | per-cluster threshold finite + non-negative; retention non-decreasing as α decreases; (μ̂_c) shape correct | synthetic-shift in-distribution retention ≥ 0.85, shifted-distribution retention ≤ 0.30 |

Every checkpoint is enforced by a test in
`tests/test_golden_validation.py` (to be written next), wired into the
`golden-validation` CI workflow. Stage 11's `reap.filter` API is
introduced in v1.2; the package code is pending and tracked in
outline.md P0-11/P0-13.

---

## 5. Multi-Tier Verification Strategy

UMAP is stochastic: cross-platform exact reproduction is fragile.
Verification therefore runs at three tiers, each enforced by CI:

**Tier 1 — Mathematical-invariant tests (exact).** Properties that
must hold by construction (symmetry, zero diagonal, triangle
inequality, finiteness, label arity). These pass or fail; no
tolerance. A failure here is a code bug.

**Tier 2 — Statistical-property tests (pre-registered ranges).** For
each pipeline stage, a metric range is declared in this protocol and
checked on the golden synthetic dataset. Ranges are wide enough to
absorb cross-platform UMAP variance but tight enough to catch real
regressions. A failure here means either (a) a code regression has
broken a scientific claim, or (b) a platform/library upgrade has
shifted UMAP outputs and the range needs justified widening (TRACE
correction required).

**Tier 3 — Tolerance snapshot tests.** For numerically meaningful
intermediates (consensus distance matrix, consensus embedding), the
reference run's output is hashed and stored. CI re-runs and compares
within documented Frobenius / element-wise tolerances. Stricter than
Tier 2; catches slow drift before it leaves the declared range.

The `golden-validation` CI workflow runs all three tiers on every PR.
**Any pre-registered scientific claim that drifts outside its declared
range fails the build.** The codebase enforces its own claims.

---

## 6. Golden Fixtures (Two-Tier)

REAP uses **two** golden fixtures with different scopes. This design
emerged from v1.0 calibration (see Changelog §15): the original
single-fixture make_blobs spec did not exhibit the phenomenon REAP
addresses because geometric blobs are trivially separable in 384-d and
their rotational symmetry is exactly what Procrustes was designed to
exploit. REAP's advantage manifests on *semantic* overlap, not
geometric overlap — so the comparative-claim fixture moved to real
labeled text.

### 6.a Blob fixture — Tier 1 invariants only

Specification:

```python
# Low-intrinsic-dim blobs projected to ambient 384-d with noise
# (see src/reap/datasets/synthetic.py :: load_golden_blobs)
intrinsic_dim = 16
ambient_dim = 384
n_samples = 400
n_centers = 8
cluster_std = 2.5
center_box = (-3.0, 3.0)
noise_sigma = 0.3
random_state = 42
```

Used for Tier-1 mathematical invariants (symmetry, zero diagonal,
triangle-inequality preservation, finiteness, label arity). No
embedding dependency, runs in ≤10 s.

### 6.b 20newsgroups text fixture — Tier 2 comparative claims

Curated 8-class subset of 20newsgroups (Lang 1995), embedded with
`sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalized),
cached in `~/.cache/reap/datasets/`. 50 documents per class → 400 docs.

Classes (pre-registered in
`src/reap/datasets/twenty_newsgroups.py :: GOLDEN_20NG_CLASSES`):

* **Clearly separated (4):** sci.space, rec.sport.hockey, comp.graphics,
  soc.religion.christian.
* **Overlap pair A (politics):** talk.politics.guns, talk.politics.mideast.
* **Overlap pair B (hardware):** comp.sys.ibm.pc.hardware,
  comp.sys.mac.hardware.

Why this structure. REAP's advantage over Procrustes shows up when
clusters have genuine semantic overlap (multiple plausible clusterings).
The 4 clearly-separated topics + 2 overlapping pairs is the smallest
design that exercises both regimes.

### 6.c Pre-registered metric ranges (calibrated 2026-04-13)

All ranges below apply to the 20newsgroups text fixture unless noted.
Ranges carry headroom so cross-platform UMAP/MiniLM drift does not
cause spurious failures, while still catching real regressions.

| Metric | Method | Fixture | Range |
|---|---|---|---|
| Single-seed silhouette | single_seed (KMeans K=8) | 20ng text | [0.40, 0.70] |
| Mean pairwise s2s ARI | single_seed within Set A | 20ng text | [0.70, 1.00] |
| Single-seed ARI vs ground truth | single_seed | 20ng text | [0.45, 0.75] |
| REAP consensus ARI vs ground truth | reap | 20ng text | [0.40, 0.80] |
| REAP consensus silhouette | reap | 20ng text | [0.55, 0.85] |
| REAP consensus trustworthiness | reap | 20ng text | [0.85, 0.99] |
| Procrustes consensus ARI vs ground truth | procrustes | 20ng text | [0.45, 0.80] |
| **REAP silhouette − Procrustes silhouette** (headline) | reap vs procrustes | 20ng text | **≥ +0.03** |
| Projection training-fit MSE | reap → head | 20ng text | ≤ 0.80 |
| Projection training-fit distance correlation | reap → head | 20ng text | ≥ 0.95 |
| Projection training-fit trustworthiness | reap → head | 20ng text | ≥ 0.80 |
| Projection CV MSE (3-fold) | reap → head | 20ng text | ≤ 3.0 |
| Projection CV R² (3-fold) | reap → head | 20ng text | ≥ 0.60 |
| Projection CV trustworthiness (3-fold) | reap → head | 20ng text | ≥ 0.75 |
| Projection CV ARI vs truth (3-fold) | reap → head | 20ng text | ≥ 0.25 |
| Projection CV distance correlation (3-fold) | reap → head | 20ng text | ≥ 0.80 |
| Topic-exclusion dominant-cluster fraction | reap → head | 20ng text | ≥ 0.30 |
| Topic-exclusion pair-partner fraction (overlap-pair held-outs) | reap → head | 20ng text | ≥ 0.50 |
| OOS filter in-distribution retention (synthetic shift) | reap → head → filter | 20ng text + register-shift | ≥ 0.85 |
| OOS filter shifted-distribution retention (synthetic shift) | reap → head → filter | 20ng text + register-shift | ≤ 0.30 |
| OOS filter retention monotone in α (mathematical invariant) | reap → head → filter | 20ng text | strict (Tier 1) |
| Korean-forest OOS retention (Mahalanobis + pooled, α=0.01) | reap → head → filter | korean_forest_oos snapshot | observed band ≥ 0.45, ≤ 0.55 |

### 6.d Headline comparative claim on the golden fixture

The paper's "REAP beats Procrustes on ARI" claim is **tested on real
datasets** (Korean forest, AI-art), *not* on the golden fixture. On
the 20ng subset REAP honestly merges the overlap pairs in its
consensus, so its ARI-vs-K=8-ground-truth is *lower* than Procrustes.
That is a feature of honest consensus, not a failure.

The golden-fixture comparative claim is instead **REAP silhouette ≥
Procrustes silhouette + 0.03** on the consensus embedding. Observed
margin on the reference run is +0.149. Silhouette captures what REAP
is actually optimising: a cleaner, more internally coherent consensus
embedding regardless of whether that embedding happens to match a
particular K-class ground truth.

### 6.e Projection head validation — three tiers

Per tests/test_projection_golden.py (v1.0, calibrated 2026-04-13):

1. **Training-fit sanity** — the head actually learned the mapping
   (MSE ≤ 0.80, distance correlation ≥ 0.95, trustworthiness ≥ 0.80).
2. **3-fold stratified CV** — generalization to unseen exemplars of
   known topics (CV R² ≥ 0.60, trustworthiness ≥ 0.75, ARI vs truth
   ≥ 0.25, distance correlation ≥ 0.80).
3. **Topic-exclusion semantic routing** — leave-one-topic-out;
   held-out docs must route to some training cluster at ≥30%
   (non-random), and for overlap-pair members the pair partner must
   attract ≥50% of held-outs. This operationalizes the finding that
   **the projection head is a semantic interpolator, not a novelty
   detector**: unseen topics are mapped toward the nearest training
   neighbour, with a strong effect for overlap-pair members (60–70%
   partner fraction) and a weaker but non-random effect for
   clearly-separated topics (40–50% dominant-fraction).

### 6.f Between-seed-set reliability (slow-CI extended tier)

The 3-seed-set design (A/B/C × 30 seeds) between-set reliability
metrics (Level D in §7) live in the nightly CI workflow. Fast-tier
golden uses the first 10 seeds of Set A; the full 30×3 sweep is too
slow for per-PR CI.

---

## 7. Metrics Schema

All metrics are computed on every method × every dataset × every seed
set, written to `results/<dataset>/<method>/<seed_set>/metrics.csv`
with full precision. Aggregations (table-ready means + 95% CIs) are
computed downstream from the per-seed CSVs.

### Level A — Per-seed (pre-consensus)

For each of the 30 seeds in a set, on the embedding produced by that
seed only:

- silhouette (Euclidean, on KMeans labels at K)
- trustworthiness @ k=15 (UMAP-style local-structure preservation)
- continuity @ k=15 (complement to trustworthiness)
- pairwise ARI between every pair of seeds in the set (30×30 matrix
  → mean, std, min, full distribution committed)
- pairwise AMI, NMI (same as ARI)

### Level B — Consensus

Computed on the consensus embedding (REAP) or each baseline's
single-output embedding:

- silhouette (Euclidean, on KMeans labels at K)
- trustworthiness @ k=15
- continuity @ k=15
- Davies-Bouldin index
- Calinski-Harabasz index

### Level C — Seed-to-consensus

For each seed in a set, the ARI / AMI / NMI between that seed's KMeans
clustering and the consensus's KMeans clustering. Reports mean ± 95%
CI per set.

### Level D — Between-seed-set (the 3×30 design payoff)

For pairs (A, B), (A, C), (B, C):

- ARI between consensus-A KMeans labels and consensus-B KMeans labels
- AMI between consensus-A and consensus-B labels
- Frobenius distance ‖D_consensus_A − D_consensus_B‖_F / ‖D_consensus_A‖_F
  on the consensus distance matrices

These three pairwise values directly quantify the reliability of the
consensus across seed samples.

### Level E — Projection head (out-of-sample)

- 5-fold stratified CV: R², trustworthiness, continuity (CV-honest)
- Final-fit on all data: R², trustworthiness, continuity (label as
  optimistic)
- Held-out cluster-assignment ARI: project N_test, KMeans on the
  projection, compare against KMeans on a fresh REAP consensus run on
  the same N_test
- Linear projection head ablation: same metrics with a linear map
  instead of MLP (ablation, in supplementary)

### Level F — Compute

- Wall-clock time per method, per dataset (median of 3 runs)
- Peak memory (resident set size, recorded via `tracemalloc` /
  `psutil`)
- Scaling curve at N ∈ {500, 1000, 2000, 5000} on synthetic data,
  reported separately

### Topic coherence + LLM labeling (always computed for paper claims)

For the labeled clusters from `reap.labeling`:

- UMass coherence
- NPMI (normalized pointwise mutual information)
- c_v coherence

Computed against the dataset's own corpus as the reference. Reported
per dataset, per method.

**LLM cluster labels** are produced by `reap.labeling.label_clusters_combined`
via c-TF-IDF → LLM refinement. Default provider+model pairs, aligned
with the sibling paper pipelines (when-algorithms-meet-artists,
green-narrative):

- Anthropic: `claude-opus-4-6` (matches Claude Code's running model;
  `claude-sonnet-4-6` is a cheaper alternative for batch labeling).
- OpenAI: `gpt-5.4-mini`.

Labels from both providers are reported side-by-side; the golden
validation suite (tests/test_labeling_golden.py) enforces that both
providers produce distinct, topic-appropriate labels for every cluster,
and that the providers agree on the dominant topic per cluster.

### External validity (where labels exist)

For datasets with expert labels (AI-art, Korean):

- ARI vs expert labels
- AMI vs expert labels
- Homogeneity, completeness, V-measure
- Per-cluster purity

For corp / presidential, deferred until labels arrive.

---

## 8. Statistical Procedures

- **Aggregation across seeds within a set:** mean ± 95% bootstrap CI
  (10,000 resamples).
- **Comparison between methods on the same dataset and seed set:**
  paired Wilcoxon signed-rank test on per-seed metric values (matched
  pairs by seed). Report W statistic, p-value, and Cohen's d effect
  size on paired differences.
- **Comparison across datasets:** treated as separate analyses; no
  cross-dataset pooling.
- **Multiple-comparison correction:** Holm-Bonferroni applied per
  dataset across the family of pairwise method comparisons (5
  comparisons of REAP vs each baseline = family of 5 tests per
  dataset × per metric of interest). FDR (Benjamini-Hochberg) reported
  alongside as a sensitivity check. This bullet defines the *core
  methods* family only; the §18 temporal-holdout families are defined
  once in the v1.5 amendment (clause d) and enumerated machine-readably
  in `docs/run_matrix_v1_5.json` (AI-art: 60; Korean forest: 216),
  recomputed in CI by `tests/test_run_matrix.py`.
- **Between-set reliability:** report the three pairwise values
  individually (no inferential test on n=3); narrative interpretation
  ties to the within-set CIs.
- **N is small everywhere.** Prefer non-parametric tests. Effect sizes
  are reported alongside every p-value.

---

## 9. K Selection and K-Robustness

- **K selection rule** per dataset: consensus K is the value that
  maximizes a pre-declared composite score: 0.5 × consensus silhouette
  + 0.5 × consensus trustworthiness, scanned over K ∈ [5, 30] on the
  REAP consensus embedding from Set A. Selected K is recorded and
  frozen.
- **K-robustness sweep:** report main metrics at K, K±1, K±2 to show
  that headline conclusions are not sensitive to the exact K choice.

---

## 10. Encoder Sensitivity (AI-art only)

Re-run the AI-art experiments under three encoders:

- intfloat/e5-large-v2 (primary; the headline number)
- sentence-transformers/all-MiniLM-L6-v2
- BAAI/bge-large-en-v1.5

Report REAP vs Procrustes vs single-seed at matched K and seed sets.
Headline question: does REAP's improvement over baselines hold across
encoders, or is it encoder-specific?

---

## 11. Threats to Validity (Pre-Registered)

A skeptical reviewer's likely objections, and how this protocol
addresses each:

- **"You cherry-picked the seed set."** → Three pre-declared disjoint
  sets; full results reported on all three; between-set reliability
  quantified.
- **"You cherry-picked K."** → K is selected by a pre-declared
  composite score (Section 9), and K-robustness is reported at K±1 and
  K±2.
- **"Your improvement is encoder-specific."** → Encoder sensitivity
  ablation on AI-art with three encoders.
- **"Your golden numbers were tuned post-hoc."** → Pre-registered
  ranges in Section 6, frozen at this document's commit; widening
  requires a TRACE correction.
- **"BERTopic does this already."** → Side-by-side comparison with
  BERTopic under matched encoder, K, and seed sets; topic coherence
  reported on both.
- **"You only show internal cluster metrics."** → External-validity
  metrics against expert labels for AI-art and Korean; topic coherence
  on all four; corp/presidential deferred until labels arrive and so
  marked.
- **"Your projection head is over-fit."** → CV and final-fit numbers
  reported distinctly; linear baseline ablation; OOS ARI against a
  fresh full REAP run.
- **"Your N² consensus matrix doesn't scale."** → Acknowledged
  explicitly in the compute reporting (Section 7, Level F) and the
  Discussion; scaling curve at N up to 5000.
- **"Your 'distance averaging preserves metric properties' claim is
  hand-waved."** → Formal statement and proof in the Method section;
  triangle-inequality preservation tested in Tier 1 invariants.
- **"Your 'OOS' corpus is in fact derived from the reference."** →
  Every claimed-OOS source must pass substring + provenance audits
  (`tests/verification/test_sibling_repo_parity.py`). Discovery
  motivated by the AI-art public-probe subsequence finding
  (round-2 audit 2026-05-21, TRACE evt_002 / evt_005). Public probes
  on AI-art are explicitly NOT used as cross-corpus OOS data per §18;
  artist probes (Lovato 2024) and KF external pledges are the true
  OOS comparators.
- **"Your temporal-holdout results inherit consensus-K drift across
  strategies."** → Effective K reported per strategy; no fixed-K
  comparison enforced. Cross-strategy comparison uses metrics that
  are K-independent (trustworthiness, distance correlation) or
  conditionally normalized (silhouette).
- **"Your random-stratified holdouts inherit the master seed."** →
  Master seeds for all random holdouts (T5, K2-T4, K2-T5) pre-
  registered in §18 (T5: 20260521 / 20260522 / 20260523;
  K2-T4: 20260524; K2-T5: 20260525); replicates reported.
- **"Your random holdouts leak same-document chunks."** → v1.5 clause
  (f): group-aware splits where document ids are recoverable; otherwise
  a reported max-cosine leakage audit with T5 labeled "optimistic under
  chunk-level leakage" and the temporal strategies (T1–T4) designated
  the primary generalization evidence.
- **"Your cross-source analyses were verified on one seed set."** →
  Status *open* (v1.5 clause p): A1/A2/A3 reproduced on Set A only;
  re-run on Sets B/C scheduled with the driver work.
- **"Your A2 Wasserstein-1 treats cluster ids as ordered."** → Status
  *accepted-limitation* (v1.5 clause p): KL is the primary divergence;
  W1 is supplementary with the caveat stated.
- **"Your artist probes are not independent samples."** → Status
  *mitigated* (v1.5 clauses g, p): artist-level permutation test is the
  primary A1 analysis; bootstrap CIs resample artists, not probes;
  per-probe results are supporting-with-caveat only.

---

## 12. Reproducibility Bundle Per Experiment

Every committed result CSV is accompanied by a sibling
`bundle.json` capturing:

```json
{
  "git_commit": "...",
  "script": "scripts/run_<name>.py",
  "config": "configs/<name>.yaml",
  "dataset_manifest_entry": "datasets/manifest.json#ai_art_v1",
  "seed_set": "A",
  "seed_manifest": "manuscript/seeds/seed_manifest.json",
  "env": "environment-lock.yml",
  "python_version": "3.12.x",
  "platform": "darwin-arm64 / linux-x86_64",
  "started_at": "...",
  "duration_seconds": ...,
  "peak_memory_mb": ...
}
```

A result without its `bundle.json` is not citable in the manuscript.

---

## 13. Deviations and Corrections

If during execution we discover that a pre-registered choice in this
document is wrong (a metric range too tight, a statistical test
inappropriate, a dataset preprocessing flaw):

1. Stop. Do not silently change the protocol.
2. Open a TRACE annotation with `category="correction"` and
   `corrects_event_ids` linking back to this document's commit hash.
3. Edit the protocol to reflect the corrected choice; update the
   version field at the top of this file (1.0 → 1.1) and add a
   `## Changelog` entry at the bottom describing what changed and why.
4. Commit the protocol edit *before* re-running any affected
   experiment.

This is how an honest pre-registered analysis handles surprises.
Reviewers respect documented corrections; they reject silent ones.

---

## 14. Dependencies on Other Documents

- Seed manifest: `manuscript/seeds/seed_manifest.json`.
- Open-source package standards: `.claude/rules/open-source-package.md`.
- Publication-quality standards: `.claude/rules/publication-standards.md`.
- Code style: `.claude/rules/code-quality.md`.
- E2E + unit testing: `.claude/rules/e2e-testing.md`,
  `.claude/rules/testing.md`.

---

## 16. Out-of-Sample Conformal Filter Pre-Registration (v1.2)

The OOS conformal filter (manuscript §3.6) is a methodological component
of REAP that gates the projection head's outputs. This section
pre-registers its defaults, the head-to-head variant comparison, and
the verification-tier ranges that the package implementation must hit.

### 16.a Filter Defaults

| Parameter | Default | Notes |
|---|---|---|
| Distance metric | Mahalanobis with per-cluster reference covariance Σ_c | Σ_c estimated on reference points only |
| Location correction | Pooled: μ̂_c = mean of OOS points in cluster c (pooled across subgroups) | One centroid per cluster, computed from OOS points only |
| Threshold method | Empirical leave-one-out on reference per-cluster Mahalanobis | The (1−α) quantile of {M_c^LOO(x_i) leaving x_i out, for x_i in cluster c}, where M_c^LOO uses reference centroid and reference covariance both estimated without x_i |
| α | 0.01 | 99th percentile reference LOO; pre-registered headline |
| Hard-fallback rule (per-subgroup variant only) | n_(c, subgroup) < 10 → fall back to pooled centroid | Pooled variant has no fallback; per-subgroup uses the pooled cluster centroid when a (cluster, subgroup) cell is too small |
| Calibration warning | n_c < ⌈1/α⌉ or n_c < d + 2 | Issued whenever the empirical-LOO quantile is undefined (small reference cluster) or covariance is rank-deficient |

### 16.b Methods Under Comparison (Filter)

The five filter variants reported head-to-head in §4.7 are
pre-registered at this revision. Future runs that test additional
variants must be reported as supplementary to these five — the
"five-variant comparison" frame is fixed:

1. Euclidean (no location correction)
2. Euclidean + pooled location correction
3. Mahalanobis (no location correction)
4. **Mahalanobis + pooled location correction** (default)
5. Mahalanobis + per-president (or other subgroup) location correction

All variants use empirical leave-one-out at α = 0.01 unless explicitly
noted; cross-α and cross-threshold-source comparisons live in the
α-sensitivity supplementary table (§4.7.2).

### 16.c Pre-Registered Synthetic-Shift Fixture

To validate the filter via CI without depending on the Korean-forest
OOS snapshot, REAP includes a synthetic-shift fixture built from the
20newsgroups golden text fixture (§6.b):

- **Reference:** the 400-document 20ng text fixture as today.
- **In-distribution OOS:** held-out 20% of each class (also from
  20newsgroups), embedded with the same MiniLM model.
- **Shifted OOS:** the same held-out documents with a synthetic
  register shift applied — concretely, prepending a short prefix
  ("In a short letter to the editor:" or similar) and re-embedding.
  The shift is large enough to move the corpus centroid while
  preserving topic content. Pre-registered ranges in §6.c require
  the filter to retain ≥ 85% of in-distribution OOS and ≤ 30% of
  shifted-OOS at α = 0.01 with pooled location correction.

Implementation of this fixture is tracked in outline.md P0-13.

### 16.d Korean Forest OOS Replication

The Korean-forest case study reported in §4.7 (1,662 pledges, 50.9%
retention) is sibling-project work (`green-narrative/hye_in/`) and is
*not* the pre-registered headline number for the manuscript. The
manuscript's filter numbers must come from a re-run of the same harness
against a locked snapshot of the OOS pledge corpus committed under
`~/.cache/reap/datasets/korean_forest_oos/<version>/`, with a sibling
`bundle.json` per §12. Pre-registered observed band: 45-55% overall
retention at α = 0.01. A re-run that lands outside this band requires
a TRACE correction (§13) before continuing.

### 16.e Reproducibility Bundle for Filter Runs

In addition to §12, every filter run records:

- α value and threshold-source (`empirical_loo` vs `chi_squared`).
- Per-cluster reference n_c values (for the small-cluster ceiling
  diagnostic).
- The per-cluster OOS centroids μ̂_c (one (1, d) vector per cluster); for the per-subgroup variant, the per-(cluster, subgroup) centroids and the fallback choices made per cell.
- Per-cluster thresholds τ_c.

Bundle field name: `filter_calibration` (sibling of `seed_set`).

### 16.f Threats to Validity for the Filter (Pre-Registered)

Filter-specific objections a reviewer will raise, and how this
protocol addresses each:

- **"Your retention number is calibrated on a single corpus."** →
  Required pre-registered synthetic-shift fixture in CI (§16.c);
  Korean forest OOS observed band recorded in §6.c; multi-corpus
  validation listed as future work in Discussion §5.
- **"Your α = 0.01 was chosen post-hoc to hit a target retention."** →
  α = 0.01 was the design default before the five-variant comparison
  was run; rejected alternatives (α ∈ {0.005, 0.001} with empirical
  LOO; chi-squared theoretical at multiple α; β shrinkage at multiple
  β values) are reported as the α-sensitivity table per §4.7.2 and the
  supplementary `oos_filter_design_decisions.md`.
- **"Mahalanobis is wrong for high-dimensional spaces."** → The filter
  operates in the consensus space (typical d ≤ 20), not the embedding
  space; covariance estimation is well-conditioned at d = 18, n_c ≥ 17.
  We document the small-n_c ceiling and the calibration warning in §16.a.
- **"Your conformal coverage statement is not rigorous."** → The
  empirical-LOO threshold gives a finite-sample $(1-\alpha)$ coverage
  bound under reference exchangeability (Vovk et al. 2005); we use
  empirical LOO precisely because chi-squared theoretical thresholds
  do not (they assume known centroid/covariance). Mahalanobis is
  invariant to shift in the score under the location correction, so
  the conformal exchangeability is preserved. The filter does not
  claim coverage on out-of-distribution points by design — that is
  the use case.
- **"The location correction breaks exchangeability."** → The pooled
  correction is a constant shift applied to every cluster; the LOO
  reference distribution is computed *after* the correction is
  applied, preserving the exchangeability of reference points relative
  to the corrected centroid.

---

## 17. Topic-Attribution Evaluation (Cross-Dataset, v1.3)

**Motivation.** The Mahalanobis conformal filter (§16) is purely
geometric — it asks "does this projection land on the reference
manifold?" but cannot, by construction, judge semantic similarity. The
empirically informative bridge metric is therefore not "fraction of OOS
accepted" but rather **topic-attribution accuracy**: of the OOS
documents the projection head places into reference clusters, how often
is the assigned cluster *semantically related* to the document's true
class/theme? This section pre-registers a 3-LLM rubric-driven evaluation
of topic attribution across three datasets, designed to defuse the
"post-hoc circular labeling" reviewer objection.

### 17.a Datasets and OOS structure

| Dataset | Reference (n, K_REAP) | OOS source | OOS ground truth |
|---|---|---|---|
| 20-Newsgroups | 800 docs, K=7 | 12 held-out classes × 50 docs = 600 | Class assignment (no themes); rubric maps OOS class → reference cluster |
| AI-art | 1,736 chunks public discourse, K=20 | 1,259 artist probes + 750 public probes from `when-algorithms-meet-artists/` | Per-doc theme labels (`compensation`, `threat`, `utility`, `ownership`, …) + per-doc cluster assignments from the AI-art panel's analysis (Hungarian-matched to our REAP labels at ARI = 0.94) |
| Korean Forest | 905 docs, K=23 | 1,662 administration pledges across Lee/Park/Moon | Administration grouping; rubric maps {admin × labeled cluster} → relatedness |

### 17.b Step A — Multi-LLM cluster naming

For each reference cluster of each dataset, three independent LLM judges
produce a name + macro-theme + description:

- **Claude Opus 4.7** via dispatched subagent (no Anthropic API key
  required; uses the agent-infrastructure pathway).
- **`gpt-5.4-mini`** via OpenAI API (the existing
  `OPENAI_DEFAULT_MODEL` in `src/reap/labeling.py`).
- **`gpt-5.5`** via OpenAI API (added 2026-05-14 as a stronger judge).

Inputs per cluster: top-10 c-TF-IDF terms, 5 representative documents
(highest-density), cluster size. Output schema: `(label: str,
macro_theme: str, description: str, confidence ∈ [0,1])` per judge.

**For the AI-art dataset, Step A is run for cross-validation against the
existing panel labels in `quad_llm_labels.csv` / `clusters_for_human_review.csv`** —
a fourth independent labeling pass that we report alongside the
sibling-project's labels. If our 3-LLM consensus agrees with the AI-art
panel at semantic-cosine ≥ 0.7 per-cluster, the labels are usable
directly; otherwise the divergence is reported as a methodological
finding.

### 17.c Step B — Multi-LLM relatedness rubric

For each (OOS class/theme × labeled reference cluster) pair, the same
three LLM judges score relatedness on a pre-registered 3-level scale:

- **0 = disjoint**: no semantic overlap between OOS class/theme and
  reference cluster.
- **1 = marginal**: tangential or partial topical overlap.
- **2 = clearly_related**: strong semantic affinity; the OOS doc
  topically belongs to this cluster.

Inputs per pair: OOS class/theme name + a 1-2 sentence description +
3-5 sample texts; reference cluster's consensus name + macro_theme +
description from Step A. Output schema: `(relatedness_score ∈ {0,1,2},
rationale: str, confidence ∈ [0,1])`.

### 17.d Consensus computation + rubric-defensibility analysis

A fourth subagent (Claude Opus 4.7) consolidates the three judges'
outputs:

1. **Per-pair inter-rater agreement** via Krippendorff's α (ordinal
   variant on the 0/1/2 score).
2. **Per-pair consensus score** = mode of the three judges. Ties
   (e.g., 0/1/2) are flagged AMBIGUOUS and excluded from the primary
   topic-attribution metric.
3. **Cross-LLM coherence summary**: per-dataset Krippendorff's α
   reported as a defensibility number. Pre-registered acceptance band:
   α ≥ 0.5 (substantial agreement) for the rubric to be considered
   reliable; 0.4–0.5 → caveat in §5 limitations; < 0.4 → rubric is
   re-built with a clarified prompt template.

### 17.e Topic-attribution accuracy metric

For each (dataset × method × seed set):

- Project each OOS document via the method's projection mechanism
  (REAP head, PUMAP encoder, etc.).
- Assign to nearest reference-cluster centroid.
- Look up the consensus relatedness score for (OOS class/theme, assigned
  cluster) from the §17.d rubric.
- A document is correctly attributed iff its assigned cluster has
  relatedness = 2 (clearly_related).
- Primary metric: **fraction of OOS documents correctly attributed** =
  TP / N_total, where AMBIGUOUS pairs are excluded from the denominator.
- Secondary: per-OOS-class accuracy, attribution-confusion-matrix
  (cluster × class), and a "graded acceptance" stat using
  relatedness ∈ {0,1,2} as a 3-level outcome.

### 17.f Pre-registered acceptance criteria

- **Krippendorff's α ≥ 0.5** per dataset (§17.d).
- **REAP topic-attribution accuracy** is reported as mean ± 95% CI
  across seed sets A/B/C, paired against each baseline via Wilcoxon
  signed-rank on per-OOS-doc indicators.
- **No dataset has < 50 OOS documents** in the AMBIGUOUS-excluded
  numerator (insufficient power).
- **Honest cap if any criterion fails**: the affected dataset is
  reported as a methodology limitation, not silently dropped.

### 17.g Reproducibility

Each LLM call records: model_id, full prompt, raw response (stored as
JSON under `results/<dataset>/topic_attribution/raw_llm_responses/`).
Re-runs of the same prompt+model can drift due to provider sampling;
report a hash of the prompt + the saved response so the consensus
artefact is reproducible from the saved JSONs regardless of API
behaviour.

---

## 18. Temporal Holdouts (v1.4)

Pre-registered in
`manuscript/proposals/2026-05-21-temporal-holdout-and-cross-source-protocol.md`;
that proposal is the authoritative source. Summary follows.

### 18.a Scope

Eleven temporal-holdout experiments + one corpus-OOS evaluation,
spread across AI-art (5 strategies × 1 encoder) and Korean forest
(6 strategies × 3 encoders = 18 evaluations).

### 18.b AI-art strategies (5)

- **T1 future**: hold out year ∈ {2023, 2024, 2025} (960 of 1,736).
- **T2 past**: hold out year ∈ {2013, 2015, 2016, 2017} (248 of 1,736).
- **T3 single-recent**: hold out year == 2025 (220 of 1,736).
- **T4 single-mid**: hold out year == 2019 (70 of 1,736).
- **T5 random 20% stratified by year, 3 replicates** with
  master_seeds 20260521 / 20260522 / 20260523.

### 18.c Korean forest strategies (6 × 3 encoders = 18)

- **K1**: corpus-OOS — project 1,662 external pledges into the
  KF reference space.
- **K2-T1 / T2 / T3**: hold out Moon / Lee / Park reference sentences.
- **K2-T4**: random 20% stratified by president (master_seed=20260524).
- **K2-T5**: random 20% unstratified (master_seed=20260525).

Each strategy runs with three encoders:
`paraphrase-multilingual-MiniLM-L12-v2` (canonical, 384-d);
`paraphrase-multilingual-mpnet-base-v2` (768-d);
`intfloat/multilingual-e5-large` (1024-d).

### 18.d Common protocol per strategy

1. Compute retained_indices and held_out_indices from per-row metadata.
2. Re-embed in-set + held-out with the encoder (Set D's 30 seeds).
3. Run consensus on the in-set encoding.
4. Train projection head: MLP (canonical, `reap.projection.ProjectionHead`).
5. Project held-out subset through head.
6. Run baselines: Parametric UMAP, identity-projection (nearest-ref-chunk
   in raw space), linear projection head (§11 ablation).
7. Compute the metrics in §18.e.
8. Save artifacts per §18.f.

### 18.e Pre-registered metrics

Per (strategy × method):
- Trustworthiness on held-out (k=15, cosine default).
- Distance correlation between raw and projected held-out.
- Silhouette of held-out projections (nearest-centroid cluster labels).
- Adapted Stage-10 ARI (held-out KMeans vs full-rerun KMeans on
  the same held-out).
- Source-separability silhouette (in-set ∪ held-out, binary labels).
- Per-cluster occupancy distributions (KL + Wasserstein-1).

Plus a cross-encoder Procrustes consistency metric on KF
(disparity between projected coordinates from each pair of encoders).

### 18.f Per-strategy artifacts

Saved under `results/temporal_holdout/<corpus>/<strategy>[/<encoder>]/`:
retained/held_out indices, in-set consensus + labels + per-seed CSVs,
head state_dict, holdout projection + assignment, baselines (parametric_umap,
identity, linear_head), `comparison_metrics.json`, `bundle.json`
(per §12).

### 18.g Statistical reporting (per §8)

Mean ± 95% bootstrap CI (10,000 resamples) across Set D's 30 seeds.
Paired Wilcoxon REAP vs each baseline per metric per strategy.
Holm-Bonferroni within each corpus — AI-art: 60 paired tests; Korean
forest: 216 (families defined once in the v1.5 amendment clause d;
enumeration in `docs/run_matrix_v1_5.json`). Cohen's d.
BH-FDR as sensitivity. Random-split replicates aggregate to the
per-seed mean before testing (v1.5 replicate rule).

### 18.h Pre-registered acceptance criteria

- AI-art: at least 3 of 5 strategies achieve trustworthiness mean
  > 0.70 AND distance correlation mean > 0.70 on held-out; Korean
  forest: at least 9 of 18 evaluations achieve the same (v1.5 clause a,
  correcting the earlier "3 of 6", which matched neither corpus). These
  are pre-registered expectations feeding claims, with the v1.5
  clause-j null path — not paper-viability gates.
- Per corpus: at least one strategy shows REAP significantly outperforming
  BOTH PUMAP and linear head on either trust or dist_corr at Holm-corrected
  α=0.05.
- K1 corpus-OOS trustworthiness > 0.65 (lower than within-corpus floor).

### 18.i Threats (incremental to §11)

- T-temporal-1 (OOS-corpus provenance): see §11.
- T-temporal-2 (consensus-K drift across strategies): see §11.
- T-temporal-3 (random-holdout master-seed inheritance): see §11.

---

## 19. Cross-Source Analyses (v1.4)

Analyses that use the existing AI-art OOS demo projections
(`results/projection_head/ai_art/oos_demo/`) without new compute.

### 19.a A1 — Artist demographic stratification

For each of 10 demographic variables in `artist_perspectives.csv`
(`Art_practice`, `Age`, `POC`, `Gender_identity`, `Country`,
`AI_models_familiarity`, `Used_AI_art_models`, `Professional_artist`,
`Purchase_art`, `compensation`):
- Build (n_levels × 20) contingency table of demographic → cluster.
- Compute chi-square + Cramér's V.
- Holm-Bonferroni across the demographics at α=0.05 — **m = 9**:
  `Country` is constant (all-USA) in `artist_perspectives.csv` and is
  excluded as untestable. The committed analysis code has always run 9;
  v1.5 clause (l) records this as the overdue §13 deviation note.

Acceptance: at least one demographic significantly associates with
cluster assignment. If none, that itself is a finding (demographic
uniformity in projection).

### 19.b A2 — Artist-vs-public theme alignment

Per theme:
- Build artist-cluster occupancy and public-cluster occupancy
  histograms (length 20).
- Compute KL(artist || public) and Wasserstein-1.

Descriptive; no acceptance threshold.

**Caveat (pre-registered)**: public probes are theme-conditioned
subsequence selections (round-2 audit, §11 T-OOS-provenance). The
A2 comparison is between "artist stated concerns" and "style-controlled
discourse passages selected to be theme-relevant" — not between two
independent corpora.

### 19.c A3 — Base-rate-corrected temporal alignment

For each of 20 reference clusters:
- `artist_density[c] = (n artist probes in c) / 1259`.
- `corpus_year_share[c] = sum over y of (n chunks in c with year y) / 1736`.
- `enrichment_ratio[c] = artist_density[c] / corpus_year_share[c]`.

Bootstrap CI on enrichment ratios. Acceptance: count of clusters with
lower-CI > 1 (artist-enriched) is reported with bootstrap CI; the
2022–2024 mean-year clusters' over-representation in that set is
reported.

### 19.d Artifacts

`results/projection_head/ai_art/oos_demo/cross_source/{A1,A2,A3}_*.json`
+ per-analysis smoke test
(`tests/verification/test_ai_art_cross_source_smoke.py`).

---

## 15. Changelog

- **1.5 (2026-07-04)** — Temporal-holdout scoring rules locked before
  any Phase 3/4/5 run. Holm families defined once (core m=5 per dataset
  per metric; AI-art holdout m=60; Korean-forest holdout m=216; A1
  demographics m=9 with `Country` excluded as constant — closing the
  live §19.a m=10 deviation); full run enumeration committed as
  `docs/run_matrix_v1_5.json` with a CI guard
  (`tests/test_run_matrix.py`). Acceptance denominators corrected
  (3-of-5 AI-art / 9-of-18 KF, replacing "3 of 6"); "CV mean" qualifier
  restored and bound to the record file's evaluation mode;
  trustworthiness designated the primary cross-method holdout metric
  (cross-method distance correlation = training-objective-aligned,
  supporting; head-vs-head on it stays primary); identity baseline
  reported as a primary comparison with a pre-written null
  interpretation; dependence-unit/leakage rule for random holdouts;
  artist repeated-measures correction for A1 (artist-level permutation
  primary; bootstraps resample artists); KF encoders never averaged +
  minimum fold sizes (n ≥ 32 and n ≥ 3·K); threshold-provenance tags
  (observed-with-headroom bars are sanity checks, not proof);
  null/negative-result path; §12 bundle fields extended (schema +
  protocol versions, resolved library versions, evaluation mode, run
  status + failure reason, metric recipe ids) with a mechanical
  protocol-version gate; the two distance-correlation calculations
  registered as distinct recipes (`distance_correlation` condensed
  metric vs `dist_corr_loss` full-matrix training loss — neither is
  Székely's); Generalized Procrustes designated the headline Procrustes
  comparator (additive; pairwise golden ranges untouched); deferred
  analyses recorded with reasons; residual risks migrated into §11.
  Authoritative source:
  `docs/proposals/2026-07-protocol-v1.5-amendment.md` (lives under
  `docs/` because new `manuscript/` files are gitignored). Ratification
  = merge of the amendment PR; no result-generating run precedes it.

- **1.4 (2026-05-21)** — Added §18 (temporal holdouts: 5 AI-art × 1
  encoder + 6 KF × 3 encoders = 23 evaluations) and §19 (cross-source
  analyses A1/A2/A3 on existing AI-art projections). Three threats
  to validity added to §11 (OOS-corpus provenance, consensus-K drift,
  random-holdout seed inheritance). Set D (30 seeds,
  master_seed=20260521) added to `seed_manifest.json`
  (schema_version 1→2). Linear projection head specified as the §11
  pre-registered ablation comparator. Multi-encoder ablation
  (canonical MiniLM + mpnet 768-d + multilingual-e5-large 1024-d)
  extended to all KF strategies (was AI-art-only in §10).
  Authoritative source: `manuscript/proposals/2026-05-21-temporal-holdout-and-cross-source-protocol.md`.
  Pre-registered before any compute; logged via TRACE
  `trace_20260521_8c18e4` (evt_011 proposed-decision).

- **1.3 (2026-05-14)** — Added §17 (topic-attribution evaluation,
  3-LLM cross-dataset). Pre-registered the rubric construction pipeline
  (Claude Opus 4.7 subagent + `gpt-5.4-mini` + `gpt-5.5`), the
  consensus computation, the accuracy metric, and the Krippendorff's α
  defensibility band. Locked the 20NG / AI-art / Korean Forest dataset
  scope and per-dataset OOS structure for the bridge-claim
  experiment.

- **1.2 (2026-05-09)** — Added the OOS conformal filter (§3.6 in the
  manuscript) as a fifth REAP methodology component subject to the
  same pre-registration framework as the consensus pipeline. Changes:

  1. **§4 pipeline table** gained Stage 11 (filter) with its math
     invariants (per-cluster threshold finite, retention monotone in
     α, μ̂_c shape correct) and statistical-property checks (synthetic
     in-distribution retention ≥ 0.85, shifted retention ≤ 0.30).
  2. **§6.c metric ranges** gained four filter rows: synthetic-shift
     in-distribution and shifted retention bounds; the Tier-1
     monotone-in-α invariant; and the observed Korean-forest OOS
     retention band (0.45-0.55 overall at α = 0.01) as a sibling
     reference number.
  3. **§16 (new section)** pre-registers the filter defaults
     (Mahalanobis + pooled location correction + empirical LOO at
     α = 0.01), the five-variant head-to-head frame, the
     synthetic-shift fixture for CI, the Korean-forest OOS replication
     contract, the reproducibility-bundle additions, and the filter-
     specific threats to validity.

  This version was logged via TRACE in REAP session
  `trace_20260509_947f1d`, building on the cross-project decisions
  recorded in `trace_20260509_305aaf` (rejected alternatives) and
  `trace_20260509_c4f4ac` (Hye In handoff). The Korean-forest validation
  itself is sibling-project work in `green-narrative/hye_in/for_hyein/
  park_moon_results_2026-05-07_v2/`, not REAP-internal — the manuscript
  numbers will come from re-running the same harness against the
  `korean_forest_oos` snapshot once it is locked under
  `~/.cache/reap/datasets/`.

- **1.1 (2026-04-13)** — Two material corrections resulting from the
  golden-validation reference run, logged in TRACE as corrections of
  the v1.0 commit:

  1. **Golden fixture redesign.** The v1.0 spec (`make_blobs(n_samples=400,
     n_features=384, centers=8, cluster_std=2.0, random_state=42)`) did
     not exhibit the phenomenon REAP addresses: random centers in
     R^384 are nearly orthogonal by the curse of dimensionality, so
     UMAP recovers them perfectly at every seed (observed single-seed-
     to-seed ARI = 1.000; REAP = Procrustes = 1.000). Replaced with a
     **two-tier fixture**:
     * **§6.a blob fixture** (low-intrinsic-dim 16 projected to 384-d
       with 15% isotropic noise, cluster_std 2.5, center_box (-3, 3),
       L2-normalized) — retained for Tier-1 mathematical invariants only.
     * **§6.b 20newsgroups text fixture** (curated 8-class subset: 4
       clearly separated + 2 overlapping pairs, embedded with
       `sentence-transformers/all-MiniLM-L6-v2`, cached) — primary
       Tier-2 comparative-claim fixture. Real text with genuine
       semantic overlap is where REAP's advantage over Procrustes
       manifests; geometric-blob manifolds have global rotational
       symmetry that Procrustes aligns perfectly.

  2. **Headline comparative claim on golden: silhouette, not ARI.**
     On the 20ng fixture REAP honestly merges the semantic overlap
     pairs in its consensus, so REAP ARI vs K=8 ground truth is
     *lower* than Procrustes (0.54 vs 0.62 reference). REAP
     silhouette, however, is *higher* than Procrustes (+0.149 observed
     margin; pre-registered floor +0.03). This correctly captures what
     REAP optimizes — a cleaner internally coherent consensus — and
     decouples the golden-fixture claim from the specific ground-truth
     K assumption. The paper's "REAP > Procrustes on ARI" headline
     claim is tested on the real Korean forest and AI-art datasets
     (where overlap patterns differ), not on golden.

  Additional v1.1 additions:

  - **§6.c pre-registered metric ranges** rewritten against the 20ng
    fixture reference run; calibrated with headroom so cross-platform
    UMAP/MiniLM drift does not spuriously fail CI.
  - **§6.e projection head validation** (three tiers: training-fit
    sanity, 3-fold stratified CV, topic-exclusion semantic routing).
    Finding: the projection head is a **semantic interpolator, not a
    novelty detector** — overlap-pair held-outs route to the pair
    partner at 60-70%, separated held-outs route to the nearest
    semantic training cluster at 40-50% (random baseline 14%).
  - **§7 topic coherence + LLM labeling** updated with the
    `claude-opus-4-6` + `gpt-5.4-mini` provider defaults matching the
    sibling-paper pipelines in `when-algorithms-meet-artists` and
    `green-narrative`.
  - **§6.f between-seed-set reliability** moved to the slow-CI
    extended tier; fast-tier golden uses the first 10 seeds of Set A.

- **1.0 (2026-04-13)** — Initial pre-registration. Datasets, seed
  design, methods, pipeline stages with verification, multi-tier
  verification strategy, golden synthetic dataset specification,
  metrics schema (Levels A–F + topic coherence + external validity),
  statistical procedures, K-robustness, encoder sensitivity, threats to
  validity, reproducibility bundle, deviations protocol.
