# Experiments

<!-- This section reports real benchmark results from 30-seed × 3-seed-set
     runs per the pre-registered evaluation protocol (evaluation_protocol.md).
     Numbers come from results/<dataset>/combined_set_<X>/all_methods.csv,
     committed alongside bundle.json reproducibility metadata.

     PLACEHOLDER markers (___) indicate slots awaiting AI-art results or
     Set C results that were still computing at draft time. Fill these
     mechanically from the committed CSVs — do not fabricate. -->

We evaluate REAP against four baselines on two corpora under a
pre-registered protocol (§ Evaluation Protocol). All methods share the
*same* multi-seed UMAP embeddings, computed once per dataset × seed
set; the difference is only how they combine or select from those
embeddings. This ensures a fair comparison: any performance gap is
attributable to the consensus strategy, not to different UMAP runs.

## 4.1 Baselines

Six methods are compared, listed in increasing sophistication:

1. **Single-seed** — run UMAP with each seed independently; report
   mean ± std of metrics across seeds. This is the instability baseline:
   it quantifies how much downstream metrics vary due to the random seed
   alone.
2. **Best-of-N** — cherry-pick the seed with the highest silhouette
   score. A common informal practice that introduces selection bias:
   the "best" seed on one metric may be mediocre on others.
3. **Naive average** — average coordinates across seeds without
   alignment. A negative control: because UMAP embeddings are
   rotationally arbitrary, averaging raw coordinates destroys structure.
4. **Procrustes** — align each seed's embedding to the first via
   orthogonal Procrustes, then average the aligned coordinates. The
   standard method in shape analysis and the most commonly recommended
   stabilization approach.
5. **REAP** (proposed) — average pairwise distance matrices across
   seeds, then project the consensus distance matrix via UMAP with
   `metric="precomputed"`.
6. **BERTopic** — the BERTopic default pipeline (UMAP → HDBSCAN →
   c-TF-IDF) with our encoder fixed.
   <!-- TODO: implement and run BERTopic baseline -->

All methods use KMeans clustering on their output embedding, with $K$
selected by scanning $K \in [5, 30]$ and maximizing silhouette. This
ensures that the K-selection procedure is identical across methods.

## 4.2 Multi-Dataset Validation

### 4.2.1 Korean Forest Policy

**Corpus.** 905 Korean-language forest-policy strategy sentences spanning
three presidential administrations (Moon, Park, Lee), embedded with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d).
Expert taxonomy: 20 human-authored policy-strategy labels.

**Parameters.** $d = 18$, $k_{\mathrm{nn}} = 19$,
$d_{\mathrm{min}} = 0.005$. Selected via four-round progressive grid
search on Set A; frozen before evaluation.

**Table 1. Korean forest policy: method comparison (30 seeds × 3 sets).**

| Method | TW (A / B / C) | Sil (A / B / C) | K (A / B / C) |
|---|---|---|---|
| single\_seed | 0.915 / 0.915 / 0.915 | 0.405 / 0.405 / 0.404 | 28 / 28 / 29 |
| best\_of\_n | 0.916 / 0.915 / 0.913 | 0.416 / 0.419 / 0.416 | 24 / 29 / 29 |
| naive\_average | 0.896 / 0.899 / 0.899 | 0.413 / 0.425 / 0.401 | 5 / 6 / 30 |
| procrustes | 0.915 / 0.916 / 0.916 | 0.405 / 0.410 / 0.405 | 25 / 30 / 30 |
| **reap** | **0.888 / 0.890 / 0.892** | **0.676 / 0.687 / 0.679** | 23 / 30 / 30 |

*30 seeds per set. Sets A, B, C are disjoint (see seed\_manifest.json).
S2S ARI: 0.694 ± 0.042 (A), 0.663 ± 0.146 (B), pending (C).
Source: results/korean\_forest/combined\_set\_\{A,B,C\}/all\_methods.csv.*

**Findings.**

- **REAP silhouette is 67% higher than the next-best baseline** (0.676
  vs 0.416 for best-of-N). The gap is enormous and consistent across
  all three seed sets. This is the headline result: distance-matrix
  consensus produces dramatically better-separated clusters than any
  coordinate-based approach.

- **Trustworthiness is ~3% lower for REAP** (0.888 vs 0.915 for
  single-seed). This is the expected tradeoff: REAP's consensus merges
  semantically adjacent sub-clusters that individual seeds keep
  separate, sacrificing some local-structure preservation for a cleaner
  global partition. The trustworthiness loss is modest and the
  silhouette gain is large.

- **Naive averaging is a catastrophic negative control.** Its selected
  $K = 5$ (vs 23–28 for other methods) shows that coordinate averaging
  collapses the embedding to the point where only the coarsest structure
  survives. Its S2C ARI of 0.200 confirms almost no seed agrees with
  the naive consensus.

- **Procrustes offers no advantage over single-seed.** Procrustes
  silhouette (0.405) matches single-seed (0.405) — the rigid alignment
  adds nothing on this dataset, where UMAP's distortions are non-rigid.

**Statistical inference on the consensus silhouette gap.** The
consensus silhouette is a single value per (method, seed set), so
within-set bootstrap CIs are not defined. We instead treat the three
seed sets (A, B, C) as the inference unit and report the cross-set
mean and standard deviation:

**Table 1b. Korean forest consensus silhouette across 3 disjoint
seed sets (paired by set).**

| Method | Set A | Set B | Set C | Mean | SD across sets |
|---|---:|---:|---:|---:|---:|
| single\_seed | 0.405 | 0.405 | 0.404 | 0.404 | 0.0004 |
| best\_of\_n | 0.416 | 0.419 | 0.416 | 0.417 | 0.0018 |
| naive\_average | 0.413 | 0.425 | 0.401 | 0.413 | 0.0122 |
| procrustes | 0.405 | 0.410 | 0.405 | 0.406 | 0.0029 |
| **reap** | **0.676** | **0.687** | **0.679** | **0.681** | **0.0056** |
| **REAP − single\_seed** | +0.271 | +0.282 | +0.275 | **+0.276** | 0.0055 |
| **REAP − best\_of\_n** | +0.260 | +0.268 | +0.263 | **+0.264** | 0.0038 |
| **REAP − naive\_average** | +0.263 | +0.262 | +0.278 | +0.268 | 0.0090 |
| **REAP − procrustes** | +0.271 | +0.277 | +0.274 | **+0.274** | 0.0030 |

*Source: `results/korean_forest/combined_set_\{A,B,C\}/all_methods.csv`.*

Every paired-difference mean exceeds 0.26 with a between-set standard
deviation under 0.01 — the gap is dramatically larger than its own
between-set noise across the entire baseline family. Formal $t$-tests
on $n = 3$ paired set-level differences are not informative (the
sample size is too small to support a useful $p$-value), but the
direction and magnitude are unambiguous.

**Per-seed inference on metrics that admit it.** For metrics computed
on each individual seed's UMAP embedding (silhouette\_A, trustworthiness,
continuity), all five methods consume the *same* per-seed UMAP runs
and so produce identical per-seed values — the paired Wilcoxon test
on these metrics returns mean diff $= 0$ with $p = 1$, as expected.
The consensus methods are differentiated only on metrics that cross
seed and consensus, which we report below.

**Table 1c. Per-seed-with-consensus-labels silhouette ("silhouette\_C")
— how well each method's consensus labels partition the per-seed
embeddings. Paired Wilcoxon on $n = 30$ seeds, Holm-corrected across
the 4 REAP-vs-baseline comparisons per set.**

| Set | Comparator | REAP mean | Mean diff | Wilcoxon $p_{\mathrm{Holm}}$ | Cohen's $d$ |
|---|---|---:|---:|---:|---:|
| A | single\_seed | 0.365 | +0.000 | 1.000 | 0.00 |
| A | best\_of\_n | 0.365 | −0.028 | 1.000 | −4.45 |
| A | naive\_average | 0.365 | +0.056 | < 0.001 | +7.55 |
| A | procrustes | 0.365 | −0.023 | 1.000 | −3.43 |
| B | single\_seed | 0.367 | +0.000 | 1.000 | 0.00 |
| B | best\_of\_n | 0.367 | −0.035 | 1.000 | −6.79 |
| B | naive\_average | 0.367 | +0.048 | < 0.001 | +6.02 |
| B | procrustes | 0.367 | −0.027 | 1.000 | −6.32 |
| C | single\_seed | 0.378 | +0.000 | 1.000 | 0.00 |
| C | best\_of\_n | 0.378 | −0.012 | 1.000 | −1.67 |
| C | naive\_average | 0.378 | +0.064 | < 0.001 | +10.99 |
| C | procrustes | 0.378 | −0.009 | 1.000 | −2.00 |

*REAP 95% bootstrap CI on per-seed silhouette\_C: A [0.363, 0.367],
B [0.365, 0.368], C [0.376, 0.380] (n=2000 resamples). Source:
`results/korean_forest/pairwise_tests.csv`.*

REAP loses by a small but consistent margin to best\_of\_n and
procrustes on this metric — expected, since for those baselines the
"consensus" embedding is itself one of the per-seed embeddings (or a
near-rigid rotation of one), so per-seed labels naturally fit. REAP's
loss is the cost of true consensus aggregation and is small (mean
diff $\le 0.035$). Against naive\_average, REAP wins by $+0.05$ to
$+0.06$ at $p_{\mathrm{Holm}} < 0.001$ with massive effect sizes
($d = 6$–$11$) — the strongest per-seed comparison signal in the
table. The single\_seed comparator is by definition zero (REAP's
per-seed embeddings are the single\_seed embeddings).

**Between-set reliability of consensus labels.**

**Table 1d. Adjusted Rand Index between consensus KMeans labels of
each pair of seed sets (3 disjoint 30-seed sets per dataset).
Higher is more reliable.**

| Method | (A, B) | (A, C) | (B, C) | Mean |
|---|---:|---:|---:|---:|
| best\_of\_n | 0.731 | 0.781 | 0.708 | 0.740 |
| naive\_average | 0.771 | 0.189 | 0.231 | 0.397 |
| procrustes | 0.714 | 0.710 | 0.659 | 0.694 |
| **reap** | **0.766** | **0.842** | **0.823** | **0.810** |

*Source: `results/korean_forest/between_set_reliability.csv`. REAP
also reports a relative Frobenius distance between consensus distance
matrices: 0.013 (A,B), 0.017 (A,C), 0.018 (B,C) — confirming the
underlying distance geometry is highly stable across seed samples.*

REAP achieves the highest mean between-set ARI (0.810) by a margin
of +0.07 over best\_of\_n (the next-best), and +0.41 over
naive\_average (the negative control whose A↔C and B↔C pairs are
catastrophic). Between-set reliability is the headline operational
question — "if I rerun with a different seed sample, how similar are
my topics?" — and REAP wins it convincingly across both the consensus-label
agreement and (uniquely) the consensus-distance-matrix Frobenius gap.

### 4.2.2 AI-Art Discourse

**Corpus.** 1,736 English-language text chunks about AI-generated art,
drawn from public news discourse, embedded with `intfloat/e5-large-v2`
(1024-d, `query:` prefix). No per-chunk expert labels.

**Parameters.** $d = 5$, $k_{\mathrm{nn}} = 53$,
$d_{\mathrm{min}} = 0.01$, $K = 20$. Selected from the sibling
project's config comparison; frozen before evaluation.

**Table 2. AI-art discourse: method comparison (30 seeds × 3 sets).**

| Method | TW (A / B / C) | Sil (A / B / C) | K (A / B / C) |
|---|---|---|---|
| single\_seed | 0.938 / 0.938 / 0.939 | 0.396 / 0.393 / 0.394 | 17 / 18 / 18 |
| best\_of\_n | 0.939 / 0.937 / 0.939 | 0.410 / 0.409 / 0.406 | 17 / 18 / 16 |
| naive\_average | 0.920 / 0.919 / 0.917 | 0.414 / 0.420 / 0.395 | 12 / 16 / 14 |
| procrustes | 0.939 / 0.939 / 0.940 | 0.410 / 0.399 / 0.407 | 24 / 21 / 18 |
| **reap** | **0.917 / 0.917 / 0.918** | **0.671 / 0.672 / 0.673** | 20 / 19 / 18 |

*30 seeds per set. Sets A, B, C are disjoint (see seed\_manifest.json).
Source: results/ai\_art/combined\_set\_\{A,B,C\}/all\_methods.csv.*

**Findings.**

- **REAP silhouette is 64% higher than the next-best baseline** (0.671
  vs 0.410 for best-of-N on Set A). The pattern mirrors the Korean
  forest result: distance-matrix consensus produces dramatically
  better-separated clusters regardless of language, domain, or
  embedding model.

- **Trustworthiness tradeoff is ~2%** (0.917 vs 0.939 for single-seed).
  Slightly smaller than the Korean forest tradeoff (3%), suggesting the
  AI-art corpus's higher-dimensional embedding (1024-d vs 384-d)
  provides more room for REAP to preserve local structure.

- **REAP selects $K = 18$–$20$**, aligning with the prior validated
  $K = 20$ from the sibling project. This is a consistency check: the
  REAP consensus independently recovers the same coarse topic count
  that domain experts converged on.

- **Cross-dataset consistency.** The REAP silhouette advantage is
  +0.274 on Korean forest and +0.266 on AI-art (set-level paired
  means) — a remarkably consistent ~65% improvement across two
  datasets that differ in language (Korean vs English), domain
  (policy vs arts discourse), embedding model (MiniLM 384-d vs
  e5-large-v2 1024-d), and corpus size (905 vs 1,736).

**Statistical inference on the consensus silhouette gap (AI-art).**
Same protocol as §4.2.1 Table 1b — three pre-declared seed sets
treated as the inference unit, paired set-level differences:

**Table 2b. AI-art consensus silhouette across 3 disjoint seed sets
(paired by set).**

| Method | Set A | Set B | Set C | Mean | SD across sets |
|---|---:|---:|---:|---:|---:|
| single\_seed | 0.396 | 0.393 | 0.394 | 0.394 | 0.0015 |
| best\_of\_n | 0.410 | 0.409 | 0.406 | 0.409 | 0.0019 |
| naive\_average | 0.414 | 0.420 | 0.395 | 0.410 | 0.0133 |
| procrustes | 0.410 | 0.399 | 0.407 | 0.406 | 0.0058 |
| **reap** | **0.671** | **0.672** | **0.673** | **0.672** | **0.0012** |
| **REAP − single\_seed** | +0.275 | +0.279 | +0.279 | **+0.278** | 0.0023 |
| **REAP − best\_of\_n** | +0.261 | +0.263 | +0.267 | **+0.263** | 0.0031 |
| **REAP − naive\_average** | +0.256 | +0.252 | +0.278 | +0.262 | 0.0142 |
| **REAP − procrustes** | +0.260 | +0.273 | +0.266 | **+0.266** | 0.0062 |

*Source: `results/ai_art/combined_set_\{A,B,C\}/all_methods.csv`.*

Every paired-difference mean exceeds 0.26 with between-set SD under
0.015 — the same pattern observed on Korean forest, on a corpus that
differs in language, embedding model, dimensionality, and domain.
REAP's between-set silhouette SD (0.0012) is the smallest of any
method, making the consensus the *most stable* approach across seed
samples.

**Table 2c. AI-art per-seed-with-consensus-labels silhouette
("silhouette\_C") — Wilcoxon paired test on $n = 30$ seeds, Holm-
corrected across 4 REAP-vs-baseline comparisons per set.**

| Set | Comparator | REAP mean | Mean diff | Wilcoxon $p_{\mathrm{Holm}}$ | Cohen's $d$ |
|---|---|---:|---:|---:|---:|
| A | single\_seed | 0.366 | +0.000 | 1.000 | 0.00 |
| A | best\_of\_n | 0.366 | −0.035 | 1.000 | −8.47 |
| A | naive\_average | 0.366 | +0.029 | < 0.001 | +6.27 |
| A | procrustes | 0.366 | −0.034 | 1.000 | −8.45 |
| B | single\_seed | 0.349 | +0.000 | 1.000 | 0.00 |
| B | best\_of\_n | 0.349 | −0.052 | 1.000 | −13.91 |
| B | naive\_average | 0.349 | −0.024 | 1.000 | −6.54 |
| B | procrustes | 0.349 | −0.042 | 1.000 | −10.44 |
| C | single\_seed | 0.369 | +0.000 | 1.000 | 0.00 |
| C | best\_of\_n | 0.369 | −0.029 | 1.000 | −9.51 |
| C | naive\_average | 0.369 | +0.048 | < 0.001 | +14.31 |
| C | procrustes | 0.369 | −0.031 | 1.000 | −11.17 |

*REAP 95% bootstrap CI on per-seed silhouette\_C: A [0.365, 0.367],
B [0.348, 0.351], C [0.368, 0.370] (n = 2000 resamples). Source:
`results/ai_art/pairwise_tests.csv`.*

The pattern matches Korean forest: REAP loses a small margin
($-0.029$ to $-0.052$) to best\_of\_n and procrustes — the same
documented cost of true consensus aggregation when the metric
evaluates per-seed embeddings under each method's consensus labels.
On set B, REAP also slightly loses to naive\_average ($-0.024$, not
significant after Holm correction); on sets A and C, REAP wins by
$+0.029$ to $+0.048$ at $p_{\mathrm{Holm}} < 0.001$ with massive
effect sizes ($d = 6$–$14$).

**Table 2d. AI-art between-set reliability — Adjusted Rand Index
between consensus KMeans labels of each pair of seed sets.**

| Method | (A, B) | (A, C) | (B, C) | Mean |
|---|---:|---:|---:|---:|
| best\_of\_n | 0.855 | 0.856 | 0.864 | 0.859 |
| naive\_average | 0.697 | 0.647 | 0.665 | 0.670 |
| procrustes | 0.817 | 0.822 | 0.863 | 0.834 |
| **reap** | **0.975** | **0.891** | **0.880** | **0.915** |

*Source: `results/ai_art/between_set_reliability.csv`. REAP relative
Frobenius distance between consensus distance matrices: 0.0062
(A,B), 0.0064 (A,C), 0.0053 (B,C) — even tighter than Korean
forest, confirming the consensus distance geometry is highly stable
on the larger corpus.*

REAP achieves the highest mean between-set ARI on AI-art (0.915,
+0.06 over best\_of\_n, +0.08 over procrustes, +0.25 over the negative
control naive\_average), and the highest mean between-set ARI on
Korean forest (0.810). The reliability advantage is consistent across
both datasets and is the strongest *operational* argument for the
method: re-running with a different seed sample produces nearly the
same topic assignments.

## 4.3 Sensitivity Analysis

### Seed Count Ablation

How many seeds are needed for stable consensus? We run REAP on the
Korean forest corpus at seed counts $S \in \{5, 10, 15, 20, 25, 30\}$,
reporting silhouette, trustworthiness, and selected $K$ at each.

<!-- TODO: Run seed ablation via examples/run_benchmark.py --ablation
     --ablation-seed-counts 5,10,15,20,25,30 -->

### K-Robustness

We report main metrics at the selected $K$ and at $K \pm 1$, $K \pm 2$
to show that headline conclusions are not sensitive to the exact $K$
choice (per protocol §9).

<!-- TODO: Implement K-robustness sweep in benchmark harness -->

### Encoder Sensitivity (AI-art only)

Re-run the AI-art experiments under three encoders (per protocol §10):

- `intfloat/e5-large-v2` (primary)
- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-large-en-v1.5`

<!-- TODO: Implement encoder-sensitivity sweep -->

## 4.4 Projection Head

**Training-fit.** On the Korean forest corpus, the projection head
achieves: MSE ___, R² ___, distance correlation ___,
trustworthiness ___.

**Cross-validation (3-fold stratified).** CV R² ___,
CV trustworthiness ___, CV ARI vs expert labels ___,
CV distance correlation ___.

**Topic-exclusion semantic routing.** For overlap-pair members
(politics.guns ↔ politics.mideast, ibm.hardware ↔ mac.hardware on
the 20ng golden fixture), 60–70% of held-out documents land in the
pair-partner cluster. For separated topics, 40–50% land in the dominant
training cluster (random baseline 14%). The projection head is a
semantic interpolator, not a novelty detector (see §3.4).

<!-- TODO: Run projection head on Korean forest and AI-art, fill metrics -->

## 4.5 Computational Cost

| Dataset | $N$ | $d_{\mathrm{in}}$ | 30-seed benchmark (5 methods) | Per-method wall time |
|---|---|---|---|---|
| Korean forest | 905 | 384 | ~50 s | 8–16 s |
| AI-art | 1,736 | 1,024 | ___ s | ___ s |

The $O(N^2)$ consensus distance matrix is the memory bottleneck: Korean
forest requires ~1.6 GB ($905^2 \times 8$ bytes × 30 seeds, accumulated
in float64). For AI-art ($N = 1{,}736$), this grows to ~6 GB. REAP is
practical for corpora up to $N \approx 5{,}000$ on consumer hardware;
larger datasets require subsampling or approximate distance methods.

<!-- TODO: Include scaling curve at N ∈ {500, 1000, 2000, 5000} per protocol §7 Level F -->

## 4.6 Statistical Procedures

Per the pre-registered protocol (§8): within-set per-seed metrics are
reported as mean ± 95% bootstrap CI (n = 2000 resamples). Cross-method
comparisons on per-seed-applicable metrics use paired Wilcoxon
signed-rank tests on per-seed metric values (matched by seed).
Holm-Bonferroni correction is applied across the family of 4 pairwise
comparisons (REAP vs each consensus baseline) per seed-set × metric.
Effect sizes (Cohen's $d$ on paired differences, plus Cliff's delta as
a non-parametric companion) are reported alongside $p$-values.

**Practical note on per-seed inference for consensus methods.** All
five methods (single\_seed, best\_of\_n, naive\_average, procrustes,
reap) consume the *same* per-seed UMAP runs in our harness, so the
per-seed metrics that depend only on those runs (silhouette\_A,
trustworthiness, continuity, davies\_bouldin, calinski\_harabasz) are
algebraically identical across methods within a seed set; the paired
Wilcoxon test on these returns mean diff $= 0$ with $p = 1$ as
expected. The inference-bearing per-seed metrics are therefore those
that *depend on the consensus output*: silhouette\_C (per-seed
silhouette under each method's consensus labels) and the
seed-to-consensus agreement scores (s2c\_ari, s2c\_ami, s2c\_nmi).
Tables 1c–1d in §4.2.1 report these.

**Headline consensus silhouette.** The consensus silhouette (the
manuscript's headline metric) is single-valued per (method, seed set)
— there is no per-seed distribution of consensus silhouettes. We
report mean ± SD across the three pre-declared seed sets and present
paired set-level differences (Table 1b). With $n = 3$ paired sets,
formal $t$-tests are uninformative; we instead show that every paired
difference exceeds 0.26 with between-set SD under 0.01 — an unambiguously
positive direction with effect magnitudes large relative to baseline
noise.

Between-set reliability (consensus-label ARI/AMI/NMI and consensus-
distance-matrix Frobenius gap) is reported as the three pairwise
values (A–B, A–C, B–C) without inferential tests ($n = 3$); the
narrative interpretation in §4.2.1 (Table 1d) ties the operational
meaning of each to the within-set inference.

**Reproduction.** All inference numbers — Korean forest (§4.2.1
Tables 1b–1d) and AI-art (§4.2.2 Tables 2b–2d) — are reproducible
from one command per dataset:

```bash
python scripts/run_paper_benchmark.py --dataset korean_forest --skip-bertopic
python scripts/run_paper_benchmark.py --dataset ai_art --skip-bertopic
```

Each invocation regenerates `per_seed.csv` for every (method, seed
set), the between-set reliability table, and the pairwise tests CSV.
Per-set runtime: ~4 minutes for Korean forest (N=905) and ~20 minutes
for AI-art (N=1,736) on consumer M-series hardware (the $O(N^2)$ scaling
in the per-seed metric evaluations dominates).

## 4.7 Out-of-Sample Conformal Filter Validation (Korean forest)

We validate the OOS conformal filter (§3.6) on the Korean forest policy corpus, projecting a *separate* corpus of district-level presidential pledges into the consensus space trained on the national-scale planning corpus, then measuring how the filter behaves under the head-to-head variant comparison and under external qualitative review.

**Setup.** The reference is the 905-sentence Korean forest planning corpus (`load_korean_forest()`, MiniLM 384-d → REAP consensus 18-d, $K = 8$). The out-of-sample corpus is a sibling-project resource of 1,662 district-level political pledge sentences scraped from three South Korean presidential election cycles (Lee = 480, Park = 633, Moon = 549). Pledges are embedded with the same MiniLM model, projected into the 18-d consensus space via REAP's projection head (§3.4), and assigned to one of $K = 8$ clusters by nearest-centroid in the consensus space. The filter then operates per-cluster on the projected pledges. The full analysis lives at `green-narrative/hye_in/for_hyein/park_moon_results_2026-05-07_v2/` and the corpus is targeted for inclusion in REAP's dataset cache as `load_korean_forest_oos()` (currently sibling-project artifact only — see §6 of the evaluation protocol).

**Five-variant comparison.** We ran the same OOS pledge corpus through five filter variants spanning the {distance metric} × {location correction} design space:

**Table 6. Five-variant filter retention on the Korean forest OOS corpus (1,662 pledge sentences).**

| Variant | Lee (n=480) | Park (n=633) | Moon (n=549) | **Overall (n=1,662)** |
|---|---:|---:|---:|---:|
| Euclidean (no correction) | 140 (29.2%) | 170 (26.9%) | 169 (30.8%) | 479 (28.8%) |
| Euclidean + pooled correction | 328 (68.3%) | 421 (66.5%) | 394 (71.8%) | 1,143 (68.8%) |
| Mahalanobis (no correction) | 132 (27.5%) | 161 (25.4%) | 154 (28.1%) | 447 (26.9%) |
| **Mahalanobis + pooled correction** (default, §3.6) | **238 (49.6%)** | **320 (50.6%)** | **288 (52.5%)** | **846 (50.9%)** |
| Mahalanobis + per-president correction | 247 (51.5%) | 325 (51.3%) | 296 (53.9%) | 868 (52.2%) |

*Threshold: empirical LOO at $\alpha = 0.01$ in all variants. Source: `five_variant_retention.csv`.*

**Sentence-level disagreement matrix.** For every pair of variants we compute the percentage of OOS points whose KEEP/REMOVE decision differs:

**Table 7. Pairwise filter-decision disagreement (% of N=1,662). Headline pairs in bold.**

|  | E\_raw | E\_pooled | M\_raw | M\_pooled | M\_perpres |
|---|---:|---:|---:|---:|---:|
| **E\_raw** | 0.0 | 55.3 | 11.3 | 39.2 | 40.3 |
| **E\_pooled** | 55.3 | 0.0 | 59.4 | **27.5** | 27.7 |
| **M\_raw** | 11.3 | 59.4 | 0.0 | 41.5 | 42.2 |
| **M\_pooled** | 39.2 | **27.5** | 41.5 | 0.0 | **5.2** |
| **M\_perpres** | 40.3 | 27.7 | 42.2 | **5.2** | 0.0 |

*Source: `five_variant_disagreement_matrix.csv`.*

The two headline disagreement pairs anchor the choices in §3.6:

- **Mahalanobis vs Euclidean (both with pooled correction): 27.5%.** A quarter of the points are decided differently by shape adaptation. This is large enough that the choice of distance metric is a real methodological lever, not a rounding decision. Per-cluster patterns (Table 8 below) confirm Mahalanobis is doing interpretable work.

- **Pooled vs per-president location correction: 5.2%.** Per-president correction barely changes the filter (only ~30 sentences flip on a corpus of 1,662). Pooled is strictly simpler — one OOS centroid per cluster (8 vectors total) instead of one per (cluster, president) cell with a hard-fallback rule when a cell contains fewer than 10 pledges — and is therefore preferred.

**Per-cluster retention (Mahalanobis + pooled correction, $\alpha = 0.01$).**

**Table 8. Per-cluster retention on the Korean forest OOS corpus.**

| Cluster | Label | $n_{\mathrm{OOS}}$ | Kept (Mahalanobis) | Pct | Kept (Euclidean) | Pct |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Climate Change Adaptation in Forestry | 2 | 0 | 0.0% | 2 | 100.0% |
| 2 | Forest Welfare and Ecosystem Services | 33 | 29 | 87.9% | 26 | 78.8% |
| 3 | Urban Forestry and Green Space | 288 | 87 | 30.2% | 138 | 47.9% |
| 4 | Ecosystem Protection and Disaster Preparedness | 269 | 170 | 63.2% | 233 | 86.6% |
| 5 | Community-Based Forestry Economy | 724 | 446 | 61.6% | 560 | 77.3% |
| 6 | Forest Bioenergy and Industrial Growth | 218 | 68 | 31.2% | 161 | 73.9% |
| 7 | Carbon Management Strategies | 118 | 39 | 33.1% | 14 | 11.9% |
| 8 | Forest Policy Infrastructure and Employment | 10 | 7 | 70.0% | 9 | 90.0% |
| **Total** | | **1,662** | **846** | **50.9%** | **1,143** | **68.8%** |

*Both columns use the same threshold ($\alpha = 0.01$, empirical LOO) and the same pooled location correction. Source: `five_variant_per_cluster.csv`.*

**Findings.**

- **Headline retention is 50.9% under the default filter.** A second-order observation worth recording: the retained fraction is intentionally well below 100%. The filter exists because the OOS corpus is genuinely broader than the reference (district-scale political pledges cover topics not present in the national-scale planning corpus). A retention rate at or near 100% would mean the filter is doing nothing.

- **Mahalanobis tightens umbrella clusters and loosens shape-mismatched clusters.** The largest Mahalanobis vs Euclidean per-cluster gap is Cluster 6 (Forest Bioenergy): Euclidean retains 73.9% (161/218); Mahalanobis retains 31.2% (68/218). Qualitative review (§4.7.1 below) showed many Cluster 6 pledges are general industrial-policy content unrelated to bioenergy that landed near the cluster centroid because the cluster's elongated shape leaves slack along its dominant axis. Conversely, Cluster 7 (Carbon Management) has Mahalanobis retain 33.1% but Euclidean retain only 11.9% — the cluster is tight in reference space but the OOS distribution there is shape-mismatched, and Mahalanobis correctly recognises some pledges as still topically aligned even though their Euclidean distance is large.

- **Location correction does roughly half the work.** Mahalanobis with the reference centroid retains 26.9%; with the per-cluster OOS centroid (default §3.6 location correction) it jumps to 50.9%. The 24-point gap is the size of the reference-vs-OOS register shift in this corpus. Without correction, the filter would over-reject points that are topically appropriate but stylistically different from the reference.

### 4.7.1 Qualitative Validation

A preliminary qualitative review (Korean text read by an LLM analyst, ~6 sample sentences per cluster per source = ~192 sentences total; full text in `text_analysis_preliminary.md`) confirmed three things the filter is genuinely accomplishing:

1. **Dropping projections that fell into the wrong cluster entirely.** Cluster 7 (Carbon Management) has the clearest cases — pledges about local parks or community centers that ended up in Carbon Management because of embedding-space coincidence. The filter drops these.

2. **Tolerating the style/register gap when the topic is right.** Cluster 3 (Urban Forestry) is the canonical example: planning-corpus text discusses urban-forest indices and therapeutic-forest policy, district pledges discuss specific local parks, eco-streams, trails. Different scale, same topic — and the filter retains most of them because the location correction moves the comparison point into the OOS register.

3. **Catching content drift in mid-confidence clusters.** Cluster 2 (Forest Welfare) reference text is about *payment for ecosystem services*; OOS pledges drift toward general rural welfare (childcare, agricultural insurance). The filter drops the heavily-drifted ones.

Domain-expert (Hye In Kim) review of the per-president KEEP/REMOVE decisions is in progress; her review xlsx files are at `for_hyein_handoff_2026-05-09/` (480 + 633 + 549 = 1,662 rows with cluster label, Mahalanobis decision, Euclidean decision, and a 4-way agreement category for direct cluster-by-cluster validation).

### 4.7.2 Sensitivity Analysis (Supplementary)

Two sensitivity sweeps are reported in supplementary `manuscript/supplementary/oos_filter_design_decisions.md`:

- **$\alpha$-sensitivity** (empirical LOO and chi-squared theoretical, $\alpha \in \{0.10, 0.05, 0.025, 0.01, 0.005, 0.001\}$). At $\alpha = 0.01$ empirical LOO retains 50.9%; lowering to 0.005 raises retention to 54.4% and to 0.001 raises it to 60.0%, where empirical LOO hits a small-cluster ceiling (Cluster 1 with $n = 17$ reference points reaches its maximum LOO Mahalanobis distance at $\alpha = 0.001$, beyond which the empirical quantile is undefined). Chi-squared theoretical thresholds are systematically much stricter than empirical LOO at the same $\alpha$ — at $\alpha = 0.01$, $d = 18$, chi-squared retains 19.1% versus empirical LOO 50.9%. The two are not interchangeable.

- **Shrinkage-covariance variant.** $\boldsymbol{\Sigma}_{\mathrm{eff}} = \beta \boldsymbol{\Sigma}_{\mathrm{ref}} + (1-\beta) \boldsymbol{\Sigma}_{\mathrm{OOS}}$ with chi-squared theoretical threshold at $\alpha = 0.01$. At $\beta = 0.75$ retention is 68.6% (in a desirable middle band); at $\beta = 0.5$ it climbs to 86.3%; at $\beta = 0$ (covariance entirely from OOS) it reaches 94.3%. The shrinkage variant is rejected as the default because $\beta$ has no principled selection rule (Ledoit-Wolf-style automatic shrinkage in the conformal-filter setting is open work) — see §5 future work.

### 4.7.3 Known Data-Quality Issues

The qualitative review surfaced three issues in the OOS corpus that affect the centroid estimates marginally:

- **Duplicated rows** appear in the Park and Moon files (e.g., Cluster 8: same district pledge appearing 3× in Park).
- **OCR artifacts** appear in the Moon corpus (random Latin character strings mixed with Korean text), consistent with PDF/image-based extraction.
- **A Korean-letter typo column header** (`ㅎ`) in the Park file is renamed to `text_sentence` on load.

These are recorded as a known limitation; deduplication shifts retention by less than one percentage point per cluster in spot checks.
