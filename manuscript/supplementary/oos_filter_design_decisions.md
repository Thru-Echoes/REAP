# OOS Conformal Filter — Design Decisions

This supplementary document records the rejected alternatives considered
during the design of REAP's out-of-sample conformal filter (manuscript
§3.6, §4.7). It is reviewer-ammunition: every decision below was made
deliberately, the alternative was tested empirically on the Korean
forest case study, and the rationale for rejection is explicit.

The filter validation work happened in `green-narrative/hye_in/for_hyein/
park_moon_results_2026-05-07_v2/` and the cross-project TRACE provenance
is in sessions `trace_20260509_305aaf` (decisions and rejected
alternatives) and `trace_20260509_c4f4ac` (Hye In handoff
contributions). The REAP-side adoption is logged in
`trace_20260509_947f1d`.

---

## Accepted defaults (recap)

| Choice | Default | Where defined |
|---|---|---|
| Distance metric | Mahalanobis with per-cluster reference Σ_c | §3.6 |
| Location correction | Pooled per-cluster OOS centroid (mean of OOS points in cluster c, pooled across subgroups) | §3.6 |
| Threshold method | Empirical leave-one-out on reference Mahalanobis distribution | §3.6 |
| α | 0.01 | §3.6 |

Headline retention on Korean forest OOS (1,662 pledges):
**846 (50.9%)** under these defaults. Per-president: Lee 49.6%, Park 50.6%,
Moon 52.5%.

---

## Rejected alternative 1: Lower α with empirical LOO

**Empirical retention at lower α (Mahalanobis + pooled correction):**

| α | Lee | Park | Moon | Overall |
|---|---:|---:|---:|---:|
| 0.10  | 15.2% | 17.7% | 19.5% | 17.6% |
| 0.05  | 28.7% | 30.0% | 35.3% | 31.4% |
| 0.025 | 38.3% | 40.4% | 40.6% | 39.9% |
| **0.01**  | **49.6%** | **50.6%** | **52.5%** | **50.9%** |
| 0.005 | 53.1% | 54.2% | 55.7% | 54.4% |
| 0.001 | 58.5% | 57.5% | 64.1% | 60.0% |

**Why rejected as default.** α = 0.005 raises overall retention by
+3.5 points and α = 0.001 by +9.1 points relative to the default. Both
are mathematically valid extensions of the conformal framework. Neither
has a principled basis for choosing one specific α value over another
on this corpus. Reporting α = 0.001 as the default would be
indistinguishable from reverse-engineering toward a target retention,
and a journal reviewer would reasonably treat it as p-hacking.

**Why we documented it anyway.** Reporting α-sensitivity in the
supplement is reviewer ammunition: it shows we *considered* higher
retention and explicitly chose conservatism. The corpus statistics
also surface a useful fact — empirical LOO has a **hard ceiling at
~60% retention** for this corpus because Cluster 1 (n_ref = 17)
reaches its maximum LOO Mahalanobis distance at α ≤ 0.001, beyond
which the empirical quantile is undefined.

**Future-work hook.** A retention-pushing extension that *is* journal-
defensible would require a principled α-selection rule (e.g., chosen
to satisfy a coherence-based per-cluster criterion before retention
is observed). Listed in Discussion §5 as the per-cluster α future
work.

Source: `higher_retention_exploration.csv` (lines for `Mahalanobis+pooled
(empirical LOO)`).

---

## Rejected alternative 2: Chi-squared theoretical threshold

**Empirical retention with chi-squared theoretical (Mahalanobis +
pooled correction, d = 18):**

| α | Lee | Park | Moon | Overall |
|---|---:|---:|---:|---:|
| 0.10  | 9.0%  | 9.6%  | 11.1% | 9.9%  |
| 0.05  | 10.6% | 14.1% | 13.7% | 12.9% |
| 0.025 | 13.1% | 16.4% | 17.1% | 15.7% |
| 0.01  | 15.0% | 20.1% | 21.7% | 19.1% |
| 0.005 | 17.3% | 22.6% | 25.1% | 21.9% |
| 0.001 | 23.3% | 28.6% | 31.1% | 27.9% |
| 0.0001 | 29.4% | 34.6% | 36.2% | 33.6% |

**Why rejected as default.** Chi-squared theoretical thresholds at α =
0.01 retain 19.1% — much stricter than the 50.9% retained by empirical
LOO at the same nominal α. The two are not interchangeable: chi-squared
assumes the cluster centroid and covariance are known *exactly*, while
empirical LOO captures the additional variance from estimating those
parameters from finite reference samples. In this regime (per-cluster
n_c ranging from 17 to 307, d = 18), the gap between the two is
roughly a factor of 2.6 in the threshold itself.

Using chi-squared would be over-rejection, and switching the threshold
source to recover an in-distribution retention rate would invalidate
the conformal coverage statement (chi-squared coverage is asymptotic
in n_c, not finite-sample).

**Why we documented it anyway.** The chi-squared α-sensitivity sweep
demonstrates the *range* of retention achievable across nominal α
values within the chi-squared framework, and shows that even at
α = 0.0001 the chi-squared threshold retains only 33.6% — well below
the empirical-LOO retention at α = 0.01. This is the cleanest evidence
that the threshold-source choice matters more than α tuning at this
sample size.

Source: `higher_retention_exploration.csv` (lines for `Mahalanobis+pooled
(chi-squared theoretical)`).

---

## Rejected alternative 3: Shrinkage covariance

**Empirical retention with shrinkage covariance (chi-squared
theoretical at α = 0.01):**

| β (weight on Σ_ref) | Lee | Park | Moon | Overall |
|---|---:|---:|---:|---:|
| 1.0  | 15.0% | 20.1% | 21.7% | 19.1% |
| 0.75 | 69.8% | 66.4% | 70.1% | 68.6% |
| 0.5  | 86.2% | 85.6% | 87.1% | 86.3% |
| 0.25 | 92.1% | 91.2% | 93.1% | 92.1% |
| 0.0  | 95.0% | 93.4% | 94.9% | 94.3% |

The shrinkage definition: Σ_eff = β · Σ_ref + (1 − β) · Σ_OOS, with
the threshold from chi-squared theoretical at α = 0.01.

**Why rejected as default.** β = 0.75 lands directly in an
intuitively desirable retention band (65-75%) and reads cleanly as
"blend reference shape with OOS shape" — but the choice of β = 0.75
specifically is unjustified. Why not β = 0.6 or 0.85? There is no
principled rule on this corpus alone. Mixing the threshold source
(chi-squared theoretical) with the covariance source (shrinkage)
also makes attribution unclear: a reviewer cannot tell which change
contributes how much to the retention shift.

The conformal coverage guarantee from Shafer & Vovk (2008) and
Angelopoulos & Bates (2021) does not extend cleanly to shrinkage-
covariance variants without additional theoretical work — specifically,
the exchangeability argument breaks because Σ_OOS is computed on the
test corpus, not the calibration corpus.

**Future-work hook.** A defensible version requires a derivation-not-
tuning β rule. Two leads documented in Discussion §5:

1. **Ledoit-Wolf-style shrinkage** [Ledoit & Wolf 2004] adapted to the
   two-distribution setting where the "target" is Σ_OOS and the
   "shrinkage source" is Σ_ref.
2. **Cross-validated β selection** on held-out reference points,
   combined with an exchangeability argument that preserves a
   conformal coverage statement.

Either route is a non-trivial methodological contribution that needs
its own validation before adoption.

Source: `higher_retention_exploration.csv` (lines for `Shrinkage
Mahalanobis β=...`).

---

## Rejected alternative 4: Per-president location correction

**Empirical retention.** Per-president location correction (separate
per-(cluster, president) OOS centroid μ̂_(c,p)^OOS for each president p,
with hard-fallback to the pooled per-cluster centroid when
n_(c,p) < 10): **52.2%** overall.

| | Pooled | Per-president | Difference |
|---|---:|---:|---:|
| Lee | 49.6% | 51.5% | +1.9 |
| Park | 50.6% | 51.3% | +0.7 |
| Moon | 52.5% | 53.9% | +1.4 |
| **Overall** | **50.9%** | **52.2%** | **+1.3** |

**Sentence-level disagreement against pooled:** 5.2% (Mahalanobis +
pooled vs Mahalanobis + per-president). Out of 1,662 pledges, only
about 86 flip KEEP/REMOVE between the two variants.

**Why rejected as default.** Per-president correction adds methodological
complexity (per-(cluster, president) centroids plus a fallback rule for
thin cells) for a 1.3-point retention gain that is barely a different
filter (5.2% disagreement). Pooled is *one consistent rule across
presidents* — eight per-cluster centroids regardless of which president
the OOS point came from — and reads cleaner in the methods section. For
a journal audience, "we apply the same filter to every president" is
strictly better than "we apply different filters per president, except
when the president has too few pledges in a cluster, in which case we
fall back to the pooled rule."

**Hidden risk of per-president (more important than the surface-level
1.3 points).** The per-president centroid correction *partially absorbs
topic-mismatch into the centroid*: if Lee's pledges in Cluster 6 are
systematically off-topic (which qualitative review confirmed), the
per-president shift centers the cluster on Lee's off-topic pledges,
effectively defeating the filter for that cell. Pooled correction does
not have this failure mode because it averages across presidents.

**Where it lives in the manuscript.** Reported as variant #5 in §4.7
Table 6. Per-president filter outputs are also retained in the
sibling-project deliverable `all_pooled_and_perpres_filtered.xlsx`
for transparency.

Source: `five_variant_retention.csv` (line 6).

---

## Rejected alternative 5: Euclidean + pooled location correction

**Empirical retention.** Euclidean + pooled correction at α = 0.01
(empirical LOO): **68.8%** overall (Lee 68.3%, Park 66.5%, Moon 71.8%).
Lands directly in the intuitively desirable 65-75% band using the
simplest possible distance metric.

**Sentence-level disagreement against Mahalanobis + pooled:** 27.5%.
This is the largest disagreement among the headline variants — out of
1,662 pledges, ~457 flip KEEP/REMOVE between the two filters.

**Per-cluster patterns confirming Mahalanobis is doing real work.**

| Cluster | Label | Euclidean retention | Mahalanobis retention | Δ |
|---:|---|---:|---:|---:|
| 1 | Climate Change Adaptation | 100.0% (2/2) | 0.0% (0/2) | −100.0 |
| 2 | Forest Welfare | 78.8% | 87.9% | +9.1 |
| 3 | Urban Forestry | 47.9% | 30.2% | −17.7 |
| 4 | Ecosystem Protection | 86.6% | 63.2% | −23.4 |
| 5 | Community Forestry | 77.3% | 61.6% | −15.7 |
| 6 | Forest Bioenergy | 73.9% | 31.2% | **−42.7** |
| 7 | Carbon Management | 11.9% | 33.1% | **+21.2** |
| 8 | Forest Policy Infra. | 90.0% | 70.0% | −20.0 |

**Why rejected as default.** Cluster 6 (Forest Bioenergy) shows the
most damaging case: Euclidean retains 73.9% (161/218) while Mahalanobis
retains 31.2% (68/218). Qualitative review confirmed that most Cluster 6
OOS pledges are general industrial-policy content (defense industry,
electronics, urban industrial complexes) that landed near the cluster
centroid because the cluster's elongated reference shape leaves slack
along its dominant axis. Euclidean ignores the cluster's anisotropy
and keeps these off-topic pledges; Mahalanobis correctly recognises
them as far from the cluster's *shape-adjusted* centroid.

Conversely, Cluster 7 (Carbon Management) shows the inverse pattern:
Mahalanobis 33.1%, Euclidean 11.9%. The cluster is tight in reference
space (n_ref = 38 highly technical carbon-management strategies) but
the OOS distribution is shape-mismatched, with carbon-related Lee
pledges (e.g., "carbon credit market") landing along an axis where
Euclidean penalises them but Mahalanobis recognises the axis as a
legitimate cluster direction.

Selecting Euclidean as the default would defeat the filter's purpose
for Cluster 6 (the cluster where the filter is doing most of its
useful work) in exchange for a higher overall retention rate. The
total retention figure 68.8% obscures this per-cluster failure.

**Where it lives in the manuscript.** Reported as variant #2 in §4.7
Table 6 and as the "Euclidean_decision" column in the per-president
review xlsx files Hye In is reviewing. Disagreement rows in those
files are explicitly flagged for domain-expert adjudication.

Source: `five_variant_retention.csv` (line 3) and
`five_variant_per_cluster.csv`.

---

## Future work (re-stated for ease of reference)

The four future-work items in Discussion §5 are direct extensions of
the rejected alternatives above:

1. **Shrinkage covariance with principled β** (Discussion §5;
   continuation of rejected alternative 3).
2. **Ensemble filter** combining Mahalanobis with silhouette ≥ 0 or
   projection-head confidence (Discussion §5; new dimension not
   covered by any single rejected alternative).
3. **Per-cluster α from objective criteria** (silhouette, topic
   coherence) — continuation of rejected alternative 1.
4. **Hybrid using projection-head confidence** (assign_sim, sub_sim
   columns) (Discussion §5; new dimension).
5. **Multi-corpus filter validation** beyond Korean forest
   (Discussion §5).

Each is a non-trivial methodological contribution that merits its own
validation. None is required for the present manuscript's claims.

---

## Provenance

- Filter validation analysis: `green-narrative/hye_in/for_hyein/
  park_moon_results_2026-05-07_v2/`
- Five-variant retention: `five_variant_retention.csv`
- Pairwise disagreement: `five_variant_disagreement_matrix.csv`
- Per-cluster retention: `five_variant_per_cluster.csv`
- α-sensitivity + shrinkage: `higher_retention_exploration.csv`
- Qualitative validation: `text_analysis_preliminary.md`
- Hye In review xlsx files: `for_hyein_handoff_2026-05-09/`
- Cross-project TRACE: `trace_20260509_305aaf` (decisions / rejected
  alternatives), `trace_20260509_c4f4ac` (handoff contributions)
- REAP-side adoption TRACE: `trace_20260509_947f1d`
