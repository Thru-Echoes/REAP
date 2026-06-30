# Discussion

## When to Use REAP

REAP is most valuable when downstream decisions depend on the specific
cluster structure of the embedding — i.e., when it matters *which*
documents are assigned to *which* topic, not just that there exist
$K$ groups. This includes: topic-modeling studies where labels will be
published, cross-corpus comparisons where embeddings must be comparable,
longitudinal analyses where new data must land in an established topic
space, and any setting where a reviewer might re-run the analysis with
a different seed and arrive at different conclusions.

REAP is unnecessary when the downstream task is robust to embedding
perturbation (e.g., broad visualization for exploratory analysis,
nearest-neighbor retrieval with a stable index, or classification tasks
where the embedding feeds into a supervised model that is itself
trained end-to-end).

## Consensus K vs Seed K

A consistent finding across both corpora is that the consensus-optimal
$K$ is lower than the per-seed-optimal $K$. On the Korean forest
corpus, individual seeds prefer $K \in [24, 30]$ while the consensus
embedding produces its best silhouette at $K = 8$ (Hye In's validated
configuration) or $K \in [23, 30]$ under free $K$-selection. The
explanation is straightforward: REAP's distance-matrix averaging damps
the fine-grained structure that differs across seeds (noise) while
preserving the coarse-grained structure that is consistent (signal).
The consensus is inherently coarser-grained than any individual seed —
it trades resolution for reliability.

This is a feature, not a bug: the coarse-grained structure is what
survives adversarial replication (run again with new seeds, get the
same clusters). The fine-grained structure that single seeds exhibit
is real local topology but is not reliably recoverable across seeds,
and therefore should not be the basis of published claims.

## Limitations

**$O(N^2)$ memory.** The consensus distance matrix has $N^2$ entries.
For $N = 905$ (Korean forest) this is tractable; for $N = 1{,}736$
(AI-art) it requires ~6 GB in float64 during accumulation; for
$N > 5{,}000$ it becomes impractical on consumer hardware. Approximate
methods (random landmark subsets, Nyström approximation, or
nearest-neighbor graph sparsification) could reduce this to
$O(N \log N)$ but would change the consensus properties.

**Parameters do not transfer.** The optimal $(d, k_{\mathrm{nn}},
d_{\mathrm{min}})$ triple is dataset-specific and must be selected via
grid search for each new corpus. The four-round progressive search
(§3.3) reduces the search cost but does not eliminate it. Users
importing REAP on a new corpus should budget compute time for
parameter selection, not assume that published configurations
generalize.

**Trustworthiness tradeoff.** REAP consistently trades 2–3% of
trustworthiness for large silhouette gains. Users who prioritize
local-structure preservation over cluster separation should consider
whether this tradeoff is acceptable for their task. In our experience,
the silhouette gain dominates for topic-modeling applications, but
tasks that depend on fine-grained neighborhood structure (e.g.,
trajectory analysis) may prefer single-seed or Procrustes approaches.

**The projection head is an interpolator, not a detector.** As
demonstrated in §3.4 and §4.4, the projection head maps novel content
to the nearest training-cluster centroid. It does not alert the user
that the content is novel or create new clusters. The OOS conformal
filter (§3.6, §4.7) addresses this directly: it accepts or rejects the
projection head's placement using a calibrated per-cluster threshold
on Mahalanobis distance. Users projecting genuinely novel content
should always pair the projection head with the filter; using the
projection head alone reproduces the no-novelty-detection limitation.

**Filter retention varies by cluster alignment.** The OOS filter's
default ($\alpha = 0.01$ empirical LOO, Mahalanobis with pooled
location correction) retains 50.9% of pledges on the Korean forest
case study (§4.7), but per-cluster retention spans 30.2% (Cluster 3
Urban Forestry) to 87.9% (Cluster 2 Forest Welfare). This variability
reflects real differences in how well the OOS corpus aligns with each
reference cluster, not a flaw in the filter — qualitative review
confirmed the low-retention clusters genuinely contain more off-topic
content. Users reporting filter-retained samples should always
disaggregate by cluster, not aggregate to a single retention rate.

**Empirical LOO has a small-cluster ceiling.** The empirical leave-one-out
threshold at level $\alpha$ is undefined when the reference cluster has
fewer than $\lceil 1/\alpha \rceil$ points: the per-cluster LOO
distribution simply does not have a $(1-\alpha)$ quantile beyond its
maximum value. In the Korean forest case (§4.7.2), Cluster 1
($n_{\mathrm{ref}} = 17$) reaches its maximum LOO Mahalanobis distance
at $\alpha = 0.001$, so the filter cannot push retention beyond ~60%
at any $\alpha$ within the empirical-LOO framework. The chi-squared
theoretical alternative is much stricter at the same $\alpha$ (19.1%
retention vs 50.9% on this corpus, both at $\alpha = 0.01$, $d = 18$)
because chi-squared assumes the reference centroid and covariance are
known exactly and ignores finite-sample estimation noise. The two
threshold sources are not interchangeable; mixing them across cells
breaks the conformal coverage statement. We document this in the
package's filter API as a hard requirement.

**Two corpora, two languages.** We validate on English (AI-art) and
Korean (forest policy) but have not tested on corpora with fundamentally
different structural properties (e.g., highly uniform embeddings where
all topics overlap, or extremely high-dimensional spaces without clear
manifold structure). The corporate sustainability and US presidential
datasets (pending) will extend coverage but do not eliminate this
limitation.

## Future Work

Several extensions are well-motivated by the present results but
deferred to future work because each requires additional methodology
that we cannot stress-test on a single corpus.

**Shrinkage-covariance filter with principled $\beta$.** Section 4.7.2
shows that $\boldsymbol{\Sigma}_{\mathrm{eff}} = \beta
\boldsymbol{\Sigma}_{\mathrm{ref}} + (1 - \beta)
\boldsymbol{\Sigma}_{\mathrm{OOS}}$ at $\beta = 0.75$ raises retention
on the Korean forest corpus from 50.9% to 68.6% — directly in the
intuitively desirable 65-75% band — but $\beta = 0.75$ has no
principled basis (why not 0.6 or 0.85?). A defensible version of this
variant requires a derivation-not-tuning rule for $\beta$. Two leads:
(a) Ledoit-Wolf shrinkage [Ledoit & Wolf 2004], adapted to the
two-distribution setting where the "target" is the OOS covariance and
the "shrinkage source" is the reference covariance; (b) cross-validated
$\beta$ selection on held-out reference points, combined with an
exchangeability argument that preserves a conformal coverage statement.
Either route is a non-trivial methodological contribution and merits
its own validation before adoption.

**Ensemble filtering.** The current filter uses a single non-conformity
score (Mahalanobis with location correction). An ensemble could combine
this score with one or more orthogonal signals: silhouette $\ge 0$ (a
purely intra-OOS-corpus signal), assignment confidence from the
projection head's similarity to the assigned cluster centroid (the
`assign_sim` field in REAP's projection output), or topic-coherence-based
quality scores. Two combination rules are natural: *strict intersection*
(keep iff all signals agree) for high-precision use cases, and
*permissive union* (keep iff any signal agrees) for high-recall use
cases. The challenge is preserving a conformal coverage statement under
combination — a known-hard problem for conjunctions, easier for
unions [Romano et al. 2020].

**Per-cluster $\alpha$ from objective criteria.** Section 4.7's
per-cluster retention spread (30.2%-87.9%) suggests that different
clusters tolerate different filter thresholds. Post-hoc per-cluster
$\alpha$ tuning is not defensible (it would be reverse-engineered to
hit retention targets), but criterion-driven per-cluster $\alpha$
*could* be: stricter $\alpha$ for clusters with low silhouette or low
topic coherence (the ones likeliest to attract genuinely off-topic
projections), looser $\alpha$ for high-coherence clusters with broad
shape. The criterion must be set before retention is observed, and the
resulting per-cluster retention rates must be reported alongside the
overall figure.

**Hybrid with projection-head confidence.** The projection head outputs
both a cluster assignment and a similarity score against that cluster's
centroid; the current filter ignores the similarity score and relies
only on the per-cluster Mahalanobis-with-correction score. A hybrid
could relax the conformal threshold for high-confidence assignments
("trust the head" + "trust the cluster shape" combined). The risk is
that the projection head's confidence score is itself a function of
the same geometry the Mahalanobis filter measures, so the two signals
may not be sufficiently independent to combine usefully. An empirical
study would resolve this.

**Multi-corpus filter validation.** The filter has been validated on
one corpus (Korean forest planning vs district pledges, §4.7) with
domain-expert review pending. Replication on the AI-art discourse
corpus (where reference and OOS would need a comparable register
shift) and on the corporate-sustainability corpus (when locked) would
test whether the 50.9% headline retention generalizes or whether it is
specific to the planning/pledge register gap. Until then, the filter's
parameter defaults are calibrated on a single corpus and should be
treated as such.

**Stricter conformal guarantee via OOS-LOO threshold.** The default
filter (§3.6) calibrates the threshold $\tau_c$ from the *reference*
LOO Mahalanobis distribution but scores OOS points against the per-
cluster *OOS* centroid. Reference LOO scores condition on $n_c - 1$
reference samples; OOS scores condition on $|\mathcal{O}_c|$ OOS
samples. When the two counts are comparable the discrepancy is small,
but it breaks the formal exchangeability argument required for a
strict conformal coverage statement. A stricter variant uses
leave-one-out within the OOS sample as well: for each OOS point $j$
in cluster $c$, compute the OOS centroid excluding $j$, score $j$
against this leave-one-out centroid (with reference covariance), and
calibrate $\tau_c$ from the OOS-LOO Mahalanobis distribution. This
restores exchangeability conditional on the OOS sample being a
location-shifted version of the reference distribution. The cost is
$O(|\mathcal{O}_c|^2 d^2)$ versus the current $O(n_c^2 d^2 +
|\mathcal{O}_c| d^2)$ — typically a wash. Empirical retention should
be similar to the default, but the coverage guarantee is rigorous
rather than heuristic.

## Connection to Ensemble Methods

REAP occupies a specific position in the ensemble landscape. Classical
cluster ensembles [Fred & Jain 2005; Strehl & Ghosh 2002] aggregate
discrete label partitions, discarding metric structure. Embedding
ensembles (e.g., multi-view learning) combine different feature spaces,
not different runs of the same reduction. REAP combines multiple runs
of the *same* stochastic algorithm on the *same* features, in *metric
space* rather than label space. The closest formal analogue is the
observation that the mean of a collection of metrics (with
non-negative weights summing to 1) is itself a metric — the
consensus distance matrix inherits this property (§3.2).

The Condorcet jury theorem provides the asymptotic guarantee: if each
seed's distance matrix is a noisy estimate of the true relational
structure with per-entry error smaller than the signal, the average
converges. In practice, 30 seeds suffice for stable consensus on both
corpora tested, with diminishing returns beyond 15 (seed-count
ablation results, if available, in §4.3).


# Conclusion

We have presented REAP, a consensus-based approach to stabilizing
stochastic dimensionality reduction for embedding-based topic modeling.
By averaging pairwise distance matrices across multi-seed UMAP runs —
an operation that is invariant to the rotation/reflection ambiguity of
UMAP embeddings by construction — REAP produces stable, reproducible
topic spaces without requiring alignment.

On two corpora spanning different languages (English, Korean), domains
(AI-art discourse, forest policy), and embedding models (e5-large-v2,
multilingual MiniLM), REAP achieves substantially higher cluster
separation than all baselines while preserving most of the local
structure. The silhouette improvement over Procrustes consensus is
consistent across three disjoint 30-seed sets — the improvement is
not an artifact of seed selection.

The neural projection head extends REAP's utility to out-of-sample
data, enabling new documents to be placed in an established topic
space in milliseconds. Topic-exclusion experiments show that the head
generalizes meaningfully to semantically adjacent content,
interpolating between training clusters rather than scattering novel
inputs randomly.

REAP is released as an open-source Python package (`pip install
reap-topics`) with typed APIs, dataset loaders, a benchmark harness,
and a golden-validation test suite that enforces pre-registered
scientific claims as CI checks. We hope it serves as both a practical
tool for researchers who depend on stable topic assignments and a
contribution to the broader question of how to build reproducible
analyses on stochastic foundations.
