# Related Work

<!-- NOTE: Citation placeholders below use the [CITE: AuthorYear — claim] format.
     Every placeholder names the specific claim being attributed so that Phase 5
     citation verification can check whether the cited paper actually supports
     that claim before promoting it to a real BibTeX entry. Do not fabricate
     DOIs, arXiv IDs, or BibTeX. See publication-standards.md. -->

REAP sits at the intersection of four literatures: stochastic
dimensionality reduction, ensemble methods, embedding-based topic
modeling, and distribution-free uncertainty quantification. This
section reviews each in the order in which it bears on the method.

## UMAP and the Rotation/Reflection Ambiguity

UMAP [CITE: McInnesEtAl2018 — Uniform Manifold Approximation and
Projection, fuzzy simplicial set objective] has become the de-facto
nonlinear dimensionality reduction tool for text-embedding pipelines.
It constructs a fuzzy simplicial set representation of the
high-dimensional data and optimizes a cross-entropy objective between
that representation and a low-dimensional layout via stochastic
gradient descent. Two properties of this construction matter for our
work. First, the objective is invariant under rotation, reflection,
and translation of the output coordinates: any orthogonal transform
of an optimal layout is itself optimal. Second, the SGD optimization
is initialized stochastically (typically by spectral embedding of the
fuzzy graph), so different random seeds reach different members of
the equivalence class — and, because UMAP's objective is non-convex,
also different *non-equivalent* local optima that disagree on which
neighborhoods to preserve and which to compress. Existing analyses of
UMAP behavior [CITE: KobakLindermanCommentary — UMAP stability
analyses] document the practical impact of seed choice on downstream
metrics. REAP takes this ambiguity as load-bearing and operates on a
quantity (the pairwise distance matrix) that is exactly invariant to
the orthogonal part of the ambiguity by construction, while averaging
absorbs the non-rigid local disagreements as noise.

## t-SNE Stabilization

The closest methodological ancestor of REAP is the line of work on
stabilizing t-SNE [CITE: vanDerMaatenHinton2008 — t-SNE objective
shares the orthogonal ambiguity property]. Two strategies dominate.
Kobak & Linderman [CITE: KobakLinderman2021 — initializing t-SNE from
PCA stabilizes layouts across runs and across datasets] show that
initializing t-SNE from the first two principal components of the
input substantially reduces between-run variance and preserves global
structure. Belkina et al. [CITE: BelkinaEtAl2019 — opt-SNE adaptive
perplexity selection improves t-SNE stability on large datasets]
introduce automated perplexity selection that adapts to local density
and produces more reproducible layouts at scale. Both interventions
reduce the *magnitude* of stochastic variation but do not eliminate
it; both also rely on objective-specific properties of t-SNE that do
not transfer to UMAP's distinct optimization landscape (the fuzzy
simplicial set objective has different topology of local minima than
t-SNE's KL divergence over Student-$t$ similarities). REAP is
complementary: it treats stochasticity as something to *aggregate*
rather than dampen, and the aggregation operates after the
optimization rather than inside it.

## Ensemble Dimensionality Reduction and Consensus Clustering

The idea that multiple unstable runs of a dimensionality reduction or
clustering algorithm can be combined into a more stable estimator is
classical. Fred & Jain [CITE: FredJain2005 — evidence accumulation
clustering aggregates co-association matrices across runs] introduced
evidence-accumulation clustering, which aggregates binary
co-association matrices (was pair $(i, j)$ in the same cluster on
this run?) across runs and then clusters the accumulated matrix.
Strehl & Ghosh [CITE: StrehlGhosh2002 — cluster ensembles via
mutual-information combination] formalized the cluster-ensemble
problem and proposed mutual-information-based partition combination.
These methods operate at the *label* level — they discard the metric
structure that produced the partitions. The continuous analogue —
combining the embeddings themselves rather than the labels they
induce — has received less attention; existing proposals typically
align embeddings into a shared coordinate frame [CITE: WangEtAl2020 —
embedding alignment via orthogonal Procrustes] and then average, which
inherits the limitations described under Procrustes below. REAP
extends the ensemble-clustering tradition to the continuous-metric
regime: distance matrices are real-valued (not binary), preserving
the full magnitude of relational structure, and they live in a vector
space closed under non-negative weighted averaging (preserving metric
properties; §3.2). This makes the consensus matrix directly
re-embeddable via UMAP's precomputed-metric pathway, removing the
need for a separate consensus-partition step.

## BERTopic and Embedding-Based Topic Modeling

Embedding-based topic modeling has converged on a three-stage
pipeline: sentence-transformer encoding, UMAP reduction, density-based
or partition-based clustering, and term-frequency labeling. BERTopic
[CITE: Grootendorst2022 — BERTopic combines UMAP + HDBSCAN + c-TF-IDF
in a single-seed pipeline] is the canonical instantiation: a frozen
UMAP run (single seed by default) feeds HDBSCAN, which feeds c-TF-IDF
for topic labels. Top2Vec [CITE: Angelov2020 — Top2Vec joint
embedding-topic discovery] takes a structurally similar approach with
a different encoder and dimensionality-reduction choice. Both inherit
the seed-dependence described above: rerunning the pipeline with a
different UMAP random state changes which documents cluster together,
which topics merge, and what labels emerge. REAP is designed as a
drop-in replacement for the single-seed UMAP stage in any such
pipeline. The downstream clusterer (KMeans, HDBSCAN, or any other) is
free to operate on the consensus embedding exactly as it would on a
single-seed embedding. Section 4 reports results under KMeans;
HDBSCAN is supported in the package and reported in supplementary
materials.

## Procrustes Analysis and Coordinate-Space Alignment

The default response to coordinate-frame ambiguity is Procrustes
alignment. Gower [CITE: Gower1975 — generalized Procrustes analysis
aligns multiple shape configurations by orthogonal transformation +
scaling, minimising the sum of squared residuals] introduced
generalized Procrustes analysis, which aligns $S$ point
configurations of matched size by jointly optimizing rotations,
reflections, and a uniform scale to minimize the residual sum of
squares against a (typically iteratively refined) mean configuration.
Procrustes has been applied to embedding alignment in cross-lingual
word vectors, in multi-modal latent spaces, and in single-cell
analysis [CITE: WangEtAl2020 — embedding alignment via orthogonal
Procrustes]. Its limitation in the UMAP setting is structural rather
than tunable. Procrustes recovers the best *rigid* (or
similarity-class) transform between configurations, but UMAP's
inter-seed disagreements are not rigid: local neighborhoods stretch
and compress differently across seeds because the SGD trajectory
visits different non-convex basins. Aligning two configurations to
minimize their global Procrustes distance does not (and cannot)
correct *local* discrepancies where different seeds disagree about
which neighborhoods to preserve. Distance-matrix averaging bypasses
this by operating on a representation that is invariant to the rigid
part of the ambiguity to begin with, so the only quantity averaged is
the local-structure disagreement itself — exactly the noise the
consensus is designed to suppress.

## Wisdom of Crowds and the Condorcet Jury Theorem

The intuition that averaging many noisy estimators produces an
estimator better than any individual is formalized by the Condorcet
jury theorem [CITE: Condorcet1785 — majority of competent independent
voters outperforms any individual voter] and its modern extensions
[CITE: ListGoodin2001 — Condorcet jury theorem extensions to
multi-alternative settings]. Translated into our setting: if each
seed's UMAP run is a noisy estimator of the data manifold's
relational structure with per-entry error bounded below the signal
magnitude, the element-wise average of distance matrices converges to
the true relational structure as $S \to \infty$. The assumptions are
strong (per-entry errors that are noise-like in expectation and
roughly independent across seeds) and we do not claim they hold
exactly. They hold approximately well enough in practice that 30
seeds — well below any asymptotic regime — yield consensus distance
matrices with relative Frobenius distance below 0.02 across disjoint
seed sets on both datasets we test (§4.2). The Condorcet analogue is
the heuristic justification for *why* averaging works; the formal
guarantees that REAP provides are the metric-space-preservation proof
(§3.2) and the empirical demonstration of between-set stability
(§4.2.1 Table 1d, §4.2.2 Table 2d).

## Knowledge Distillation and Neural Projection Heads

REAP's projection head (§3.4) trains an MLP to map sentence
embeddings directly into the consensus coordinate space, bypassing
the expensive multi-seed consensus procedure at inference time. This
construction can be viewed through the lens of knowledge distillation
[CITE: HintonEtAl2015 — knowledge distillation: train a student to
match a teacher's output on a fixed training distribution]: the
multi-seed REAP pipeline is the "teacher" (slow, non-parametric,
producing the consensus coordinates as a fixed target), and the MLP
is the "student" (fast, parametric, learning to approximate the
teacher's output for new inputs). The closest direct analogue in
dimensionality reduction is parametric UMAP [CITE: SainburgEtAl2021 —
parametric UMAP trains a neural network to reproduce UMAP's
nonlinear embedding for out-of-sample data], which trains a network
to approximate a single-seed UMAP layout. REAP's projection head
differs from parametric UMAP in two ways. First, the target is the
*consensus* embedding rather than a single-seed embedding, so the
network learns the more stable target. Second, the loss function
combines pointwise MSE with a distance-correlation term that
preserves relational geometry; topic-modeling applications depend on
relational structure more than on exact coordinates. As §4.4 reports,
the resulting head is a *semantic interpolator* rather than a
*novelty detector* — held-out topics route to the nearest semantic
neighbor in the training set rather than being assigned a "new
cluster" identity. This behavior is what motivates the conformal
filter (§3.6).

## Conformal Prediction and Out-of-Distribution Rejection

The OOS filter introduced in §3.6 instantiates the split-conformal
prediction framework [CITE: VovkEtAl2005 — algorithmic learning in a
random world, foundational split-conformal framework] [CITE:
ShaferVovk2008 — tutorial on conformal prediction's exchangeability
requirement and finite-sample coverage guarantee]. Split conformal partitions a
labeled exchangeable sample into a calibration set and a test set,
computes a non-conformity score for each calibration point, and uses
the empirical $(1 - \alpha)$ quantile of those scores as the
threshold for accepting new test points. The guarantee is
distribution-free in the sense that no parametric assumption on the
data is required, and finite-sample in the sense that the threshold
controls the expected coverage at any sample size where the quantile
is well-defined. The framework has since been adapted to selective
classification, anomaly detection, and prediction-set construction
for deep models [CITE: AngelopoulosBates2021 — gentle introduction to
conformal prediction, including selective classification and OOD
applications] [CITE: RomanoEtAl2020 — conformalized quantile
regression and selective-classification adaptations]. The choice of non-conformity
score determines what the filter is sensitive to: in our case, a
per-cluster squared Mahalanobis distance with a pooled OOS-centroid
location correction (§3.6). The Mahalanobis choice is grounded in
the classical multivariate-statistics literature on ellipsoidal
density estimation [CITE: MardiaEtAl1979 — multivariate analysis,
chapter on Mahalanobis distance and ellipsoidal cluster fits]; it
adapts to anisotropic cluster shapes that an isotropic Euclidean
score would conflate. The location correction is motivated by the
register/style shift between reference and OOS corpora endemic to
deployment-time NLP topic modeling and is decoupled from the shape
estimate by construction, so the cluster-shape geometry remains
anchored in the reference data. The empirical-LOO threshold (rather
than chi-squared theoretical) is the choice that preserves
finite-sample exchangeability of reference points after the location
correction; §3.6 lays out the conditions under which the heuristic
coverage statement holds. REAP's filter is, to our knowledge, the
first conformal application to gating out-of-sample placements in a
multi-seed UMAP consensus space.
