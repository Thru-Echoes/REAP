# Introduction

<!-- NOTE: Citation keys are bracketed placeholders [Author YYYY]. Real BibTeX
     entries must be verified against the actual papers before submission — see
     publication-standards.md. Do not fabricate DOIs or page numbers. -->

Embedding-based topic modeling has become the dominant paradigm for
discovering thematic structure in text corpora. The standard pipeline —
encode documents with a pretrained transformer, reduce dimensionality
via UMAP [McInnes et al. 2018], cluster in the low-dimensional
space, and label the resulting groups — underpins widely-adopted
systems such as BERTopic [Grootendorst 2022] and Top2Vec [Angelov 2020].
Its appeal is its modularity: each component can be swapped or tuned
independently, and the resulting topics are interpretable as regions of
a continuous semantic space.

A foundational assumption of this pipeline, however, is rarely
examined: that the UMAP step produces a *stable* embedding. In
practice, UMAP is stochastic. Different random seeds yield embeddings
that differ by rotation, reflection, and — critically — by local
non-rigid distortions of the manifold. Two runs on the same data can
partition a political-discourse corpus into 20 distinct topics or 18,
can merge or separate semantically adjacent themes, and can assign the
same document to different clusters. This instability propagates into
every downstream artifact: topic assignments, cluster labels,
comparative analyses across corpora, and the scientific claims built
on them.

The standard mitigation is Procrustes alignment: align each seed's
embedding to a reference via orthogonal transformation, then average
the aligned coordinates. This approach, rooted in shape analysis
[Gower 1975], recovers a consensus when the inter-seed variation is
limited to rigid transforms. But UMAP's distortions are not rigid —
local neighborhoods stretch and compress differently across seeds — and
when clusters have genuine semantic overlap, different seeds may
resolve the ambiguity in incompatible ways. Procrustes alignment
followed by coordinate averaging blurs these divergent local solutions,
producing a consensus that is less informative than any individual
seed. On our Korean forest policy corpus, Procrustes consensus achieves
an adjusted Rand index (ARI) of 0.56 against expert labels; single
seeds individually achieve similar scores. The consensus adds no
information.

We propose **REAP** (Reproducible Embedding via Averaged Projection),
a consensus-based approach that operates on *pairwise distance
matrices* rather than coordinates. Because Euclidean distances are
invariant to rotation and reflection, averaging distance matrices
across seeds preserves relational structure without requiring
alignment. The consensus distance matrix is then re-embedded via UMAP
in a single final projection, producing a stable, coherent embedding.
On the Korean forest corpus, REAP achieves an ARI of 0.75 — a 32%
improvement over Procrustes — and a silhouette score that substantially
exceeds all baselines. We additionally train a lightweight neural
projection head that maps new documents into the consensus space
without re-running the multi-seed procedure, enabling efficient
out-of-sample topic assignment.

Our contributions are:

1. **A distance-matrix consensus algorithm for UMAP stabilization.**
   We show formally that the element-wise average of Euclidean distance
   matrices preserves metric-space properties (§3.2) and demonstrate
   empirically that it outperforms Procrustes consensus, best-of-N
   selection, naive coordinate averaging, and single-seed baselines
   across two corpora spanning different languages, domains, and
   embedding models (§4).

2. **A neural projection head for out-of-sample inference.** The
   projection head (§3.4) learns a direct mapping from sentence
   embeddings to consensus coordinates, enabling new documents to be
   placed in an established topic space without the O($N^2$) cost of
   full consensus recomputation. Cross-validation experiments
   demonstrate that the head generalizes to unseen exemplars of known
   topics and interpolates meaningfully for semantically adjacent novel
   content.

3. **An out-of-sample conformal filter for principled rejection.** The
   projection head places every input somewhere; the filter (§3.6)
   decides which placements should be trusted. Built on the
   split-conformal framework, it uses a per-cluster Mahalanobis
   non-conformity score with a pooled location correction (to handle
   reference/OOS register shift) and an empirical leave-one-out
   threshold at level $\alpha$. On a Korean forest case study (§4.7)
   projecting 1,662 district-level political pledges into a topic
   space trained on 905 national-scale planning sentences, the default
   filter retains 50.9% of pledges; head-to-head against four
   alternative variants confirms that shape adaptation (Mahalanobis
   over Euclidean) and location correction each do material work.

4. **Multi-dataset validation under a pre-registered protocol.** We
   evaluate REAP on two corpora — public discourse about AI-generated
   art (1,736 English chunks, e5-large-v2 1024-d) and Korean forest
   policy strategy sentences (905 Korean chunks, multilingual MiniLM
   384-d) — using a pre-registered evaluation protocol that specifies
   datasets, seed design (three disjoint 30-seed sets), metrics,
   statistical procedures, and K-selection rules before any results
   are computed.

5. **An open-source package for adoption and extension.** REAP is
   released as `pip install reap-topics`, with a typed API, dataset
   loaders, a benchmark harness, and a golden-validation test suite
   that enforces pre-registered scientific claims as CI checks.

The remainder of this paper is organized as follows. Section 2 reviews
related work in dimensionality-reduction stabilization, ensemble
clustering, neural projection, and conformal prediction. Section 3
presents the REAP method: the consensus algorithm (§3.1–3.2),
parameter search (§3.3), projection head (§3.4), cluster labeling
(§3.5), and out-of-sample conformal filter (§3.6). Section 4 reports
experimental results on two datasets with six comparison methods plus
a five-variant filter validation case study. Section 5 discusses when
to use REAP, its limitations, future work on filter extensions, and
connections to the ensemble and conformal-prediction literature.
Section 6 concludes.


# Related Work

<!-- NOTE: Every citation claim must be verified against the actual paper.
     Placeholder keys below need real BibTeX from the publisher / arXiv.
     See publication-standards.md: "Every citation has a DOI, arXiv ID,
     or canonical URL. BibTeX entries are copied from the publisher." -->

**UMAP and stochastic dimensionality reduction.** UMAP [McInnes et al. 2018]
constructs a fuzzy simplicial set representation of the high-dimensional
data and optimizes a cross-entropy objective to find a low-dimensional
layout. The objective has multiple equivalent minima related by rotation,
reflection, and non-rigid distortions, making the output seed-dependent.
t-SNE [van der Maaten & Hinton 2008] shares this property; efforts to
stabilize it include initialization from PCA [Kobak & Linderman 2021]
and optimized perplexity selection [Belkina et al. 2019]. These
approaches reduce but do not eliminate stochastic variation, and they do
not transfer to UMAP's distinct optimization landscape.

**Procrustes analysis and shape alignment.** Generalized Procrustes
analysis [Gower 1975] aligns point configurations by minimizing the sum
of squared distances under rotation, reflection, and scaling. It is the
standard approach for comparing landmark-based shapes and has been
applied to embedding alignment [Wang et al. 2020]. Its limitation in the
UMAP setting is that UMAP distortions are non-rigid: Procrustes recovers
the best *global* rigid transform but cannot correct *local*
discrepancies where neighborhoods have stretched or rotated differently
across seeds.

**Ensemble clustering and consensus.** Fred & Jain [2005] introduced
evidence-accumulation clustering, which aggregates co-association
matrices across multiple clusterings. Strehl & Ghosh [2002] proposed
cluster ensembles via mutual-information-based combination. These
methods operate on discrete label partitions or binary co-occurrence
matrices and discard metric structure. REAP extends this line of work
to continuous distance matrices, retaining relational magnitude
information that enables re-embedding rather than requiring a separate
consensus-partition step.

**BERTopic and embedding-based topic models.** BERTopic [Grootendorst 2022]
combines UMAP, HDBSCAN, and c-TF-IDF into an end-to-end topic-modeling
pipeline. It uses a single UMAP seed by default, inheriting the
instability described above. Top2Vec [Angelov 2020] takes a similar
approach with a different embedding strategy. Neither system addresses
UMAP instability directly. REAP can serve as a drop-in replacement for
the single-seed UMAP step in any such pipeline, with the consensus
embedding passed to any downstream clusterer.

**Knowledge distillation and neural projection.** The REAP projection
head can be viewed through the lens of knowledge distillation [Hinton
et al. 2015]: the multi-seed consensus procedure is the "teacher" and
the MLP is the "student" that approximates the teacher's output at
inference time. Similar distillation approaches have been used to
compress non-parametric dimensionality reduction into parametric
models [Sainburg et al. 2021] (parametric UMAP).

**Condorcet jury theorem and wisdom of crowds.** The intuition behind
REAP — that averaging many noisy estimates yields a better estimate
than any individual — is formalized by the Condorcet jury theorem and
its extensions [List & Goodin 2001]. REAP applies this principle in
metric space: each seed's distance matrix is a noisy estimate of the
data manifold's relational structure, and their average converges toward
the true structure as the number of seeds grows.

**Conformal prediction and out-of-distribution rejection.** Split
conformal prediction [Vovk et al. 2005; Shafer & Vovk 2008] provides
distribution-free finite-sample coverage guarantees for prediction
sets, given a calibration set of exchangeable in-distribution samples.
The framework has been adapted to selective classification, anomaly
detection, and prediction-set construction for deep models
[Angelopoulos & Bates 2021; Romano et al. 2020]. REAP's out-of-sample
filter (§3.6) instantiates split conformal in the consensus space:
the reference corpus serves as the calibration set, the per-cluster
Mahalanobis distance with location correction is the non-conformity
score, and the threshold is the $(1-\alpha)$ quantile of the
empirical leave-one-out reference distribution. The Mahalanobis
choice is grounded in classical multivariate-statistics literature on
ellipsoidal cluster fits [Mardia et al. 1979]; the location-shift
correction is motivated by the register/style gap between training
and OOS corpora that is endemic to NLP topic-modeling deployment
contexts.
