# Abstract

<!-- NOTE: All numerical claims below are copied to three decimal places
     from manuscript/sections/experiments.md Tables 1b, 1d, 2b, 2d, and §4.7.
     Do not edit the numbers without re-checking the source tables.
     Word count target: 200-280. -->

UMAP is the standard nonlinear dimensionality reduction component
of embedding-based topic-modeling pipelines, but its stochastic
optimization produces seed-dependent embeddings: different random
seeds yield different rotations, reflections, and — critically —
non-rigid local distortions of the manifold. Downstream clustering
inherits this instability: the same corpus yields different topic
assignments depending on the seed. Procrustes alignment recovers
only the rigid part of the disagreement and provides no consistent
improvement over single-seed runs on real text corpora. We present
**REAP** (Reproducible Embedding via Averaged Projection), a
consensus-based approach that averages pairwise *distance matrices*
across multi-seed UMAP runs rather than coordinates. Because
Euclidean distances are invariant to rotation and reflection by
construction, the average is invariant to the orthogonal ambiguity
without requiring alignment, and the average preserves metric
properties. The consensus distance matrix is then re-embedded via a
single final UMAP projection. A neural projection head, trained with
a combined MSE + distance-correlation loss, places new documents in
the consensus space without re-running the multi-seed procedure; a
split-conformal filter with a Mahalanobis non-conformity score and
per-cluster empirical leave-one-out threshold gates the projection
head's placements under register-shifted out-of-sample inputs. On
two pre-registered corpora — AI-art discourse (1,736 English
chunks, e5-large-v2 1024-d) and Korean forest policy (905 Korean
chunks, MiniLM 384-d) — REAP achieves consensus silhouette
0.672-0.681 versus 0.394-0.404 for single-seed UMAP, with
between-set consensus-label ARI 0.810 (Korean forest) and 0.915
(AI-art) across three disjoint 30-seed sets; the conformal filter
retains 50.9% of register-shifted OOS pledges in the Korean forest
case study at $\alpha = 0.01$. REAP is released as `pip install
reap-topics` with a typed API, dataset loaders, a benchmark harness,
and a golden-validation CI suite.
