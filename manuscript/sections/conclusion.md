# Conclusion

<!-- NOTE: Citation placeholders below use the [CITE: AuthorYear — claim] format.
     Phase 5 verification replaces these with verified BibTeX entries.
     See publication-standards.md. -->

We presented REAP, a consensus-based approach to stabilizing
stochastic dimensionality reduction for embedding-based topic
modeling. By averaging pairwise distance matrices across multi-seed
UMAP runs — an operation invariant to the rotation/reflection
ambiguity of UMAP embeddings by construction — REAP produces stable,
reproducible topic spaces without requiring coordinate-frame
alignment.

Our five contributions are:

1. **A distance-matrix consensus algorithm for UMAP stabilization
   (§3.1-3.2).** Element-wise averaging of Euclidean distance
   matrices is exactly invariant to UMAP's orthogonal ambiguity and
   preserves metric-space properties (symmetry, non-negativity, and
   the triangle inequality) by a convex-combination argument. The
   resulting consensus distance matrix is directly re-embeddable via
   UMAP's precomputed-metric pathway, requiring no separate
   consensus-partition step.

2. **A neural projection head for out-of-sample inference (§3.4).**
   The MLP head maps sentence embeddings directly into the consensus
   coordinate space, removing the $O(N^2)$ cost of full consensus
   recomputation. A combined MSE + distance-correlation loss preserves
   relational geometry. Topic-exclusion ablations confirm the head is
   a semantic interpolator rather than a novelty detector.

3. **An out-of-sample conformal filter for principled rejection
   (§3.6).** A per-cluster squared-Mahalanobis non-conformity score
   with a pooled OOS-centroid location correction and an empirical
   leave-one-out threshold at level $\alpha$ gates the projection
   head's placements. The five-variant comparison on the Korean
   forest case study (§4.7) confirms shape adaptation and location
   correction each do material work; supplementary materials
   enumerate rejected alternatives and their failure modes.

4. **Multi-dataset validation under a pre-registered protocol (§4).**
   Two corpora differing in language, domain, embedding model, and
   size — AI-art discourse (1,736 English chunks, e5-large-v2
   1024-d) and Korean forest policy (905 Korean chunks, MiniLM
   384-d) — evaluated under a protocol specifying datasets, three
   disjoint 30-seed sets, metrics, statistical procedures, and
   K-selection rules before any results were computed. REAP's
   silhouette advantage replicates across both datasets and all three
   seed sets, and its between-set consensus-label ARI exceeds every
   baseline.

5. **An open-source package for adoption and extension.** REAP is
   released as `pip install reap-topics`, with a typed public API,
   dataset loaders, a benchmark harness, a golden-validation test
   suite enforcing pre-registered scientific claims as CI checks, and
   a reproducibility bundle attached to every committed result.

**Planned applications.** Two follow-on papers build directly on
this infrastructure. The first develops the Alignment-Resistance
Index for measuring drift between aligned topic spaces over time,
using REAP consensus spaces as the stable substrate [CITE:
ARIPaper2026 — forthcoming companion paper on alignment-resistance
measurement, cross-ref §6]. The second is a disclosure-audit study
of corporate sustainability reporting using the BDVGA framework over
a 69-report pilot corpus, with REAP consensus topics serving as the
disclosure-category taxonomy [CITE: BDVGAPilot2026 — forthcoming
BDVGA pilot, cross-ref §6].

REAP is available now via `pip install reap-topics`; the package,
benchmark harness, and pre-registered protocol are under version
control at the repository linked from the arXiv preprint [CITE:
REAPArxivPlaceholder — arXiv preprint with repository link, to be
assigned at submission]. We hope it serves both as a practical tool
for researchers who depend on stable topic assignments and as a
contribution to the broader question of how to build reproducible
analyses on stochastic algorithmic foundations.
