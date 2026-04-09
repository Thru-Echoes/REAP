# Introduction

<!-- TODO: Draft introduction prose -->
<!-- Key points to cover:
  - UMAP is the de facto standard for dimensionality reduction in NLP/topic modeling
  - But UMAP is stochastic: different random seeds produce different embeddings
  - This undermines reproducibility of downstream clustering and topic assignments
  - Existing mitigation (Procrustes alignment + coordinate averaging) is insufficient
  - REAP solves this by averaging pairwise distance matrices, which are invariant to rotation/reflection
  - The projection head enables new data to be mapped without re-running multi-seed UMAP
  - Validated across 4 datasets spanning 3 languages and 4 domains
-->
