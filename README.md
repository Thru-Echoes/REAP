# REAP: Reproducible Embedding via Averaged Projection

Stabilize stochastic dimensionality reduction through distance-matrix consensus.

UMAP embeddings are rotationally arbitrary — different random seeds produce geometrically equivalent solutions differing by rigid transforms. **REAP** averages *pairwise distance matrices* across seeds, which is invariant to these transforms, producing stable embeddings that coordinate averaging destroys.

**Result:** +32% ARI improvement over Procrustes consensus (0.75 vs 0.56).

## Installation

```bash
pip install reap-topics
```

With optional dependencies:
```bash
pip install reap-topics[projection]   # Neural projection head (PyTorch)
pip install reap-topics[labeling]     # LLM cluster labeling
pip install reap-topics[all]          # Everything
```

## Quick Start

```python
import numpy as np
from reap import run_consensus_pipeline, find_best_k, run_kmeans

# Your sentence embeddings (n_samples, n_features)
X = np.load("embeddings.npy")

# Run REAP consensus: multi-seed UMAP → distance-matrix averaging → final projection
embedding, metadata = run_consensus_pipeline(
    X=X,
    seeds=list(range(30)),   # 30 random seeds
    n_components=18,         # target dimensions
    n_neighbors=19,          # UMAP neighborhood size
    min_dist=0.005,          # minimum distance
)

# Find optimal K and cluster
result = find_best_k(embedding, k_range=range(3, 20))
labels, kmeans_model = run_kmeans(embedding, k=result.best_k)
```

## How It Works

1. **Multi-seed UMAP**: Run UMAP N times with different random seeds
2. **Distance-matrix averaging**: Compute pairwise distance matrices for each seed, then average them (invariant to rotation/reflection)
3. **Final UMAP**: Run UMAP on the consensus distance matrix with `metric="precomputed"`
4. **Clustering**: KMeans with silhouette-based K selection
5. **Projection head** (optional): Train an MLP to map new data directly to consensus space

## API Reference

### Core Consensus
- `run_consensus_pipeline(X, seeds, ...)` — Full pipeline: multi-seed → consensus → embedding
- `get_multi_seed_embeddings(X, seeds, ...)` — Run UMAP with each seed
- `get_consensus_distance_matrix(embeddings)` — Average pairwise distances
- `get_consensus_embedding(D, ...)` — UMAP on precomputed consensus distances

### Evaluation
- `compute_trustworthiness(X_high, X_low)` — Neighborhood preservation
- `compute_silhouette(X, labels)` — Cluster separation quality
- `compute_pairwise_ari(labels_list)` — Seed-to-seed agreement
- `compute_seed_stability(seed_labels, consensus_labels)` — Stability statistics

### Clustering
- `find_best_k(X, k_range)` — Silhouette-based K selection
- `run_kmeans(X, k)` — KMeans clustering

### Validation
- `validate_embeddings(X)` — Check for NaN, duplicates, normalization
- `validate_cluster_sizes(labels)` — Cluster size distribution checks

## Citation

```bibtex
@article{muellerklein2026reap,
  title={REAP: Stabilizing Semantic Topic Spaces via Distance-Matrix Consensus and Neural Projection},
  author={Muellerklein, Oliver},
  year={2026}
}
```

## License

Apache 2.0
