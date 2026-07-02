# REAP: Reproducible Embedding via Averaged Projection

**REAP** stabilizes stochastic dimensionality reduction by averaging
*pairwise distance matrices* across multi-seed UMAP runs — a step that
is invariant to UMAP's rotation/reflection ambiguity — then trains a
neural projection head that maps new data into the consensus space
without re-running the multi-seed pipeline. The result is a
reproducible topic geometry suitable for downstream clustering,
labeling, and out-of-sample inference. Method comparisons — against
Procrustes alignment, naive coordinate averaging, best-of-N seed
selection, and parametric UMAP — run under a pre-registered benchmark
protocol, and every reported number traces to a committed result
artifact rather than living in this README.

## Installation

```bash
pip install reap-topics
```

Optional extras:

```bash
pip install "reap-topics[projection]"     # Neural projection head (PyTorch)
pip install "reap-topics[labeling]"       # LLM cluster labeling (OpenAI, Anthropic)
pip install "reap-topics[baselines]"      # BERTopic + HDBSCAN baselines
pip install "reap-topics[coherence]"      # Topic coherence (gensim)
pip install "reap-topics[text-fixtures]"  # sentence-transformers for examples
pip install "reap-topics[all]"            # everything above
```

Developer setup:

```bash
git clone https://github.com/Thru-Echoes/reap-topics
cd reap-topics
pip install -e ".[dev]"
pytest tests/ -v --tb=short
```

## Minimal Working Example

Load the AI-art discourse snapshot, run REAP consensus, find the best
*K*, cluster, and print the silhouette of the resulting partition:

```python
from reap import run_consensus_pipeline, find_best_k, run_kmeans
from reap.datasets import load_ai_art
from reap.evaluation import compute_silhouette

data = load_ai_art()                          # DatasetSnapshot (cached locally)
embedding, meta = run_consensus_pipeline(
    X=data.embeddings,
    seeds=list(range(30)),                    # pre-declared 30-seed set
    n_components=18,
    n_neighbors=19,
    min_dist=0.005,
)
best = find_best_k(embedding, k_range=range(3, 20))
labels, _ = run_kmeans(embedding, k=best.best_k)
print(f"k={best.best_k}  silhouette={compute_silhouette(embedding, labels):.3f}")
```

Bring-your-own-embeddings is the standard path: pass any
`np.ndarray (N, d)` of `float32`/`float64` to `run_consensus_pipeline`.

## How It Works

1. **Multi-seed UMAP** — run UMAP `N` times with different random seeds.
2. **Distance-matrix averaging** — compute pairwise distance matrices
   for each seed and average them. This is invariant to the
   rotation/reflection ambiguity that breaks coordinate averaging.
3. **Final UMAP** — fit UMAP on the averaged distance matrix with
   `metric="precomputed"`.
4. **Clustering** — KMeans with silhouette-based *K* selection.
5. **Projection head** *(optional)* — train an MLP that maps new
   embeddings directly into consensus space, enabling out-of-sample
   inference without re-running the consensus pipeline.

## Datasets

The package ships first-class loaders that return Pydantic-validated
snapshots with content-addressed metadata (SHA256, version,
embedding-model name, license, citation):

```python
from reap.datasets import load_ai_art, load_korean_forest

ai_art = load_ai_art()
print(ai_art.embeddings.shape, ai_art.metadata.embedding_model)
```

See `examples/load_real_datasets.py` for a fuller walk-through and
`scripts/build_datasets.py` for the snapshot-build workflow.

## API Reference

### Core consensus
- `run_consensus_pipeline(X, seeds, ...)` — full pipeline: multi-seed -> consensus -> embedding
- `get_multi_seed_embeddings(X, seeds, ...)` — run UMAP with each seed
- `get_consensus_distance_matrix(embeddings)` — average pairwise distances
- `get_consensus_embedding(D, ...)` — UMAP on a precomputed consensus distance matrix

### Evaluation
- `compute_trustworthiness(X_high, X_low)` — neighborhood preservation
- `compute_silhouette(X, labels)` — cluster-separation quality
- `compute_pairwise_ari(labels_list)` — seed-to-seed agreement
- `compute_seed_stability(seed_labels, consensus_labels)` — stability statistics

### Clustering
- `find_best_k(X, k_range)` — silhouette-based *K* selection
- `run_kmeans(X, k)` — KMeans clustering

### Validation
- `validate_embeddings(X)` — checks for NaN, duplicates, normalization
- `validate_cluster_sizes(labels)` — cluster size distribution checks

### Datasets
- `load_ai_art()`, `load_korean_forest()`, `load_korean_forest_oos()`
- `DatasetSnapshot`, `DatasetMetadata` — Pydantic v2 carriers

## Links

- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Examples: [`examples/`](examples/)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Documentation: forthcoming alongside the `v1.0.0` release
- Paper: forthcoming on arXiv

## Citation

```bibtex
@article{muellerklein2026reap,
  title  = {REAP: Stabilizing Semantic Topic Spaces via Distance-Matrix
            Consensus and Neural Projection},
  author = {Muellerklein, Oliver},
  year   = {2026}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
