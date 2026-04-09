# REAP Methods Paper — Outline

**REAP** = **R**eproducible **E**mbedding via **A**veraged **P**rojection

**Working Title:** REAP: Stabilizing Semantic Topic Spaces via Distance-Matrix Consensus and Neural Projection

**Status:** Outline complete — implementation and validation in progress.

---

## Paper Structure

### 1. Introduction (2-3 pages)
- The UMAP instability problem: stochastic outputs undermine downstream clustering
- Why it matters: reproducibility in topic modeling, semantic analysis, cross-dataset comparison
- REAP overview: distance-matrix consensus + projection head = stable, projectable semantic spaces
- Contributions: (1) distance-matrix consensus algorithm, (2) neural projection head, (3) multi-dataset validation, (4) open-source package

### 2. Related Work (2-3 pages)
- UMAP (McInnes et al., 2018) — REAP wraps UMAP, stabilizing its output
- t-SNE stabilization (Kobak & Linderman, 2021; Belkina et al., 2019)
- Ensemble dimensionality reduction — REAP contributes distance averaging
- BERTopic (Grootendorst, 2022) — uses single-seed UMAP; REAP solves its instability
- Procrustes analysis (Gower, 1975) — standard alignment; insufficient (ARI = 0.56)
- Wisdom of crowds / Condorcet jury theorem — theoretical grounding
- Knowledge distillation (Hinton et al., 2015) — projection head as distilled procedure

### 3. Method (5-7 pages)
- 3.1 Distance-matrix consensus — the core algorithm, rotation/reflection invariance
- 3.2 Mathematical justification — formal proof that distance averaging preserves relational structure
- 3.3 Parameter search framework — progressive grid search
- 3.4 Projection head — architecture, combined MSE + distance correlation loss, CV
- 3.5 Cluster labeling — c-TF-IDF + LLM combined pipeline

### 4. Experiments (5-7 pages)
- 4.1 Baselines — single-seed, Procrustes, coordinate-avg, majority-vote
- 4.2 Multi-dataset validation:
  - AI-art discourse (primary, 1742 chunks, English, e5-large-v2)
  - Korean forest policy (secondary, 905 chunks, Korean, MiniLM)
  - Corporate sustainability (secondary, 1012 reports)
  - US presidential language (secondary, infrastructure ready)
- 4.3 Sensitivity analysis — seed count, parameters, encoder choice
- 4.4 Projection head — CV vs final metrics, linear baseline ablation
- 4.5 Computational cost — N² memory, runtime scaling

### 5. Discussion (2-3 pages)
- When to use REAP vs single-seed
- Consensus K vs seed K: emergent coarse structure
- Limitations: N² memory, parameter non-transferability
- Connection to ensemble methods literature

### 6. Conclusion (1 page)
- Summary of contributions
- Open-source availability: `pip install reap-embeddings`

---

## Key Results to Report

| Metric | Single-seed | Procrustes | Coord-avg | REAP |
|--------|:-----------:|:----------:|:---------:|:----:|
| ARI    | ~0.56       | ~0.56      | ~0.56     | **0.75** |

## Datasets

| Dataset | n | d_input | d_consensus | K | Status |
|---------|---|---------|-------------|---|--------|
| AI-art discourse | 1,742 | 1024 | 5 | 20 | Complete |
| Korean forest policy | 905 | 384 | 18 | 8 | Complete |
| Corporate sustainability | 1,012 | TBD | TBD | TBD | Extraction in progress |
| US presidential | ~1,000 | TBD | TBD | TBD | Infrastructure ready |

## Implementation Priority

1. P0-1: Clean consensus API in `src/reap/consensus.py` — **DONE**
2. P0-2: Dead code removal — **DONE** (fresh codebase)
3. P0-3: Baseline comparison script — **TODO**
4. P0-4: Mathematical justification document — **TODO**
5. P0-5: Metric consistency (euclidean silhouette) — **DONE** (euclidean default)
6. P0-6: Package structure — **DONE**

## Cross-Citation

- **Cites Hye-In:** Domain expert validation of Korean 8-cluster taxonomy
- **Cited by Hye-In:** "We used REAP (Author, 2026) to construct a stable 8-topic semantic space..."
- **Cites AI-art paper:** Primary validation dataset
