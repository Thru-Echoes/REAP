# REAP Methods Paper — Outline

**REAP** = **R**eproducible **E**mbedding via **A**veraged **P**rojection

**Working Title:** REAP: Stabilizing Semantic Topic Spaces via Distance-Matrix Consensus and Neural Projection

**Status:** Outline complete — implementation and validation in progress.

---

## Paper Structure

### 1. Introduction (2-3 pages)
- The UMAP instability problem: stochastic outputs undermine downstream clustering
- Why it matters: reproducibility in topic modeling, semantic analysis, cross-dataset comparison
- REAP overview: distance-matrix consensus + projection head + conformal filter = stable, projectable, gated semantic spaces
- Contributions: (1) distance-matrix consensus algorithm, (2) neural projection head, (3) **OOS conformal filter** (NEW), (4) multi-dataset validation, (5) open-source package

### 2. Related Work (2-3 pages)
- UMAP (McInnes et al., 2018) — REAP wraps UMAP, stabilizing its output
- t-SNE stabilization (Kobak & Linderman, 2021; Belkina et al., 2019)
- Ensemble dimensionality reduction — REAP contributes distance averaging
- BERTopic (Grootendorst, 2022) — uses single-seed UMAP; REAP solves its instability
- Procrustes analysis (Gower, 1975) — standard alignment; insufficient (ARI = 0.56)
- Wisdom of crowds / Condorcet jury theorem — theoretical grounding
- Knowledge distillation (Hinton et al., 2015) — projection head as distilled procedure
- **Conformal prediction (Vovk et al. 2005; Shafer & Vovk 2008; Angelopoulos & Bates 2021)** — theoretical grounding for §3.6 OOS filter

### 3. Method (5-7 pages)
- 3.1 Distance-matrix consensus — the core algorithm, rotation/reflection invariance
- 3.2 Mathematical justification — formal proof that distance averaging preserves relational structure
- 3.3 Parameter search framework — progressive grid search
- 3.4 Projection head — architecture, combined MSE + distance correlation loss, CV
- 3.5 Cluster labeling — c-TF-IDF + LLM combined pipeline
- **3.6 Out-of-sample conformal filter — Mahalanobis + pooled location correction + empirical-LOO threshold (NEW)**

### 4. Experiments (5-7 pages)
- 4.1 Baselines — single-seed, Procrustes, coordinate-avg, majority-vote
- 4.2 Multi-dataset validation:
  - AI-art discourse (primary, 1736 chunks, English, e5-large-v2)
  - Korean forest policy (secondary, 905 chunks, Korean, MiniLM)
  - Corporate sustainability (secondary, 1012 reports)
  - US presidential language (secondary, infrastructure ready)
- 4.3 Sensitivity analysis — seed count, parameters, encoder choice
- 4.4 Projection head — CV vs final metrics, linear baseline ablation
- 4.5 Computational cost — N² memory, runtime scaling
- 4.6 Statistical procedures
- **4.7 OOS conformal filter validation — Korean forest case study, 5-variant retention, per-cluster, qualitative validation, sensitivity (NEW)**

### 5. Discussion (2-3 pages)
- When to use REAP vs single-seed
- Consensus K vs seed K: emergent coarse structure
- Limitations: N² memory, parameter non-transferability, projection-head interpolator behavior, **filter retention varies by cluster, empirical-LOO ceiling for small clusters (NEW)**
- **Future Work: shrinkage covariance with principled β, ensemble filtering, per-cluster α from coherence, projection-head-confidence hybrid, multi-corpus filter validation (NEW)**
- Connection to ensemble methods literature

### 6. Conclusion (1 page)
- Summary of contributions
- Open-source availability: `pip install reap-topics`

---

## Key Results to Report

| Metric | Single-seed | Procrustes | Coord-avg | REAP |
|--------|:-----------:|:----------:|:---------:|:----:|
| ARI    | ~0.56       | ~0.56      | ~0.56     | **0.75** |

## Datasets

| Dataset | n | d_input | d_consensus | K | Status |
|---------|---|---------|-------------|---|--------|
| AI-art discourse | 1,736 | 1024 | 5 | 20 | Complete |
| Korean forest policy | 905 | 384 | 18 | 8 | Complete |
| Corporate sustainability | 1,012 | TBD | TBD | TBD | Extraction in progress |
| US presidential | ~1,000 | TBD | TBD | TBD | Infrastructure ready |

## Implementation Priority

1. P0-1: Clean consensus API in `src/reap/consensus.py` — **DONE**
2. P0-2: Dead code removal — **DONE** (fresh codebase)
3. P0-3: Baseline comparison script — **DONE** (`scripts/run_paper_benchmark.py`, protocol §7 layout)
4. P0-4: Mathematical justification document — **DONE** (Method §3.2 drafted)
5. P0-5: Metric consistency (euclidean silhouette) — **DONE** (euclidean default)
6. P0-6: Package structure — **DONE**
7. P0-7: Korean forest 30-seed × 3-set benchmark — **DONE** (results/ committed)
8. P0-8: AI-art 30-seed × 3-set benchmark — **RUNNING**
9. P0-9: Full paper draft (Introduction, Method, Experiments, Discussion, Conclusion) — **DONE** (first draft; AI-art numbers pending)
10. P0-10: OOS conformal filter design — **DONE** (validated on Korean forest, manuscript §3.6 / §4.7 drafted)
11. P0-11: OOS conformal filter package code — **TODO** (port `reap.filter` from sibling-project script; add unit + golden tests)
12. P0-12: `load_korean_forest_oos()` loader — **TODO** (capture 1,662 OOS pledges as a locked snapshot under `~/.cache/reap/datasets/`)
13. P0-13: Filter golden tests — **TODO** (Tier-1 invariants: per-cluster threshold finite, retention monotone in α; Tier-2: 20ng-style synthetic-shift pre-registered ranges)
14. P1-1: Topic coherence metrics (UMass, NPMI, c_v) — **TODO** (pre-registered §7)
15. P1-2: BERTopic baseline — **TODO** (pre-registered §3)
16. P1-3: Seed ablation + K-robustness + encoder sensitivity — **TODO** (protocol §4.3)
17. P2-1: Shrinkage-covariance filter variant with principled β — **FUTURE WORK** (Discussion §5)
18. P2-2: Ensemble filtering (Mahalanobis ⊕ silhouette ⊕ projection-head confidence) — **FUTURE WORK** (Discussion §5)
19. P2-3: Per-cluster α from objective criteria (coherence-driven) — **FUTURE WORK** (Discussion §5)
20. P2-4: Filter validation on AI-art + corporate-sustainability — **FUTURE WORK** (Discussion §5)

## Cross-Citation

- **Cites Hye-In:** Domain expert validation of Korean 8-cluster taxonomy
- **Cited by Hye-In:** "We used REAP (Author, 2026) to construct a stable 8-topic semantic space..."
- **Cites AI-art paper:** Primary validation dataset
