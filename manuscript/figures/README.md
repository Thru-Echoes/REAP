# Manuscript Figures — Source-to-Render Crosswalk

This directory contains the publication-ready PNG/PDF artifacts for the
REAP methods paper. Every rendered figure is generated from a committed
script under `manuscript/figures/src/` per the binding rule in
`.claude/rules/publication-standards.md` (Figures and Tables section)
and `.claude/rules/diagrams.md` (Always Create Both Source and Rendered
Output).

To regenerate any figure, activate the `reap` conda env, set
`KMP_DUPLICATE_LIB_OK=TRUE`, and run the script. Each script prints the
output paths on success and a one-line `logging` warning per missing
optional input.

## Crosswalk

| # | Figure file | Source script | Data source | Caption stub |
|---|---|---|---|---|
| 1 | `pipeline.png` | `src/pipeline.md` (Mermaid; mirror at `src/pipeline.mmd` for `mmdc`) | None (architecture diagram) | REAP pipeline: raw embeddings &rarr; multi-seed UMAP &rarr; per-seed pairwise distance matrices &rarr; averaged distance matrix &rarr; final UMAP (precomputed metric) &rarr; consensus embedding. Dashed branches show training of the projection head and the OOS conformal filter (calibration only). |
| 2 | `consensus_illustration.{png,pdf}` | `src/consensus_illustration.py` | Synthetic (deterministic via `np.random.seed(0)`) | Methodological illustration on three 2-D blobs: four simulated UMAP runs (top row), naive coordinate average and Procrustes average (which Procrustes cannot fully resolve when per-run distortion is non-rigid), REAP distance-matrix consensus, and a histogram contrasting across-run pairwise-distance variance vs Procrustes-aligned coordinate variance. Empirical magnitudes on real corpora are reported in Figures 3/4 against the actual benchmark CSVs. |
| 3 | `metrics_bars_ai_art.{png,pdf}` | `src/metrics_bars_ai_art.py` | `results/ai_art/combined_set_{A,B,C}/all_methods.csv` | AI-art discourse (`intfloat/e5-large-v2`, *N* = 1,736). Cross-set mean &pm; SD across three disjoint 30-seed sets per method (single seed, best-of-N, naive average, Procrustes, REAP; BERTopic when Lane A finishes). Headline consensus metrics: silhouette, trustworthiness. Topic coherence columns plotted automatically when populated in the source CSVs; otherwise a `logging.warning` is emitted and the figure renders without them. |
| 4 | `metrics_bars_korean_forest.{png,pdf}` | `src/metrics_bars_korean_forest.py` | `results/korean_forest/combined_set_{A,B,C}/all_methods.csv` | Korean forest policy (MiniLM, *N* = 905). Same plot structure as Figure 3; see manuscript Table 1b for the underlying paired-set differences. |
| 5 | `filter_retention.{png,pdf}` | `src/filter_retention.py` | `five_variant_retention.csv`, `higher_retention_exploration.csv` (sibling-project until a snapshot lands under `results/korean_forest_oos/`; cached metadata at `~/.cache/reap/datasets/korean_forest_oos/2026-05-09/metadata.json`). | OOS conformal filter on Korean forest pledges (*N* = 1,662). Panel (a): retention at &alpha; = 0.01 for the five pre-registered variants of §16.b (Euclidean &plusmn; pooled correction, Mahalanobis &plusmn; pooled correction, Mahalanobis + per-president correction); the pre-registered observed band 45-55% (§16.d) is drawn as a horizontal span. Panel (b): retention vs &alpha; for the empirical-LOO and chi-squared-theoretical thresholds (default variant). |
| 6 | `projection_head_scatter.{png,pdf}` | `src/projection_head_scatter.py` | 20newsgroups golden text fixture (auto-cached via `reap.datasets.load_golden_text`); 10 seeds from `manuscript/seeds/seed_manifest.json`. | Projection head validation on the 20ng golden fixture (*N* = 400, *d_in* = 384, *d_out* = 8). Panel (a): 3-fold CV R&sup2; distribution (boxplot + per-fold dots) with the pre-registered &ge; 0.60 floor (protocol §6.c). Panel (b): predicted vs target consensus coordinates on the first held-out CV fold; top-4 high-variance output dimensions. |

## Reproducibility

* All scripts are deterministic: `np.random.seed` and (where applicable)
  `torch.manual_seed`. Cross-platform UMAP variance is absorbed by the
  pre-registered metric ranges in `manuscript/evaluation_protocol.md`
  §6.c; the *visual* figure may differ minutely across machines but the
  numerical claims it depicts are tested by `tests/test_golden_validation.py`
  and `tests/test_projection_golden.py`.
* The metrics bar charts read the *combined* CSVs (one row per method),
  not the per-seed CSVs. Cross-set mean &pm; SD is reported because each
  (method, seed-set) yields a single consensus value — the same
  inference unit used in manuscript Tables 1b / 2b.
* `pipeline.png` is generated via `mmdc` (mermaid-cli). It is available
  on this machine via `npx -y -p @mermaid-js/mermaid-cli mmdc -i src/pipeline.mmd -o pipeline.png`; the markdown variant `src/pipeline.md` carries the
  same diagram with explanatory prose so reviewers without `mmdc` can
  still read the source.

## When source data is missing

The figure scripts degrade gracefully:

* `metrics_bars_*.py` skip metrics whose CSV column is absent or empty
  (e.g., topic-coherence columns while Lane A is in progress) with a
  one-line warning via `logging.getLogger(__name__).warning(...)`.
* `filter_retention.py` looks in `results/korean_forest_oos/` first,
  then falls back to the sibling-project directory. Panels (a) and (b)
  render independently; missing one does not abort the other.
* `projection_head_scatter.py` raises with a clear "torch missing" /
  "fixture missing" message if its dependencies are unavailable, so the
  failure surfaces loudly rather than silently rendering nothing.

## File locations

```
manuscript/figures/
    pipeline.png
    consensus_illustration.{png,pdf}
    metrics_bars_ai_art.{png,pdf}
    metrics_bars_korean_forest.{png,pdf}
    filter_retention.{png,pdf}
    projection_head_scatter.{png,pdf}
    src/
        pipeline.md           (mermaid source + caption)
        pipeline.mmd          (mermaid CLI input)
        consensus_illustration.py
        metrics_bars_ai_art.py
        metrics_bars_korean_forest.py
        _bench_plot.py        (shared helper for the two bar charts)
        filter_retention.py
        projection_head_scatter.py
```
