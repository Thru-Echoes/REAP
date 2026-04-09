# Experiments

<!-- TODO: Draft experiments section -->
<!-- Key subsections:
  4.1 Baselines
    - Single-seed UMAP (seed=42)
    - Procrustes consensus (align + average coordinates)
    - Coordinate averaging (no alignment)
    - Majority-vote clustering
    - Linear projection head baseline
    
  4.2 Multi-dataset validation
    AI-art discourse (primary):
      - n=1742, d_in=1024, d_out=5, K=20
      - TW=0.916, Sil=0.657, ARI=0.668
      - Projection R²=0.904
      
    Korean forest policy (secondary):
      - n=905, d_in=384, d_out=18, K=8
      - TW=0.892, Sil=0.841, ARI=0.758
      - Key finding: consensus K=8 vs seed K mode=29
      
    Corporate sustainability (secondary):
      - n=1012 reports, extraction in progress
      
    US presidential (secondary):
      - Infrastructure ready, data collection pending
      
  4.3 Sensitivity analysis
    - Seed count: how many seeds needed for convergence?
    - Parameter sensitivity: how robust to nn/md/d choices?
    - Encoder choice: does the benefit hold across embedding models?
    
  4.4 Projection head evaluation
    - CV metrics (honest) vs final metrics (optimistic)
    - Korean: CV ARI=0.445 vs Final ARI=0.899
    - AI-art: R²=0.904, TW=0.916
    - Linear baseline ablation
    
  4.5 Computational cost
    - O(N² × n_seeds) for distance matrices
    - Runtime benchmarks at N=500, 1000, 2000, 5000
-->
