# Method

<!-- TODO: Draft method section -->
<!-- Key subsections:
  3.1 Distance-matrix consensus
    - Algorithm pseudocode (43 lines)
    - Why distances are rotation/reflection invariant
    - Why coordinate averaging fails (with example)
    
  3.2 Mathematical justification
    - UMAP embedding non-identifiability (rotation/reflection invariance of loss)
    - Coordinate averaging failure mode
    - Distance-matrix invariance proof
    - Triangle inequality preservation (convex combination of metrics is a metric)
    - Connection to consensus literature (Fred & Jain 2005)
    
  3.3 Parameter search framework
    - 4-round progressive grid search
    - 8 selection criteria including Pareto front
    - Parameters are dataset-specific (do NOT transfer)
    
  3.4 Projection head
    - Architecture: MLP with BatchNorm + GELU + Dropout
    - Loss: alpha*MSE + (1-alpha)*(1-distance_correlation)
    - 5-fold stratified cross-validation
    - When to use projection head vs full consensus
    
  3.5 Cluster labeling
    - c-TF-IDF (statistical)
    - LLM-based (exemplar sampling)
    - Combined approach
    - Point stratification (core vs peripheral)
-->
