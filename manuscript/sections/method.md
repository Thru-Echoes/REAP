# Method

## 3.1 Distance-Matrix Consensus

UMAP (McInnes et al., 2018) is a stochastic algorithm: different random seeds produce embeddings that differ by rotation, reflection, and local distortion. These solutions are geometrically equivalent — they minimize the same cross-entropy objective over fuzzy simplicial sets — but their coordinate representations are not comparable across runs. Downstream clustering inherits this instability: the same data, encoded by the same model, yields different topic assignments depending on the seed.

REAP addresses this by observing that while *coordinates* are seed-dependent, *pairwise distances* in the embedding space are not. Two UMAP runs may place a pair of points at different absolute positions, but the distance between them is determined by the local geometry of the data manifold, not by the global orientation of the embedding. Averaging pairwise distance matrices across seeds therefore combines information from multiple runs without requiring alignment.

**Algorithm.** Given $N$ samples in a $d_{\mathrm{in}}$-dimensional embedding space:

1. Run UMAP $S$ times with distinct random seeds $\{s_1, \ldots, s_S\}$, each producing an embedding $\mathbf{E}^{(s)} \in \mathbb{R}^{N \times d}$ at target dimensionality $d$.
2. For each seed, compute the Euclidean pairwise distance matrix $\mathbf{D}^{(s)}_{ij} = \|\mathbf{e}^{(s)}_i - \mathbf{e}^{(s)}_j\|_2$.
3. Average: $\bar{\mathbf{D}} = \frac{1}{S} \sum_{s=1}^{S} \mathbf{D}^{(s)}$.
4. Project: run UMAP on $\bar{\mathbf{D}}$ with `metric="precomputed"` to obtain the consensus embedding $\mathbf{Y} \in \mathbb{R}^{N \times d}$.

The result is a single, stable embedding that encodes the relational structure agreed upon across seeds.

**Why not coordinate averaging?** Coordinate averaging — even after Procrustes alignment — suffers from two failure modes. First, Procrustes finds the best rigid transform (rotation + reflection + uniform scaling) between a pair of embeddings, but UMAP's distortions are *non-rigid*: local neighborhoods stretch and compress differently across seeds. Aligning globally cannot correct local discrepancies. Second, when clusters have genuine semantic overlap (e.g., two political-discourse topics that share vocabulary and embedding-space neighbors), different seeds may resolve the overlap differently — sometimes merging the topics, sometimes separating them. Procrustes alignment followed by coordinate averaging blurs these locally divergent solutions into an uninformative mean, destroying exactly the fine-grained structure that topic modeling depends on.

Distance-matrix averaging sidesteps both problems. Each seed's distance matrix captures the relational geometry of *that* seed's solution. Averaging over seeds preserves relationships that are consistent across runs (signal) while damping relationships that are seed-specific (noise). The resulting consensus distance matrix $\bar{\mathbf{D}}$ is then re-embedded via UMAP in a single final projection, producing a coherent low-dimensional representation.

## 3.2 Mathematical Justification

**Rotation/reflection invariance.** Let $\mathbf{Q} \in \mathbb{R}^{d \times d}$ be an orthogonal matrix ($\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$). If embedding $\mathbf{E}'$ is a rotated/reflected copy of $\mathbf{E}$, i.e., $\mathbf{E}' = \mathbf{E}\mathbf{Q}$, then

$$\|\mathbf{e}'_i - \mathbf{e}'_j\|_2 = \|\mathbf{Q}^\top(\mathbf{e}_i - \mathbf{e}_j)\|_2 = \|\mathbf{e}_i - \mathbf{e}_j\|_2$$

so $\mathbf{D}^{(\mathbf{E}')} = \mathbf{D}^{(\mathbf{E})}$. Distance matrices are exactly invariant to the orthogonal ambiguity of UMAP embeddings. Coordinate-based methods (Procrustes averaging, naive averaging) must *estimate and remove* this ambiguity; distance-matrix consensus is invariant to it by construction.

**Metric-space preservation.** The element-wise average of Euclidean distance matrices is itself a valid metric. Symmetry and non-negativity are trivially preserved. For the triangle inequality: if each $\mathbf{D}^{(s)}$ satisfies $D^{(s)}_{ij} \le D^{(s)}_{ik} + D^{(s)}_{kj}$ for all $i, j, k$, then

$$\bar{D}_{ij} = \frac{1}{S}\sum_s D^{(s)}_{ij} \le \frac{1}{S}\sum_s \bigl(D^{(s)}_{ik} + D^{(s)}_{kj}\bigr) = \bar{D}_{ik} + \bar{D}_{kj}.$$

This guarantees that the consensus distance matrix is a proper metric and can be passed to UMAP's precomputed-metric pathway without introducing geometric pathologies. The convex-combination argument generalizes beyond the mean: any non-negative weighted average of metrics with weights summing to 1 produces a metric.

**Connection to ensemble methods.** REAP can be viewed as a metric-space ensemble method. Where Fred & Jain (2005) accumulate co-association matrices (binary: "were $i$ and $j$ in the same cluster?"), REAP accumulates continuous distance matrices (real-valued: "how far apart were $i$ and $j$?"). The co-association approach discards magnitude information — it treats a pair that is barely separated the same as a pair at opposite ends of the embedding — while distance averaging retains the full metric structure. This richer consensus signal is what enables REAP to re-embed via UMAP rather than requiring a separate clustering-consensus step.

## 3.3 Parameter Search Framework

UMAP's behavior is sensitive to three parameters: the number of components $d$, the number of neighbors $k_{\mathrm{nn}}$, and the minimum distance $d_{\mathrm{min}}$. REAP treats these as dataset-specific — parameters tuned on one corpus do not transfer to another — and selects them via a progressive grid search.

The search operates in four rounds of increasing resolution, each narrowing the search region around the best configuration from the previous round. At each round, every candidate configuration is evaluated by running the full REAP consensus pipeline (30 seeds) and computing a composite score: $0.5 \times \text{silhouette} + 0.5 \times \text{trustworthiness}$. This composite balances cluster separation (silhouette) against local-structure preservation (trustworthiness). The K-selection rule scans $K \in [5, 30]$ and picks the $K$ maximizing the composite on the consensus embedding.

Eight selection criteria (including Pareto optimality) are reported to ensure the chosen configuration is robust and not an artifact of the composite weighting. Final parameters are frozen before downstream evaluation begins: the benchmark uses only the winning configuration, not a post-hoc selection across configs.

**Validated configurations.** The Korean forest policy corpus was best served by $d=18$, $k_{\mathrm{nn}}=19$, $d_{\mathrm{min}}=0.005$ (consensus $K=8$). The AI-art discourse corpus used $d=5$, $k_{\mathrm{nn}}=53$, $d_{\mathrm{min}}=0.01$ (consensus $K=20$). These configurations were frozen before the benchmark runs reported in Section 4.

## 3.4 Projection Head

Once a consensus embedding is established, new documents can be projected into the same space without re-running the multi-seed UMAP procedure. REAP trains a lightweight MLP to learn the mapping $f: \mathbb{R}^{d_{\mathrm{in}}} \to \mathbb{R}^d$ from high-dimensional sentence embeddings to consensus coordinates.

**Architecture.** The default projection head for a 384-dimensional input (MiniLM) and an 18-dimensional consensus target is:

$$\mathrm{Linear}(384, 128) \to \mathrm{BN} \to \mathrm{GELU} \to \mathrm{Dropout}(0.3) \to \mathrm{Linear}(128, 64) \to \mathrm{BN} \to \mathrm{GELU} \to \mathrm{Dropout}(0.3) \to \mathrm{Linear}(64, 18)$$

where BN denotes batch normalization. This produces approximately 59K trainable parameters — small enough to train in seconds on CPU and cheap enough to deploy alongside the embedding model.

**Loss function.** The projection head minimizes a combined loss:

$$\mathcal{L} = \alpha \cdot \mathrm{MSE}(\hat{\mathbf{Y}}, \mathbf{Y}) + (1 - \alpha) \cdot \bigl(1 - \mathrm{dCorr}(\hat{\mathbf{Y}}, \mathbf{Y})\bigr)$$

where $\mathrm{MSE}$ is the mean squared error between predicted and target coordinates, $\mathrm{dCorr}$ is the Pearson correlation of flattened pairwise-distance vectors, and $\alpha = 0.7$ by default. The MSE term encourages pointwise accuracy; the distance-correlation term preserves the relational geometry of the consensus space, which is what downstream clustering depends on. In practice, training converges in 100–200 epochs with early stopping (patience 50) on a held-out fold.

**Cross-validation.** The projection head is evaluated under stratified $k$-fold CV to separate training-fit from generalization. Reported metrics include CV R², trustworthiness, ARI against true labels, and distance correlation. Final-fit metrics (trained on all data, evaluated on all data) are reported separately and explicitly labeled as optimistic. This distinction is enforced in the evaluation protocol (§6.e) and in the golden-validation test suite.

**Semantic interpolation, not novelty detection.** Topic-exclusion experiments on the golden fixture reveal that the projection head maps held-out topics toward their nearest semantic neighbor in the training set. For overlap-pair topics (e.g., `talk.politics.guns` held out, `talk.politics.mideast` retained), 60–70% of held-out documents land in the pair-partner cluster. For clearly separated topics, the effect is weaker (40–50%) but still non-random (random baseline 14%). The projection head is therefore best understood as a semantic interpolator: it generalizes meaningfully to related content but does not create novel cluster structure for content that is orthogonal to the training distribution. Users should be aware that projecting genuinely novel topics will route them to the nearest training cluster, not to a new one.

## 3.5 Cluster Labeling

REAP provides a two-stage labeling pipeline that does not depend on any specific topic-modeling framework.

**Stage 1: c-TF-IDF.** For each cluster, class-based term frequency–inverse document frequency (c-TF-IDF) ranks terms by their discriminative power within the cluster relative to the corpus. The top $n$ terms (default 10) serve as statistical candidate labels. This stage requires no API calls and runs in milliseconds.

**Stage 2: LLM refinement.** The candidate terms, together with a sample of representative texts from the cluster, are passed to a large language model (default: `gpt-5.4-mini` for OpenAI, `claude-opus-4-6` for Anthropic) to synthesize a coherent, human-readable cluster label. The LLM is prompted to return a short label, a one-sentence description, and a confidence score. Running both providers and comparing labels serves as a cross-provider consistency check.

**Point stratification.** Before labeling, points can be stratified into *core* (silhouette score $\ge 0$) and *peripheral* (silhouette $< 0$). Labeling on core points only produces more interpretable labels by excluding boundary cases that straddle multiple clusters. Stratification summaries (core fraction, mean silhouette per cluster) are logged alongside the labels.

## 3.6 Out-of-Sample Conformal Filtering

The projection head (§3.4) places every out-of-sample point somewhere in the consensus space — by design, it has no opt-out. Topic-exclusion experiments showed that genuinely novel content is routed to the nearest training cluster rather than rejected; the head is a semantic interpolator, not a novelty detector. Many downstream analyses, however, require deciding *which* projected placements should be trusted — for example, when projecting a domain-shifted corpus (presidential pledges) into a topic space trained on a stylistically different reference (national forest-policy planning documents), and asking which pledges are genuinely typical of their assigned cluster.

REAP therefore ships an out-of-sample conformal filter that operates on the consensus-space coordinates produced by the projection head. The design follows the split-conformal framework of Vovk et al. [Shafer & Vovk 2008] and Angelopoulos & Bates [2021]: the reference corpus serves as the calibration set; a per-cluster non-conformity score is computed; a per-cluster threshold is calibrated at level $\alpha$; new points are accepted iff their non-conformity score falls below the threshold.

**Distance metric.** For a point $\mathbf{y} \in \mathbb{R}^d$ assigned to cluster $c$ with covariance $\boldsymbol{\Sigma}_c$ estimated from the reference points in cluster $c$, the non-conformity score is the squared Mahalanobis distance

$$M_c(\mathbf{y}) = (\mathbf{y} - \hat{\boldsymbol{\mu}}_c^{\mathrm{OOS}})^\top \boldsymbol{\Sigma}_c^{-1} (\mathbf{y} - \hat{\boldsymbol{\mu}}_c^{\mathrm{OOS}}),$$

where $\hat{\boldsymbol{\mu}}_c^{\mathrm{OOS}}$ is a *per-cluster OOS centroid* (defined below). Mahalanobis is preferred over plain Euclidean because consensus clusters are anisotropic in non-trivial ways: some are tight along several axes and broad along one (specialised topics with long tails of related but distinct content), others are uniformly broad (umbrella topics covering several sub-themes). Section 4.7 reports a head-to-head comparison showing the two metrics disagree on 27.5% of points on the Korean forest validation set, and per-cluster patterns confirm Mahalanobis is doing real work — it tightens for clusters where Euclidean retains too much (e.g., umbrella clusters) and loosens for tight clusters where the OOS distribution is shape-mismatched.

**Location correction.** The reference corpus and the OOS corpus may live in systematically different regions of the consensus space — a *style/register shift* in addition to topic content. On the Korean forest data, the reference is national-scale strategic-planning prose; the OOS is district-scale political pledges. Even when topic alignment is good (e.g., urban-forestry references and urban-park pledges discuss the same domain), the embedding-space centroid of OOS points within a cluster is offset from the reference centroid for that cluster. Without correction, the filter would conflate this register shift with topic mismatch and reject too aggressively.

REAP's filter therefore decouples *shape* (anchored in the reference geometry, $\boldsymbol{\Sigma}_c$) from *location* (re-estimated from the OOS sample). The corrected centroid is

$$\hat{\boldsymbol{\mu}}_c^{\mathrm{OOS}} = \frac{1}{|\mathcal{O}_c|} \sum_{j \in \mathcal{O}_c} \mathbf{y}_j$$

where $\mathcal{O}_c = \{j : \mathrm{label}(\mathbf{y}_j) = c\}$ is the set of OOS points assigned to cluster $c$. We refer to this as the *pooled* variant: it pools the OOS-centroid estimate across any subgroup structure within the OOS corpus (e.g., across the three presidents in the Korean forest case). Pooling is preferred over per-subgroup correction: a per-subgroup centroid (separate $\hat{\boldsymbol{\mu}}_{c,p}^{\mathrm{OOS}}$ for each subgroup $p$) requires a hard-fallback rule when $|\mathcal{O}_{c,p}|$ is small, adds complexity, and barely changes the filter (5.2% sentence-level disagreement in §4.7). The reference covariance $\boldsymbol{\Sigma}_c$ is unchanged — shape information remains anchored in the reference geometry — so the score $M_c$ asks "is $\mathbf{y}$ shape-typical of cluster $c$, given that cluster $c$ in OOS is centered at $\hat{\boldsymbol{\mu}}_c^{\mathrm{OOS}}$?"

**Threshold calibration.** The threshold $\tau_c$ is calibrated empirically via leave-one-out (LOO) on the reference points in cluster $c$: for each reference point $\mathbf{x}_i^{(c)}$ we compute a Mahalanobis score against the *reference* centroid and covariance estimated *without* $\mathbf{x}_i$,

$$M_c^{\mathrm{LOO}}(\mathbf{x}_i^{(c)}) = (\mathbf{x}_i^{(c)} - \boldsymbol{\mu}_{c, -i})^\top \boldsymbol{\Sigma}_{c, -i}^{-1} (\mathbf{x}_i^{(c)} - \boldsymbol{\mu}_{c, -i}),$$

yielding a per-cluster reference distribution $\{M_c^{\mathrm{LOO}}(\mathbf{x}_i^{(c)}) : i = 1, \dots, n_c\}$. The threshold is the $(1-\alpha)$ quantile of this distribution. We use $\alpha = 0.01$ (99th percentile) by default, which yields the headline retention reported in §4.7.

We use empirical LOO rather than the chi-squared theoretical $\chi^2_{d, 1-\alpha}$ for two reasons. First, the chi-squared threshold assumes the reference centroid and covariance are *known exactly*; finite-sample estimation inflates the LOO distribution above the chi-squared null, by amounts that depend on $n_c$ and the cluster's intrinsic shape. Empirically calibrated thresholds therefore correctly reflect the calibration-set sample size, while chi-squared thresholds are systematically too tight (cf. §4.7 ablation: at $\alpha = 0.01$ on the Korean forest data, chi-squared retains 19.1% while empirical LOO retains 50.9%). Second, the empirical-LOO threshold gives a heuristic coverage statement: under the null hypothesis "OOS in cluster $c$ is drawn from the reference shape (covariance $\boldsymbol{\Sigma}_c$) centered at $\hat{\boldsymbol{\mu}}_c^{\mathrm{OOS}}$ — exchangeable with reference points after location centering", the probability that a fresh OOS point exceeds $\tau_c$ is approximately $\alpha$. The approximation arises because reference LOO scores condition on $n_c - 1$ centroid samples while OOS scores condition on $|\mathcal{O}_c|$ samples; the discrepancy is small when both counts are comparable.

**Boundaries of the guarantee.** The heuristic coverage statement above applies to *in-distribution-modulo-shift* OOS points — i.e., points drawn from a Gaussian with the same covariance as reference cluster $c$ but possibly different mean. For genuinely off-shape OOS points (different covariance), the filter does not provide a coverage statement; the empirical retention rate on such points is what one measures, not what one bounds. This is the right behavior for the use case: the filter exists precisely to catch off-topic placements that the projection head cannot reject. A formally rigorous variant uses leave-one-out within the OOS sample as well (see §5 future work).

**Parameter choices and rejected alternatives.** Five filter variants were validated head-to-head on the Korean forest corpus (§4.7). Lower $\alpha$ (0.005, 0.001) and shrinkage covariance ($\boldsymbol{\Sigma}_{\mathrm{eff}} = \beta \boldsymbol{\Sigma}_{\mathrm{ref}} + (1-\beta) \boldsymbol{\Sigma}_{\mathrm{OOS}}$, $\beta = 0.75$) yield higher retention but were rejected as the default because $\alpha$ tuning lacks a principled justification and shrinkage covariance lacks a principled $\beta$ selection rule. Per-president location correction (separate $\bar{\mathbf{y}}_{\mathrm{OOS}}^{(p)}$ for each subgroup $p$) was rejected because it adds complexity without changing the filter materially (5.2% sentence-level disagreement against pooled). The Discussion (§5) and supplementary `manuscript/supplementary/oos_filter_design_decisions.md` enumerate these rejected alternatives, the empirical retention they produce, and why each was deferred to future work rather than adopted as a default.
