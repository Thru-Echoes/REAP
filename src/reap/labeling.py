"""Cluster labeling via c-TF-IDF + LLM refinement + human validation.

Three-stage labeling pipeline (core REAP methodology):
1. c-TF-IDF: statistical candidate term extraction (no API needed)
2. LLM refinement: synthesize coherent labels from candidate terms (requires
   openai or anthropic — pip install reap-topics[labeling])
3. Human validation: domain expert reviews and finalizes labels (external)

Point stratification (core vs peripheral) can be applied before labeling
to focus on the most representative points.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_samples

from reap._types import (
    CTfidfClusterResult,
    CTfidfTerm,
    LLMClusterLabel,
    PointStratification,
    StratificationSummary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point stratification
# ---------------------------------------------------------------------------


def stratify_points(
    X: np.ndarray,
    labels: np.ndarray,
    silhouette_threshold: float = 0.0,
) -> tuple[list[PointStratification], StratificationSummary]:
    """Split points into core (high silhouette) and peripheral (low silhouette).

    Parameters
    ----------
    X : (n_samples, d) embedding coordinates.
    labels : (n_samples,) cluster assignments.
    silhouette_threshold : Points with silhouette >= threshold are core.

    Returns
    -------
    (point_list, summary) with per-point classification and aggregate stats.
    """
    sil_scores = silhouette_samples(X, labels, metric="euclidean")

    # Centroid distances
    centroids: dict[int, np.ndarray] = {}
    for label in set(labels.tolist()):
        mask = labels == label
        centroids[label] = X[mask].mean(axis=0)

    points: list[PointStratification] = []
    for i in range(len(X)):
        label = int(labels[i])
        dist = float(np.linalg.norm(X[i] - centroids[label]))
        stratum = "core" if sil_scores[i] >= silhouette_threshold else "peripheral"
        points.append(PointStratification(
            point_index=i,
            cluster=label,
            silhouette_score=float(sil_scores[i]),
            centroid_distance=dist,
            stratum=stratum,
        ))

    core_mask = np.array([p.stratum == "core" for p in points])
    core_labels = labels[core_mask]
    core_sizes = dict(Counter(core_labels.tolist()))

    summary = StratificationSummary(
        silhouette_threshold=silhouette_threshold,
        total_points=len(points),
        core_points=int(core_mask.sum()),
        peripheral_points=int((~core_mask).sum()),
        core_silhouette_mean=float(sil_scores[core_mask].mean()) if core_mask.any() else 0.0,  # type: ignore[reportArgumentType]
        all_silhouette_mean=float(sil_scores.mean()),  # type: ignore[reportAttributeAccessIssue]
        core_cluster_sizes=core_sizes,
        smallest_core_cluster=min(core_sizes.values()) if core_sizes else 0,
    )
    return points, summary


# ---------------------------------------------------------------------------
# Method A: c-TF-IDF labeling
# ---------------------------------------------------------------------------


def build_ctfidf_matrix(
    texts: list[str],
    labels: np.ndarray,
    ngram_range: tuple[int, int] = (1, 3),
    min_df: int = 2,
    max_df: float = 0.95,
    stop_words: str | list[str] | None = "english",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build a c-TF-IDF matrix for cluster labeling.

    c-TF-IDF (class-based TF-IDF) treats each cluster as a single document,
    then weights terms by their frequency within the cluster relative to
    their frequency across all clusters.

    Parameters
    ----------
    texts : Raw text for each data point.
    labels : (n_samples,) cluster assignments.
    ngram_range : N-gram sizes to extract.
    min_df : Minimum document frequency for terms.
    max_df : Maximum document frequency fraction.
    stop_words : Stop word list or "english" for built-in.

    Returns
    -------
    (ctfidf_matrix, feature_names, cluster_ids) where ctfidf_matrix is
    (n_clusters, n_terms).
    """
    unique_labels = sorted(set(labels.tolist()))

    # Concatenate texts per cluster
    cluster_docs: list[str] = []
    for label in unique_labels:
        mask = labels == label
        cluster_docs.append(" ".join(t for t, m in zip(texts, mask) if m))

    # Adapt min_df/max_df for small cluster counts. Each c-TF-IDF "document"
    # is a concatenation of all texts in a cluster, so min_df=1 is valid.
    # With few clusters (< 2*min_df), requiring min_df=2 filters out every
    # discriminative term unique to a single cluster.
    n_docs = len(unique_labels)
    effective_min_df = 1 if n_docs < 2 * min_df else min_df
    effective_max_df = max_df if n_docs > 2 else 1.0

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=effective_min_df,
        max_df=effective_max_df,
        stop_words=stop_words,
    )
    raw_matrix = vectorizer.fit_transform(cluster_docs)
    tf_matrix = np.asarray(raw_matrix.toarray(), dtype=np.float64)  # type: ignore[reportAttributeAccessIssue]
    feature_names = vectorizer.get_feature_names_out().tolist()

    # L1-normalize TF per cluster
    row_sums = tf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    tf_norm = tf_matrix / row_sums

    # c-TF-IDF IDF: log(1 + n_clusters / df) where df = number of clusters
    # containing the term. This is CLASS-based IDF (clusters as documents),
    # not standard document-frequency IDF. Terms unique to one cluster get
    # the highest IDF; terms appearing in all clusters get the lowest.
    n_clusters = len(unique_labels)
    df = (tf_matrix > 0).sum(axis=0).astype(np.float64)
    df[df == 0] = 1.0
    idf = np.log1p(n_clusters / df)

    ctfidf = tf_norm * idf
    return ctfidf, feature_names, np.array(unique_labels)


def label_clusters_ctfidf(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 10,
    ngram_range: tuple[int, int] = (1, 3),
    min_df: int = 2,
    max_df: float = 0.95,
    stop_words: str | list[str] | None = "english",
) -> list[CTfidfClusterResult]:
    """Label clusters using c-TF-IDF term extraction.

    Parameters
    ----------
    texts : Raw text for each data point.
    labels : (n_samples,) cluster assignments.
    top_n : Number of top terms to return per cluster.
    ngram_range : N-gram sizes.
    min_df : Minimum document frequency.
    max_df : Maximum document frequency fraction.
    stop_words : Stop words to exclude.

    Returns
    -------
    List of CTfidfClusterResult, one per cluster.
    """
    ctfidf, features, cluster_ids = build_ctfidf_matrix(
        texts, labels, ngram_range, min_df, max_df, stop_words
    )

    results: list[CTfidfClusterResult] = []
    for i, cluster_id in enumerate(cluster_ids):
        scores = ctfidf[i]
        sorted_idx = np.argsort(scores)[::-1]

        # Precompute max score across OTHER clusters for discriminativeness
        max_other = np.zeros(len(features))
        for k in range(len(cluster_ids)):
            if k != i:
                max_other = np.maximum(max_other, ctfidf[k])
        disc_ratio = np.divide(scores, max_other + 1e-12)

        # Top terms by c-TF-IDF score
        top_terms: list[CTfidfTerm] = []
        for j in sorted_idx[:top_n]:
            if scores[j] <= 0:
                break
            term = features[j]
            top_terms.append(CTfidfTerm(
                term=term,
                ngram_size=len(term.split()),
                ctfidf_score=float(scores[j]),
                discriminativeness=float(disc_ratio[j]),
                cluster=int(cluster_id),
            ))
        disc_idx = np.argsort(disc_ratio)[::-1]

        disc_terms: list[CTfidfTerm] = []
        for j in disc_idx[:top_n]:
            if scores[j] <= 0:
                break
            term = features[j]
            disc_terms.append(CTfidfTerm(
                term=term,
                ngram_size=len(term.split()),
                ctfidf_score=float(scores[j]),
                discriminativeness=float(disc_ratio[j]),
                cluster=int(cluster_id),
            ))

        # Shared terms (high score in multiple clusters)
        shared_terms: list[CTfidfTerm] = []
        multi_cluster = (ctfidf[:, :] > 0).sum(axis=0) > 1
        shared_idx = np.argsort(scores * multi_cluster)[::-1]
        for j in shared_idx[:top_n]:
            if scores[j] <= 0 or not multi_cluster[j]:
                break
            term = features[j]
            shared_terms.append(CTfidfTerm(
                term=term,
                ngram_size=len(term.split()),
                ctfidf_score=float(scores[j]),
                discriminativeness=float(disc_ratio[j]),
                cluster=int(cluster_id),
            ))

        n_items = int((labels == cluster_id).sum())
        results.append(CTfidfClusterResult(
            cluster=int(cluster_id),
            n_items=n_items,
            top_terms=top_terms,
            discriminative_terms=disc_terms,
            shared_terms=shared_terms,
        ))

    return results


# ---------------------------------------------------------------------------
# Stage 2: LLM label refinement
# ---------------------------------------------------------------------------


def _check_llm_provider(provider: str) -> None:
    """Raise ImportError if the LLM provider package is not installed."""
    pkg = {"anthropic": "anthropic", "openai": "openai"}
    if provider not in pkg:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic' or 'openai'.")
    try:
        __import__(pkg[provider])
    except ImportError:
        raise ImportError(
            f"{pkg[provider]} is required for LLM labeling. "
            f"Install with: pip install reap-topics[labeling]"
        ) from None


def _build_labeling_prompt(
    cluster_result: CTfidfClusterResult,
    sample_texts: list[str],
) -> str:
    """Build the LLM prompt for cluster label refinement.

    Parameters
    ----------
    cluster_result : c-TF-IDF output for a single cluster.
    sample_texts : Representative texts from the cluster.

    Returns
    -------
    Formatted prompt string.
    """
    top_terms_str = ", ".join(
        f"{t.term} ({t.ctfidf_score:.3f})" for t in cluster_result.top_terms[:10]
    )
    disc_terms_str = ", ".join(
        f"{t.term} ({t.discriminativeness:.2f}x)" for t in cluster_result.discriminative_terms[:5]
    )

    texts_section = "\n".join(f"  - {text[:300]}" for text in sample_texts)

    return (
        "You are labeling semantic clusters from a topic model. "
        "Based on the statistical term analysis and sample texts below, "
        "provide a concise, descriptive label for this cluster.\n\n"
        f"Cluster {cluster_result.cluster} ({cluster_result.n_items} items):\n\n"
        f"Top terms (by c-TF-IDF score): {top_terms_str}\n"
        f"Most discriminative terms (unique to this cluster): {disc_terms_str}\n\n"
        f"Sample texts from this cluster:\n{texts_section}\n\n"
        "Respond in this exact JSON format:\n"
        '{\n'
        '  "label": "2-5 word topic label",\n'
        '  "short_description": "One sentence describing the cluster theme",\n'
        '  "evidence_summary": "Key terms and patterns supporting this label",\n'
        '  "confidence": 0.0-1.0\n'
        '}\n\n'
        "Respond ONLY with the JSON object, no other text."
    )


def _call_llm(provider: str, model: str, prompt: str) -> str:
    """Call the LLM API and return the response text.

    Parameters
    ----------
    provider : "anthropic" or "openai".
    model : Model identifier.
    prompt : User prompt.

    Returns
    -------
    Raw response text from the LLM.

    Side effects: API call to external service.
    """
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text  # type: ignore[reportAttributeAccessIssue]

    if provider == "openai":
        import openai

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    raise ValueError(f"Unknown provider: {provider!r}")


def _parse_llm_response(response_text: str, cluster_id: int) -> LLMClusterLabel:
    """Parse LLM JSON response into an LLMClusterLabel.

    Falls back to raw text as the label if JSON parsing fails.

    Parameters
    ----------
    response_text : Raw response from the LLM.
    cluster_id : Cluster ID to attach to the label.

    Returns
    -------
    Parsed LLMClusterLabel.
    """
    import json

    try:
        text = response_text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        return LLMClusterLabel(
            cluster=cluster_id,
            label=data.get("label", "unknown"),
            short_description=data.get("short_description", ""),
            evidence_summary=data.get("evidence_summary", ""),
            confidence=float(data.get("confidence", 0.5)),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Failed to parse LLM response for cluster %d: %s", cluster_id, e)
        return LLMClusterLabel(
            cluster=cluster_id,
            label=response_text[:100].strip(),
            short_description="LLM response could not be parsed as JSON",
            evidence_summary=response_text[:200],
            confidence=0.0,
        )


def label_clusters_llm(
    ctfidf_results: list[CTfidfClusterResult],
    texts: list[str],
    labels: np.ndarray,
    provider: str = "anthropic",
    model: str | None = None,
    n_sample_texts: int = 5,
    random_state: int = 42,
) -> list[LLMClusterLabel]:
    """Refine c-TF-IDF labels using an LLM (Stage 2 of the labeling pipeline).

    Takes statistical c-TF-IDF candidate terms and uses an LLM to synthesize
    coherent, human-readable cluster labels.

    Parameters
    ----------
    ctfidf_results : Output from label_clusters_ctfidf().
    texts : Raw text for each data point.
    labels : (n_samples,) cluster assignments.
    provider : "anthropic" or "openai".
    model : Model name. Defaults to claude-opus-4-6 (anthropic) or gpt-5.4-mini (openai).
    n_sample_texts : Number of sample texts per cluster to include in prompt.
    random_state : Seed for reproducible text sampling.

    Returns
    -------
    List of LLMClusterLabel, one per cluster.

    Side effects: API calls to the LLM provider, logging.
    """
    _check_llm_provider(provider)
    rng = np.random.default_rng(random_state)

    # Default to Claude Opus 4.6 (best-in-class reasoning) + OpenAI gpt-5.4-mini.
    # For batch labeling at scale, claude-sonnet-4-6 is a cheaper alternative
    # that typically matches Opus on well-scoped prompts like cluster labeling.
    default_models = {"anthropic": "claude-opus-4-6", "openai": "gpt-5.4-mini"}
    model_name = model or default_models[provider]

    results: list[LLMClusterLabel] = []

    for cluster_result in ctfidf_results:
        cluster_id = cluster_result.cluster
        cluster_mask = labels == cluster_id
        cluster_texts = [t for t, m in zip(texts, cluster_mask) if m]

        # Sample texts reproducibly
        n_sample = min(n_sample_texts, len(cluster_texts))
        if n_sample > 0:
            sample_idx = rng.choice(len(cluster_texts), size=n_sample, replace=False)
            sample_texts = [cluster_texts[i] for i in sample_idx]
        else:
            sample_texts = []

        prompt = _build_labeling_prompt(cluster_result, sample_texts)
        response_text = _call_llm(provider, model_name, prompt)
        label_data = _parse_llm_response(response_text, cluster_id)
        results.append(label_data)
        logger.info(
            "Cluster %d labeled: %s (confidence=%.2f)",
            cluster_id, label_data.label, label_data.confidence,
        )

    return results


def label_clusters_combined(
    texts: list[str],
    labels: np.ndarray,
    provider: str = "anthropic",
    model: str | None = None,
    top_n: int = 10,
    ngram_range: tuple[int, int] = (1, 3),
    min_df: int = 2,
    max_df: float = 0.95,
    stop_words: str | list[str] | None = "english",
    n_sample_texts: int = 5,
    random_state: int = 42,
) -> tuple[list[CTfidfClusterResult], list[LLMClusterLabel]]:
    """Full labeling pipeline: c-TF-IDF → LLM refinement.

    Stage 1: Extract statistical candidate terms via c-TF-IDF.
    Stage 2: Refine into coherent labels via LLM.
    Stage 3 (not automated): Human expert validates and finalizes labels.

    Parameters
    ----------
    texts : Raw text for each data point.
    labels : (n_samples,) cluster assignments.
    provider : LLM provider ("anthropic" or "openai").
    model : LLM model name (defaults based on provider).
    top_n : Number of top c-TF-IDF terms per cluster.
    ngram_range : N-gram sizes for c-TF-IDF.
    min_df : Minimum document frequency for c-TF-IDF terms.
    max_df : Maximum document frequency fraction.
    stop_words : Stop words to exclude.
    n_sample_texts : Number of sample texts per cluster for LLM context.
    random_state : Seed for reproducible text sampling.

    Returns
    -------
    (ctfidf_results, llm_labels) — statistical terms and LLM-refined labels.
    Human validation is the expected next step.

    Side effects: API calls to LLM provider, logging.
    """
    # Stage 1: c-TF-IDF
    ctfidf_results = label_clusters_ctfidf(
        texts, labels, top_n, ngram_range, min_df, max_df, stop_words
    )
    logger.info("c-TF-IDF complete: %d clusters labeled", len(ctfidf_results))

    # Stage 2: LLM refinement
    llm_labels = label_clusters_llm(
        ctfidf_results, texts, labels, provider, model, n_sample_texts, random_state
    )
    logger.info("LLM labeling complete: %d clusters refined", len(llm_labels))

    return ctfidf_results, llm_labels
