"""Run the REAP projection head on a production corpus, recording CV metrics.

The projection head's load-bearing claim is OOS preservation: a small MLP
trained on (X_high, Y_consensus) for the corpus should generalize to held-out
documents, projecting them into the same consensus space and preserving their
high-d neighborhood structure.

This script trains the head with deterministic seeds, runs 5-fold stratified
CV, and writes per-fold + summary metrics to
``results/projection_head/<corpus>/oos_metrics.json`` for downstream
verification and manuscript reporting.

Usage
-----
    python scripts/run_projection_head_oos.py --corpus twenty_newsgroups_reference
    python scripts/run_projection_head_oos.py --corpus ai_art
    python scripts/run_projection_head_oos.py --corpus korean_forest

Reproducibility
---------------
Sets torch.manual_seed, torch.cuda.manual_seed_all, numpy.random.seed, and
torch deterministic-mode flags. Outputs include git commit and a SHA256 of
the input arrays. A second run with the same seed/inputs/code is expected to
reproduce the metrics to within float-roundoff.

Provenance recorded
-------------------
- Input array SHA256s
- Git commit
- Hyperparameters (hidden layers, dropout, alpha, lr, etc.)
- Per-fold metrics: mse / trustworthiness / silhouette / ari / distance_correlation
- Summary statistics across folds: mean, std, min, max
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"

CORPUS_LOADERS = {
    "twenty_newsgroups_reference": "load_twenty_newsgroups_reference",
    "ai_art": "load_ai_art",
    "korean_forest": "load_korean_forest",
}


def _set_deterministic_seeds(seed: int) -> None:
    """Pin every relevant RNG."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _sha256_of_array(arr: np.ndarray) -> str:
    """Hex SHA256 of the raw bytes of an array."""
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sha256_of_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        return out
    except Exception:
        return "<unknown>"


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": float(fmean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def run_for_corpus(corpus: str, set_letter: str, seed: int = 42) -> dict[str, Any]:
    """Train + 5-fold-CV the projection head for one (corpus, set).

    Returns the metrics dict (also written to disk).
    """
    _set_deterministic_seeds(seed)

    import reap.datasets as ds
    from reap.projection import train_projection_head

    loader = getattr(ds, CORPUS_LOADERS[corpus])
    snapshot = loader()
    X_full = np.asarray(snapshot.embeddings, dtype=np.float32)

    consensus_emb_path = (
        RESULTS_ROOT / corpus / "reap" / f"set_{set_letter}" / "consensus_embedding.npy"
    )
    consensus_labels_path = (
        RESULTS_ROOT / corpus / "reap" / f"set_{set_letter}" / "consensus_labels.npy"
    )
    Y = np.load(consensus_emb_path).astype(np.float32)
    labels = np.load(consensus_labels_path)

    # Snapshot embeddings may include filtered-out rows; consensus is on the
    # retained subset. Re-align X to the retained rows used to build Y.
    # The retained-indices file (if present) tells us; else we assume X is
    # already aligned (n_X == n_Y).
    if X_full.shape[0] != Y.shape[0]:
        # Look for a retained-indices file (the filter step may have written one).
        retained_path = (
            RESULTS_ROOT / corpus / "reap" / f"set_{set_letter}" / "retained_indices.npy"
        )
        if retained_path.exists():
            retained = np.load(retained_path)
            X = X_full[retained]
        else:
            # Fall back: take the first n_Y rows (the filter stage produces
            # consensus on a contiguous prefix in some pipelines).
            X = X_full[: Y.shape[0]]
        if X.shape[0] != Y.shape[0]:
            raise RuntimeError(
                f"{corpus} set {set_letter}: cannot align X ({X_full.shape[0]} -> "
                f"{X.shape[0]}) with Y ({Y.shape[0]})"
            )
    else:
        X = X_full

    print(
        f"[{corpus}/set_{set_letter}] X={X.shape} Y={Y.shape} labels={labels.shape} "
        f"n_unique_labels={len(np.unique(labels))}"
    )

    result = train_projection_head(
        X=X,
        Y=Y,
        labels=labels,
        hidden_layers=[128, 64],
        dropout=0.3,
        alpha=0.7,
        n_folds=5,
        batch_size=64,
        max_epochs=500,
        patience=50,
        lr=1e-3,
        weight_decay=1e-4,
        device="cpu",
    )

    metric_keys = ["mse", "trustworthiness", "silhouette", "ari", "distance_correlation"]
    per_fold = result["cv_metrics"]
    cv_summary = {
        key: _summarize([fold[key] for fold in per_fold]) for key in metric_keys
    }

    payload = {
        "corpus": corpus,
        "set": set_letter,
        "git_commit": _current_git_commit(),
        "seed": seed,
        "input_X_shape": list(X.shape),
        "input_X_sha256": _sha256_of_array(X),
        "input_Y_shape": list(Y.shape),
        "input_Y_sha256_from_disk": _sha256_of_path(consensus_emb_path),
        "input_labels_shape": list(labels.shape),
        "input_labels_n_unique": int(len(np.unique(labels))),
        "config": result["config"],
        "per_fold": per_fold,
        "cv_summary": cv_summary,
        "final_metrics": result["final_metrics"],
    }

    out_dir = RESULTS_ROOT / "projection_head" / corpus / f"set_{set_letter}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "oos_metrics.json"
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"[{corpus}/set_{set_letter}] wrote {out_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", required=True, choices=list(CORPUS_LOADERS.keys())
    )
    parser.add_argument("--set", default="A", choices=["A", "B", "C"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    payload = run_for_corpus(args.corpus, args.set, seed=args.seed)
    cv = payload["cv_summary"]
    print(
        f"\n[{args.corpus}/set_{args.set}] CV summary:\n"
        f"  trustworthiness  mean={cv['trustworthiness']['mean']:.4f} "
        f"std={cv['trustworthiness']['std']:.4f}\n"
        f"  silhouette       mean={cv['silhouette']['mean']:.4f} "
        f"std={cv['silhouette']['std']:.4f}\n"
        f"  ARI              mean={cv['ari']['mean']:.4f} "
        f"std={cv['ari']['std']:.4f}\n"
        f"  distance_corr    mean={cv['distance_correlation']['mean']:.4f} "
        f"std={cv['distance_correlation']['std']:.4f}\n"
        f"  MSE              mean={cv['mse']['mean']:.4f} "
        f"std={cv['mse']['std']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
