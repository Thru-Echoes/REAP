# REAP — Claude Code Instructions

## Project Context

**REAP** (Reproducible Embedding via Averaged Projection) is an open-source Python package and methods paper for stabilizing stochastic dimensionality reduction. Core contribution: averaging pairwise distance matrices across multi-seed UMAP runs, which is invariant to the rotation/reflection ambiguity of UMAP embeddings.

This is PhD research at UC Berkeley. The package accompanies a methods paper.

## Environment

```bash
conda activate nlp_sent_trans_notebook
export KMP_DUPLICATE_LIB_OK=TRUE   # macOS OMP workaround
```

Python 3.10+. Core deps: numpy, scipy, scikit-learn, umap-learn, pydantic v2.
Optional: torch (projection head), openai/anthropic (LLM labeling).

## Package Structure

```
src/reap/
├── __init__.py        # Public API exports
├── consensus.py       # Core algorithm: distance-matrix averaging
├── evaluation.py      # Metrics: trustworthiness, silhouette, ARI
├── clustering.py      # KMeans + K selection
├── projection.py      # Neural projection head (PyTorch)
├── search.py          # Grid search framework
├── labeling.py        # c-TF-IDF + LLM cluster labeling
├── visualization.py   # PCA 2D projection
├── validation.py      # Data quality gates
└── _types.py          # Pydantic models
```

## Code Standards

### Functional Programming Style (MANDATORY)
- **Pure functions** with explicit inputs and outputs — no hidden state
- **Naming**: `get_*` (data retrieval/computation), `compute_*` (metrics), `run_*` (pipelines with side effects), `find_*` (search/optimization)
- **Document ALL side effects** — if a function writes to disk, mutates state, calls an API, or uses randomness, say so in the docstring
- **Document all inputs and outputs** for every function; module-level purpose at the top of each file
- **Prefer immutable data**: Pydantic `BaseModel(frozen=True)` for all structured data crossing function boundaries
- **`Field(..., description="...")`** on all public-facing model fields
- **`model_validate()` / `model_dump()`** — never manual dict construction for Pydantic models
- **Avoid class hierarchies** when functions + data models suffice. Classes OK when state is genuinely needed (e.g., ProjectionHead wraps a PyTorch model).
- Three clear lines of code beat one clever abstraction. No premature abstractions.

### Type Safety (MANDATORY)
- **Run `pyright`** on all modified Python files — zero errors in src/reap/
- Use `from __future__ import annotations` in all files
- Use modern syntax: `list[int]`, `dict[str, float]`, `X | None` (not `List`, `Dict`, `Optional`)
- `# type: ignore[specific-code]` with error code — never bare `# type: ignore`

### Testing (MANDATORY)
- **All code must verify itself.** Every deliverable MUST have E2E tests.
- **Use real computations**, not mocks, for E2E tests. Mock ONLY external services (paid APIs).
- Tests must **FAIL LOUDLY** — no silent skips, no `pytest.mark.skipif` without documented reason.
- `pytest` as the test runner. `asyncio_mode = "auto"` for async tests.
- **Pre-commit check**: `pyright src/reap/ && ruff check src/ tests/ && pytest tests/ -v`

### Logging
- Use `logging.getLogger(__name__)` — **never `print()`**.
- Follow the pattern in `cluster_labels.py` from green-narrative.

### Diagrams
- When creating diagrams: always produce .md source (Mermaid) AND at least one rendered format (.html, .png, .pdf)
- Use Playwright MCP to render HTML → PNG when needed for manuscript figures

## Forbidden Actions

1. **NEVER fabricate results or metrics** — this is academic research.
2. **NEVER skip validation steps.**
3. **NEVER use magic numbers without documentation.**
4. **NEVER mix Korean/domain-specific code into the core package** — keep `src/reap/` general.
5. **NEVER add features, refactor code, or make "improvements" beyond what was asked.**

## Key Validated Results

### Korean Forest Policy (d=18, nn=19, md=0.005, K=8, 30 seeds)
- Trustworthiness: 0.892
- Silhouette: 0.841
- Distance-matrix ARI: 0.75 (+32% over Procrustes 0.56)

### AI-Art Discourse (d=5, nn=53, md=0.01, K=20, 30 seeds)
- Trustworthiness: 0.916
- Silhouette: 0.657
- Projection R²: 0.904

## Manuscript

The `manuscript/` directory contains the REAP methods paper. Multi-dataset validation across:
1. AI-art discourse (primary, 1742 chunks)
2. Korean forest policy (secondary, 905 chunks)
3. Corporate sustainability (secondary, in progress)
4. US presidential language (secondary, planned)

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
