# REAP — Claude Code Instructions

## Project Context

**REAP** (Reproducible Embedding via Averaged Projection) is an open-source Python package and accompanying methods paper for consensus-based topic modeling with out-of-sample projection. Core contribution: averaging pairwise distance matrices across multi-seed UMAP runs, which is invariant to the rotation/reflection ambiguity of UMAP embeddings, then training a neural projection head for new data.

This is PhD research at UC Berkeley. The package accompanies a methods paper.

---

## Publication-Quality Standard (READ FIRST)

All work in this repo targets a **top-tier scientific methods paper**
and a **public PyPI package**. Every number, claim, figure, and
citation must survive adversarial peer review; every line of code must
read cleanly to a stranger adopting the package tomorrow. The binding
rules live in:

- [`.claude/rules/publication-standards.md`](.claude/rules/publication-standards.md)
  — scientific rigor, reproducibility bundles, pre-registered
  protocols, verified citations, self-review checklist.
- [`.claude/rules/open-source-package.md`](.claude/rules/open-source-package.md)
  — public API stability, semver + CHANGELOG, CI matrix, license,
  datasets-as-API contract, bring-your-own-embedding contract,
  adoptability checklist.

These override anything in sibling rule files when they conflict. The
pre-registered evaluation protocol lives at
[`manuscript/evaluation_protocol.md`](manuscript/evaluation_protocol.md)
— every later script, table, figure, and paragraph points back to it.

---

## Environment

```bash
conda activate reap
export KMP_DUPLICATE_LIB_OK=TRUE   # macOS OMP workaround
```

**Conda env:** `reap` (Python 3.12)
**Full path:** `/opt/homebrew/Caskroom/miniforge/base/envs/reap/bin/python`

Core deps: numpy, scipy, scikit-learn, umap-learn, pydantic v2, torch.
Optional: openai/anthropic (LLM labeling).

---

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

---

## TRACE Protocol

Uses **trace-mcp v0.3.0**. Sessions stored in `~/.trace/sessions/`.
Knowledge store: `~/.trace/knowledge/REAP.json`.

**Absolute rule**: Never fabricate, falsify, or retroactively alter TRACE data.

### Session Lifecycle

1. **Start** a session at the beginning of any multi-step workflow.
2. Before any task, review prior context with `trace_search` or `trace_list_sessions(project="REAP")`.
3. **End** the session with a summary when done.
4. **Micro-sessions** for provenance-relevant events outside workflows.

### What to Log

| Event | When | Tool |
|-------|------|------|
| Decision | BEFORE acting | `trace_propose_decision` / `trace_resolve_decision` |
| Correction | When human catches AI mistake | `trace_log_annotation(category="correction")` |
| Contribution | AFTER artifact exists | `trace_log_contribution` |
| State change | When it occurs | `trace_log_state_change` |

**DO NOT log:** File reads, greps, directory listings, exploratory calls, TRACE's own calls.

### trace-learn (default extension)

Use `trace_learn_*` tools for cross-session knowledge:
- `trace_learn_recall` — find relevant past learnings
- `trace_learn_add` — manually add a learning
- `trace_learn_extract` — extract learnings from session events

---

## Code Style

### Functional Programming (Preferred)

The codebase prefers functional programming style without adding unnecessary
complexity. The goal is code that is easy to read, easy to extend, easy to
adopt, and easy to modify.

- **Prefer pure functions** with explicit inputs and outputs over classes with hidden state
- **Naming conventions**:
  - `get_*` — retrieve, compute, or construct data
  - `set_*` — configure or mutate state (document side effects)
  - `compute_*` — derive metrics or measurements
  - `run_*` — execute pipelines or multi-step workflows (likely has side effects)
  - `find_*` — search or optimization
- **Document every function** with a docstring that describes:
  - What the function does (one line)
  - Parameters and return values
  - Side effects, if any (disk writes, state mutation, API calls, randomness)
- **Prefer immutable data**: Use Pydantic `BaseModel(frozen=True)` for structured data
  crossing function boundaries. `Field(..., description="...")` on public fields.
- **Avoid class hierarchies** when functions + data models suffice. Use classes when
  state is genuinely needed (e.g., `ProjectionHead` wraps a PyTorch model).
- **Keep it simple**: Three clear lines of code beat one clever abstraction. No premature
  abstractions. No unnecessary complexity.

### Type Hints

- Run `pyright` on all modified Python files before considering work complete
- Use `from __future__ import annotations` in all files
- Use modern syntax: `list[int]`, `dict[str, float]`, `X | None`
- When type stubs disagree with runtime: `# type: ignore[specific-code]` with the error code

### Testing

- All code should verify itself. Prefer E2E tests that run real computations.
- Use real data (synthetic blobs, make_blobs), not mocks, for E2E tests. Mock only
  external services (paid APIs, remote databases).
- Tests should fail loudly — no silent skips without documented reason.
- `pytest` as the test runner. Pre-commit: `pyright src/reap/ && pytest tests/ -v`

### Logging

- Use `logging.getLogger(__name__)` — never `print()`.

### Diagrams

- When creating diagrams: always produce .md source (Mermaid) AND at least one rendered
  format (.html, .png, .pdf).
- Use Playwright MCP to render HTML → PNG for manuscript figures when needed.

---

## Forbidden Actions

1. **NEVER fabricate results or metrics** — this is academic research.
2. **NEVER skip validation steps.**
3. **NEVER use magic numbers without documentation.**
4. **NEVER mix domain-specific code into the core package** — keep `src/reap/` general.

---

## Key Validated Results

### Korean Forest Policy (d=18, nn=19, md=0.005, K=8, 30 seeds)
- Trustworthiness: 0.892
- Silhouette: 0.841
- Distance-matrix ARI: 0.75 (+32% over Procrustes 0.56)

### AI-Art Discourse (d=5, nn=53, md=0.01, K=20, 30 seeds)
- Trustworthiness: 0.916
- Silhouette: 0.657
- Projection R²: 0.904

---

## Manuscript

The `manuscript/` directory contains the REAP methods paper. Multi-dataset validation across:
1. AI-art discourse (primary, 1742 chunks)
2. Korean forest policy (secondary, 905 chunks)
3. Corporate sustainability (secondary, in progress)
4. US presidential language (secondary, planned)

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
