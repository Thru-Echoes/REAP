# REAP — Claude Code Instructions

## Project Context

**REAP** (Reproducible Embedding via Averaged Projection) is an open-source Python package and accompanying methods paper for consensus-based topic modeling with out-of-sample projection. Core contribution: averaging pairwise distance matrices across multi-seed UMAP runs — invariant to UMAP's rotation/reflection ambiguity — then training a neural projection head for new data.

PhD research at UC Berkeley; the package accompanies a methods paper.

---

## Binding rules (READ FIRST)

All work in this repo targets a **top-tier scientific methods paper** and a **public PyPI package**. The behavioral rules live in:

- [`.claude/rules/publication-standards.md`](.claude/rules/publication-standards.md) — scientific rigor, reproducibility bundles, pre-registered protocols, verified citations, self-review checklist.
- [`.claude/rules/open-source-package.md`](.claude/rules/open-source-package.md) — public API stability, semver + CHANGELOG, CI matrix, license, dataset API contract.
- Other `.claude/rules/*.md` (`code-quality`, `pyright-and-tooling`, `testing`, `e2e-testing`, `diagrams`, `manuscript`, `playwright`) — do not duplicate them here; treat them as the single source of truth.

These override anything that conflicts. The pre-registered evaluation protocol is at [`manuscript/evaluation_protocol.md`](manuscript/evaluation_protocol.md) — every script, table, figure, and paragraph traces back to it.

---

## TRACE is a project priority

Every AI-assisted session on this repo runs under TRACE. TRACE captures decisions, corrections, contributions, and gotchas as they happen — that provenance trail is what makes the methods paper auditable and the package's design history recoverable later. Treat it as first-class infrastructure, not optional logging.

- Start a session at the beginning of any multi-step workflow; end with a summary.
- Log decisions BEFORE acting, contributions AFTER each artifact, corrections when humans catch AI mistakes.
- **Interleave logging with the work** — do not defer to session end.
- The full protocol mechanics live in the global `~/.claude/CLAUDE.md` "TRACE Protocol v0.4.1" section; do not restate them here.

**Project-specific TRACE facts:**
- Sessions: `~/.trace/sessions/`
- Knowledge store: `~/.trace/knowledge/REAP.json`
- trace-mcp version: v0.4.1

---

## Environment

```bash
conda activate reap
export KMP_DUPLICATE_LIB_OK=TRUE   # macOS OMP workaround
```

**Conda env:** `reap` (Python 3.12)
**Full path:** `/opt/homebrew/Caskroom/miniforge/base/envs/reap/bin/python`

Core deps: numpy, scipy, scikit-learn, umap-learn, pydantic v2, torch.
Optional extras: `[projection]` (torch), `[labeling]` (openai, anthropic), `[text-fixtures]` (sentence-transformers), `[dev]`.

---

## Where things live

- **Loaders + cache**: `src/reap/datasets/`, `~/.cache/reap/datasets/<name>/<version>/`
- **Build a snapshot**: `python scripts/build_datasets.py --dataset <name> --source-path <sibling-project>`
- **Locked snapshot manifest**: `manuscript/datasets/manifest.json`
- **Pre-registered protocol** (binding): `manuscript/evaluation_protocol.md`
- **Seed manifest** (Sets A/B/C × 30): `manuscript/seeds/seed_manifest.json`
- **Examples**: `examples/load_real_datasets.py`, `examples/run_benchmark.py`
- **Reference validated results** (sibling-project baselines, not paper finals): `manuscript/key_validated_results.md`

---

## Reminders (TLDR of binding rules — see rules files for details)

1. **Never fabricate** results, metrics, citations, or dataset stats.
2. **Never skip validation** or silence failing tests.
3. **Never use magic numbers** without a comment citing the source.
4. **Never mix domain-specific code** into `src/reap/` core modules — those live in `src/reap/datasets/<name>.py`.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

<!-- trace-mcp:claude-code -->

## TRACE Audit Protocol (v0.4.1+)

This project uses [TRACE](https://github.com/Thru-Echoes/TRACE) for transparent
documentation of AI-human collaboration. The TRACE MCP server is configured in
`.mcp.json` and enforced via `.claude/hooks/`.

**Absolute rule**: Never fabricate, falsify, or retroactively alter TRACE
data. A sparse honest record beats a dense fabricated one.

**Session lifecycle**

- **Start** a TRACE session at the beginning of any multi-step workflow.
- **End** with a summary when the workflow is complete. Review the
  Attribution Audit returned by `trace_end_session` before closing.

**What to log**

- **Decisions** (propose BEFORE acting, resolve when the human responds).
  - **Proposer Identity Rule (v0.4.1, spec §3.6)**: set `proposed_by` to the
    actor who authored the proposal *content* (whose words populate
    `description`), not the speaker of the resolving directive.
    Question→AI-proposal→accept means `proposed_by=ai`, `resolved_by=human`.
- **Corrections** when a participant catches a mistake.
  - If the corrected entity is not a TRACE event (subagent output, tool
    result, external claim), use a URI-form reference per spec §3.7.1:
    `external:<uri>` (universal fallback), `jsonl:<path>#L<line>`,
    `subagent:<id>`, or `tool-result:<id>`. `related_event_ids` is NOT
    for the correction relationship.
- **Discoveries (v0.4.1, `category="discovery"`)**: non-trivial findings
  from autonomous work — log AT THE MOMENT of discovery, not in a
  post-hoc summary.
- **Contributions** — one per artifact, with `direction` (who had the idea)
  and `execution` (who did the work). Always set `conversation_snippet`
  to the relevant user message (~200 chars). If no user message
  motivated the event (autonomous-execution stretch), use
  `<autonomous-stretch>` rather than omitting. Silent omission is a
  v0.4.1 protocol violation per spec §3.4.1.
- **Subagent dispatches** when their outcome is summarized by a
  contribution — `trace_log_tool_call(host="internal", server="claude-code",
  parent_event_id=...)` per spec §3.5. Skip routine file reads, greps,
  or TRACE's own calls.

Full protocol, including attribution rules, URI-form references, and
worked examples, lives at the [TRACE specification](https://github.com/Thru-Echoes/TRACE/blob/main/docs/specification.md).

<!-- /trace-mcp:claude-code -->
