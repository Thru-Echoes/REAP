# Contributing to REAP

Thanks for your interest in REAP. The package ships as
`pip install reap-topics` and accompanies an in-progress methods paper,
so contributions are held to publication-grade standards for
reproducibility and provenance. Please skim
[`.claude/rules/`](.claude/rules/) before opening a substantive PR —
those documents are the binding source of truth for code style,
testing, manuscript work, and the public-package contract.

## Development Setup

```bash
git clone https://github.com/Thru-Echoes/reap-topics
cd reap-topics
python -m venv .venv && source .venv/bin/activate   # or use conda env `reap`
pip install -e ".[dev]"
```

On macOS you may need:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Running the Test Suite

The full pre-commit triage is three commands:

```bash
pytest tests/ -v --tb=short        # unit + E2E (must pass)
pyright src/reap/                  # type-check (must be zero errors)
ruff check src/ tests/             # lint (must be zero errors)
```

Slow and LLM-gated tests are excluded by default via pytest markers
(`-m "not slow"`, `-m "not llm"`). Run them explicitly when you change
code they cover:

```bash
pytest tests/ -v -m slow
pytest tests/ -v -m llm            # requires ANTHROPIC_API_KEY / OPENAI_API_KEY
```

The golden-validation suite is the manuscript-companion gate. Run it
locally when changing anything that touches the consensus algorithm,
metric computations, or the dataset loaders:

```bash
pytest tests/test_golden_validation.py -v --tb=short
```

## Pull Request Expectations

Every PR should include:

1. **Tests.** New behavior gets tests. Bug fixes get a regression
   test that fails before the fix and passes after. Prefer E2E
   tests that run the real pipeline on small synthetic data
   over mock-heavy unit tests
   (see [`.claude/rules/e2e-testing.md`](.claude/rules/e2e-testing.md)).
2. **CHANGELOG entry.** Add a bullet under `[Unreleased]` describing
   the user-visible change. Use Keep-a-Changelog sections
   (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`).
3. **Docs updated.** If you change a public API, update the README's
   API reference and any affected example under `examples/`.
4. **Breaking-change flag.** If the change breaks any public name,
   say so explicitly in the PR description and update CHANGELOG
   under `### Removed` / `### Changed` with a migration note.
5. **Type-clean.** `pyright src/reap/` must report zero errors.
6. **Lint-clean.** `ruff check src/ tests/` must report zero errors.

The pull-request template will prompt you for each of these.

## Coding Standards

The short version (the long version lives in
[`.claude/rules/`](.claude/rules/)):

- Functional-programming style by default: pure functions with
  explicit inputs and outputs, named with `get_*`, `compute_*`,
  `run_*`, `find_*`, `build_*`, `validate_*` prefixes.
- Pydantic v2 `BaseModel(frozen=True)` for all structured data
  crossing function boundaries — no raw dicts on the public surface.
- Type hints on every public function; `from __future__ import annotations`
  at the top of every file.
- No domain-specific code in `src/reap/` core modules — domain logic
  lives under `src/reap/datasets/<name>.py`.
- Logging via `logging.getLogger(__name__)`; never `print()`.
- Three clear lines beat one clever abstraction.

## Reporting Issues

Use the issue templates under
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE):

- **Bug report** — for unexpected behavior in installed code.
- **Feature request** — for new functionality.
- **Dataset request** — for proposing a new corpus loader under
  `reap.datasets`.

For security-sensitive issues, please email the maintainer directly
rather than filing a public issue.

## Code of Conduct

Participation in this project is governed by the
[Contributor Covenant v2.1](CODE_OF_CONDUCT.md). By contributing,
you agree to abide by its terms.

## License

By contributing, you agree that your contributions will be licensed
under the [Apache License 2.0](LICENSE).
