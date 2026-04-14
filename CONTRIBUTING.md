# Contributing to pybhatlib

Thanks for your interest in contributing. This guide covers the pull request
workflow, local development setup, and the standards we maintain for
numerical correctness. It is written to be usable even if you are new to
GitHub — read top to bottom the first time.

## Quick overview

- All changes go through a **pull request (PR)** — nobody pushes directly to `main`.
- Each PR covers **one logical change**, not a batch of unrelated edits.
- CI runs `pytest` automatically on every PR. **CI must pass before merging.**
- Because pybhatlib reproduces published log-likelihood values exactly, any
  change that alters numerical results must be justified and verified.

## Setting up locally

```bash
git clone https://github.com/UMN-Choi-Lab/pybhatlib.git
cd pybhatlib
pip install -e ".[dev]"
```

Verify the install:

```bash
pytest tests/ -m "not slow"
```

All tests should pass. If they do not, open an issue before making changes —
something in your environment is off, and PRs built on a broken baseline
waste review time.

## The pull request workflow

### 1. Create a branch

One branch per logical change. Short, descriptive names — not your username,
not a date.

```bash
git checkout main
git pull origin main
git checkout -b fix-mixture-shared-coefs
```

Good branch names: `fix-mixture-shared-coefs`, `add-mvncd-j5-benchmark`,
`docs-contributing-guide`.

Avoid: `dales-changes`, `update`, `wip`, `patch-1`.

### 2. Make changes and commit

```bash
# Edit files...
pytest tests/ -m "not slow"        # Run tests before committing
git add <files>
git commit -m "Short imperative summary of the change"
```

Commit message style:

- First line: imperative mood, under 72 characters
  (`"Add shared-coefficient support for mixture-of-normals"`, not
  `"Added ..."` or `"Changes to mixture"`).
- Optional body after a blank line explains *why*, not *what*.
- One commit per logical step is fine; squash trivial fixup commits before
  pushing (`git rebase -i main`).

### 3. Push and open the PR

```bash
git push -u origin fix-mixture-shared-coefs
```

GitHub prints a URL for opening the PR after the push — click it, or visit
the repo on github.com and look for the "Compare & pull request" banner.

PR description should cover:

- **What** changed (one or two sentences).
- **Why** the change is needed (link to an issue or email thread if relevant).
- **How it was tested** — which tests you ran, which tutorials you re-executed,
  any benchmark numbers.
- **Verification impact** — if your change could affect log-likelihoods,
  explicitly state whether you re-ran the Table 1 verification (see below).

### 4. Review and merge

A maintainer reviews the PR, may request changes, and merges it. While the
PR is open:

- Push new commits to the same branch — they update the PR automatically.
- Do not force-push unless asked.
- Do not merge `main` into your branch unless requested; rebase if you must
  update.

After merge, delete your branch:

```bash
git checkout main
git pull origin main
git branch -d fix-mixture-shared-coefs
```

## Numerical verification (important)

pybhatlib reproduces exact log-likelihood values from BHATLIB Table 1:

| Model | LL |
|-------|-----|
| (a)(i) IID | −670.956 |
| (a)(ii) Flexible | −661.111 |
| (b) + AGE45 | −659.285 |
| (c) Random coefficient | −635.871 |
| (d) Mixture (nseg=2) | ≈−634.975 |

If your change touches any code under `src/pybhatlib/gradmvn/`,
`src/pybhatlib/matgradient/`, `src/pybhatlib/vecup/`, or
`src/pybhatlib/models/mnp/`, run the Table 1 tutorial locally before
requesting review:

```bash
jupyter nbconvert --to notebook --execute docs/tutorials/t04h_bhatlib_table1.ipynb
```

Include the resulting LL values in the PR description. If they differ from
the table, explain why the change is intentional — do not merge a regression
without discussion.

## Code standards

- **Python**: PEP 8, 88-char line length (matches `ruff` config).
- **Type hints** on all public function signatures.
- **Docstrings** in NumPy style on all public functions.
- **Backend-aware numerical code** accepts an optional `xp` keyword for
  NumPy / PyTorch backend selection.
- **Numba-compatible** in hot paths — use `scipy.special.ndtr`, not
  `scipy.stats.norm.cdf`; avoid Python-level scipy.stats objects inside
  JIT-compiled functions.
- **No module-level mutable state** — thread safety matters for parallel
  gradient evaluations.
- **Private modules** are prefixed with an underscore (`_vec_ops.py`).

## Running specific test subsets

```bash
pytest tests/                      # Everything including slow integration tests
pytest tests/ -m "not slow"        # Skip slow tests (what CI runs)
pytest tests/ -m torch             # Only PyTorch-backend tests (needs torch installed)
pytest tests/test_models/          # Just the model tests
pytest tests/ -k mnp               # Tests matching "mnp" in name
```

## Scope rules

- **One logical change per PR.** Bundling a bug fix, a feature, and a refactor
  into one PR makes review painful and makes `git bisect` useless if a
  regression shows up later.
- **Do not reformat unrelated code** in your PR. If you notice style issues
  elsewhere, open a separate PR for cleanup.
- **Performance changes need before/after numbers.** "Faster" is not
  reviewable; "1.40s → 0.82s on Table 1 Model (a)(ii)" is.
- **Feature additions that change model semantics** (e.g., allowing
  coefficients to be shared across mixture segments) should include a test
  that pins the new behavior.

## Getting help

- **Questions about an existing PR**: comment on the PR directly.
- **Questions about where to start**: open an issue with the `question` label.
- **Something is broken**: open an issue with a minimal reproducer and the
  output of `pip freeze`.

## License

By contributing, you agree that your contributions are licensed under the
MIT license (see `LICENSE`).
