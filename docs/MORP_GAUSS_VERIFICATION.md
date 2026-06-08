# MORP — Python vs GAUSS Verification Plan

**Audience:** run on a machine with GAUSS + BHATLIB installed (e.g. Anna/Dale at
UTA, or the MacBook with a local GAUSS). The Python side runs anywhere
`pybhatlib` is installed.

**Purpose:** a concrete checklist for confirming the Python reimplementation
reproduces GAUSS BHATLIB MORP outputs, covering (Part A) the round‑2 changes
made in response to Anna's 2026‑06 follow‑up report, (Part B) the regression
anchors that must *not* move, and (Part C) the adjacent areas Anna is likely to
exercise next so we can get ahead of them.

> Each item lists **what changed**, the **Python command**, the **GAUSS
> correspondence**, and an explicit **PASS criterion**. Where exact numeric
> equality isn't expected (MVN‑CDF approximations differ slightly), the tolerance
> is stated.

---

## 0. Reference artifacts

Local (not committed — `Gauss Files and Comparison/MORP/` and `anna0605/` are
git‑ignored):

| File | Role |
|---|---|
| `Gauss Files and Comparison/MORP/MORP_DINING.gss` | GAUSS Dining model (`nobs=1092`, 3 outcomes `[11,7,7]`, `_indep=0`, `_Spher=0`) |
| `Gauss Files and Comparison/MORP/MORP_WALK.gss` | GAUSS Walk model (`nobs=1583`, 4 outcomes `[3,3,3,3]`, `_indep=0`, `_Spher=0`) |
| `Gauss Files and Comparison/MORP/MORP_WALK_ATE.gss` | GAUSS ATE driver — plugs in a final `est` vector, writes joint combo probs |
| `Gauss Files and Comparison/MORP/ate1.csv` | GAUSS ATE output: 81 rows `combo(4) ~ mean joint prob` |
| `Gauss Files and Comparison/MORP/Example_{Dining,Walk}.csv` | Datasets |
| `anna0605/MORP_test1.py`, `MORP_test2.py` | Anna's Python Dining / Walk scripts (currently `iid=True`) |
| `anna0605/MORP_test_outputs.docx` | Anna's side‑by‑side Python/GAUSS output capture |

**Important GAUSS settings to mirror in Python:**
`_indep=0` ⇒ `MORPControl(iid=False)`. `_Spher=0` (radial) — the Python default is
`spherical=True`; **the correlation matrix and log‑likelihood are identical
either way** (different unconstrained parameterisation of the *same* correlation
matrix), so leave the Python default unless reproducing raw optimiser parameters.
GAUSS uses unit error variances (no free scales) ⇒ Python default
`fix_scales=True`.

---

## 1. Environment

```bash
cd pybhatlib
pip install -e ".[dev]"
# full regression (incl. slow parity anchors), run once:
pytest tests/ -q
```

All Python snippets below assume:

```python
import numpy as np, pandas as pd
from pybhatlib.models.morp import (
    MORPModel, MORPControl, MORPResults,
    morp_ate, morp_ate_from_params, morp_joint_probs,
)
```

---

## Part A — Round‑2 changes to verify

### A1. `corr_*` rows report the actual correlation entries (Anna issue #1)

**What changed.** The `corr_*` rows in `summary()` / `to_dataframe()` previously
printed the *raw optimiser parameters* (spherical angles by default, `atanh`
pre‑images under `spherical=False`), which do **not** equal the correlation
values printed in the "Estimated error correlation matrix" block. They now print
the actual correlation entries with delta‑method standard errors. (This mirrors
the round‑1 fix that made the `thresh_*` rows print cut‑points instead of the raw
`tau`/log‑increment slots.)

**Python.** Fit the Walk model non‑independently:

```python
df = pd.read_csv("Gauss Files and Comparison/MORP/Example_Walk.csv")
spec = {
    "Hfemale": {"Happy": "Female"},   "Hage20": {"Happy": "Age20"},
    "Hutil":   {"Happy": "Util"},     "Mfemale":{"Meaning": "Female"},
    "Mage65":  {"Meaning": "Age65"},  "Sage65": {"Stress": "Age65"},
    "Smorning":{"Stress": "Morning"}, "Sutil":  {"Stress": "Util"},
    "Tage65":  {"Tired": "Age65"},    "Tmorning":{"Tired": "Morning"},
    "Tutil":   {"Tired": "Util"},
}
m = MORPModel(df, ["Happy","Meaning","Stress","Tired"], spec,
              n_categories=[3,3,3,3], control=MORPControl(iid=False, seed=42))
r = m.fit()
print(r.summary())
print(r.correlation_matrix)
```

**GAUSS correspondence.** `MORP_WALK.gss` (`_indep=0`) prints the estimated
correlation matrix and the per‑parameter table.

**PASS.**
1. Each `corr_<a>_<b>` value in the Python table **equals** the corresponding
   off‑diagonal entry of the printed "Estimated error correlation matrix" block
   (exact — both come from the same transform; the regression test asserts
   `atol=1e-12`).
2. That correlation matrix matches GAUSS's printed correlation matrix (tolerance
   ~`1e-2`; depends on convergence + MVNCD method).
3. `corr_*` standard errors are finite and in a sane range, and match GAUSS's
   reported correlation SEs to ~`1e-2`.

Automated: `pytest tests/test_models/test_morp.py -k Corr`.

---

### A2. ATE from final coefficients without re‑fitting (Anna issue #3)

**What changed.** Anna wanted the GAUSS workflow: paste the converged estimates
and compute ATEs from them, no re‑optimisation. The old `morp_ate(results, …)`
read `results.params` in **raw optimiser space** (thresholds as log‑increments,
correlations as spherical angles), so a hand‑built `MORPResults` with natural
coefficients produced wrong ATEs. New entry points accept the **reported**
(natural) coefficients directly:

* `MORPResults.from_estimates(beta, thresholds, correlation, *, dep_vars, n_categories)`
  — `thresholds` are **cut‑points** (the printed THRESH values), `correlation` is
  the error correlation matrix (`None` ⇒ IID). Returns a results object that
  drives `morp_ate` / `morp_predict` / `morp_joint_probs` / `summary` unchanged.
* `morp_joint_probs(results, X, n_dims, n_categories, n_beta)` — mean **joint**
  category‑combination probabilities = the GAUSS `ate1.csv` quantity. The MVN
  rectangle kernel is the same one used in the log‑likelihood.
  `.marginal(d)` collapses the joint to per‑dimension marginals.
* `morp_ate_from_params(...)` — convenience wrapper (marginal by default,
  `joint=True` for the joint distribution).

**Recommended GAUSS comparison (reproduce `ate1.csv`).** Use the values GAUSS
*prints* (betas by name, THRESH cut‑points, correlation matrix) — this avoids any
ambiguity about the raw `est` parameterisation/ordering inside `MORP_WALK_ATE.gss`.

```python
# 1) natural coefficients read from the GAUSS MORP_WALK output:
beta = np.array([...11 betas in coefficient order...])
thresholds = [np.array([t1,t2]),  # Happy  cut-points (increasing)
              np.array([t1,t2]),  # Meaning
              np.array([t1,t2]),  # Stress
              np.array([t1,t2])]  # Tired
correlation = np.array([[1, ...],[...]])   # 4x4 from GAUSS

res = MORPResults.from_estimates(beta, thresholds, correlation,
        dep_vars=["Happy","Meaning","Stress","Tired"], n_categories=[3,3,3,3])

# 2) design tensor under the SAME scenario MORP_WALK_ATE.gss used
#    (changevar=female, changeval=1 -> set Female:=1 for everyone):
df_t = df.copy(); df_t["Female"] = 1
mt = MORPModel(df_t, ["Happy","Meaning","Stress","Tired"], spec,
               n_categories=[3,3,3,3], control=MORPControl(iid=False))
X = mt.X                      # built in __init__, no fit needed
n_beta = mt.n_beta

J = morp_joint_probs(res, X, 4, [3,3,3,3], n_beta)
# J.combos: (81,4) 1-based; J.probs: (81,) mean joint prob — compare to ate1.csv
ate = pd.read_csv("Gauss Files and Comparison/MORP/ate1.csv", header=None).to_numpy()
order = np.lexsort(J.combos.T[::-1])     # match GAUSS row order if needed
print("max |python - gauss| joint prob:", np.max(np.abs(J.probs[order] - ate[:,4])))
print("python joint sum:", J.probs.sum())   # ~1 (MVNCD approx)
```

**GAUSS correspondence.** `MORP_WALK_ATE.gss` sets `changevar={female}`,
`changeval={1}`, evaluates `combprob[.,c]=exp(lpr1(est',dta))` for every combo and
writes `combo ~ meancombprob` to `ate1.csv`; it also prints the marginal
"level 3" probabilities (`happy3prob` …), which equal `J.marginal(d)[2]`.

**PASS.**
1. **Round‑trip (internal, exact):** for a fitted model,
   `from_estimates(reported betas, r.thresholds, r.correlation_matrix)` then
   `morp_ate` reproduces `morp_ate(r, …)` to `atol=1e-12`. (Test:
   `test_round_trip_marginal_ate_is_exact`.)
2. **vs GAUSS:** `J.probs` matches `ate1.csv` joint probabilities, and
   `J.marginal(d)` matches GAUSS's printed `*3prob` marginals, to ~`1e-2`
   (limited by the MVN‑CDF approximation each side uses — confirm both use the
   same `method`, see caveats).
3. **Confirm the scenario:** verify whether `ate1.csv` is the treatment
   (`Female=1`) or base scenario by matching against both; document which.

Automated (internal consistency): `pytest tests/test_models/test_morp.py -k FromEstimates`.

---

### A3. Faster non‑independent fit; standard errors unchanged (Anna issue #2)

**What changed.** The BHHH per‑observation score matrix (used for the default
reported SEs) was computed by central finite differences — `2·n_params`
full‑data MVNCD passes after every fit, the dominant post‑convergence cost. It is
now computed analytically in a **single pass** (reusing the analytic gradient's
per‑observation contributions) when `analytic_grad=True` and `method ∈ {me,ovus}`
(the defaults), falling back to finite differences otherwise. (Round‑1 already
made the 3‑estimator SE *diagnostic* opt‑in via `se_diagnostic`.)

**Python.** Time a non‑independent fit and check it converges to the same place:

```python
import time
t0 = time.time(); r = m.fit(); print("fit minutes:", (time.time()-t0)/60)
print("LL:", r.loglik, "se:", r.se)
```

**GAUSS correspondence.** Compare wall‑clock against GAUSS on the same machine
(Anna reported ~1.5 min GAUSS vs ~3.5–4 min Python on a laptop). Numbers won't
match exactly (interpreted Python vs compiled GAUSS, no GPU), but the gap should
narrow substantially after this change.

**PASS.**
1. **SEs unchanged** by the analytic‑score switch: the reported BHHH SEs equal
   the finite‑difference SEs to ~`1e-5` rel (test:
   `test_bhhh_se_unchanged`); analytic scores match FD scores to ~`1e-5`.
2. **Estimates/LL unchanged** vs the previous Python version and vs GAUSS
   (Part B anchors hold).
3. **Speed:** the `_per_obs_scores`/SE step is no longer a multi‑minute tail.
   (Reference: a 4‑dim `n=400` profiling fit went 34 s → 13 s; the score step
   11.3 s → 0.48 s.)

Automated: `pytest tests/test_models/test_morp.py -k AnalyticScores`.

---

## Part B — Regression anchors (must NOT change)

These are fixed targets; any drift here is a real regression. (From the project
`CLAUDE.md` verification list and the GAUSS Walk/Dining LLs from round‑1.)

| Model | Target |
|---|---|
| BHATLIB Table 1 (a)(i) IID | LL = **−670.956** |
| (a)(ii) Flexible | LL = **−661.111** |
| (b) + AGE45 | LL = **−659.285** |
| (c) Random coeff | LL = **−635.871** |
| (d) Mixture | LL ≈ **−634.975** (±2, local optima) |
| MORP Dining `iid=False` | LL = **−4.6599** (mean), correlation matrix matches GAUSS |
| MORP Walk `iid=False` | LL = **−3.7591** (mean), correlation matrix matches GAUSS |
| Threshold delta‑method SE | e.g. `thresh2` 0.0455 vs GAUSS 0.0454 |

Run: `pytest tests/ -q` (the slow MNP Table‑2 SE‑parity anchors are included in
the full run). Confirm the Dining/Walk `iid=False` mean LLs and printed
correlation matrices still match GAUSS.

---

## Part C — Adjacent scope Anna is likely to probe next

Pre‑emptive checklist — things that commonly differ between a reimplementation
and GAUSS, and that Anna may hit while exercising the executive‑course specs.

**C1. Per‑parameter table parity (beyond corr).**
For Dining and Walk, compare *every* row of the Python table to GAUSS:
`Estimates`, `Std. err.`, `Est./s.e.`, and the new `Gradient` column. Betas and
`thresh_*` were addressed in round‑1; `corr_*` in round‑2. Watch sign
conventions and the order of outcomes/coefficients.

**C2. `iid=True` (independent) models** (Anna's current `MORP_test1/2.py`).
These were the round‑1 focus; re‑confirm betas, `thresh_*` cut‑points, gradient
column, and LL match GAUSS now that the table machinery changed again.

**C3. MVN‑CDF `method`.** GAUSS Walk/Dining use a specific approximation
(`"ovus"`/`"me"`/`"tvbs"`…). Set the Python `MORPControl(method=...)` to the same
one before comparing probabilities/LL/ATE; small cross‑method differences are
expected and explain residual ATE/joint‑prob gaps (and why `J.probs` sums to
~1.000 not exactly 1).

**C4. Standard‑error estimator choice.** Python defaults to BHHH
(`se_method="bhhh"`) to match GAUSS. If Anna compares Hessian/sandwich SEs, she
must set `se_method=` and `se_diagnostic=True`; note the observed‑Hessian path is
still finite‑difference (slow) and opt‑in.

**C5. `fix_scales=False` (free latent scales).** Default is `True` (GAUSS
unit‑variance). If anyone sets `fix_scales=False`, the `scale_*` rows are still
reported in **raw (log) space** (only `thresh_*` and `corr_*` are transformed to
natural space). Flag if GAUSS‑style natural scale reporting is wanted — easy
follow‑up (`exp` transform + Jacobian), out of round‑2 scope.

**C6. ATE scenario semantics.** Confirm base‑vs‑treatment construction matches
GAUSS `changevar`/`changeval` for binary, multinomial, and count treatment
variables. Provide a small helper if Anna wants the GAUSS `changevar/changeval`
sugar rather than building `X_base`/`X_treat` herself.

**C7. Joint vs marginal ATE.** GAUSS reports the joint distribution (`ate1.csv`)
and derives marginals. `morp_joint_probs` gives both; `morp_ate` gives marginals
directly (correlation‑independent). Make sure Anna compares like with like.

**C8. Larger / unbalanced data and missing outcomes.** GAUSS supports
"outcome not available for some individuals" via availability columns
(`davordname`). Confirm Python handling of partially‑observed outcomes if her
executive‑course data uses it. (Possible gap — verify.)

**C9. Other models.** If she moves on to MNP / MDCEV: MNP just changed too
(PR #25 dropped the `2·atanh` kernel transform; PR #26 made Σscale²=1 the default
reporting normalisation). Re‑verify MNP Table‑2 betas/correlations/SEs against
GAUSS under the new normalisation (`published_beta == reported/scale01`).

---

## Appendix — numerical caveats

* **MVN‑CDF approximation.** Both GAUSS and Python use analytic MVNCD
  *approximations* (ME/OVUS/TVBS…). Probabilities, LL, and ATE agree only up to
  the method's accuracy; joint probabilities sum to ~1.000, not exactly 1. Always
  compare with the **same `method`** on both sides.
* **Local optima.** Mixture / random‑coefficient models have multiple optima;
  use the same start values / seed where comparing.
* **Parameterisation ≠ values.** `spherical` (radial vs spherical) and the
  threshold log‑increment encoding change the *raw optimiser* numbers but not the
  reported natural quantities (correlations, cut‑points) or the LL.
* **SEs are NaN for `from_estimates`.** Fixed user‑input coefficients carry no
  covariance, so `from_estimates` results report `NaN` SEs by design — that is
  expected, not a bug.
