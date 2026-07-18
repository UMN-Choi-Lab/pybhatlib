#!/usr/bin/env python
"""Micro-benchmark harness for the mixed MORPFlex & MDCEVMixed engines.

Reconstructs each engine at its converged GAUSS fixture *exactly* as the parity
tests do (``tests/test_mixed/test_morp_flex_gauss_parity.py`` and
``test_mdcev_mixed_gauss_parity.py``), using ``FixtureDrawSource`` on the fixture
``ass``/``b``, then for each model times the **median of 9 warm evals** of
``estimator.simulated_loglik(b, want_grad=False)`` and ``want_grad=True``.

Prints a table (model, ll_ms, grad_ms, ll_value, grad_norm) and the ratio to the
recorded baselines, and appends one CSV row per run to
``bench/mixed_perf_log.csv``.

Run::

    .venv/bin/python bench/bench_mixed.py [opt_id]

``opt_id`` defaults to ``baseline`` and is written into the CSV log so each
optimization's delta is tracked alongside any parity drift in ``ll_val``.

This is a pure timing harness (plan MIXED_PERF_PLAN.md section 5). It changes no
numerics; the ``ll_val`` it prints IS the parity anchor and must stay at
MORPFlex -7400.62680 / MDCEVMixed -14979.909828528662.
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

from pybhatlib.mixed._draws import FixtureDrawSource  # noqa: E402
from pybhatlib.vecup._panel import PanelIndex  # noqa: E402

# --- Recorded baselines (plan section, median of warm evals) ------------------
_BASELINES = {
    "MORPFlex": {"ll_ms": 5043.0, "grad_ms": 8106.0, "ll_val": -7400.626799592934},
    "MDCEVMixed": {"ll_ms": 1183.0, "grad_ms": 5018.0, "ll_val": -14979.9098285287},
}

_N_WARM = 9        # median over this many warm evals
_N_WARMUP = 1      # untimed warmup eval(s) to prime caches / JIT
_CSV_LOG = os.path.join(_HERE, "mixed_perf_log.csv")


# =============================================================================
# MORPFlex reconstruction (mirrors test_morp_flex_gauss_parity.py)
# =============================================================================
_MORP_FIX = os.path.join(_REPO, "tests", "fixtures", "mixed", "morp", "panelcommute_converged")
_MORP_DATA = os.path.join(_REPO, "0714_UTA_request", "panel_commute_dataset_with_exog.csv")

_MORP_SPEC = {
    "UNO_M":  {"M_stop_b": "UNO",          "E_stop_b": "SERO"},
    "UNO_E":  {"M_stop_b": "SERO",         "E_stop_b": "UNO"},
    "AGE65M": {"M_stop_b": "age_above_64", "E_stop_b": "SERO"},
    "AGE45E": {"M_stop_b": "SERO",         "E_stop_b": "age_45_64"},
    "AGE65E": {"M_stop_b": "SERO",         "E_stop_b": "age_above_64"},
    "BLACKM": {"M_stop_b": "black_only",   "E_stop_b": "SERO"},
    "CHILD":  {"M_stop_b": "pres_child",   "E_stop_b": "pres_child"},
    "HYBRID": {"M_stop_b": "hybrid",       "E_stop_b": "hybrid"},
    "TT":     {"M_stop_b": "M_TT",         "E_stop_b": "E_TT"},
}


def _build_morp():
    import pandas as pd

    from pybhatlib.models.morp_flex import MORPFlexControl, MORPFlexModel

    df = pd.read_csv(_MORP_DATA, low_memory=False)
    df = df.sort_values("INDID", kind="mergesort").reset_index(drop=True)

    ctrl = MORPFlexControl(
        person_id="INDID",
        normvar=("UNO_M", "UNO_E"),
        copula=False,
        yj_kernel=False,
        n_rep=10,
        spher=False,
        scal=1.0,
        floor_pcomp=0.0,
        floor_z=0.0,
    )
    model = MORPFlexModel(
        data=df,
        dep_vars=["M_stop_b", "E_stop_b"],
        spec=_MORP_SPEC,
        n_categories=[2, 2],
        control=ctrl,
    )
    spec, layout = model._build_spec_layout()
    panel = PanelIndex.from_ids(model.person_ids)
    draws = FixtureDrawSource(os.path.join(_MORP_FIX, "ass.csv"))
    est = model._build_estimator(spec, layout, panel, draws=draws)
    b = np.loadtxt(os.path.join(_MORP_FIX, "b.csv"))
    return est, b


# =============================================================================
# MDCEVMixed reconstruction (mirrors test_mdcev_mixed_gauss_parity.py)
# =============================================================================
_MDCEV_FIX = os.path.join(_REPO, "tests", "fixtures", "mixed", "mdcev", "scag_converged")
_MDCEV_DATA_CANDIDATES = [
    os.path.join(_REPO, "0714_UTA_request", "Workshop_SCAG_Est.csv"),
    os.path.join(_MDCEV_FIX, "Workshop_SCAG_Est.csv"),
]

_MDCEV_IVMT = np.array(
    [
        ["sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero"],
        ["uno",  "sero", "sero", "sero", "sero", "gend1", "sero", "sero", "Lin",  "sero", "sero", "sero"],
        ["sero", "uno",  "sero", "sero", "sero", "sero", "gend1", "sero", "sero", "Lin",  "sero", "sero"],
        ["sero", "sero", "uno",  "sero", "sero", "sero", "sero", "gend1", "sero", "sero", "sero", "sero"],
        ["sero", "sero", "sero", "uno",  "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero"],
        ["sero", "sero", "sero", "sero", "uno",  "sero", "sero", "sero", "sero", "sero", "Lin",  "Min"],
    ],
    dtype=str,
)
_MDCEV_VARNAM = [
    "ASC_Esc", "ASC_ho", "ASC_Soc", "ASC_AR", "ASC_Eo",
    "male_Esc", "male_ho", "male_Soc", "Lin_Esc", "Lin_ho", "Lin_Eo", "Min_Eo",
]
_MDCEV_IVGT = np.array(
    [
        ["uno",  "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero"],
        ["sero", "uno",  "sero", "sero", "sero", "sero", "sero", "sero", "sero"],
        ["sero", "sero", "uno",  "sero", "sero", "sero", "gend1", "sero", "sero"],
        ["sero", "sero", "sero", "uno",  "sero", "sero", "sero", "gend1", "sero"],
        ["sero", "sero", "sero", "sero", "uno",  "sero", "sero", "sero", "sero"],
        ["sero", "sero", "sero", "sero", "sero", "uno",  "sero", "sero", "gend1"],
    ],
    dtype=str,
)
_MDCEV_VARNGAM = [
    "G_Out", "G_Esc", "G_ho", "G_Soc", "G_AR", "G_Eo",
    "male_ho", "male_Soc", "male_Eo",
]
_MDCEV_ALTS = ["alt_out", "Esc", "Ho", "Soc", "AR", "Eo"]


def _mdcev_data_path() -> str:
    for p in _MDCEV_DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Workshop_SCAG_Est.csv not found (searched {_MDCEV_DATA_CANDIDATES})"
    )


def _build_mdcev():
    from pybhatlib.models.mdcev_mixed._mdcev_mixed_control import MDCEVMixedControl
    from pybhatlib.models.mdcev_mixed._mdcev_mixed_model import MDCEVMixedModel

    ctrl = MDCEVMixedControl(
        utility="trad",
        logvar=("male_ho",),
        yjvar=("male_Soc",),
        varneg=("male_ho",),
        n_rep=10,
        intordn1=20,
        floor_pcomp=0.0,
        floor_z=0.0,
        verbose=0,
    )
    model = MDCEVMixedModel(
        data=_mdcev_data_path(),
        alternatives=_MDCEV_ALTS,
        price=["uno"] * len(_MDCEV_ALTS),
        utility_spec=_MDCEV_IVMT,
        gamma_spec=_MDCEV_IVGT,
        param_names=_MDCEV_VARNAM,
        gamma_names=_MDCEV_VARNGAM,
        control=ctrl,
        obs_id_var="ID",
    )
    spec, layout = model._build_spec_layout()
    panel = PanelIndex.from_ids(model.person_ids)
    draws = FixtureDrawSource(os.path.join(_MDCEV_FIX, "ass.csv"))
    est = model.build_estimator(spec, layout, panel, draws=draws)
    b = np.loadtxt(os.path.join(_MDCEV_FIX, "b.csv"))
    return est, b


# =============================================================================
# Timing
# =============================================================================
def _time_median(fn, n_warmup: int, n_warm: int) -> tuple[float, object]:
    """Run ``fn`` ``n_warmup`` untimed times, then time ``n_warm`` and return
    (median_ms, last_result)."""
    result = None
    for _ in range(n_warmup):
        result = fn()
    samples = []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), result


def _grad_norm(score) -> float:
    """L2 norm of the total (summed-over-individuals) analytic score vector."""
    score = np.asarray(score, dtype=np.float64)
    total = score.sum(axis=0) if score.ndim == 2 else score
    return float(np.linalg.norm(total))


def _bench_one(name: str, est, b):
    ll_ms, ll_res = _time_median(
        lambda: est.simulated_loglik(b, want_grad=False), _N_WARMUP, _N_WARM
    )
    grad_ms, grad_res = _time_median(
        lambda: est.simulated_loglik(b, want_grad=True), _N_WARMUP, _N_WARM
    )
    ll_val = float(np.asarray(ll_res[0]).sum())
    grad_norm = _grad_norm(grad_res[1])
    return {
        "model": name,
        "ll_ms": ll_ms,
        "grad_ms": grad_ms,
        "ll_val": ll_val,
        "grad_norm": grad_norm,
    }


# =============================================================================
# Reporting
# =============================================================================
def _print_table(rows):
    header = (
        f"{'model':<12} {'ll_ms':>10} {'grad_ms':>10} "
        f"{'ll_value':>20} {'grad_norm':>14}  {'ll_x':>7} {'grad_x':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        base = _BASELINES.get(r["model"], {})
        ll_x = r["ll_ms"] / base["ll_ms"] if base.get("ll_ms") else float("nan")
        grad_x = r["grad_ms"] / base["grad_ms"] if base.get("grad_ms") else float("nan")
        print(
            f"{r['model']:<12} {r['ll_ms']:>10.1f} {r['grad_ms']:>10.1f} "
            f"{r['ll_val']:>20.9f} {r['grad_norm']:>14.6f}  "
            f"{ll_x:>6.2f}x {grad_x:>6.2f}x"
        )
    print()
    print("Baselines (recorded): "
          "MORPFlex 5043/8106 ms, MDCEVMixed 1183/5018 ms  (ll_ms/grad_ms)")


def _append_csv(opt_id: str, rows):
    new_file = not os.path.exists(_CSV_LOG)
    os.makedirs(os.path.dirname(_CSV_LOG), exist_ok=True)
    with open(_CSV_LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["opt_id", "model", "ll_ms", "grad_ms", "ll_val", "grad_norm", "timestamp"])
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for r in rows:
            w.writerow([
                opt_id, r["model"],
                f"{r['ll_ms']:.3f}", f"{r['grad_ms']:.3f}",
                f"{r['ll_val']:.12f}", f"{r['grad_norm']:.9f}", ts,
            ])


def main(argv):
    opt_id = argv[1] if len(argv) > 1 else "baseline"
    print(f"[bench_mixed] opt_id={opt_id}  warm evals={_N_WARM} (median), warmup={_N_WARMUP}\n")

    rows = []

    print("[bench_mixed] building MDCEVMixed engine at scag_converged fixture ...")
    est, b = _build_mdcev()
    rows.append(_bench_one("MDCEVMixed", est, b))

    print("[bench_mixed] building MORPFlex engine at panelcommute_converged fixture ...")
    est, b = _build_morp()
    rows.append(_bench_one("MORPFlex", est, b))

    print()
    _print_table(rows)
    _append_csv(opt_id, rows)
    print(f"\n[bench_mixed] appended {len(rows)} row(s) to {_CSV_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
