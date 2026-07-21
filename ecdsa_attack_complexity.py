#!/usr/bin/env python3
"""
ECDSA two-predictor subset-attack complexity model
==================================================

Threat model
------------
A signature is considered CLEAN if and only if BOTH of the following hold:

  1. The ML predictor correctly recovers the first two decimal digits of the
     real nonce k  (accuracy = p_b).
  2. The magnitude predictor, given those (correct) leading digits, correctly
     recovers the decimal length of the real k  (accuracy = p_m).

Only a completely clean subset of size m_req can be safely fed into the
lattice reduction.  Any single poisoned signature destroys the reduction.

This script evaluates the combinatorial cost of obtaining such a clean subset
under the two-predictor model.

Primary parameters (all have CLI defaults; override as needed)
-------------------------------------------------------------
  --curve-bits            (default 256)
  --leak-bits             (default [6])
  --cores                 (default [1, 8, 64, 124])
  --t-per-run-sec         (default 2.0 s -- a flat placeholder; calibrate it)
  --p-ml-start / --p-ml-stop / --p-ml-step
  --p-mag                 (default 90.0)

Everything else is calculated automatically. Note that --t-per-run-sec is NOT
measured here: unless you pass a calibrated value, the reported wall-clock
times rest on a flat 2.0 s-per-reduction placeholder and are order-of-magnitude
only.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from dataclasses import dataclass, field
from math import ceil, lgamma, log, log10
from typing import Sequence


# ---------------------------------------------------------------------------
# Pure mathematics
# ---------------------------------------------------------------------------

def log10_binom(n: int, k: int) -> float:
    """log10(C(n, k)). Returns -inf when the binomial is undefined."""
    if k < 0 or k > n:
        return float("-inf")
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(10)


def joint_clean_probability(p_b: float, p_m: float, rho: float = 0.0) -> float:
    """
    Probability that a signature is clean.

    Primary model (rho = 0):
        p_clean = p_b * p_m
        where p_m is the accuracy of the magnitude predictor
        conditioned on the leading digits being correct.

    Optional correlation (rho ≠ 0) is kept for future experiments.
    """
    base = p_b * p_m
    if abs(rho) < 1e-12:
        return base

    cov = rho * math.sqrt(p_b * (1.0 - p_b) * p_m * (1.0 - p_m))
    joint = base + cov
    lo = max(0.0, p_b + p_m - 1.0)
    hi = min(p_b, p_m)
    return max(lo, min(hi, joint))


def required_clean_signatures(curve_bits: int, leak_bits: int,
                              safety_factor: float) -> int:
    """Information-theoretic lower bound with practical safety margin."""
    return ceil(ceil(curve_bits / leak_bits) * safety_factor)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Config:
    curve_bits: int
    leak_bits: int
    safety_factor: float
    surplus: float
    p_mag: float
    correlation: float
    acc_start: float
    acc_stop: float
    acc_step: float
    cores: list[int]
    t_per_run_sec: float
    dim_overhead: int
    filter_pass_rate: float


@dataclass
class Row:
    leak_bits: int
    p_ml_pct: float
    p_clean: float
    m_req: int
    dimension: int
    t_filtered: int
    t_raw: int
    clean: int
    log10_runs: float | None
    t_per_run_sec: float
    times: dict[int, float | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_LOG10_SEC_PER_YEAR = log10(31_536_000.0)


def fmt_count(log10_x: float | None) -> str:
    if log10_x is None or log10_x == float("-inf"):
        return "n/a"
    if log10_x < 4.0:
        return f"{10 ** log10_x:,.0f}"
    return f"1e{log10_x:.2f}"


def fmt_duration(log10_sec: float | None) -> str:
    if log10_sec is None or log10_sec == float("-inf"):
        return "n/a"
    if log10_sec > 30:
        return f"1e{log10_sec - _LOG10_SEC_PER_YEAR:.1f} yr"
    sec = 10.0 ** log10_sec
    if sec < 60:
        return f"{sec:.2f} s"
    if sec < 3600:
        return f"{sec / 60:.2f} min"
    if sec < 86400:
        return f"{sec / 3600:.2f} h"
    if sec < 2_592_000:
        return f"{sec / 86400:.2f} d"
    if sec < 31_536_000:
        return f"{sec / 2_592_000:.2f} mo"
    years = sec / 31_536_000
    if years < 1e6:
        return f"{years:.2f} yr"
    return f"1e{log10(years):.1f} yr"


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def accuracy_sweep(start: float, stop: float, step: float) -> list[float]:
    values = []
    cur = start
    while cur <= stop + 1e-9:
        values.append(round(cur, 6))
        cur += step
    return values


def build_rows(cfg: Config) -> list[Row]:
    m_req = required_clean_signatures(cfg.curve_bits, cfg.leak_bits, cfg.safety_factor)
    dim = m_req + cfg.dim_overhead
    p_m = cfg.p_mag / 100.0
    log10_t = log10(cfg.t_per_run_sec)

    rows: list[Row] = []
    for pct in accuracy_sweep(cfg.acc_start, cfg.acc_stop, cfg.acc_step):
        p_b = pct / 100.0
        p_clean = joint_clean_probability(p_b, p_m, cfg.correlation)

        if p_clean <= 0.0:
            continue

        t_filt = ceil(cfg.surplus * m_req / p_clean)
        t_raw = ceil(t_filt / cfg.filter_pass_rate)
        clean = round(p_clean * t_filt)

        if clean < m_req:
            log_runs = None
            times = {c: None for c in cfg.cores}
        else:
            log_runs = log10_binom(t_filt, m_req) - log10_binom(clean, m_req)
            times = {
                c: log_runs + log10_t - log10(c)
                for c in cfg.cores
            }

        rows.append(Row(
            leak_bits=cfg.leak_bits,
            p_ml_pct=pct,
            p_clean=p_clean,
            m_req=m_req,
            dimension=dim,
            t_filtered=t_filt,
            t_raw=t_raw,
            clean=clean,
            log10_runs=log_runs,
            t_per_run_sec=cfg.t_per_run_sec,
            times=times,
        ))
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_legend(cfg: Config) -> None:
    bar = "=" * 96
    print(bar)
    print("ECDSA two-predictor subset-attack complexity")
    print(bar)
    print()
    print("Definition of a CLEAN signature:")
    print("  • ML predictor correctly recovers the first two decimal digits of k")
    print("  • Magnitude predictor (given those correct digits) recovers the true")
    print("    decimal length of k")
    print()
    print(f"  p_b  = ML accuracy          (swept {cfg.acc_start:g} … {cfg.acc_stop:g} %)")
    print(f"  p_m  = magnitude accuracy   ({cfg.p_mag:.2f} %)  [conditioned on correct digits]")
    print(f"  p_clean = p_b * p_m         (correlation p = {cfg.correlation:+.2f})")
    print()
    print("Combinatorial model:")
    print("  m_req   = ceil(ceil(curve_bits / L) * safety_factor)")
    print("  T_filt  = ceil(surplus * m_req / p_clean)")
    print("  clean   = round(p_clean * T_filt)")
    print("  runs    = C(T_filt, m_req) / C(clean, m_req)")
    print("  time    = runs * t_per_run_sec / cores")
    print()
    print(f"curve_bits     = {cfg.curve_bits}")
    print(f"leak_bits (L)  = {cfg.leak_bits}")
    print(f"safety_factor  = {cfg.safety_factor}")
    print(f"surplus        = {cfg.surplus}")
    print(f"t_per_run_sec  = {cfg.t_per_run_sec:.4f} s   "
          f"(calibrate per hardware; pass --t-per-run-sec)")
    print(f"cores          = {cfg.cores}")
    print(bar)


def print_table(rows: list[Row], cores: Sequence[int]) -> None:
    if not rows:
        print("  (no rows)")
        return

    head = rows[0]
    print()
    print(f"  L = {head.leak_bits} bit   |   m_req = {head.m_req}   |   "
          f"dim = {head.dimension}   |   t/run = {head.t_per_run_sec:.4f} s")
    print()

    # Header
    cols = ["p_ml%", "p_clean", "T_filt", "T_raw", "clean", "runs"]
    cols += [f"t@{c}c" for c in cores]
    widths = [7, 9, 9, 9, 8, 12] + [14] * len(cores)

    print("  " + "".join(h.rjust(w) for h, w in zip(cols, widths)))
    print("  " + "-" * sum(widths))

    for r in rows:
        cells = [
            f"{r.p_ml_pct:g}",
            f"{r.p_clean:.5f}",
            f"{r.t_filtered}",
            f"{r.t_raw}",
            f"{r.clean}",
            fmt_count(r.log10_runs),
        ]
        for c in cores:
            cells.append(fmt_duration(r.times.get(c)))
        print("  " + "".join(c.rjust(w) for c, w in zip(cells, widths)))

    print("  " + "-" * sum(widths))


def write_csv(all_rows: list[Row], cores: Sequence[int], path: str) -> None:
    fieldnames = [
        "leak_bits", "p_ml_pct", "p_clean", "m_req", "dimension",
        "t_filtered", "t_raw", "clean", "log10_runs", "expected_runs",
        "t_per_run_sec",
    ]
    for c in cores:
        fieldnames += [f"t_{c}cores_human", f"log10_seconds_{c}cores"]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in all_rows:
            row = [
                r.leak_bits, r.p_ml_pct, f"{r.p_clean:.6f}",
                r.m_req, r.dimension,
                r.t_filtered, r.t_raw, r.clean,
                "" if r.log10_runs is None else f"{r.log10_runs:.6f}",
                fmt_count(r.log10_runs),
                f"{r.t_per_run_sec:.6f}",
            ]
            for c in cores:
                lsec = r.times.get(c)
                row.append(fmt_duration(lsec))
                row.append("" if lsec is None else f"{lsec:.6f}")
            w.writerow(row)
    print(f"\n  [CSV written → {path}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ECDSA two-predictor subset-attack complexity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required / primary parameters (exactly as requested) ---
    g = p.add_argument_group("primary parameters")
    g.add_argument("--curve-bits", type=int, default=256,
                   help="bit length of the curve order n")
    g.add_argument("--leak-bits", type=int, nargs="+", default=[6],
                   help="number of leaked / predicted bits L (one table per value)")
    g.add_argument("--cores", type=int, nargs="+", default=[1, 8, 64, 124],
                   help="core counts for which wall-clock time is reported")
    g.add_argument("--t-per-run-sec", type=float, default=2.0,
                   help="calibrated wall-clock seconds of one lattice reduction")
    g.add_argument("--p-ml-start", type=float, default=80.0)
    g.add_argument("--p-ml-stop", type=float, default=99.0)
    g.add_argument("--p-ml-step", type=float, default=1.0)
    g.add_argument("--p-mag", type=float, default=82.0,
                   help="magnitude predictor accuracy (%%) conditioned on correct leading digits")

    # --- Secondary parameters with safe defaults ---
    s = p.add_argument_group("secondary parameters (safe defaults)")
    s.add_argument("--safety-factor", type=float, default=1.3)
    s.add_argument("--surplus", type=float, default=2.0)
    s.add_argument("--dim-overhead", type=int, default=2)
    s.add_argument("--correlation", type=float, default=0.0,
                   help="correlation p between the two predictors (-1…+1)")
    s.add_argument("--filter-pass-rate", type=float, default=1.0,
                   help="fraction of signatures that pass the leading-digit pre-filter")

    # --- Output ---
    o = p.add_argument_group("output")
    o.add_argument("--csv", type=str, default=None)
    o.add_argument("--no-legend", action="store_true")

    args = p.parse_args(argv)

    # Validation
    if args.curve_bits <= 0:
        p.error("--curve-bits must be positive")
    if any(L <= 0 for L in args.leak_bits):
        p.error("--leak-bits values must be positive")
    if any(c <= 0 for c in args.cores):
        p.error("--cores values must be positive")
    if args.t_per_run_sec <= 0:
        p.error("--t-per-run-sec must be positive")
    if not (0.0 < args.p_ml_start <= args.p_ml_stop <= 100.0):
        p.error("require 0 < --p-ml-start ≤ --p-ml-stop ≤ 100")
    if args.p_ml_step <= 0:
        p.error("--p-ml-step must be positive")
    if not (0.0 < args.p_mag <= 100.0):
        p.error("--p-mag must be in (0, 100]")
    if not (-1.0 <= args.correlation <= 1.0):
        p.error("--correlation must be in [-1, 1]")
    if args.safety_factor <= 0 or args.surplus <= 0:
        p.error("--safety-factor and --surplus must be positive")
    if args.dim_overhead < 0:
        p.error("--dim-overhead must be >= 0 (a negative overhead makes the "
                "lattice dimension smaller than m_req)")
    if not (0.0 < args.filter_pass_rate <= 1.0):
        p.error("--filter-pass-rate must be in (0, 1]")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    base = Config(
        curve_bits=args.curve_bits,
        leak_bits=args.leak_bits[0],
        safety_factor=args.safety_factor,
        surplus=args.surplus,
        p_mag=args.p_mag,
        correlation=args.correlation,
        acc_start=args.p_ml_start,
        acc_stop=args.p_ml_stop,
        acc_step=args.p_ml_step,
        cores=args.cores,
        t_per_run_sec=args.t_per_run_sec,
        dim_overhead=args.dim_overhead,
        filter_pass_rate=args.filter_pass_rate,
    )

    if not args.no_legend:
        print_legend(base)

    all_rows: list[Row] = []
    for L in args.leak_bits:
        cfg = dataclasses.replace(base, leak_bits=L)
        rows = build_rows(cfg)
        print_table(rows, cfg.cores)
        all_rows.extend(rows)

    if args.csv:
        write_csv(all_rows, args.cores, args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())