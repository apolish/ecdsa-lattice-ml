#!/usr/bin/env python3
"""Two-filter subset-attack complexity for an HNP lattice attack on ECDSA.

THREAT MODEL
============
A signature is "clean" iff BOTH predictors are correct:
  (1) ML predictor recovers top L bits of nonce k       -- accuracy p_b (swept)
  (2) Magnitude predictor recovers decimal length of k  -- accuracy p_m (fixed)
Both must hit; otherwise the interval bound the lattice receives for k is wrong
and the reduction yields nothing.

Optional pre-filter on the predicted leading two decimal digits of k:
  keep only signatures whose predicted leading pair lies in some range.
  --filter-pass-rate is the fraction of raw signatures the filter passes.
  Example: filter [32, 99] on secp256k1 keeps ~75.6% and enforces L >= 6.

JOINT PROBABILITY
=================
    p_clean = p_b * p_m + rho * sqrt(p_b*(1-p_b) * p_m*(1-p_m))
clamped to [max(0, p_b+p_m-1), min(p_b, p_m)].
  rho = 0  -> independence  (default; conservative for the attacker)
  rho > 0  -> realistic: predictors fail together more often than chance
  rho < 0  -> exotic

CORE FORMULAS (per row)
=======================
    m_req      = ceil(ceil(curve_bits / L) * safety_factor)
    T_filt     = ceil(surplus * m_req / p_clean)        -- usable pool
    T_raw      = ceil(T_filt / pass_rate)               -- raw signatures to collect
    clean      = round(p_clean * T_filt)
    runs       = C(T_filt, m_req) / C(clean, m_req)
    time/core  = runs * t_per_run_sec / cores           -- embarrassingly parallel

PARALLELIZATION
===============
Each lattice reduction is independent: different m_req-subsets of T_filt are
tried on different cores with no inter-core communication. Time scales LINEARLY
in --cores: the formula is mathematically tight, not optimistic. Caps come only
from memory per worker (trivial at dim ~58) and process setup latency
(negligible for multi-minute batches). Default --cores = [1] is the single-core
baseline; pass several core counts to get one column each.

CALIBRATION OF t_per_run_sec
============================
This is the single most important number in the model. MUST be measured on YOUR
hardware and solver. Workflow:
  1) Build a synthetic all-clean HNP instance of dimension m_req + 2.
  2) Run fpylll LLL.reduction or BKZ.reduction(block_size=beta) once.
  3) Take wall-clock seconds; pass via --t-per-run-sec.
Without --t-per-run-sec the script falls back to a dim-dependent heuristic that
is order-of-magnitude only.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from dataclasses import dataclass, field
from math import ceil, lgamma, log, log10


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------
def log10_binom(n: int, k: int) -> float:
    """log10 of binomial C(n, k); -inf when out of range."""
    if k < 0 or k > n:
        return float("-inf")
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(10)


def joint_clean_prob(p_b: float, p_m: float, rho: float) -> float:
    """P(both predictors correct), with correlation hyperparameter rho."""
    base = p_b * p_m
    cov = rho * (p_b * (1.0 - p_b) * p_m * (1.0 - p_m)) ** 0.5
    joint = base + cov
    lo = max(0.0, p_b + p_m - 1.0)
    hi = min(p_b, p_m)
    return max(lo, min(hi, joint))


def required_clean_signatures(curve_bits: int, leak_bits: int,
                              safety_factor: float, override: int | None) -> int:
    if override is not None:
        return override
    return ceil(ceil(curve_bits / leak_bits) * safety_factor)


def t_per_run_heuristic(dim: int, base_sec: float, growth: float, ref_dim: int) -> float:
    """Order-of-magnitude time-per-reduction. Calibrate before trusting."""
    return base_sec * (10.0 ** (growth * (dim - ref_dim) / log(10)))


def lattice_type_label(dim: int) -> str:
    if dim <= 50:
        return "LLL / BKZ-20"
    if dim <= 80:
        return "BKZ-30"
    return "BKZ-40+"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Config:
    curve_bits: int
    leak_bits: int
    m_req_override: int | None
    safety_factor: float
    surplus: float
    filter_pass_rate: float
    p_mag: float
    correlation: float
    acc_start: float
    acc_stop: float
    acc_step: float
    cores: list[int]
    t_per_run_sec: float | None
    dim_overhead: int
    run_base_sec: float
    run_growth: float
    run_ref_dim: int


@dataclass
class Row:
    leak_bits: int
    accuracy_pct: float
    p_clean: float
    m_req: int
    dimension: int
    lattice_type: str
    t_filtered: int
    t_raw: int
    clean: int
    log10_combinations: float
    log10_runs: float | None
    t_per_run_sec: float
    cores_times: dict[int, float | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt_count(log10_x: float | None) -> str:
    if log10_x is None or log10_x == float("-inf"):
        return "n/a"
    if log10_x < 4:
        return f"{10 ** log10_x:,.0f}"
    return f"1e{log10_x:.2f}"


_LOG10_SEC_PER_YEAR = log10(31_536_000.0)  # ~7.4988


def fmt_duration(log10_sec: float | None) -> str:
    # FIX B4: the original returned f"1e{log10_sec:.1f} yr" using log10 of
    # SECONDS while labelling it "yr" -- a 7.5-order-of-magnitude error
    # (log10 seconds per year ~= 7.499). It never surfaced only because every
    # value in the shipped table sat below the >30 branch. Convert to years.
    if log10_sec is None or log10_sec == float("-inf"):
        return "n/a"
    if log10_sec > 30:
        log10_years = log10_sec - _LOG10_SEC_PER_YEAR
        return f"1e{log10_years:.1f} yr"
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
# Row construction
# ---------------------------------------------------------------------------
def accuracy_sweep(start: float, stop: float, step: float) -> list[float]:
    out: list[float] = []
    cur = start
    while cur <= stop + 1e-9:
        out.append(round(cur, 6))
        cur += step
    return out


def build_rows(cfg: Config) -> list[Row]:
    m_req = required_clean_signatures(cfg.curve_bits, cfg.leak_bits,
                                       cfg.safety_factor, cfg.m_req_override)
    dim = m_req + cfg.dim_overhead
    lat_type = lattice_type_label(dim)
    if cfg.t_per_run_sec is not None:
        t_per_run = cfg.t_per_run_sec
    else:
        t_per_run = t_per_run_heuristic(dim, cfg.run_base_sec, cfg.run_growth, cfg.run_ref_dim)
    log10_t_per_run = log10(t_per_run)
    p_m = cfg.p_mag / 100.0

    rows: list[Row] = []
    for pct in accuracy_sweep(cfg.acc_start, cfg.acc_stop, cfg.acc_step):
        p_b = pct / 100.0
        p_clean = joint_clean_prob(p_b, p_m, cfg.correlation)
        if p_clean <= 0:
            continue
        t_filt = ceil(cfg.surplus * m_req / p_clean)
        t_raw = ceil(t_filt / cfg.filter_pass_rate)
        clean = round(p_clean * t_filt)
        log_CT = log10_binom(t_filt, m_req)
        if clean < m_req:
            log_runs: float | None = None
            times = {c: None for c in cfg.cores}
        else:
            log_runs = log_CT - log10_binom(clean, m_req)
            times = {c: log_runs + log10_t_per_run - log10(c) for c in cfg.cores}
        rows.append(Row(
            leak_bits=cfg.leak_bits, accuracy_pct=pct, p_clean=p_clean,
            m_req=m_req, dimension=dim, lattice_type=lat_type,
            t_filtered=t_filt, t_raw=t_raw, clean=clean,
            log10_combinations=log_CT, log10_runs=log_runs,
            t_per_run_sec=t_per_run, cores_times=times,
        ))
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_legend(cfg: Config) -> None:
    bar = "=" * 100
    print(bar)
    print("ECDSA two-filter subset-attack complexity  (order-of-magnitude unless calibrated)")
    print(bar)
    print("Both predictors must be correct per signature for it to be 'clean':")
    print(f"  ML  (top L bits)         p_b swept   ({cfg.acc_start:g}..{cfg.acc_stop:g}%, "
          f"step {cfg.acc_step:g})")
    print(f"  magnitude (digit count)  p_m = {cfg.p_mag:g}% (fixed)")
    print(f"  correlation rho          = {cfg.correlation:+.2f}")
    print()
    print(f"Filter: leading-pair pass rate = {cfg.filter_pass_rate:.4f}")
    if cfg.filter_pass_rate < 1.0 - 1e-9:
        print("  T_raw = T_filt / pass_rate   (raw signatures to collect from the wild)")
    print()
    print("Per row:")
    print("  m_req       = ceil(ceil(curve_bits / L) * safety_factor)")
    print("  T_filt      = ceil(surplus * m_req / p_clean)")
    print("  T_raw       = ceil(T_filt / pass_rate)")
    print("  clean       = round(p_clean * T_filt)")
    print("  runs        = C(T_filt, m_req) / C(clean, m_req)")
    print("  time@Nc     = runs * t_per_run_sec / N      (embarrassingly parallel)")
    print()
    if cfg.t_per_run_sec is not None:
        print(f"t/run         = {cfg.t_per_run_sec:.3f} s  (USER-PROVIDED via --t-per-run-sec)")
    else:
        print(f"t/run         = heuristic  (base {cfg.run_base_sec:g}s @ dim {cfg.run_ref_dim}, "
              f"growth {cfg.run_growth:g})  -- CALIBRATE before trusting numbers")
    print(f"cores         = {cfg.cores}")
    print(f"curve         = {cfg.curve_bits}-bit  (e.g. 256 for secp256k1)")
    print(bar)


def print_table(rows: list[Row], cfg: Config) -> None:
    if not rows:
        return
    head = rows[0]
    print()
    print(f"  L = {head.leak_bits} bit   m_req = {head.m_req}   dim = {head.dimension}   "
          f"solver = {head.lattice_type}   t/run = {head.t_per_run_sec:.3f} s")
    static = ["p_ml%", "p_clean", "T_filt", "T_raw", "clean", "C(T,m)", "runs"]
    core_cols = [f"t@{c}c" for c in cfg.cores]
    headers = static + core_cols
    widths = [7, 9, 8, 8, 7, 11, 11] + [14] * len(core_cols)
    total_w = sum(widths)
    print("  " + "-" * total_w)
    print("  " + "".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  " + "-" * total_w)
    for r in rows:
        cells = [f"{r.accuracy_pct:g}", f"{r.p_clean:.4f}",
                 str(r.t_filtered), str(r.t_raw), str(r.clean),
                 fmt_count(r.log10_combinations), fmt_count(r.log10_runs)]
        for c in cfg.cores:
            cells.append(fmt_duration(r.cores_times.get(c)))
        print("  " + "".join(c.rjust(w) for c, w in zip(cells, widths)))
    print("  " + "-" * total_w)


def write_csv(all_rows: list[Row], cfg: Config, path: str) -> None:
    core_cols: list[str] = []
    for c in cfg.cores:
        core_cols += [f"t_{c}cores_human", f"log10_seconds_{c}cores"]
    fieldnames = ["leak_bits", "accuracy_pct", "p_clean", "m_req", "dimension",
                  "lattice_type", "t_filtered", "t_raw", "clean",
                  "log10_combinations", "log10_runs", "expected_runs",
                  "t_per_run_sec"] + core_cols
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(fieldnames)
        for r in all_rows:
            row: list = [r.leak_bits, r.accuracy_pct, f"{r.p_clean:.6f}",
                         r.m_req, r.dimension, r.lattice_type,
                         r.t_filtered, r.t_raw, r.clean,
                         f"{r.log10_combinations:.6f}",
                         "" if r.log10_runs is None else f"{r.log10_runs:.6f}",
                         fmt_count(r.log10_runs), f"{r.t_per_run_sec:.6f}"]
            for c in cfg.cores:
                lsec = r.cores_times.get(c)
                row.append(fmt_duration(lsec))
                row.append("" if lsec is None else f"{lsec:.6f}")
            w.writerow(row)
    print(f"  [CSV written to {path}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-filter HNP attack complexity for ECDSA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g1 = p.add_argument_group("problem parameters")
    g1.add_argument("--curve-bits", type=int, default=256,
                    help="bit length of the curve order n (256 for secp256k1)")
    g1.add_argument("--leak-bits", type=int, nargs="+", default=[6],
                    help="leak L per nonce; pass several to get one table each")
    g1.add_argument("--m-req", type=int, default=None,
                    help="override clean-signatures requirement; else computed")
    g1.add_argument("--safety-factor", type=float, default=1.3,
                    help="multiplier on the information floor to get practical m_req")
    g1.add_argument("--surplus", type=float, default=2.0,
                    help="usable pool size as a multiple of m_req")

    g2 = p.add_argument_group("predictor accuracies")
    g2.add_argument("--p-ml-start", type=float, default=80.0)
    g2.add_argument("--p-ml-stop", type=float, default=99.0)
    g2.add_argument("--p-ml-step", type=float, default=1.0)
    g2.add_argument("--p-mag", type=float, default=89.5,
                    help="magnitude predictor accuracy in percent (fixed across sweep)")
    g2.add_argument("--correlation", type=float, default=0.0,
                    help="rho between ML and magnitude predictor errors, in [-1, 1]")

    g3 = p.add_argument_group("filter")
    g3.add_argument("--filter-pass-rate", type=float, default=1.0,
                    help="fraction of raw signatures kept by leading-pair filter "
                         "(use 0.756 for the [32, 99] filter on secp256k1)")

    g4 = p.add_argument_group("hardware & timing")
    g4.add_argument("--cores", type=int, nargs="+", default=[1],
                    help="core counts to report time for; one column each (default: 1)")
    g4.add_argument("--t-per-run-sec", type=float, default=None,
                    help="CALIBRATED wall-clock of one lattice reduction on one core; "
                         "if omitted, the script falls back to a dim-dependent heuristic. "
                         "FIX B5: default was 2.0, which made the heuristic branch dead "
                         "code and mislabelled the fallback as 'USER-PROVIDED'.")
    g4.add_argument("--dim-overhead", type=int, default=2,
                    help="lattice dimension = m_req + this overhead")
    g4.add_argument("--run-base-sec", type=float, default=2.0,
                    help="heuristic only: reduction time at --run-ref-dim")
    g4.add_argument("--run-growth", type=float, default=0.047,
                    help="heuristic only: exponential growth rate in dim")
    g4.add_argument("--run-ref-dim", type=int, default=42,
                    help="heuristic only: dim at which run time equals --run-base-sec")

    g5 = p.add_argument_group("output")
    g5.add_argument("--csv", type=str, default=None,
                    help="optional path to write the full table as CSV")
    g5.add_argument("--no-legend", action="store_true",
                    help="suppress the explanatory legend before tables")

    args = p.parse_args(argv)

    if args.curve_bits <= 0 or any(L <= 0 for L in args.leak_bits):
        p.error("--curve-bits and --leak-bits must be positive")
    if not (0 < args.p_ml_start <= args.p_ml_stop <= 100):
        p.error("require 0 < --p-ml-start <= --p-ml-stop <= 100")
    if args.p_ml_step <= 0:
        p.error("--p-ml-step must be positive")
    if not (0 < args.p_mag <= 100):
        p.error("--p-mag must be in (0, 100]")
    if not (-1 <= args.correlation <= 1):
        p.error("--correlation must be in [-1, 1]")
    if not (0 < args.filter_pass_rate <= 1):
        p.error("--filter-pass-rate must be in (0, 1]")
    if any(c <= 0 for c in args.cores):
        p.error("--cores values must be positive")
    # FIX D: the original left these unvalidated. --surplus -5 produced a table
    # of negative T_filt/clean and all-"n/a" rows; negative safety-factor and
    # dim-overhead were silently accepted.
    if args.surplus <= 0:
        p.error("--surplus must be positive")
    if args.safety_factor <= 0:
        p.error("--safety-factor must be positive")
    if args.m_req is not None and args.m_req <= 0:
        p.error("--m-req must be positive when given")
    if args.dim_overhead < 0:
        p.error("--dim-overhead must be non-negative")
    if args.t_per_run_sec is not None and args.t_per_run_sec <= 0:
        p.error("--t-per-run-sec must be positive when given")
    if args.run_base_sec <= 0:
        p.error("--run-base-sec must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = Config(
        curve_bits=args.curve_bits, leak_bits=args.leak_bits[0],
        m_req_override=args.m_req, safety_factor=args.safety_factor,
        surplus=args.surplus, filter_pass_rate=args.filter_pass_rate,
        p_mag=args.p_mag, correlation=args.correlation,
        acc_start=args.p_ml_start, acc_stop=args.p_ml_stop, acc_step=args.p_ml_step,
        cores=args.cores, t_per_run_sec=args.t_per_run_sec,
        dim_overhead=args.dim_overhead, run_base_sec=args.run_base_sec,
        run_growth=args.run_growth, run_ref_dim=args.run_ref_dim,
    )
    if not args.no_legend:
        print_legend(base)

    all_rows: list[Row] = []
    for L in args.leak_bits:
        cfg = dataclasses.replace(base, leak_bits=L)
        rows = build_rows(cfg)
        print_table(rows, cfg)
        all_rows.extend(rows)

    if args.csv:
        write_csv(all_rows, base, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())