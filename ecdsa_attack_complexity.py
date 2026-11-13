#!/usr/bin/env python3
"""
ECDSA two-predictor subset-attack complexity model  (v3, beginner-legible)
==========================================================================

PLAIN-LANGUAGE SUMMARY (read this first)
----------------------------------------
This script estimates how much WORK a hypothetical attack on ECDSA would take
IF an attacker already had a machine-learning "oracle" that guesses the top bits
of each secret nonce. It is a COST CALCULATOR, not an attack: it runs no lattice
reduction and breaks no key. The oracle's accuracy is an ASSUMPTION you type in.

The headline question it answers is: "given a nonce-bit oracle of accuracy X,
how many independent lattice reductions, and how much wall-clock time, until the
attack succeeds?" Every number is conditional on that oracle existing.

WHAT "CLEAN" MEANS
------------------
The lattice attack needs a batch of m_req signatures in which EVERY signature
was predicted correctly ("clean"). A signature is clean only if BOTH predictors
are right:
  1. the leading-digit predictor   (accuracy p_b -- this is the swept variable)
  2. the magnitude predictor        (accuracy p_m -- a fixed input)
A single wrong ("poisoned") signature destroys the whole reduction, so the batch
must be 100% clean. That all-or-nothing rule is why a good-looking accuracy like
73.8% still turns into millions of attempts: 0.738 raised to the 56th power.

WHAT THE SCRIPT DOES NOT DO (do not over-read the output)
---------------------------------------------------------
  * It does NOT show that secp256k1 is breakable. The oracle is the hard part
    and is assumed away; building it is a partial ECDLP (~2^128), untouched here.
  * safety_factor (1.3) is a fudge with no lattice-theoretic basis.
  * Confidence ranking is not modelled; a real attacker who ranked signatures by
    predictor confidence would do strictly BETTER (cheaper) than these numbers.
  * The per-reduction time is a guess you must calibrate on your own hardware.

WHAT CHANGED IN v3  (output clarity only -- the mathematics is byte-for-byte the
same model as v2 and produces identical numbers)
--------------------------------------------------------------------------------
  * A caveat BANNER now sits with the numbers, so a "90% -> 1 year" row can no
    longer be misread as "the curve falls in a year".
  * A PROVENANCE block records the exact command, timestamp, git commit, curve
    and corpus, so any table can be reproduced and dated.
  * A "HOW TO READ" block decodes every column and every unit in plain words.
  * Counts are printed in human form ("16M", "1.3B") instead of the ambiguous
    "1e7.2" notation; where a value exceeds 10^15 it falls back to STANDARD
    scientific notation ("4.0e17"), never the non-standard "1e17.6".
  * The per-draw success probability q is shown as PER-TRY odds ("odds/try =
    1 in 24M"), with an explicit AVERAGE column (N:avg) next to the median and
    percentiles so the odds number has a visible count twin. The text spells out
    that the attempt count is geometric: it varies, has NO maximum, and the
    unlucky-95% value is expected to exceed the average (a common misreading the
    earlier layout invited).
  * The reduction columns are labelled lucky-5% / median / avg / unlucky-95%,
    and the attempt count is named a COUNT OF ATTEMPTS, never a combination.
  * min_raw_for_pool now searches pool size exactly (step 1) instead of in
    coarse jumps of m_req//8, so the "minimum pool" advice is no longer rounded
    up by as much as ~7 signatures.

Nothing below the "Pure mathematics" line was changed in substance from v2; the
model and every formula are preserved. Only presentation was reworked.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil, floor, lgamma, log, log10, log1p
from typing import Sequence

_LN10 = log(10.0)
_LOG10_SEC_PER_YEAR = log10(31_536_000.0)

KNOWN_UNVALIDATED = (
    "p_b is an assumed oracle accuracy, not a measured capability "
    "(recovering it is a partial ECDLP)",
    "safety_factor is a fudge with no lattice-theoretic justification",
    "confidence ranking is not modelled; a real attacker would do better",
    "t_per_run_sec must be calibrated on your own hardware and solver",
)


# ===========================================================================
# Pure mathematics   (UNCHANGED from v2 -- verified correct)
# ===========================================================================

def log10_binom(n: int, k: int) -> float:
    """log10 C(n, k); -inf when the binomial is undefined."""
    if k < 0 or k > n:
        return float("-inf")
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / _LN10


def log10_binom_pmf(n: int, p: float, k: int) -> float:
    """log10 P[Bin(n, p) = k]."""
    if k < 0 or k > n:
        return float("-inf")
    if p <= 0.0:
        return 0.0 if k == 0 else float("-inf")
    if p >= 1.0:
        return 0.0 if k == n else float("-inf")
    return log10_binom(n, k) + k * log10(p) + (n - k) * log10(1.0 - p)


def joint_clean_probability(p_b: float, p_m: float, rho: float) -> float:
    """
    P(signature clean).

    conditional mode (rho == 0): p_clean = p_b * p_m, with p_m already
    conditioned on the leading digits being correct.

    marginal mode: p_m is the marginal magnitude accuracy and rho is the
    Pearson correlation of the two Bernoulli indicators. The result is clamped
    to the Frechet-Hoeffding bounds so a joint distribution with these
    marginals is guaranteed to exist.
    """
    base = p_b * p_m
    if abs(rho) < 1e-12:
        return base
    cov = rho * math.sqrt(p_b * (1.0 - p_b) * p_m * (1.0 - p_m))
    lo = max(0.0, p_b + p_m - 1.0)
    hi = min(p_b, p_m)
    return max(lo, min(hi, base + cov))


def bayes_post_clean(p_clean: float, pass_rate: float,
                     recall: float) -> tuple[float, float]:
    """
    Purity of the pool *after* an informative pre-filter.

        P(clean | pass) = p_clean * P(pass | clean) / P(pass)

    Returns (posterior clean probability, implied false-pass rate), where
    P(pass | not clean) is recovered from the law of total probability

        pi = p_clean * recall + (1 - p_clean) * fpr

    and raises ValueError when (pi, recall, p_clean) admit no valid fpr.
    """
    post = p_clean * recall / pass_rate
    if post > 1.0 + 1e-12:
        raise ValueError(
            f"inconsistent filter: P(clean|pass) = {post:.4f} > 1. "
            f"pass_rate ({pass_rate:.4f}) cannot be below "
            f"p_clean*recall ({p_clean * recall:.4f})."
        )
    post = min(1.0, post)
    if p_clean >= 1.0:
        return post, 0.0
    fpr = (pass_rate - p_clean * recall) / (1.0 - p_clean)
    if not (-1e-9 <= fpr <= 1.0 + 1e-9):
        raise ValueError(
            f"inconsistent filter: implied P(pass|not clean) = {fpr:.4f} is "
            f"outside [0, 1] for pass_rate={pass_rate}, recall={recall}, "
            f"p_clean={p_clean:.4f}."
        )
    return post, min(1.0, max(0.0, fpr))


def required_clean_signatures(order_bits: int, leak_bits: int,
                              safety_factor: float) -> int:
    """
    m_req = ceil( ceil(log2(n) / L) * safety_factor )

    The inner term is the information-theoretic NECESSARY condition
    m_req * L >= log2(n). It is not sufficient: whether the hidden number is
    actually a short vector depends on the lattice determinant versus the
    target norm, which this model does not evaluate. safety_factor is a fudge
    standing in for that gap.
    """
    return ceil(ceil(order_bits / leak_bits) * safety_factor)


def log10_success_per_draw(t_filt: int | None, m_req: int, clean: int | None,
                           p_clean: float, p_succ: float) -> float:
    """
    log10 of q = P[one reduction attempt succeeds].

    unlimited supply (t_filt is None): draws are i.i.d., so
        q = p_clean^m_req * p_succ
    scarce supply: draws are m_req-subsets of a fixed pool of t_filt holding
    `clean` clean signatures, i.e. hypergeometric,
        q = C(clean, m_req) / C(t_filt, m_req) * p_succ
    """
    lp = log10(p_succ) if p_succ > 0.0 else float("-inf")
    if t_filt is None:
        if p_clean <= 0.0:
            return float("-inf")
        return m_req * log10(p_clean) + lp
    num = log10_binom(clean, m_req)
    if num == float("-inf"):
        return float("-inf")
    return num - log10_binom(t_filt, m_req) + lp


def _survival(log10_n: float, components: Sequence[tuple[float, float]]) -> float:
    """
    P[N > n] for N a mixture of geometric variables.

    `components` is a sequence of (weight, log10_q) pairs. Evaluated in log
    space because q routinely underflows float64 in the regimes of interest.
    """
    total = 0.0
    for w, log10_q in components:
        if w <= 0.0:
            continue
        if log10_q == float("-inf"):
            total += w                       # this pool never succeeds
            continue
        if log10_q >= 0.0:
            if log10_n < 0.0:
                total += w
            continue
        log10_lam = log10_n + log10_q
        if log10_lam > 3.0:
            continue                         # exp(-lam) underflows to 0
        q = 10.0 ** log10_q
        if q < 1e-4:
            total += w * math.exp(-(10.0 ** log10_lam))
        else:
            total += w * math.exp((10.0 ** log10_n) * log1p(-q))
    return total


def runs_quantile(level: float,
                  components: Sequence[tuple[float, float]]) -> float | None:
    """
    log10 of the smallest n with P[N <= n] >= level, by bisection on log10 n.
    Returns None when the level is unreachable (too much mass on q = 0).
    """
    if _survival(300.0, components) > 1.0 - level:
        return None
    lo, hi = -3.0, 300.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _survival(mid, components) > 1.0 - level:
            lo = mid
        else:
            hi = mid
    return hi


def log10_mean_runs(components: Sequence[tuple[float, float]]) -> float | None:
    """
    log10 of E[N], the AVERAGE number of attempts, for the geometric mixture.

    For a single geometric this is log10(1/q) -- i.e. exactly the per-try odds
    read as a count. Returns +inf when any never-succeeding component carries
    weight (an unlucky pool can hold too few clean signatures, making the mean
    unbounded even though the median is finite).
    """
    terms = []
    for w, lq in components:
        if w <= 0.0:
            continue
        if lq == float("-inf"):
            return float("inf")
        terms.append(log10(w) - lq)          # log10(w / q_i)
    if not terms:
        return None
    m = max(terms)
    return m + log10(sum(10.0 ** (t - m) for t in terms))


def log10_t_run(dim: int, t0_sec: float, dim_ref: int,
                doubling_dims: float | None) -> float:
    """
    log10 of the wall-clock cost of one reduction at lattice dimension `dim`.

    Flat (doubling_dims is None) reproduces v1's placeholder and is wrong:
    reduction cost grows steeply with dimension. Supplying doubling_dims turns
    on the exponential model t_run(dim) = t0 * 2^((dim - dim_ref)/doubling_dims).
    """
    base = log10(t0_sec)
    if doubling_dims is None:
        return base
    return base + ((dim - dim_ref) / doubling_dims) * log10(2.0)


def min_raw_for_pool(m_req: int, p_post: float, pass_rate: float,
                     confidence: float = 0.95) -> int | None:
    """
    Smallest T_raw such that P[at least m_req clean signatures survive the
    filter] >= confidence.

    v3: searches the filtered-pool size in steps of 1 (v2 used m_req//8 and
    could overshoot the true minimum by several signatures).
    """
    if p_post <= 0.0:
        return None
    t_filt = m_req
    while t_filt <= 400 * m_req + 1000:
        tail = 0.0
        for k in range(m_req, t_filt + 1):
            lg = log10_binom_pmf(t_filt, p_post, k)
            if lg != float("-inf"):
                tail += 10.0 ** lg
            if tail >= confidence:
                break
        if tail >= confidence:
            return ceil(t_filt / pass_rate)
        t_filt += 1
    return None


# ===========================================================================
# Human-readable formatting  (NEW in v3)
# ===========================================================================

_SUFFIX = {3: "K", 6: "M", 9: "B", 12: "T"}


# Retained public utilities (compact log-form). The tables no longer use these
# -- they switched to the human_* formatters below -- but they remain part of
# the module's API, so external callers and the test suite keep working.
def fmt_log_count(x: float | None) -> str:
    if x is None or x == float("-inf"):
        return "--"
    if x < 3.0:
        return f"{10.0 ** x:,.0f}"
    return f"1e{x:.1f}"


def fmt_duration(log10_sec: float | None, precision: str = "orders") -> str:
    """Order-of-magnitude by default; the model cannot support more."""
    if log10_sec is None or log10_sec == float("-inf"):
        return "--"
    if log10_sec > 9.0:                       # beyond ~30 yr, report in years
        yr = log10_sec - _LOG10_SEC_PER_YEAR
        return f"1e{yr:.1f} yr" if precision == "orders" else f"1e{yr:.3f} yr"
    sec = 10.0 ** log10_sec
    d = 0 if precision == "orders" else 2
    if sec < 1.0:
        return "<1 s"
    for lim, div, unit in ((60, 1, "s"), (3600, 60, "min"),
                           (86400, 3600, "h"), (2_592_000, 86400, "d"),
                           (31_536_000, 2_592_000, "mo")):
        if sec < lim:
            return f"{sec / div:.{d}f} {unit}"
    return f"{sec / 31_536_000:.{d}f} yr"


def human_count(log10_x: float | None) -> str:
    """
    A count given by its base-10 log, printed the way a person reads it:
      <1000 -> exact ('842'); then '16K', '1.3M', '12.6B', '4.1T';
      >=10^15 -> STANDARD scientific ('4.0e17'), never the ambiguous '1e17.6'.
    """
    if log10_x is None or log10_x == float("-inf"):
        return "--"
    if log10_x < 0.0:
        return "<1"
    exp = int(floor(log10_x))
    if exp < 3:
        return f"{10.0 ** log10_x:,.0f}"
    if exp >= 15:
        mant = 10.0 ** (log10_x - exp)
        return f"{mant:.1f}e{exp}"
    g = (exp // 3) * 3
    scaled = 10.0 ** (log10_x - g)            # in [1, 1000)
    suf = _SUFFIX[g]
    return f"{scaled:.0f}{suf}" if scaled >= 100 else f"{scaled:.1f}{suf}"


def human_mean(log10_x: float | None) -> str:
    """Average attempt count; 'unbounded' when unlucky pools never finish."""
    if log10_x == float("inf"):
        return "unbounded"
    return human_count(log10_x)


def human_odds(log10_q: float | None) -> str:
    """Per-draw success probability q shown as intuitive odds '1 in N'."""
    if log10_q is None or log10_q == float("-inf"):
        return "never"
    if log10_q >= 0.0:
        return "1 in 1"
    return "1 in " + human_count(-log10_q)


def human_duration(log10_sec: float | None) -> str:
    """
    A duration given by log10(seconds), in the largest sensible unit:
      '<1 s', '45 s', '18 h', '6 d', '2.0 mo', '790 yr';
      beyond 1000 years -> STANDARD scientific years ('2.5e10 yr').
    """
    if log10_sec is None or log10_sec == float("-inf"):
        return "--"
    log10_yr = log10_sec - _LOG10_SEC_PER_YEAR
    if log10_yr >= 3.0:
        exp = int(floor(log10_yr))
        mant = 10.0 ** (log10_yr - exp)
        return f"{mant:.1f}e{exp} yr"
    sec = 10.0 ** log10_sec
    if sec < 1.0:
        return "<1 s"
    for lim, div, unit in ((60, 1, "s"), (3600, 60, "min"),
                           (86400, 3600, "h"), (2_592_000, 86400, "d"),
                           (31_536_000, 2_592_000, "mo")):
        if sec < lim:
            v = sec / div
            return f"{v:.0f} {unit}" if v >= 10 else f"{v:.1f} {unit}"
    yr = sec / 31_536_000.0
    return f"{yr:.0f} yr" if yr >= 10 else f"{yr:.1f} yr"


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class Config:
    order_bits: int
    curve_name: str
    leak_bits: int
    safety_factor: float
    dim_overhead: int
    p_mag: float
    p_mag_mode: str
    correlation: float
    acc_start: float
    acc_stop: float
    acc_step: float
    supply: str
    t_raw: int | None
    filter_pass_rate: float
    filter_recall: float
    p_reduction_success: float
    t_per_run_sec: float
    t_run_dim_ref: int | None
    t_run_doubling_dims: float | None
    cores: list[int]
    corpus: str | None


@dataclass
class Row:
    leak_bits: int
    p_ml_pct: float
    p_clean: float
    p_clean_post: float
    implied_fpr: float | None
    m_req: int
    dimension: int
    t_raw: int | None
    t_filtered: int | None
    clean: int | None
    log10_q: float
    log10_runs_p05: float | None
    log10_runs_p50: float | None
    log10_runs_p95: float | None
    log10_t_run: float
    note: str = ""
    times_p50: dict[int, float | None] = field(default_factory=dict)
    log10_runs_mean: float | None = None


# ===========================================================================
# Core calculation   (UNCHANGED from v2)
# ===========================================================================

def accuracy_sweep(start: float, stop: float, step: float) -> list[float]:
    """Index-based, so the grid cannot drift with floating accumulation."""
    n = int(math.floor((stop - start) / step + 1e-9))
    return [round(start + i * step, 6) for i in range(n + 1)]


def build_rows(cfg: Config) -> list[Row]:
    m_req = required_clean_signatures(cfg.order_bits, cfg.leak_bits,
                                      cfg.safety_factor)
    dim = m_req + cfg.dim_overhead
    p_m = cfg.p_mag / 100.0
    rho = 0.0 if cfg.p_mag_mode == "conditional" else cfg.correlation
    dim_ref = cfg.t_run_dim_ref if cfg.t_run_dim_ref is not None else dim
    lt_run = log10_t_run(dim, cfg.t_per_run_sec, dim_ref,
                         cfg.t_run_doubling_dims)

    rows: list[Row] = []
    for pct in accuracy_sweep(cfg.acc_start, cfg.acc_stop, cfg.acc_step):
        p_b = pct / 100.0
        p_clean = joint_clean_probability(p_b, p_m, rho)
        if p_clean <= 0.0:
            continue

        note = ""
        if cfg.supply == "unlimited":
            p_post, fpr = p_clean, None
            t_filt = clean = None
            log10_q = log10_success_per_draw(None, m_req, None, p_clean,
                                             cfg.p_reduction_success)
            components = [(1.0, log10_q)]
        else:
            p_post, fpr = bayes_post_clean(p_clean, cfg.filter_pass_rate,
                                           cfg.filter_recall)
            t_filt = floor(cfg.filter_pass_rate * cfg.t_raw)
            if t_filt < m_req:
                need = min_raw_for_pool(m_req, p_post, cfg.filter_pass_rate)
                rows.append(Row(
                    leak_bits=cfg.leak_bits, p_ml_pct=pct, p_clean=p_clean,
                    p_clean_post=p_post, implied_fpr=fpr, m_req=m_req,
                    dimension=dim, t_raw=cfg.t_raw, t_filtered=t_filt,
                    clean=None, log10_q=float("-inf"),
                    log10_runs_p05=None, log10_runs_p50=None,
                    log10_runs_p95=None, log10_t_run=lt_run,
                    note=(f"filtered pool {t_filt} < m_req {m_req}: cannot "
                          f"even form one batch"
                          + (f"; need T_raw >= {need} raw signatures"
                             if need else "")),
                    times_p50={c: None for c in cfg.cores},
                ))
                continue

            components = []
            for k in range(m_req, t_filt + 1):
                lw = log10_binom_pmf(t_filt, p_post, k)
                if lw < -12.0:
                    continue
                components.append((
                    10.0 ** lw,
                    log10_success_per_draw(t_filt, m_req, k, p_post,
                                           cfg.p_reduction_success),
                ))
            mass = min(1.0, sum(w for w, _ in components))
            if mass <= 1e-12:
                components = [(1.0, float("-inf"))]
                note = "pool almost surely holds fewer than m_req clean"
            elif mass < 1.0 - 1e-9:
                components.append((1.0 - mass, float("-inf")))
            if 1e-12 < mass < 0.99:
                need = min_raw_for_pool(m_req, p_post, cfg.filter_pass_rate)
                note = (f"P(pool holds >= m_req clean) = {mass:.3f}; "
                        f"the 95% column is unreachable"
                        + (f"; need T_raw >= {need} raw signatures for 95%"
                           if need else ""))
            clean = round(p_post * t_filt)
            log10_q = log10_success_per_draw(t_filt, m_req, clean, p_post,
                                             cfg.p_reduction_success)

        q05 = runs_quantile(0.05, components)
        q50 = runs_quantile(0.50, components)
        q95 = runs_quantile(0.95, components)
        qmean = log10_mean_runs(components)
        times = {c: (None if q50 is None else q50 + lt_run - log10(c))
                 for c in cfg.cores}

        rows.append(Row(
            leak_bits=cfg.leak_bits, p_ml_pct=pct, p_clean=p_clean,
            p_clean_post=p_post, implied_fpr=fpr, m_req=m_req, dimension=dim,
            t_raw=cfg.t_raw, t_filtered=t_filt, clean=clean, log10_q=log10_q,
            log10_runs_p05=q05, log10_runs_p50=q50, log10_runs_p95=q95,
            log10_t_run=lt_run, note=note, times_p50=times,
            log10_runs_mean=qmean,
        ))
    return rows


# ===========================================================================
# Output  (REWRITTEN in v3 for zero-ambiguity, beginner-legible tables)
# ===========================================================================

def _git_commit() -> str | None:
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def print_banner() -> None:
    bar = "#" * 104
    print(bar)
    print("#  WHAT THIS TABLE IS  --  read before quoting any number")
    print("#")
    print("#  A COST ESTIMATE for a HYPOTHETICAL attack, not a demonstration and")
    print("#  not a break. Every row ASSUMES an oracle that already predicts the")
    print("#  top bits of each secret nonce at the stated accuracy. No such oracle")
    print("#  exists for a correct ECDSA implementation; building one is equivalent")
    print("#  to (partially) solving the elliptic-curve discrete log (~2^128 work),")
    print("#  which this script does NOT attempt.")
    print("#")
    print("#  So a row like '90% -> ~1 year' does NOT mean secp256k1 breaks in a")
    print("#  year. It means: GIVEN a 90%/82% nonce-bit oracle, the lattice")
    print("#  book-keeping alone would take ~1 year on 1 core. The oracle is the")
    print("#  hard part and is assumed for free.")
    print(bar)


def print_provenance(cfg: Config) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cmd = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    commit = _git_commit()
    print()
    print("PROVENANCE (so this table can be reproduced and dated)")
    print("-" * 104)
    print(f"  generated     : {ts}")
    print(f"  script        : ecdsa_attack_complexity.py  (v3, beginner-legible)")
    if commit:
        print(f"  git commit    : {commit}")
    print(f"  command       : {cmd}")
    print(f"  curve         : {cfg.curve_name}  (order n, log2 n = "
          f"{cfg.order_bits} bits)")
    if cfg.corpus:
        print(f"  corpus        : {cfg.corpus}")


def print_parameters(cfg: Config, leak_bits: Sequence[int]) -> None:
    print()
    print("INPUTS YOU CHOSE")
    print("-" * 104)
    print(f"  leak_bits L          = {list(leak_bits)}   "
          f"(bits of each nonce the oracle is assumed to reveal)")
    if cfg.p_mag_mode == "conditional":
        print(f"  magnitude accuracy   = {cfg.p_mag:.2f}%   "
              f"(p_m, held FIXED; combined as p_clean = p_b * p_m)")
    else:
        print(f"  magnitude accuracy   = {cfg.p_mag:.2f}%   "
              f"(p_m marginal), correlation rho = {cfg.correlation:+.3f}")
    print(f"  leading-digit sweep  = {cfg.acc_start:g}%..{cfg.acc_stop:g}% "
          f"step {cfg.acc_step:g}%   (p_b, the SWEPT variable = table rows)")
    print(f"  safety_factor        = {cfg.safety_factor}   "
          f"[UNVALIDATED FUDGE on the information bound]")
    print(f"  supply model         = {cfg.supply}"
          + (f"   (T_raw = {cfg.t_raw} signatures available)"
             if cfg.supply == "scarce" else
             "   (i.i.d. limit: assumes a large pool, hundreds+; see note)"))
    if cfg.supply == "scarce":
        print(f"  filter pass / recall = {cfg.filter_pass_rate} / "
              f"{cfg.filter_recall}")
    print(f"  reduction success    = {cfg.p_reduction_success}"
          + ("   [OPTIMISTIC: reduction assumed infallible]"
             if cfg.p_reduction_success >= 1.0 else ""))
    ref = cfg.t_run_dim_ref if cfg.t_run_dim_ref is not None else "first L's dim"
    print(f"  seconds per reduction= {cfg.t_per_run_sec:g} s at dim {ref}  "
          f"[CALIBRATE THIS on your own hardware]")
    if cfg.t_run_doubling_dims is None:
        print(f"                         [FLAT model: cost does NOT grow with "
              f"lattice size -- WRONG; do not compare rows across different L]")
    else:
        print(f"  cost doubling every  = {cfg.t_run_doubling_dims} dimensions")
    print(f"  CPU cores reported   = {cfg.cores}")


def print_how_to_read(cfg: Config) -> None:
    print()
    print("HOW TO READ EACH COLUMN")
    print("-" * 104)
    print("  p_ml%        accuracy of the leading-digit predictor p_b. This is")
    print("               the SWEPT variable: each row is one assumed accuracy.")
    print("  p_clean      chance a single signature is usable (both predictors")
    print("               right) = p_b * p_m. One bad signature spoils a batch.")
    print(f"  m_req        signatures that must ALL be clean in one batch "
          f"(= {_first_m_req(cfg)} here).")
    if cfg.supply == "scarce":
        print("  P(cln|pass)  pool purity after the pre-filter (Bayes).")
    print("  odds/try     chance that ONE random batch is 100% clean, PER TRY")
    print("               (= q). '1 in 24.5M' = each attempt has a 1/24,500,000")
    print("               chance. This is NOT the total you need -- it is the odds")
    print("               of a single coin flip.")
    print("  reductions   how many attempts (coin flips) until the first success.")
    print("    N:lucky5%  This is a COUNT OF ATTEMPTS, not a combination C(.,.),")
    print("    N:median   and it has NO MAXIMUM. Being a 1-in-q coin, N VARIES:")
    print("    N:avg        lucky 5% = you finish unusually early,")
    print("    N:bad95%     median   = half the runs finish by here,")
    print("               avg      = the long-run average (equals the odds number),")
    print("               bad 95%  = 5% of runs still run PAST here.")
    print("               So N:bad95% is LARGER than N:avg. That is expected for a")
    print("               geometric process, not a contradiction.")
    cores = ", ".join(str(c) for c in cfg.cores)
    print(f"  t @ N cores  wall-clock = reductions * seconds-per-reduction / cores,")
    print(f"               at the MEDIAN N, for N cores in [{cores}].")
    print("               Units: s, min, h, d (day), mo (month), yr (year).")
    print("  NUMBER FORMAT: '16K'=16 thousand, '16M'=million, '13B'=billion,")
    print("               '2.5T'=trillion; above that, standard sci '4.0e17'.")
    if cfg.supply == "unlimited":
        print("  POOL SIZE: unlimited mode is the best-case infinite-pool limit;")
        print("             a batch of exactly m_req clean signatures cannot be")
        print("             guaranteed from a tiny pool. Run  --supply scarce")
        print("             --t-raw N  to size the actual signature pool you need.")


def _first_m_req(cfg: Config) -> int:
    return required_clean_signatures(cfg.order_bits, cfg.leak_bits,
                                     cfg.safety_factor)


def print_table(rows: list[Row], cfg: Config) -> None:
    if not rows:
        print("\n  (no rows)")
        return
    head = rows[0]
    print()
    print("=" * 104)
    line = (f"RESULTS   L = {head.leak_bits} bits  |  m_req = {head.m_req} "
            f"clean signatures per batch  |  lattice dim = {head.dimension}  |  "
            f"t/run = {10.0 ** head.log10_t_run:g} s")
    print(line)
    if cfg.supply == "scarce":
        print(f"          pool: T_raw = {head.t_raw} available, "
              f"T_filt = {head.t_filtered} after filter")
    print("=" * 104)
    print()

    cols = [("p_ml%", 7), ("p_clean", 9)]
    if cfg.supply == "scarce":
        cols.append(("P(cln|pass)", 12))
    cols += [("odds/try", 12),
             ("N:lucky5%", 11), ("N:median", 11), ("N:avg", 11),
             ("N:bad95%", 11)]
    cols += [(f"t@{c}core", 10) for c in cfg.cores]

    print("  " + "".join(h.rjust(w) for h, w in cols))
    print("  " + "-" * sum(w for _, w in cols))
    for r in rows:
        cells = [f"{r.p_ml_pct:g}", f"{r.p_clean:.4f}"]
        if cfg.supply == "scarce":
            cells.append(f"{r.p_clean_post:.4f}")
        cells.append(human_odds(r.log10_q))
        cells += [human_count(r.log10_runs_p05),
                  human_count(r.log10_runs_p50),
                  human_mean(r.log10_runs_mean),
                  human_count(r.log10_runs_p95)]
        cells += [human_duration(r.times_p50.get(c)) for c in cfg.cores]
        print("  " + "".join(c.rjust(w) for c, (_, w) in zip(cells, cols)))
        if r.note:
            print(f"      ! {r.note}")
    print("  " + "-" * sum(w for _, w in cols))
    print("  N is a COUNT OF ATTEMPTS (lattice reductions), not a combination")
    print("  C(.,.) and not a ceiling -- it varies and has NO maximum. 'odds/try'")
    print("  is the per-attempt chance and equals the AVERAGE count (N:avg);")
    print("  unlucky runs (N:bad95%) exceed that average, which is normal here.")
    print("  Times are at the MEDIAN N; the band scales time the same way.")
    print("  ALL numbers assume the oracle (see banner).")


def write_csv(all_rows: list[Row], cfg: Config, path: str) -> None:
    """CSV keeps every v2 machine column AND adds human-readable twins."""
    fields = ["leak_bits", "p_ml_pct", "p_clean", "p_clean_post",
              "implied_false_pass_rate", "m_req", "dimension", "t_raw",
              "t_filtered", "clean_pointest", "log10_q", "clean_odds_human",
              "log10_runs_p05", "log10_runs_p50", "log10_runs_p95",
              "log10_runs_mean",
              "runs_p05_human", "runs_p50_human", "runs_p95_human",
              "runs_mean_human",
              "log10_t_run_sec", "note"]
    for c in cfg.cores:
        fields += [f"log10_seconds_{c}cores_p50",
                   f"t_{c}cores_p50_human"]

    def g(x):
        return "" if x is None or x == float("-inf") else f"{x:.6f}"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in all_rows:
            row = [r.leak_bits, r.p_ml_pct, f"{r.p_clean:.6f}",
                   f"{r.p_clean_post:.6f}",
                   "" if r.implied_fpr is None else f"{r.implied_fpr:.6f}",
                   r.m_req, r.dimension,
                   "" if r.t_raw is None else r.t_raw,
                   "" if r.t_filtered is None else r.t_filtered,
                   "" if r.clean is None else r.clean,
                   g(r.log10_q), human_odds(r.log10_q),
                   g(r.log10_runs_p05), g(r.log10_runs_p50),
                   g(r.log10_runs_p95),
                   ("inf" if r.log10_runs_mean == float("inf")
                    else g(r.log10_runs_mean)),
                   human_count(r.log10_runs_p05),
                   human_count(r.log10_runs_p50),
                   human_count(r.log10_runs_p95),
                   human_mean(r.log10_runs_mean),
                   g(r.log10_t_run), r.note]
            for c in cfg.cores:
                v = r.times_p50.get(c)
                row += [g(v), human_duration(v)]
            w.writerow(row)
    print(f"\n  [CSV written -> {path}]")


def print_footer() -> None:
    print()
    print("Unvalidated assumptions carried by EVERY number above")
    print("-" * 104)
    for a in KNOWN_UNVALIDATED:
        print(f"  - {a}")
    print()
    print("  Bottom line: this estimates the CHEAP half of the attack (the")
    print("  lattice book-keeping, which parallelises trivially). The EXPENSIVE")
    print("  half -- the nonce-bit oracle -- is assumed, and for a correct")
    print("  implementation it costs ~2^128, which no number of cores shortens.")


# ===========================================================================
# CLI
# ===========================================================================

class _Refuse(argparse.Action):
    def __call__(self, parser, ns, values, option_string=None):
        parser.error(
            f"{option_string} was removed in v2. It silently controlled the "
            "headline figure across ~11 orders of magnitude with no "
            "justification. Use --supply unlimited (no pool parameters) or "
            "--supply scarce --t-raw N (signatures actually available under "
            "the target key)."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ECDSA two-predictor subset-attack complexity (v3, "
                    "beginner-legible output)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("threat parameters")
    g.add_argument("--curve-bits", type=int, default=256,
                   help="ceil(log2 n) for the curve order n")
    g.add_argument("--curve-name", type=str, default="secp256k1",
                   help="curve name, printed in the provenance block")
    g.add_argument("--leak-bits", type=int, nargs="+", default=[6],
                   help="predicted/leaked bits L per signature (one table each)")
    g.add_argument("--p-ml-start", type=float, default=80.0)
    g.add_argument("--p-ml-stop", type=float, default=99.0)
    g.add_argument("--p-ml-step", type=float, default=1.0)
    g.add_argument("--p-mag", type=float, default=82.0,
                   help="magnitude predictor accuracy in %%")
    g.add_argument("--p-mag-mode", choices=["conditional", "marginal"],
                   default="conditional",
                   help="conditional: p_m already assumes correct digits "
                        "(rho forced to 0). marginal: p_m is the marginal "
                        "rate and --correlation applies.")
    g.add_argument("--correlation", type=float, default=0.0,
                   help="Pearson rho between the two predictors; marginal "
                        "mode only. Measured ~+0.126 on this project's corpus.")
    g.add_argument("--corpus", type=str, default=None,
                   help="name/path of the corpus p_mag was measured on "
                        "(printed in provenance, e.g. transaction_list_*.txt)")

    s = p.add_argument_group("signature supply")
    s.add_argument("--supply", choices=["unlimited", "scarce"],
                   default="unlimited",
                   help="unlimited: i.i.d. draws, q = p_clean^m_req. "
                        "scarce: a fixed pool derived from --t-raw.")
    s.add_argument("--t-raw", type=int, default=None,
                   help="signatures available under the target key "
                        "(required for --supply scarce)")
    s.add_argument("--filter-pass-rate", type=float, default=1.0,
                   help="P(signature passes the pre-filter); scarce mode only")
    s.add_argument("--filter-recall", type=float, default=None,
                   help="P(passes the pre-filter | clean); scarce mode only. "
                        "Default: equal to --filter-pass-rate (uninformative "
                        "filter). Set above the pass rate to enrich the pool.")

    lat = p.add_argument_group("lattice")
    lat.add_argument("--safety-factor", type=float, default=1.3,
                     help="UNVALIDATED multiplier on the information bound")
    lat.add_argument("--dim-overhead", type=int, default=2)
    lat.add_argument("--p-reduction-success", type=float, default=1.0,
                     help="P(reduction succeeds | subset entirely clean). "
                          "1.0 is optimistic and unverified.")

    c = p.add_argument_group("cost")
    c.add_argument("--t-per-run-sec", type=float, default=2.0,
                   help="calibrated seconds for one reduction at --t-run-dim-ref")
    c.add_argument("--t-run-dim-ref", type=int, default=None,
                   help="lattice dimension at which --t-per-run-sec was "
                        "measured (default: the dimension of the first L)")
    c.add_argument("--t-run-doubling-dims", type=float, default=None,
                   help="dimensions per doubling of reduction cost. Omit for "
                        "a FLAT (wrong) model that forbids cross-L comparison.")
    c.add_argument("--cores", type=int, nargs="+", default=[1, 8, 64],
                   help="core counts for which wall-clock is reported")

    o = p.add_argument_group("output")
    o.add_argument("--csv", type=str, default=None)
    o.add_argument("--no-banner", action="store_true",
                   help="suppress the caveat banner (not recommended)")
    o.add_argument("--no-legend", action="store_true",
                   help="suppress provenance / parameters / how-to-read blocks")

    p.add_argument("--surplus", nargs="?", action=_Refuse,
                   help=argparse.SUPPRESS)
    p.add_argument("--precision", choices=["orders", "full"], default="orders",
                   help=argparse.SUPPRESS)   # retained for fmt_duration / API

    a = p.parse_args(argv)

    if a.curve_bits <= 0:
        p.error("--curve-bits must be positive")
    if any(L <= 0 for L in a.leak_bits):
        p.error("--leak-bits values must be positive")
    if any(x <= 0 for x in a.cores):
        p.error("--cores values must be positive")
    if a.t_per_run_sec <= 0:
        p.error("--t-per-run-sec must be positive")
    if a.t_run_doubling_dims is not None and a.t_run_doubling_dims <= 0:
        p.error("--t-run-doubling-dims must be positive")
    if not (0.0 < a.p_ml_start <= a.p_ml_stop <= 100.0):
        p.error("require 0 < --p-ml-start <= --p-ml-stop <= 100")
    if a.p_ml_step <= 0:
        p.error("--p-ml-step must be positive")
    if not (0.0 < a.p_mag <= 100.0):
        p.error("--p-mag must be in (0, 100]")
    if not (-1.0 <= a.correlation <= 1.0):
        p.error("--correlation must be in [-1, 1]")
    if a.p_mag_mode == "conditional" and abs(a.correlation) > 1e-12:
        p.error("--correlation is meaningless with --p-mag-mode conditional: "
                "a conditional p_m already carries the dependence, so using "
                "both double-counts it. Pass --p-mag-mode marginal.")
    if a.safety_factor <= 0:
        p.error("--safety-factor must be positive")
    if a.dim_overhead < 0:
        p.error("--dim-overhead must be >= 0")
    if not (0.0 < a.p_reduction_success <= 1.0):
        p.error("--p-reduction-success must be in (0, 1]")
    if a.supply == "scarce":
        if a.t_raw is None or a.t_raw < 1:
            p.error("--supply scarce requires --t-raw N (N >= 1)")
        if not (0.0 < a.filter_pass_rate <= 1.0):
            p.error("--filter-pass-rate must be in (0, 1]")
        if a.filter_recall is None:
            a.filter_recall = a.filter_pass_rate
        if not (0.0 <= a.filter_recall <= 1.0):
            p.error("--filter-recall must be in [0, 1]")
    else:
        if a.t_raw is not None:
            p.error("--t-raw only applies to --supply scarce")
        if a.filter_pass_rate != 1.0 or a.filter_recall not in (None, 1.0):
            p.error("--filter-pass-rate / --filter-recall only apply to "
                    "--supply scarce (with unlimited supply the pool is not "
                    "a binding constraint)")
        a.filter_recall = 1.0
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)

    base = Config(
        order_bits=a.curve_bits, curve_name=a.curve_name,
        leak_bits=a.leak_bits[0], safety_factor=a.safety_factor,
        dim_overhead=a.dim_overhead, p_mag=a.p_mag, p_mag_mode=a.p_mag_mode,
        correlation=a.correlation, acc_start=a.p_ml_start, acc_stop=a.p_ml_stop,
        acc_step=a.p_ml_step, supply=a.supply, t_raw=a.t_raw,
        filter_pass_rate=a.filter_pass_rate, filter_recall=a.filter_recall,
        p_reduction_success=a.p_reduction_success,
        t_per_run_sec=a.t_per_run_sec, t_run_dim_ref=a.t_run_dim_ref,
        t_run_doubling_dims=a.t_run_doubling_dims,
        cores=a.cores, corpus=a.corpus,
    )

    if not a.no_banner:
        print_banner()
    if not a.no_legend:
        print_provenance(base)
        print_parameters(base, a.leak_bits)
        print_how_to_read(base)

    if len(a.leak_bits) > 1 and a.t_run_doubling_dims is None:
        print()
        print("  WARNING: several L values with a FLAT per-run cost. The lattice")
        print("  dimension differs across these tables but the model charges the")
        print("  same time for all of them. Do not compare rows across L. Pass")
        print("  --t-run-doubling-dims to enable a size-dependent cost.")

    all_rows: list[Row] = []
    for L in a.leak_bits:
        cfg = dataclasses.replace(base, leak_bits=L)
        try:
            rows = build_rows(cfg)
        except ValueError as exc:
            print(f"\n  L = {L}: {exc}")
            continue
        print_table(rows, cfg)
        all_rows.extend(rows)

    if not a.no_legend:
        print_footer()

    if a.csv:
        write_csv(all_rows, base, a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
