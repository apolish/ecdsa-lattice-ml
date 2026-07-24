#!/usr/bin/env python3
"""
ECDSA two-predictor subset-attack complexity model  (v2, audited)
=================================================================

WHAT THIS IS
------------
A cost model for a *hypothetical* subset-HNP lattice attack. It implements no
lattice reduction and no predictor. Its dominant input -- the probability that
an oracle recovers leading information about the secret nonce k from public
signature data -- is a swept ASSUMPTION. Recovering that information is
equivalent to a (partial) ECDLP; nothing here attempts it.

Read KNOWN_UNVALIDATED below before quoting any number this script prints.

THREAT MODEL
------------
A signature is CLEAN iff both assumed predictors are correct:
  1. the leading-digit predictor          (accuracy p_b, swept)
  2. the magnitude predictor              (accuracy p_m)
Any single poisoned signature destroys the reduction, so a usable subset of
size m_req must be entirely clean.

CHANGES FROM v1  (each fixes an audited defect)
-----------------------------------------------
1. SUPPLY MODEL IS NOW EXPLICIT.  v1 invented the pool size from an
   unjustified `surplus` constant and then back-derived the raw-signature
   requirement, so signature scarcity never bound anything while `surplus`
   silently controlled the headline figure across ~11 orders of magnitude.
     --supply unlimited : runs = (p_clean^m_req * p_succ)^-1, the correct
                          i.i.d. limit. No pool parameters at all.
     --supply scarce    : --t-raw is an INPUT (signatures available under the
                          target key d). The pool follows from it.
   `--surplus` is removed and now errors out with an explanation.

2. THE PRE-FILTER NOW OBEYS BAYES.  v1 charged the cost of filtering (more raw
   signatures) without crediting the benefit (a purer pool), and
   filter_pass_rate did not influence any result. Now, with pass rate
   pi = P(pass) and recall = P(pass | clean):
        P(clean | pass) = p_clean * recall / pi
   and the implied false-pass rate is derived and validated, so inconsistent
   (pi, recall, p_clean) triples are rejected instead of silently ignored.

3. REDUCTION COST DEPENDS ON LATTICE DIMENSION.  v1 computed `dim` and never
   used it: every L was charged a flat 2.0 s, which inverts reality (small L
   is hard *because* the lattice grows). Now
        t_run(dim) = t0 * 2^((dim - dim_ref) / doubling_dims)
   If --t-run-doubling-dims is omitted the model stays flat AND refuses to
   present cross-L comparisons as meaningful.

4. THE REDUCTION IS NO LONGER ASSUMED INFALLIBLE.  v1 implicitly set
   P(reduction succeeds | subset all clean) = 1. Now --p-reduction-success
   enters the denominator.

5. p_m PARAMETERISATION IS NO LONGER OVER-DETERMINED.  v1 documented p_m as
   conditional on correct leading digits *and* offered a correlation knob;
   holding both risks double-counting the dependence.
     --p-mag-mode conditional : p_clean = p_b * p_m, correlation forced to 0
     --p-mag-mode marginal    : correlation is used, Frechet-clamped
   On this project's own simulated corpus the two predictors are NOT
   independent: measured rho ~ +0.126 with a marginal p_mag ~ 88.9%.

6. THE OUTPUT IS A DISTRIBUTION, NOT A POINT.  v1 froze the pool composition
   at its mean and labelled the result "expected runs". It is in fact close to
   the MEDIAN; the true expectation is dominated by rare bad pools and the
   5-95% band spans several orders of magnitude. This version reports p05,
   p50 and p95 of the number of reductions, compounding pool randomness with
   the geometric search.

7. FALSE PRECISION REMOVED.  Wall-clock is reported as orders of magnitude by
   default (--precision full restores decimals) and every table carries its
   unvalidated assumptions in the header.

8. HYGIENE.  Docstring/CLI defaults reconciled; the legend reports every swept
   L instead of only the first; the accuracy sweep is index-based so its grid
   cannot drift; rounding modes made explicit; the dead `clean < m_req` branch
   replaced by an actionable diagnostic.

STILL NOT MODELLED (deliberately -- do not read the output as if these were
solved)
-------------------------------------------------------------------------
  * Confidence ranking. A real classifier emits calibrated scores, so an
    attacker would take the top-m_req most confident signatures rather than a
    uniform random subset. That is strictly cheaper, by an unmodelled and
    potentially large factor. This model assumes a hard 0/1 label.
  * safety_factor has no lattice-theoretic content. The condition for
    reduction success is a determinant-versus-target-norm question (Gaussian
    heuristic, BKZ block size beta); m_req * L >= log2(n) is necessary, not
    sufficient. The 1.3 multiplier is a fudge and is treated as such.
  * p_b itself. See the first paragraph.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Pure mathematics
# ---------------------------------------------------------------------------

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

    and raises ValueError when (pi, recall, p_clean) admit no valid fpr --
    exactly the case v1 accepted silently.
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
            # q >= 1: the very first attempt succeeds, so P[N > n] = 0 for
            # any n >= 1. Guarding this is not cosmetic: log1p(-1.0) raises
            # a domain error, which a fully clean pool reaches exactly.
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
    filter] >= confidence. Replaces v1's bare 'n/a', which conflated
    'infeasible' with 'the point estimate fell below a threshold'.
    """
    if p_post <= 0.0:
        return None
    t_filt = m_req
    step = max(1, m_req // 8)
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
        t_filt += step
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Config:
    order_bits: int
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
    precision: str


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


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

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
                    note=(f"filtered pool {t_filt} < m_req {m_req}"
                          + (f"; need T_raw >= {need}" if need else "")),
                    times_p50={c: None for c in cfg.cores},
                ))
                continue

            # The pool composition is a random variable. Enumerate it instead
            # of freezing it at its mean (v1's Jensen error).
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
                # remaining mass is the 'too few clean signatures' event
                components.append((1.0 - mass, float("-inf")))
            if 1e-12 < mass < 0.99:
                need = min_raw_for_pool(m_req, p_post, cfg.filter_pass_rate)
                note = (f"P(pool holds >= m_req clean) = {mass:.3f}; "
                        f"quantiles above that level are unreachable"
                        + (f", need T_raw >= {need} for 95%" if need else ""))
            clean = round(p_post * t_filt)
            log10_q = log10_success_per_draw(t_filt, m_req, clean, p_post,
                                             cfg.p_reduction_success)

        q05 = runs_quantile(0.05, components)
        q50 = runs_quantile(0.50, components)
        q95 = runs_quantile(0.95, components)
        times = {c: (None if q50 is None else q50 + lt_run - log10(c))
                 for c in cfg.cores}

        rows.append(Row(
            leak_bits=cfg.leak_bits, p_ml_pct=pct, p_clean=p_clean,
            p_clean_post=p_post, implied_fpr=fpr, m_req=m_req, dimension=dim,
            t_raw=cfg.t_raw, t_filtered=t_filt, clean=clean, log10_q=log10_q,
            log10_runs_p05=q05, log10_runs_p50=q50, log10_runs_p95=q95,
            log10_t_run=lt_run, note=note, times_p50=times,
        ))
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_legend(cfg: Config, leak_bits: Sequence[int]) -> None:
    bar = "=" * 92
    print(bar)
    print("ECDSA two-predictor subset-attack complexity  (v2, audited)")
    print(bar)
    print()
    print("CLEAN signature = leading-digit predictor correct AND magnitude")
    print("predictor correct. One poisoned signature destroys the reduction.")
    print()
    if cfg.p_mag_mode == "conditional":
        print(f"  p_clean = p_b * p_m (p_m = {cfg.p_mag:.2f}%) "
              f"[conditional; rho forced to 0]")
    else:
        print("  p_clean = clamp(p_b*p_m + rho*sqrt(p_b(1-p_b)p_m(1-p_m)), "
              "Frechet)")
        print(f"            p_m = {cfg.p_mag:.2f}% [marginal], "
              f"rho = {cfg.correlation:+.3f}")
    print("  m_req   = ceil(ceil(log2(n)/L) * safety_factor)")
    print("  dim     = m_req + dim_overhead")
    if cfg.supply == "unlimited":
        print("  q       = p_clean^m_req * p_succ [unlimited supply]")
    else:
        print("  P(clean|pass) = p_clean * recall / pass_rate [Bayes]")
        print("  T_filt  = floor(pass_rate * T_raw)")
        print("  q       = C(C,m_req)/C(T_filt,m_req) * p_succ,  "
              "C ~ Bin(T_filt, P(clean|pass))")
    print("  N       ~ Geometric(q)  ->  p05 / p50 / p95 reported")
    print("  t_run   = t0 * 2^((dim - dim_ref)/doubling_dims)")
    print("  time    = N * t_run / cores")
    print()
    print(f"order_bits (log2 n) = {cfg.order_bits}")
    print(f"leak_bits (L)       = {list(leak_bits)}")
    print(f"safety_factor       = {cfg.safety_factor} [UNVALIDATED FUDGE]")
    print(f"supply              = {cfg.supply}"
          + (f"   T_raw = {cfg.t_raw}" if cfg.supply == "scarce" else ""))
    if cfg.supply == "scarce":
        print(f"filter pass_rate    = {cfg.filter_pass_rate}   "
              f"recall = {cfg.filter_recall}")
    print(f"p_reduction_success = {cfg.p_reduction_success}"
          + (" [OPTIMISTIC: reduction assumed infallible]"
             if cfg.p_reduction_success >= 1.0 else ""))
    ref = cfg.t_run_dim_ref if cfg.t_run_dim_ref is not None else "first L's dim"
    print(f"t_per_run_sec       = {cfg.t_per_run_sec:g} s at dim {ref}")
    if cfg.t_run_doubling_dims is None:
        print("                      [FLAT -- cost does not grow with "
              "dimension, which is")
        print("                       wrong; cross-L rows are NOT comparable]")
    else:
        print(f"doubling_dims       = {cfg.t_run_doubling_dims}")
    print(f"cores               = {cfg.cores}")
    print()
    print("Unvalidated assumptions carried by every number below:")
    for a in KNOWN_UNVALIDATED:
        print(f"  - {a}")
    print(bar)


def print_table(rows: list[Row], cfg: Config) -> None:
    if not rows:
        print("  (no rows)")
        return
    head = rows[0]
    print()
    line = (f"  L = {head.leak_bits} bit  |  m_req = {head.m_req}  |  "
            f"dim = {head.dimension}  |  "
            f"t/run = {10.0 ** head.log10_t_run:.4g} s")
    if cfg.supply == "scarce":
        line += f"  |  T_raw = {head.t_raw}  T_filt = {head.t_filtered}"
    print(line)
    print()

    cols = ["p_ml%", "p_clean"]
    widths = [7, 9]
    if cfg.supply == "scarce":
        cols.append("P(cln|ps)")
        widths.append(10)
    cols += ["N p05", "N p50", "N p95"]
    widths += [10, 10, 10]
    cols += [f"t@{c}c p50" for c in cfg.cores]
    widths += [13] * len(cfg.cores)

    print("  " + "".join(h.rjust(w) for h, w in zip(cols, widths)))
    print("  " + "-" * sum(widths))
    for r in rows:
        cells = [f"{r.p_ml_pct:g}", f"{r.p_clean:.4f}"]
        if cfg.supply == "scarce":
            cells.append(f"{r.p_clean_post:.4f}")
        cells += [fmt_log_count(r.log10_runs_p05),
                  fmt_log_count(r.log10_runs_p50),
                  fmt_log_count(r.log10_runs_p95)]
        cells += [fmt_duration(r.times_p50.get(c), cfg.precision)
                  for c in cfg.cores]
        print("  " + "".join(c.rjust(w) for c, w in zip(cells, widths)))
        if r.note:
            print(f"      ! {r.note}")
    print("  " + "-" * sum(widths))
    print("  Times are at the p50 of N. The p05-p95 band scales the times by")
    print("  exactly the same factor as it scales N.")


def write_csv(all_rows: list[Row], cfg: Config, path: str) -> None:
    fields = ["leak_bits", "p_ml_pct", "p_clean", "p_clean_post",
              "implied_false_pass_rate", "m_req", "dimension", "t_raw",
              "t_filtered", "clean_pointest", "log10_q",
              "log10_runs_p05", "log10_runs_p50", "log10_runs_p95",
              "log10_t_run_sec", "note"]
    for c in cfg.cores:
        fields += [f"log10_seconds_{c}cores_p50", f"t_{c}cores_p50_human"]

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
                   g(r.log10_q), g(r.log10_runs_p05), g(r.log10_runs_p50),
                   g(r.log10_runs_p95), g(r.log10_t_run), r.note]
            for c in cfg.cores:
                v = r.times_p50.get(c)
                row += [g(v), fmt_duration(v, cfg.precision)]
            w.writerow(row)
    print(f"\n  [CSV written -> {path}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class _Refuse(argparse.Action):
    """v1 flags whose semantics were unsound: fail loudly rather than lie."""

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
        description="ECDSA two-predictor subset-attack complexity (v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("threat parameters")
    g.add_argument("--curve-bits", type=int, default=256,
                   help="ceil(log2 n) for the curve order n")
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
                        "Default: equal to --filter-pass-rate, i.e. an "
                        "UNINFORMATIVE filter that only shrinks the pool. "
                        "Set it above the pass rate to model a filter that "
                        "actually enriches the pool with clean signatures.")

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
    o.add_argument("--precision", choices=["orders", "full"], default="orders",
                   help="orders: order-of-magnitude only, matching what the "
                        "model can actually support")
    o.add_argument("--no-legend", action="store_true")

    p.add_argument("--surplus", nargs="?", action=_Refuse,
                   help=argparse.SUPPRESS)

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
            a.filter_recall = a.filter_pass_rate      # uninformative filter
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
        order_bits=a.curve_bits, leak_bits=a.leak_bits[0],
        safety_factor=a.safety_factor, dim_overhead=a.dim_overhead,
        p_mag=a.p_mag, p_mag_mode=a.p_mag_mode, correlation=a.correlation,
        acc_start=a.p_ml_start, acc_stop=a.p_ml_stop, acc_step=a.p_ml_step,
        supply=a.supply, t_raw=a.t_raw,
        filter_pass_rate=a.filter_pass_rate, filter_recall=a.filter_recall,
        p_reduction_success=a.p_reduction_success,
        t_per_run_sec=a.t_per_run_sec, t_run_dim_ref=a.t_run_dim_ref,
        t_run_doubling_dims=a.t_run_doubling_dims,
        cores=a.cores, precision=a.precision,
    )

    if not a.no_legend:
        print_legend(base, a.leak_bits)

    if len(a.leak_bits) > 1 and a.t_run_doubling_dims is None:
        print()
        print("  WARNING: several L values with a flat per-run cost. The "
              "lattice dimension")
        print("  differs across these tables and the model charges the same "
              "time for all of")
        print("  them. Do not compare rows across L. Pass "
              "--t-run-doubling-dims.")

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

    if a.csv:
        write_csv(all_rows, base, a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
