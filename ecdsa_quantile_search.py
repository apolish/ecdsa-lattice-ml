#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecdsa_quantile_search.py
========================

Run ONE real Strategy-1 search against a transaction file and classify the
result against a pre-computed plan (the quantiles an attacker expects from the
oracle accuracy).  This answers the practical question:

    "I planned for p05 / p50 / p95 draws.  I run the real search on my actual
     transaction list -- where did this run land, and was I lucky or not?"

PLAN vs ACTUAL  (the two halves, kept strictly separate)
--------------------------------------------------------
* PLAN (what you expect, from the oracle).  Three quantiles --p5 / --p50 / --p95
  of the number of draws N.  These are INPUTS and are produced by the authority
  for the plan, ecdsa_attack_complexity.py, NOT recomputed here (the experiment
  must not be judged against its own copy of the model).  Their defaults are the
  model's scarce-mode output for the canonical oracle (p_ml=90%, p_mag=86.4%,
  T=1000, leak=6 -> m_req=56): p05=69,656, p50=1,326,417, p95=13,219,421.  For a
  different oracle, re-run ecdsa_attack_complexity.py and pass the new values.

* ACTUAL (what happens).  The real search draws random m-subsets from the real
  pool and checks each against the file's own predictor columns
  (ml_match_k_val, ml_match_k_mag).  A draw succeeds iff all m drawn signatures
  are clean (both columns "Yes").  You never know in advance which signatures
  are clean; you only learn it per subset -- exactly as a real attacker learns
  it from whether the lattice recovered the key.

OUTCOME
-------
Let  ceiling = p95 * (1 + tolerance/100).  The search runs until it either finds
an all-clean subset or reaches the ceiling:

    success      -- all-clean found at N <= p95            (within budget)
    over_p95     -- all-clean found at p95 < N <= ceiling  (recovered, unlucky)
    catastrophic -- ceiling reached with no success        (clearly unlucky; stop)
    interrupted  -- stopped (crash / Ctrl-C) before either -> resume continues

RESUME  (crash-safe, "mathematical")
------------------------------------
Draws come from a deterministic BLOCK-COUNTER engine: draw i is a pure function
of (seed, block_size, i), so the checkpoint stores only the next draw index.
On restart the engine recomputes draw i directly and continues from exactly
where it stopped -- up to the ceiling.  Example: if a run passed p50 without a
success and was killed, re-running resumes past p50 and keeps going toward the
ceiling.  Checkpoints are atomic and flushed on block boundaries and on signals.

LATTICE SEAM  (not implemented on purpose)
------------------------------------------
On a found subset the script calls solve_with_lattice() -- a stub returning None
-- and can write a handoff JSON (--emit-subset) with everything a future
HNP/lattice module needs.  The reduction itself is intentionally left out.

EVERYTHING IS A PARAMETER
-------------------------
Pool size, clean-subset size, oracle accuracies, the three plan quantiles, the
catastrophe tolerance, the data-file path and all resume controls are CLI flags
with sensible defaults, so different "oracle strategies" are just different
invocations.  See --help.

WARNING
-------
Every count here is the cheap combinatorial draw stage only, conditional on a
nonce-bit predictor oracle that does not exist yet (a partial-ECDLP problem,
~2^128).  This measures luck against a plan, never the real barrier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional, Sequence

# =========================================================================== #
# Section 1 -- count parsing and formatting helpers                           #
# =========================================================================== #

_SUFFIX = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}


def parse_count(text: str) -> int:
    """Parse '69.7K', '1.3M', '13_200_000', '13200000' -> int."""
    s = str(text).strip().replace("_", "").replace(",", "")
    if not s:
        raise ValueError("empty count")
    mult = 1
    if s[-1].upper() in _SUFFIX:
        mult = _SUFFIX[s[-1].upper()]
        s = s[:-1]
    value = float(s) * mult
    if value < 0:
        raise ValueError("count must be non-negative")
    return int(round(value))


def human_count(n: float) -> str:
    """Compact human form: 1326417 -> '1.33M'."""
    if n is None:
        return "n/a"
    if n == float("inf"):
        return "inf"
    a = abs(n)
    if a >= 1_000_000_000:
        return f"{n / 1e9:.2f}G"
    if a >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if a >= 1_000:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


def _fmt(n: Optional[float]) -> str:
    """Grouped integer form: 1326417 -> '1,326,417'."""
    if n is None:
        return "n/a"
    if n == float("inf"):
        return "inf"
    return f"{n:,.0f}"


# =========================================================================== #
# Section 2 -- data-file parser (transactions -> in-memory pool)              #
# =========================================================================== #

# Whitespace-separated columns; a data row is exactly 13 tokens.
COL = {
    "s": 0, "z": 1, "rx": 2, "ry": 3, "d": 4, "k": 5, "kinv": 6,
    "ml_exp_k_val": 7, "ml_act_k_val": 8, "ml_match_k_val": 9,
    "ml_exp_k_mag": 10, "ml_act_k_mag": 11, "ml_match_k_mag": 12,
}
_N_COLS = 13


@dataclass
class Pool:
    """The transactions loaded into memory, with the clean mask precomputed."""
    rows: list[dict]           # per-signature fields (strings), length T
    clean: list[bool]          # clean[i] = both predictor columns are "Yes"
    n: Optional[int]           # curve order from the header, if present
    path: str
    content_hash: str          # hash of the T used rows (checkpoint fingerprint)

    @property
    def T(self) -> int:
        return len(self.rows)

    @property
    def clean_count(self) -> int:
        return sum(self.clean)


def _is_data_row(tok: Sequence[str]) -> bool:
    return (
        len(tok) == _N_COLS
        and tok[0].isdigit()
        and tok[COL["ml_match_k_val"]] in ("Yes", "No")
        and tok[COL["ml_match_k_mag"]] in ("Yes", "No")
    )


def load_pool(path: str, expected_t: int) -> Pool:
    """
    Parse every usable row of `path` and require the count to equal exactly
    `expected_t`.  A mismatch -- in either direction -- stops the run: the
    declared pool size must match the data file, so a run is never silently
    carried out against a different pool than the user intended.

    Cleanness is read from the file's own predictor columns; it is never
    re-rolled.  The content hash covers all rows, tying the checkpoint to this
    exact file.
    """
    rows: list[dict] = []
    n_order: Optional[int] = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("n ="):
                try:
                    n_order = int(stripped.split("=", 1)[1].strip())
                except ValueError:
                    pass
            tok = line.split()
            if not _is_data_row(tok):
                continue
            rows.append({name: tok[idx] for name, idx in COL.items()})
    if not rows:
        raise SystemExit(f"error: no data rows parsed from {path!r}")
    if len(rows) != expected_t:
        verb = "more" if len(rows) > expected_t else "fewer"
        raise SystemExit(
            f"error: --transactions {expected_t} does not match the data file.\n"
            f"       {os.path.basename(path)} holds {len(rows)} usable "
            f"transactions ({verb} than {expected_t}).\n"
            f"       Set --transactions {len(rows)} to use this file, or point "
            f"--data at a file with exactly {expected_t} transactions.")
    clean = [r["ml_match_k_val"] == "Yes" and r["ml_match_k_mag"] == "Yes"
             for r in rows]
    h = hashlib.sha256()
    for r in rows:
        h.update((r["s"] + "|" + r["ml_match_k_val"] + r["ml_match_k_mag"]
                  + "\n").encode("utf-8"))
    return Pool(rows=rows, clean=clean, n=n_order, path=path,
                content_hash=h.hexdigest())


# =========================================================================== #
# Section 3 -- the PLAN quantiles (fixed defaults sourced from the model)      #
# =========================================================================== #
#
# The plan is NOT recomputed here.  This script is the EXPERIMENT; the authority
# for the plan is ecdsa_attack_complexity.py.  Re-deriving the quantiles here
# with a second copy of the model would mean the experiment is judged against
# its own reimplementation, not against the model -- so these defaults are fixed
# constants, taken verbatim from the model's scarce-mode output for the
# canonical oracle below:
#
#   $ python3 ecdsa_attack_complexity.py --supply scarce \
#         --p-ml-start 90 --p-ml-stop 90 --t-raw 1000 --leak-bits 6
#   -> p05 = 69,656   p50 = 1,326,417   p95 = 13,219,421   (69.7K / 1.3M / 13.2M)
#
# They are valid ONLY for that canonical scenario.  For any other oracle
# strategy (different accuracies, pool size, subset size, leak bits, pre-filter
# or lattice success probability) re-run ecdsa_attack_complexity.py and pass the
# resulting quantiles with --p5/--p50/--p95; this script will not re-derive them.

# Canonical scenario the default quantiles correspond to.
CANON_P_VAL = 0.90
CANON_P_MAG = 0.864
CANON_T = 1000
CANON_M = 56
# ecdsa_attack_complexity.py scarce-mode quantiles for the canonical scenario.
DEFAULT_P5 = 69_656
DEFAULT_P50 = 1_326_417
DEFAULT_P95 = 13_219_421

HEADER_LENGTH = 93


@dataclass
class Targets:
    """The plan the actual run is judged against (never computed here)."""
    p5: int
    p50: int
    p95: int
    tolerance_pct: float
    source: str                 # "default (...)" or "user-specified"

    @property
    def ceiling(self) -> int:
        return int(math.ceil(self.p95 * (1.0 + self.tolerance_pct / 100.0)))


def resolve_targets(args: argparse.Namespace) -> Targets:
    """Take the quantiles as given (defaults are the model's canonical output)."""
    p5, p50, p95 = args.p5, args.p50, args.p95
    if not (p5 <= p50 <= p95):
        raise SystemExit(f"error: quantiles must satisfy p5 <= p50 <= p95 "
                         f"(got {p5}, {p50}, {p95})")
    is_default = (p5 == DEFAULT_P5 and p50 == DEFAULT_P50 and p95 == DEFAULT_P95)
    source = ("default (ecdsa_attack_complexity.py, canonical scenario)"
              if is_default else "user-specified")
    return Targets(p5=p5, p50=p50, p95=p95,
                   tolerance_pct=args.p95_tolerance_pct, source=source)


def oracle_is_canonical(args: argparse.Namespace) -> bool:
    """True when the oracle/pool/subset match the scenario the defaults assume."""
    return (abs(args.p_val - CANON_P_VAL) < 1e-9
            and abs(args.p_mag - CANON_P_MAG) < 1e-9
            and args.transactions == CANON_T
            and args.subset_size == CANON_M)


# =========================================================================== #
# Section 4 -- resumable block-counter draw engine                            #
# =========================================================================== #

class DrawEngine:
    """
    Deterministic, resumable source of random m-subsets of range(T).

    Draw i is in block b = i // block_size at offset i % block_size; block b is
    the stream of random.Random(f"{seed}:{b}").  The engine keeps the current
    block's RNG live and reseeds only on a boundary, so throughput is high while
    any draw i remains a pure function of (seed, block_size, i).
    """

    def __init__(self, seed: str, T: int, m: int, block_size: int):
        if m > T:
            raise ValueError("subset size m cannot exceed pool size T")
        self.seed = seed
        self.T = T
        self.m = m
        self.block_size = block_size

    def _block_rng(self, b: int) -> random.Random:
        return random.Random(f"{self.seed}:{b}")

    def subset_for(self, i: int) -> list[int]:
        """Reconstruct the full m-subset of draw i directly (for the handoff)."""
        b, o = divmod(i, self.block_size)
        rng = self._block_rng(b)
        for _ in range(o):
            self._one(rng, collect=False)
        return self._one(rng, collect=True)

    def _one(self, rng: random.Random, collect: bool) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        while len(seen) < self.m:
            x = rng.randrange(self.T)
            if x in seen:
                continue
            seen.add(x)
            if collect:
                out.append(x)
        return out

    def scan(self, clean: Sequence[bool], start_index: int
             ) -> Iterator[tuple[int, bool, Optional[list[int]]]]:
        """
        Yield (draw_index, is_success, winning_subset_or_None) from start_index.
        Losing draws short-circuit at the first dirty member; the full subset is
        returned only on a success.
        """
        i = start_index
        b = -1
        rng: Optional[random.Random] = None
        bs, m, T = self.block_size, self.m, self.T
        while True:
            blk = i // bs
            if blk != b:
                rng = self._block_rng(blk)
                for _ in range(i % bs):           # fast-forward within the block
                    s_: set[int] = set()
                    while len(s_) < m:
                        x = rng.randrange(T)
                        if x not in s_:
                            s_.add(x)
                b = blk
            seen: set[int] = set()
            picks: list[int] = []
            win = True
            while len(seen) < m:
                x = rng.randrange(T)              # type: ignore[union-attr]
                if x in seen:
                    continue
                seen.add(x)
                picks.append(x)
                if not clean[x]:
                    win = False
                    break
            yield (i, win, picks if win else None)
            i += 1


# =========================================================================== #
# Section 5 -- checkpoint (atomic, fingerprinted)                             #
# =========================================================================== #

CHECKPOINT_VERSION = 4


@dataclass
class Checkpoint:
    version: int = CHECKPOINT_VERSION
    seed: str = ""
    data_hash: str = ""
    transactions: int = 0
    subset_size: int = 0
    block_size: int = 0
    # plan the run is tied to (so a resume cannot change the budget mid-flight)
    p5: int = 0
    p50: int = 0
    p95: int = 0
    tolerance_pct: float = 0.0
    # progress
    next_index: int = 0
    attempts: int = 0
    elapsed_accum: float = 0.0
    bands_announced: list[str] = field(default_factory=list)
    # terminal state
    outcome: Optional[str] = None
    found_index: Optional[int] = None
    found_subset: Optional[list[int]] = None

    def fingerprint(self) -> tuple:
        return (self.version, self.seed, self.data_hash, self.transactions,
                self.subset_size, self.block_size, self.p5, self.p50, self.p95,
                round(self.tolerance_pct, 6))


def save_checkpoint(path: str, ck: Checkpoint) -> None:
    """Atomic write (temp + os.replace), keeping a .bak of the prior state."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(asdict(ck), fh)
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.exists(path):
        try:
            os.replace(path, path + ".bak")
        except OSError:
            pass
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Optional[Checkpoint]:
    for candidate in (path, path + ".bak"):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                return Checkpoint(**json.load(fh))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


# =========================================================================== #
# Section 6 -- lattice integration seam (stub) + handoff                      #
# =========================================================================== #

def solve_with_lattice(subset_rows: list[dict], n_order: Optional[int]
                       ) -> Optional[int]:
    """
    Future HNP / lattice-reduction entry point.  Inputs per signature: s, z, rx
    (= r) and the predicted nonce bits (ml_exp_k_val / ml_exp_k_mag).  A real
    solver builds the Nguyen-Shparlinski lattice and runs LLL/BKZ + Babai to
    recover d.  This stub does nothing (keeps the search fast); replace the body
    to integrate a real attack.
    """
    return None


def emit_handoff(path: str, pool: Pool, subset: list[int], draw_index: int,
                 m: int) -> None:
    """Write the found subset as a self-contained JSON handoff for a lattice."""
    sigs = []
    for idx in subset:
        r = pool.rows[idx]
        sigs.append({
            "pool_index": idx,
            "s": r["s"], "z": r["z"], "r_x": r["rx"], "r_y": r["ry"],
            "ml_exp_k_val": r["ml_exp_k_val"], "ml_exp_k_mag": r["ml_exp_k_mag"],
            "d_star_synthetic": r["d"],   # present only because data is synthetic
        })
    payload = {
        "schema": "ecdsa-lattice-handoff/v1",
        "curve": "secp256k1",
        "n_order": str(pool.n) if pool.n is not None else None,
        "subset_size": m,
        "draw_index": draw_index,
        "pool_path": os.path.basename(pool.path),
        "pool_hash": pool.content_hash,
        "note": ("r_y is the raw affine coordinate mod p (NOT reduced mod n); "
                 "r_x is the signature r. Verify reduction in the lattice module."),
        "signatures": sigs,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


# =========================================================================== #
# Section 7 -- search driver + outcome classification                         #
# =========================================================================== #

# outcome constants
SUCCESS = "success"
OVER_P95 = "over_p95"
CATASTROPHIC = "catastrophic"
INTERRUPTED = "interrupted"


@dataclass
class SearchResult:
    outcome: str
    attempts: int                 # draws made (N if found, else draws at stop)
    found: bool
    found_index: Optional[int]
    subset: Optional[list[int]]
    elapsed: float


class Interrupt:
    """Cooperative stop flag set by SIGINT / SIGTERM."""
    def __init__(self) -> None:
        self.stop = False

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:    # noqa: ANN001
        self.stop = True


def _band_markers(attempts: int, targets: Targets, announced: list[str],
                  quiet: bool) -> None:
    """Announce the first time the running count passes each plan quantile."""
    for name, value in (("p5", targets.p5), ("p50", targets.p50),
                        ("p95", targets.p95)):
        if name not in announced and attempts >= value:
            announced.append(name)
            if not quiet:
                print(f"  [passed {name} = {_fmt(value)} at draw "
                      f"{_fmt(attempts)} -- still searching]", flush=True)


def run_search(pool: Pool, engine: DrawEngine, ck: Checkpoint, ck_path: str,
               targets: Targets, args: argparse.Namespace,
               interrupt: Interrupt) -> SearchResult:
    """Draw until all-clean found, or the ceiling is hit, or interrupted."""
    ceiling = targets.ceiling
    checkpoint_every = max(engine.block_size,
                           (args.checkpoint_every // engine.block_size)
                           * engine.block_size)
    t0 = time.perf_counter()
    last_ck = ck.next_index
    last_prog = ck.next_index

    def session_elapsed() -> float:
        return ck.elapsed_accum + (time.perf_counter() - t0)

    def persist(outcome: Optional[str], idx: int) -> None:
        ck.next_index = idx
        ck.attempts = idx
        ck.elapsed_accum = session_elapsed()
        ck.outcome = outcome
        save_checkpoint(ck_path, ck)

    try:
        for i, win, subset in engine.scan(pool.clean, ck.next_index):
            attempts = i + 1

            if win:
                outcome = SUCCESS if attempts <= targets.p95 else OVER_P95
                ck.found_index = i
                ck.found_subset = subset
                persist(outcome, attempts)
                return SearchResult(outcome, attempts, True, i, subset,
                                    session_elapsed())

            _band_markers(attempts, targets, ck.bands_announced, args.quiet)

            if attempts >= ceiling:
                persist(CATASTROPHIC, attempts)
                return SearchResult(CATASTROPHIC, attempts, False, None, None,
                                    session_elapsed())

            if not args.quiet and (i - last_prog) >= args.progress_every:
                last_prog = i
                rate = (i - last_ck + 1) / max(1e-9, time.perf_counter() - t0)
                print(f"  .. {_fmt(attempts)} draws   {_fmt(rate)}/s   "
                      f"{100.0 * attempts / ceiling:5.1f}% of ceiling", flush=True)

            if (attempts - last_ck) >= checkpoint_every and attempts % engine.block_size == 0:
                last_ck = attempts
                persist(None, attempts)

            if interrupt.stop:
                persist(INTERRUPTED, attempts)
                print(f"\n[interrupted] checkpoint saved at draw "
                      f"{_fmt(attempts)} -> {ck_path}")
                return SearchResult(INTERRUPTED, attempts, False, None, None,
                                    session_elapsed())
    except KeyboardInterrupt:
        persist(INTERRUPTED, ck.attempts)
        print(f"\n[interrupted] checkpoint saved -> {ck_path}")
        return SearchResult(INTERRUPTED, ck.attempts, False, None, None,
                            session_elapsed())


# =========================================================================== #
# Section 8 -- statistics report                                              #
# =========================================================================== #

_VERDICT = {
    SUCCESS: "SUCCESS -- clean subset found within the p95 budget",
    OVER_P95: "OVER p95 -- found, but beyond the planned 95th percentile (unlucky)",
    CATASTROPHIC: "CATASTROPHIC -- ceiling reached with no clean subset (very unlucky)",
    INTERRUPTED: "INTERRUPTED -- stopped before a decision; rerun to resume",
}


def _deviation_line(name: str, n: int, q: int) -> str:
    ratio = n / q if q else float("inf")
    pct = (n - q) / q * 100.0 if q else float("inf")
    side = "below" if n < q else ("above" if n > q else "at")
    return (f"  vs {name:<3} = {_fmt(q):>14} : N is {abs(pct):6.1f}% {side:<5} "
            f"({ratio:.3f}x)")


def print_header(pool: Pool, args: argparse.Namespace, targets: Targets,
                 m: int) -> None:
    pc = args.p_val * args.p_mag
    print("=" * HEADER_LENGTH)
    print("ECDSA Strategy-1 search -- actual run vs plan")
    print("=" * HEADER_LENGTH)
    print(f"data file      : {os.path.basename(pool.path)}")
    print(f"transactions T : {pool.T}")
    print(f"clean subset m : {m}   (a draw wins only if all m are clean)")
    print(f"oracle (plan)  : p_val = {args.p_val:.4f}  p_mag = {args.p_mag:.4f}"
          f"  ->  p_clean = {pc:.5f}")
    print(f"seed           : {args.seed!r}")
    print("-" * HEADER_LENGTH)
    print(f"PLAN quantiles : {targets.source}")
    print(f"  p05 = {_fmt(targets.p5)}   p50 = {_fmt(targets.p50)}"
          f"   p95 = {_fmt(targets.p95)}")
    print(f"  ceiling = p95 x (1 + {targets.tolerance_pct:.0f}%) = "
          f"{_fmt(targets.ceiling)}  (catastrophe stop)")
    print("=" * HEADER_LENGTH)


def report(result: SearchResult, pool: Pool, targets: Targets,
           args: argparse.Namespace, m: int) -> None:
    N = result.attempts
    print("\n" + "=" * HEADER_LENGTH)
    print("RESULT")
    print("=" * HEADER_LENGTH)
    print(f"verdict        : {_VERDICT[result.outcome]}")
    print(f"draws N        : {_fmt(N)}  ({human_count(N)})")
    print(f"elapsed        : {result.elapsed:.2f} s"
          f"   ({_fmt(N / result.elapsed) if result.elapsed > 0 else 'n/a'}/s)")

    print("-" * HEADER_LENGTH)
    print("deviation from plan:")
    print(_deviation_line("p05", N, targets.p5))
    print(_deviation_line("p50", N, targets.p50))
    print(_deviation_line("p95", N, targets.p95))
    # placement in words
    if result.found:
        if N <= targets.p5:
            where = "in the lucky 5% tail (<= p05)"
        elif N <= targets.p50:
            where = "better than the median (p05..p50)"
        elif N <= targets.p95:
            where = "worse than median but within budget (p50..p95)"
        else:
            where = "beyond p95 but under the ceiling (unlucky tail)"
        print(f"placement      : {where}")

    print("-" * HEADER_LENGTH)
    # post-hoc description of the realized pool (NOT known before the run).
    cc = pool.clean_count
    pc = args.p_val * args.p_mag
    print("post-hoc (observed only after running; not used for the plan):")
    print(f"  realized clean in pool : {cc}/{pool.T} = {cc / pool.T:.4f}")
    print(f"  oracle-implied p_clean : {pc:.5f}  "
          f"(realized pool is {'richer' if cc / pool.T > pc else 'poorer'} "
          f"in clean signatures)")

    if result.found:
        subset = sorted(result.subset or [])
        all_clean = all(pool.clean[i] for i in subset)
        print("-" * HEADER_LENGTH)
        print(f"subset all clean : {all_clean}")
        print(f"selected {len(subset)} clean signatures:")
        print(f"  {'#':>3}  {'idx':>5}   s")
        print("  " + "-" * (HEADER_LENGTH - 2))
        for rank, idx in enumerate(subset, 1):
            print(f"  {rank:>3}  {idx:>5}   {pool.rows[idx]['s']}")
        if args.emit_subset:
            emit_handoff(args.emit_subset, pool, subset,
                         result.found_index or 0, m)
            print(f"handoff written  : {args.emit_subset}")
        if args.run_lattice:
            d = solve_with_lattice([pool.rows[i] for i in subset], pool.n)
            print(f"lattice solve    : "
                  f"{'d = ' + str(d) if d else 'stub returned None (not implemented)'}")
    elif result.outcome == CATASTROPHIC:
        print("-" * HEADER_LENGTH)
        print(f"no clean subset in {_fmt(N)} draws (the ceiling). Either this")
        print("pool is genuinely clean-poor, or the plan's p95 was optimistic.")
    print("=" * HEADER_LENGTH)


# =========================================================================== #
# Section 9 -- self-test                                                       #
# =========================================================================== #

def self_test() -> int:
    T, m, B = 1000, 56, 64
    eng = DrawEngine("unit", T, m, B)
    # (1) direct reconstruction == sequential scan (incl. across a boundary)
    direct = [eng.subset_for(i) for i in (0, 1, 63, 64, 65, 200)]
    seq: dict[int, list[int]] = {}
    for i, _w, sub in eng.scan([True] * T, 0):
        if i in (0, 1, 63, 64, 65, 200):
            seq[i] = sub
        if i > 200:
            break
    ok1 = all(direct[k] == seq[i] for k, i in enumerate((0, 1, 63, 64, 65, 200)))
    # (2) count parsing
    ok2 = (parse_count("69.7K") == 69700 and parse_count("1.3M") == 1_300_000
           and parse_count("13.2M") == 13_200_000 and parse_count("13200000")
           == 13_200_000)
    # (3) ceiling math
    tg = Targets(10, 100, 1000, 50.0, "test")
    ok3 = tg.ceiling == 1500
    # (4) fixed default quantiles are ordered and render as the documented values
    ok4 = (DEFAULT_P5 < DEFAULT_P50 < DEFAULT_P95
           and human_count(DEFAULT_P5) == "69.7K"
           and human_count(DEFAULT_P50) == "1.33M"
           and human_count(DEFAULT_P95) == "13.22M")
    all_ok = ok1 and ok2 and ok3 and ok4
    print(f"self-test: reconstruct={ok1} parse={ok2} ceiling={ok3} "
          f"defaults={ok4} -> {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


# =========================================================================== #
# Section 10 -- CLI + main                                                     #
# =========================================================================== #

DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "transaction_list_20261116220426.txt")


def _count_arg(text: str) -> int:
    try:
        return parse_count(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one real Strategy-1 search against a transaction file "
                    "and classify it against a planned p05/p50/p95 budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    p.add_argument("--data", default=DEFAULT_DATA,
                   help="path to the transaction list file")
    # scenario / pool
    p.add_argument("--transactions", type=int, default=1000,
                   help="T: declared pool size; MUST equal the number of usable "
                        "transactions in --data, or the run stops")
    p.add_argument("--subset-size", type=int, default=56,
                   help="m: clean signatures to collect (all m drawn must be clean)")
    # oracle accuracies (context the attacker has)
    p.add_argument("--p-val", type=float, default=0.90,
                   help="oracle accuracy on the nonce value channel")
    p.add_argument("--p-mag", type=float, default=0.864,
                   help="oracle accuracy on the nonce magnitude channel")
    # plan quantiles (defaults = ecdsa_attack_complexity.py canonical output)
    p.add_argument("--p5", type=_count_arg, default=DEFAULT_P5,
                   help="plan 5th pct of draws (default from the model: 69.7K)")
    p.add_argument("--p50", type=_count_arg, default=DEFAULT_P50,
                   help="plan 50th pct of draws (default from the model: 1.3M)")
    p.add_argument("--p95", type=_count_arg, default=DEFAULT_P95,
                   help="plan 95th pct of draws (default from the model: 13.2M)")
    p.add_argument("--p95-tolerance-pct", type=float, default=50.0,
                   help="catastrophe ceiling = p95 x (1 + this%%)")
    # resume / engine
    p.add_argument("--seed", default="ecdsa-lattice-ml",
                   help="master seed fixing the deterministic draw sequence")
    p.add_argument("--block-size", type=int, default=4096,
                   help="draws per reseed (resume granularity)")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint file (default: <data_dir>/.search_checkpoint.json)")
    p.add_argument("--checkpoint-every", type=int, default=1_000_000,
                   help="checkpoint cadence in draws (rounded to a block)")
    p.add_argument("--progress-every", type=int, default=500_000,
                   help="progress line cadence in draws")
    p.add_argument("--restart", action="store_true",
                   help="ignore any existing checkpoint and start fresh")
    p.add_argument("--force", action="store_true",
                   help="resume even if the checkpoint fingerprint mismatches")
    # lattice handoff
    p.add_argument("--emit-subset", default=None,
                   help="write the found subset as a lattice handoff JSON")
    p.add_argument("--run-lattice", action="store_true",
                   help="call the (stubbed) lattice seam on success")
    # misc
    p.add_argument("--quiet", action="store_true", help="suppress progress lines")
    p.add_argument("--self-test", action="store_true",
                   help="run fast internal invariants and exit")

    args = p.parse_args(argv)
    if args.transactions <= 0:
        p.error("--transactions must be positive")
    if args.subset_size <= 0:
        p.error("--subset-size must be positive")
    if args.subset_size > args.transactions:
        p.error("--subset-size cannot exceed --transactions")
    if not (0.0 < args.p_val <= 1.0) or not (0.0 < args.p_mag <= 1.0):
        p.error("--p-val and --p-mag must be in (0, 1]")
    if args.p95_tolerance_pct < 0:
        p.error("--p95-tolerance-pct must be non-negative")
    if args.block_size <= 0:
        p.error("--block-size must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    pool = load_pool(args.data, args.transactions)
    m = args.subset_size
    targets = resolve_targets(args)
    if targets.source.startswith("default") and not oracle_is_canonical(args):
        print("!" * HEADER_LENGTH)
        print("WARNING: using the DEFAULT plan quantiles, but the oracle/pool/"
              "subset differ")
        print(f"  from the canonical scenario they were computed for "
              f"(p_val={CANON_P_VAL}, p_mag={CANON_P_MAG}, T={CANON_T}, "
              f"m={CANON_M}).")
        print("  The default quantiles do NOT match this scenario. Re-run")
        print("  ecdsa_attack_complexity.py for your oracle and pass "
              "--p5/--p50/--p95,")
        print("  otherwise the deviation analysis compares against the wrong plan.")
        print("!" * HEADER_LENGTH)
    print_header(pool, args, targets, m)

    ck_path = args.checkpoint or os.path.join(
        os.path.dirname(os.path.abspath(args.data)) or ".",
        ".search_checkpoint.json")

    fresh = Checkpoint(
        seed=args.seed, data_hash=pool.content_hash, transactions=pool.T,
        subset_size=m, block_size=args.block_size, p5=targets.p5,
        p50=targets.p50, p95=targets.p95, tolerance_pct=targets.tolerance_pct)

    ck = None if args.restart else load_checkpoint(ck_path)
    if ck is not None and ck.fingerprint() != fresh.fingerprint():
        if not args.force:
            print(f"\nerror: checkpoint {ck_path} does not match this run")
            print(f"  checkpoint: {ck.fingerprint()}")
            print(f"  this run  : {fresh.fingerprint()}")
            print("  use --restart to overwrite, or --force to resume anyway.")
            return 3
        ck = None
    if ck is None:
        ck = fresh
    elif ck.outcome in (SUCCESS, OVER_P95, CATASTROPHIC):
        print(f"[done] this checkpoint already reached a terminal outcome: "
              f"{ck.outcome}. Use --restart to run again.")
        result = SearchResult(ck.outcome, ck.attempts, ck.found_index is not None,
                              ck.found_index, ck.found_subset, ck.elapsed_accum)
        report(result, pool, targets, args, m)
        return 0
    else:
        print(f"[resume] continuing from draw {_fmt(ck.next_index)} "
              f"(elapsed so far {ck.elapsed_accum:.1f} s)\n")

    interrupt = Interrupt()
    interrupt.install()

    engine = DrawEngine(args.seed, pool.T, m, args.block_size)
    result = run_search(pool, engine, ck, ck_path, targets, args, interrupt)
    if result.outcome != INTERRUPTED:
        report(result, pool, targets, args, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
