#!/usr/bin/env python3
"""Most-likely decimal length (magnitude) of a uniform k-nonce, given its first
two decimal digits, for either secp256k1 parameter set (TEST_PARAMS or
LEGACY_PARAMS).

Honest about its own limits -- and the limits are the same for both curves. The
leading two digits carry almost no information about the magnitude: the dominant
prediction comes from the curve order n alone (the bulk magnitude is reached
with no digits at all). The leading pair adds only a boundary effect, capped at
~90% accuracy, that exists purely because n is not a power of ten. Which leading
pairs are "boundary" pairs and what the dominant magnitude is both depend on the
curve, but the conclusion does not: for a correctly (uniformly) generated nonce
this is NOT a usable nonce-bit predictor. In a real attack you never observe a
secret nonce's decimal digits, and the residual magnitude uncertainty is ~3.3
bits (log2 10) per magnitude step.

Empirically, on LEGACY_PARAMS pairs 12..99 (88 of the 90 possible pairs)
predict D=77 at p=0.90; the two exceptions are pair 10 (D=78, p=0.90) and pair
11 (D=78, p=0.839). The predictor is therefore essentially a constant function:
its ~0.90 accuracy is the base rate "n has 78 decimal digits, so ~90% of values
below it have 77", not information gained from the leading pair.

CONDITIONING
============
``magnitude_distribution`` is conditioned on "k has at least two decimal
digits", because the leading-PAIR model is undefined for k in 1..9. Lengths
therefore start at D=2 and the returned probabilities sum to 1 over that
conditional space. For a 256-bit curve P(k < 10) is about 1e-76, so this is
bookkeeping, not an approximation.

CLI:
    python3 -m ecurve.k_magnitude --curve-mode {test,legacy} [--digits DIGITS]
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from typing import NamedTuple

try:  # works as a package module (python -m ecurve.k_magnitude)
    from .secp256k1 import CurveParams, LEGACY_PARAMS, TEST_PARAMS
except ImportError:  # and standalone (python k_magnitude.py, or imported flat)
    from secp256k1 import CurveParams, LEGACY_PARAMS, TEST_PARAMS

# Curves selectable via --curve-mode. Orders come straight from secp256k1.py
# (single source of truth -- no hardcoded copies here).
CURVES: dict[str, CurveParams] = {
    "test": TEST_PARAMS,
    "legacy": LEGACY_PARAMS,
}


class Prediction(NamedTuple):
    magnitude: int        # most likely number of decimal digits of k
    probability: float    # P(D = magnitude | first two decimal digits)


def _count(d: int, lead: int, n: int) -> int:
    """Count integers in [1, n-1] with exactly ``d`` decimal digits whose first
    two decimal digits equal ``lead`` (10..99)."""
    lo = max(lead * 10 ** (d - 2), 1)
    hi = min((lead + 1) * 10 ** (d - 2) - 1, n - 1)
    return max(0, hi - lo + 1)


def _normalize_lead(value) -> int:
    """Accept an int in 10..99 (e.g. 47) or any value whose decimal form starts
    with two digits (the rest is ignored), and return the leading pair.
    """
    s = str(value).strip().lstrip("+")
    if not s.isdigit():
        raise ValueError(f"expected decimal digits, got {value!r}")
    if len(s) < 2:
        raise ValueError(
            f"need at least two leading decimal digits, got {s!r}; the "
            f"leading-pair model is undefined for nonces below 10"
        )
    lead = int(s[:2])
    if not 10 <= lead <= 99:
        raise ValueError("the leading two digits must form a number in 10..99")
    return lead


def magnitude_distribution(first_two_digits, curve_n: int) -> dict[int, float]:
    """Full exact conditional distribution P(D = d | first two digits) for the
    given curve order ``n``, as {decimal_length: probability} for lengths with
    P > 0.

    Conditioned on k having at least two decimal digits; see module docstring.

    >>> dist = magnitude_distribution(47, LEGACY_PARAMS.n)
    >>> round(sum(dist.values()), 12)
    1.0
    >>> round(dist[77], 4)
    0.9
    """
    lead = _normalize_lead(first_two_digits)
    if curve_n < 100:
        raise ValueError("curve order must have at least two decimal digits")
    max_digits = len(str(curve_n))
    counts = {d: _count(d, lead, curve_n) for d in range(2, max_digits + 1)}
    total = sum(counts.values())
    if total == 0:
        raise ValueError(
            f"leading pair {lead} is impossible for an order with {max_digits} digits"
        )
    return {d: float(Fraction(c, total)) for d, c in counts.items() if c > 0}


def predict_magnitude(first_two_digits, curve_n: int) -> Prediction:
    """Most likely decimal length of a uniform k-nonce and the probability of
    that length, given its first two decimal digits and the curve order ``n``.

    On real secp256k1 (``n`` has 78 digits and begins "11"), most leading
    pairs land below ``n`` only at 77 digits:

    >>> predict_magnitude(47, LEGACY_PARAMS.n)
    Prediction(magnitude=77, probability=0.9)
    >>> predict_magnitude(10, LEGACY_PARAMS.n).magnitude
    78

    The TEST_PARAMS results below are tied to the *current* test curve and
    must be updated whenever it changes -- they are documentation of shipped
    behaviour, not curve-independent facts. ``secp17k1`` has ``n = 99667``,
    which begins "99", so every leading pair fits at full width:

    >>> predict_magnitude(47, TEST_PARAMS.n).magnitude
    5
    >>> predict_magnitude(99, TEST_PARAMS.n).magnitude
    5

    What *is* curve-independent is the shape: the predictor is essentially a
    constant function, so one magnitude dominates almost every leading pair.

    >>> from collections import Counter
    >>> counts = Counter(predict_magnitude(lead, TEST_PARAMS.n).magnitude
    ...                  for lead in range(10, 100))
    >>> counts.most_common(1)[0][1] >= 88
    True
    """
    dist = magnitude_distribution(first_two_digits, curve_n)
    best = max(dist, key=dist.get)
    return Prediction(magnitude=best, probability=dist[best])


def _format_distribution(dist: dict[int, float]) -> str:
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"D={d}: {p:.4f}" for d, p in items if p >= 5e-4)


def _lead_arg(value: str) -> int:
    try:
        return _normalize_lead(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Most likely decimal length (magnitude) of a uniform "
                    "k-nonce given its first two decimal digits.",
    )
    parser.add_argument(
        "--curve-mode", choices=["test", "legacy"], default="legacy",
        help="which secp256k1 parameter set to use (default: legacy)",
    )
    parser.add_argument(
        "--digits", nargs="?", type=_lead_arg,
        help="first two decimal digits of the nonce (10..99). If omitted, "
             "predictions for representative leading pairs are printed.",
    )
    args = parser.parse_args(argv)

    curve = CURVES[args.curve_mode]
    n = curve.n
    max_digits = len(str(n))
    print(f"curve-mode = {args.curve_mode}: n has {max_digits} decimal digits, "
          f"starts with {str(n)[:2]}")

    if args.digits is not None:
        p = predict_magnitude(args.digits, n)
        dist = magnitude_distribution(args.digits, n)
        print(f"first two digits = {args.digits}")
        print(f"  most likely magnitude = {p.magnitude} digits  (p = {p.probability:.4f})")
        print(f"  full distribution: {_format_distribution(dist)}")
        return

    print("representative predictions:")
    for lead in range(10, 100, 1):
        p = predict_magnitude(lead, n)
        dist = magnitude_distribution(lead, n)
        print(f"  {lead:>2}  ->  magnitude = {p.magnitude:>2}  (p = {p.probability:.4f})"
              f" full distribution: {_format_distribution(dist)}")


if __name__ == "__main__":
    main()
