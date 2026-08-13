#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""secp256k1 transaction data generator with a SIMULATED nonce predictor.

READ THIS BEFORE INTERPRETING ANY OUTPUT
========================================
The "ML predictor" in this file does not predict anything. It is a shuffled
list of booleans matching a target accuracy supplied on the command line:

    is_correct_list = [True] * num_correct + [False] * (total - num_correct)
    random.shuffle(is_correct_list)

The quantity it "predicts" (``str(k)[:2]``) is derived from the SECRET nonce,
which an attacker never observes. ``--ml-exp-accuracy`` is therefore an INPUT
ASSUMPTION, not a measurement. Recovering leading bits of k from public
signature data (r, s, z, Q) is equivalent to solving ECDLP; no model here
attempts it, and the real hook is still a TODO. Any downstream complexity
figure inherits this assumption wholesale.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Iterable, List, Sequence, Tuple

try:
    from ecurve.secp256k1 import (
        Secp256k1,
        TEST_PARAMS,
        LEGACY_PARAMS,
        CurveParams,
        make_bitcoin_legacy_sighash_message,
        make_bitcoin_segwit_sighash_message,
    )
    from ecurve.k_magnitude import predict_magnitude
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from secp256k1 import (  # type: ignore[no-redef]
            Secp256k1,
            TEST_PARAMS,
            LEGACY_PARAMS,
            CurveParams,
            make_bitcoin_legacy_sighash_message,
            make_bitcoin_segwit_sighash_message,
        )
        from k_magnitude import predict_magnitude  # type: ignore[no-redef]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import secp256k1/k_magnitude either as the 'ecurve' "
            "package or as flat modules next to this file. Expected one of:\n"
            "  <root>/ecurve/{__init__,secp256k1,k_magnitude}.py   (package)\n"
            "  <root>/{secp256k1,k_magnitude}.py                   (flat)\n"
            f"sys.path[0] = {sys.path[0]!r}"
        ) from exc


@dataclass(frozen=True)
class ReportConfig:
    column_widths: Sequence[int]
    line_length: int
    headers: Sequence[str] = (
        "s", "z", "r(x)", "r(y)", "d*", "k*", "k^(-1)*",
        "ml_exp_k_val", "ml_act_k_val", "ml_match_k_val",
        "ml_exp_k_mag", "ml_act_k_mag", "ml_match_k_mag",
    )


def _build_report_config(curve: CurveParams) -> ReportConfig:
    count = 7
    ml_col_width = 15
    if curve.mode == "test":
        getcontext().prec = 20
        width = 12
    else:
        getcontext().prec = 80
        width = 81
    column_widths = ([width] * count) + [ml_col_width] * 6
    line_length = (width * count) + (ml_col_width * 6)
    return ReportConfig(column_widths=column_widths, line_length=line_length)


def _print_once_demo(ec: Secp256k1, sig_type: str = "p2pkh") -> None:
    """Single key/sign/verify demonstration."""

    t0 = time.time()
    pk, pub = ec.generate_keypair()
    print("Private key:")
    print(f"  d: {pk}, ({bin(pk)[2:]})")
    print("Public key:")
    print(f"  x: {pub[0]}")
    print(f"  y: {pub[1]}")
    print(f"Spent time: {time.time() - t0:.3f} sec.\n")

    t0 = time.time()
    print(f"Is the point on curve?: {ec.is_on_curve(pub)}")
    print(f"Spent time: {time.time() - t0:.3f} sec.\n")

    t0 = time.time()
    if ec.curve.mode == "legacy":
        prev_txid = os.urandom(32)
        if sig_type == "p2wpkh":
            message = make_bitcoin_segwit_sighash_message(pub, prev_txid)
            print("Using BIP 143 SegWit v0 preimage for signing.")
        else:
            message = make_bitcoin_legacy_sighash_message(pub, prev_txid)
            print("Using legacy P2PKH preimage for signing.")
    else:
        message = b"Hello, secp256k1!"
        print("Using random message hashes and nonces (only for test curves).")

    sig = ec.sign_message(pk, message, min_start_range=1)
    z, r_x, r_y, s, k, k_inv = sig
    print("Signature parameters:")
    print(f"  z:   {hex(z)[2:]}")
    print(f"  r:   {hex(r_x)[2:]}")
    print(f"  s:   {hex(s)[2:]}")
    print(f"  k:   {hex(k)[2:]}")
    print(f"  k-1: {hex(k_inv)[2:]}")
    print(f"Spent time: {time.time() - t0:.3f} sec.\n")

    t0 = time.time()
    print(f"Signature validation: {ec.verify_signature(pub, sig)}")
    print(f"Spent time: {time.time() - t0:.3f} sec.\n")


def _format_row(row: Sequence[object], widths: Sequence[int]) -> str:
    return "".join(f"{str(val):<{w}}" for val, w in zip(row, widths))


def _write_report(
    curve: CurveParams,
    cfg: ReportConfig,
    rows: List[Sequence[object]],
    stats: dict,
    sig_type: str = "p2pkh",
    outdir: str = ".",
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = os.path.join(outdir, f"transaction_list_{timestamp}.txt")
    header_line = "-" * cfg.line_length

    with open(fname, "w", encoding="utf-8") as f:
        f.write("Elliptic curve parameters:\n")
        f.write(f"name = {curve.name}\n")
        f.write(f"mode = {curve.mode}\n")
        f.write(f"p = {curve.p}\n")
        f.write(f"a = {curve.a}\n")
        f.write(f"b = {curve.b}\n")
        f.write(f"g = {curve.g}\n")
        f.write(f"n = {curve.n}\n\n")
        f.write("Formulas:\n")
        f.write("  s = (z + r_x * d) * k^(-1)\n")
        f.write("  r(x,y) = k * G\n\n")
        f.write("NOTE: r(y) is the raw affine coordinate mod p (NOT reduced mod n).\n\n")
        if curve.mode == "legacy":
            if sig_type == "p2wpkh":
                f.write("Using BIP 143 SegWit v0 preimage for signing.\n\n")
            else:
                f.write("Using legacy P2PKH preimage for signing.\n\n")
        else:
            f.write("Using random message hashes and nonces (only for test curves).\n\n")

        f.write("WARNING: ml_act_k_val is a SIMULATION driven by --ml-exp-accuracy,\n")
        f.write("not a model output. See the module docstring.\n\n")

        if rows:
            f.write(f"{header_line}\n")
            f.write(_format_row(cfg.headers, cfg.column_widths) + "\n")
            f.write(f"{header_line}\n")
            for row in rows:
                f.write(_format_row(row, cfg.column_widths) + "\n")
            f.write(f"{header_line}\n")
            f.write(_format_row(cfg.headers, cfg.column_widths) + "\n")
            f.write(f"{header_line}\n")

        f.write("\nStatistics:\n\n")

        base_keys = [
            "total_key_count",
            "transaction_limit_per_key",
            "total_transaction_count",
            "maximum_transaction_count",
            "ml_accuracy_k_val",
            "ml_accuracy_k_mag",
        ]

        dynamic_items: list[tuple[int, str]] = []
        for key in stats:
            m = re.fullmatch(r"k_mag_(\d+)_count", key)
            if m:
                dynamic_items.append((int(m.group(1)), key))
        dynamic_items.sort(key=lambda t: t[0])

        for key in base_keys:
            f.write(f"{key.replace('_', ' ').title()}: {stats.get(key, 0)}\n")

        for _, key in dynamic_items:
            f.write(f"{key.replace('_', ' ').title()}: {stats[key]}\n")

        f.write(f"Spent time: {stats.get('spent_time_sec', 0.0):.3f} sec.\n")

    return fname


def _collect_rows(
    ec: Secp256k1,
    total_key_count: int,
    private_key: int,
    transaction_limit_per_key: int,
    min_start_range: int,
    ml_expected_accuracy: float,
    sig_type: str = "p2pkh",
) -> Tuple[List[Sequence[object]], dict]:

    start = time.time()

    if not 0.0 <= ml_expected_accuracy <= 1.0:
        raise ValueError("ml_expected_accuracy must be in [0.0, 1.0]")
    if transaction_limit_per_key < 1:
        raise ValueError("transaction_limit_per_key must be >= 1")

    if private_key != 0:
        if not 1 <= private_key < ec.curve.n:
            raise ValueError(
                f"--private-key must be in [1, n-1]; got {private_key}"
            )
        uniq_keys = [private_key]
    else:
        if total_key_count < 1:
            raise ValueError("total_key_count must be >= 1")
        uniq_keys = ec.generate_unique_keys(total_key_count, min_start_range)

    effective_key_count = len(uniq_keys)
    planned_tx = effective_key_count * transaction_limit_per_key

    rows: List[Sequence[object]] = []
    progress_step = max(1, effective_key_count // 10)

    # =========================== ML SIMULATION CODE ===============================
    # NOT A MODEL. A shuffled boolean mask sized to the transactions that will
    # actually be generated.
    num_correct = round(ml_expected_accuracy * planned_tx)
    is_correct_list = [True] * num_correct + [False] * (planned_tx - num_correct)
    random.shuffle(is_correct_list)
    correct_idx = 0
    # =========================== ML SIMULATION CODE ===============================

    magnitude_counter: Counter[int] = Counter()

    for i, current_key in enumerate(uniq_keys, start=1):
        # generate_keypair returns d*G, which is on the curve by construction
        # (or raises for the point at infinity), so no is_on_curve guard is
        # needed here -- an earlier defensive branch was dead code.
        _, public_key = ec.generate_keypair(current_key)

        for _ in range(transaction_limit_per_key):
            if ec.curve.mode == "legacy":
                prev_txid = os.urandom(32)
                prev_index = random.randint(0, 3)
                input_value_sats = random.randint(10_000, 100_0000_0000)
                fee_sats = random.randint(1_000, 50_000)
                output_value_sats = input_value_sats - fee_sats

                if sig_type == "p2wpkh":
                    msg = make_bitcoin_segwit_sighash_message(
                        public_key, prev_txid, prev_index,
                        input_value_sats, output_value_sats,
                    )
                else:
                    msg = make_bitcoin_legacy_sighash_message(
                        public_key, prev_txid, prev_index,
                        output_value_sats,
                    )
            else:
                rnd = random.randrange(1, ec.curve.n - 1)
                msg = str(rnd).encode()

            z, r_x, r_y, s, k, k_inv = ec.sign_message(current_key, msg, min_start_range)

            # TODO: Implement actual ML model prediction based on r(x, y) and the
            # public signature parameters. This is the load-bearing gap: doing so
            # is equivalent to recovering partial information about k from k*G,
            # i.e. solving (a partial) ECDLP. Nothing below attempts it.
            ml_exp_k_val = str(k)[:2]

            if len(ml_exp_k_val) < 2:
                correct_idx += 1
                continue

            # Simulation of ML model behavior with the specified accuracy
            # =========================== ML SIMULATION CODE ===============================
            is_correct = is_correct_list[correct_idx]
            correct_idx += 1
            if is_correct:
                ml_act_k_val = ml_exp_k_val
            else:
                while True:
                    ml_act_k_val = f"{random.randint(10, 99):02d}"
                    if ml_act_k_val != ml_exp_k_val:
                        break
            ml_match_k_val = "Yes" if ml_act_k_val == ml_exp_k_val else "No"
            # =========================== ML SIMULATION CODE ===============================

            k_magnitude_prediction = predict_magnitude(ml_act_k_val, ec.curve.n)
            ml_act_k_mag = k_magnitude_prediction.magnitude
            ml_exp_k_mag = len(str(k))
            ml_match_k_mag = "Yes" if ml_exp_k_mag == ml_act_k_mag else "No"

            magnitude_counter[ml_exp_k_mag] += 1

            rows.append([
                s, z, r_x, r_y, current_key, k, k_inv,
                ml_exp_k_val, ml_act_k_val, ml_match_k_val,
                ml_exp_k_mag, ml_act_k_mag, ml_match_k_mag,
            ])

        if i % progress_step == 0:
            print(f"{i} keys ({len(rows)} transactions) generated...")

    elapsed = time.time() - start

    total_transaction_count = len(rows)
    if total_transaction_count:
        ml_accuracy_k_val = sum(1 for row in rows if row[9] == "Yes") / total_transaction_count
        ml_accuracy_k_mag = sum(1 for row in rows if row[12] == "Yes") / total_transaction_count
    else:
        ml_accuracy_k_val = ml_accuracy_k_mag = 0.0

    # Cardinality of the (z, r, d, k^-1) tuple space. A combinatorial ceiling,
    # not a security bound and not an attack cost.
    maximum_transaction_count = Decimal(ec.curve.n - 1) ** 4

    stats = {
        "total_key_count": effective_key_count,
        "transaction_limit_per_key": transaction_limit_per_key,
        "total_transaction_count": total_transaction_count,
        "maximum_transaction_count": f"{maximum_transaction_count:.6e}",
        "ml_accuracy_k_val": f"{ml_accuracy_k_val:.2%} (INPUT ASSUMPTION, not measured)",
        "ml_accuracy_k_mag": f"{ml_accuracy_k_mag:.2%}",
        "spent_time_sec": elapsed,
    }
    for digits, count in magnitude_counter.items():
        stats[f"k_mag_{digits}_count"] = count

    return rows, stats


def _select_curve(mode: str) -> CurveParams:
    if mode == "test":
        return TEST_PARAMS
    if mode == "legacy":
        return LEGACY_PARAMS
    raise ValueError("Mode must be 'test' or 'legacy'!")


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {ivalue}")
    return ivalue


def _unit_float(value: str) -> float:
    fvalue = float(value)
    if not 0.0 <= fvalue <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0.0, 1.0], got {fvalue}")
    return fvalue


def _nonneg_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {ivalue}")
    return ivalue


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="secp256k1 transaction data generator (simulated predictor)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--curve-mode", choices=["test", "legacy"], default="legacy",
                   help="Curve mode: test or legacy.")
    p.add_argument("--private-key", type=_nonneg_int, default=0,
                   help="Private key to use (0 = random pool, else this single key)")
    p.add_argument("--keys", type=_positive_int, default=1,
                   help="Total unique keys to generate")
    p.add_argument("--tx-per-key", type=_positive_int, default=200,
                   help="Transactions per key")
    p.add_argument("--min-start-range", type=_positive_int, default=10,
                   help="Minimum start range for private key and k-nonce generation")
    p.add_argument("--ml-exp-accuracy", type=_unit_float, default=0.90,
                   help="ASSUMED accuracy of the simulated predictor (not measured)")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed the non-cryptographic RNG for reproducible runs")
    p.add_argument("--outdir", type=str, default=".",
                   help="Directory to write the report into")
    p.add_argument("--demo", action="store_true",
                   help="Run demo of key generation, signing, and verification")
    p.add_argument("--sig-type", choices=["p2pkh", "p2wpkh"], default="p2pkh",
                   help="Signature type: p2pkh (legacy) or p2wpkh (segwit)")

    args = p.parse_args(list(argv) if argv is not None else None)

    curve = _select_curve(args.curve_mode)
    if args.private_key and not 1 <= args.private_key < curve.n:
        p.error(f"--private-key must be in [1, {curve.n - 1}]")
    if args.min_start_range >= curve.n:
        p.error(f"--min-start-range must be < n ({curve.n})")
    available = curve.n - args.min_start_range
    if args.private_key == 0 and args.keys > available:
        p.error(
            f"--keys {args.keys} exceeds the {available} values available in "
            f"[{args.min_start_range}, {curve.n - 1}]"
        )
    return args


def main() -> None:
    args = parse_args()

    # Reproducibility (README section 5, CI "test curve corpus (reproducible)")
    # requires seeding EVERY source of randomness, not just the module-level
    # ``random`` used for the boolean mask and the test-mode message integer.
    # The private keys and, on the test curve, z and k are drawn inside
    # ``Secp256k1``; with no rng passed it defaults to ``SystemRandom`` for z/k
    # and ``secrets`` for keys, neither of which ``random.seed`` can influence.
    # So the previous code seeded the mask but left keys and nonces
    # non-deterministic, and two runs with the same --seed did NOT match.
    # Passing a seeded ``random.Random`` wires the curve's own rng into the same
    # determinism (secp256k1.py already honours an explicit rng end to end).
    rng = None
    if args.seed is not None:
        random.seed(args.seed)
        rng = random.Random(args.seed)

    curve = _select_curve(args.curve_mode)
    ec = Secp256k1(curve, rng=rng)
    cfg = _build_report_config(curve)

    os.makedirs(args.outdir, exist_ok=True)

    if args.demo:
        _print_once_demo(ec, args.sig_type)

    rows, stats = _collect_rows(
        ec=ec,
        total_key_count=args.keys,
        private_key=args.private_key,
        transaction_limit_per_key=args.tx_per_key,
        min_start_range=args.min_start_range,
        ml_expected_accuracy=args.ml_exp_accuracy,
        sig_type=args.sig_type,
    )
    path = _write_report(curve, cfg, rows, stats, args.sig_type, args.outdir)
    print(f"Wrote report: {path}")


if __name__ == "__main__":
    main()
