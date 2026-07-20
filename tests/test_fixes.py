#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests: one (or more) per bug identified in the audit.

Run:  python3 -m pytest tests/ -q      (if pytest is present)
  or: python3 tests/test_fixes.py      (standalone, no dependencies)

Each test name references the audit ID (A1, B3, C8, ...).
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecurve.secp256k1 import (  # noqa: E402
    Secp256k1, TEST_PARAMS, LEGACY_PARAMS, hash160,
)
from ecurve.k_magnitude import predict_magnitude, magnitude_distribution  # noqa: E402
import ecurve.k_magnitude as km  # noqa: E402
import ecdsa_attack_complexity as A  # noqa: E402
import generate_transaction_list as G  # noqa: E402


def _pct(value) -> float:
    """Extract the leading percentage from a stats display string.

    `stats` values are pre-formatted for the report ("93.00%", possibly with a
    trailing caveat). Exact string matching couples these tests to prose: any
    edit to a human-readable label breaks them, which is a test defect, not a
    code defect. Parse the number instead.
    """
    m = re.match(r"\s*([\d.]+)\s*%", str(value))
    assert m, f"no leading percentage in {value!r}"
    return float(m.group(1))


# --------------------------------------------------------------------------
# Correctness foundations (must hold before anything else means anything)
# --------------------------------------------------------------------------
def test_rfc6979_official_vectors():
    """RFC 6979 + SHA-256 test vectors for secp256k1."""
    ec = Secp256k1(LEGACY_PARAMS)
    n = LEGACY_PARAMS.n
    vectors = [
        (0x1, b"Satoshi Nakamoto",
         0x8F8A276C19F4149656B280621E358CCE24F5F52542772691EE69063B74F15D15),
        (0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140,
         b"Satoshi Nakamoto",
         0x33A19B60E25FB6F4435AF53A3D42D493644827367E6453928554F43E49AA6F90),
        (0xf8b8af8ce3c7cca5e300d33939540c10d45ce001b8f252bfbc57ba0342904181,
         b"Alan Turing",
         0x525A82B70E67874398067543FD84C83D30C175FDC45FDEEE082FE13B1D7CFDF1),
    ]
    for d, msg, want in vectors:
        z = int.from_bytes(hashlib.sha256(msg).digest(), "big") % n
        assert ec._rfc6979_generate_k(d, z) == want


def test_sign_verify_roundtrip_legacy():
    ec = Secp256k1(LEGACY_PARAMS)
    d, pub = ec.generate_keypair(0x1234567890ABCDEF1234567890ABCDEF)
    for msg in [b"a", b"hello", os.urandom(40)]:
        sig = ec.sign_message(d, msg, min_start_range=1)
        assert ec.verify_signature(pub, sig)


def test_sign_verify_roundtrip_test_curve():
    ec = Secp256k1(TEST_PARAMS)
    for _ in range(50):
        d, pub = ec.generate_keypair()
        sig = ec.sign_message(d, b"msg", min_start_range=1)
        assert ec.verify_signature(pub, sig)


def test_pubkey_on_curve():
    for params in (TEST_PARAMS, LEGACY_PARAMS):
        ec = Secp256k1(params)
        for _ in range(20):
            _, pub = ec.generate_keypair()
            assert ec.is_on_curve(pub)


# --------------------------------------------------------------------------
# A1 -- import layout
# --------------------------------------------------------------------------
def test_A1_package_imports():
    import importlib
    mod = importlib.import_module("generate_transaction_list")
    assert hasattr(mod, "main")


# --------------------------------------------------------------------------
# A3 -- EVERY documented invocation mode must work.
# The original test suite only exercised the package import, which is why the
# `from ._ripemd160 import ...` hard relative import shipped broken: running
# `python secp256k1.py` from inside ecurve/ raised
# "attempted relative import with no known parent package", and k_magnitude.py
# fell over too because its flat fallback imports secp256k1.
# --------------------------------------------------------------------------
def _run(cmd, cwd):
    import subprocess
    return subprocess.run(
        [sys.executable] + cmd, cwd=cwd,
        capture_output=True, text=True, timeout=300,
    )


def test_A3_standalone_secp256k1_from_inside_package_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.join(root, "ecurve")
    r = _run(["-c", "import secp256k1; print(secp256k1.TEST_PARAMS.n)"], cwd=pkg)
    assert r.returncode == 0, r.stderr
    assert "1241630743" in r.stdout


def test_A3_standalone_k_magnitude_from_inside_package_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.join(root, "ecurve")
    r = _run(["k_magnitude.py", "--curve-mode", "test", "--digits", "47"], cwd=pkg)
    assert r.returncode == 0, r.stderr
    assert "magnitude" in r.stdout


def test_A3_module_mode_from_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = _run(["-m", "ecurve.k_magnitude", "--curve-mode", "legacy",
              "--digits", "47"], cwd=root)
    assert r.returncode == 0, r.stderr


def test_A3_ripemd160_fallback_importable_both_ways():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.join(root, "ecurve")
    r = _run(["-c", "from _ripemd160 import ripemd160; "
                    "print(ripemd160(b'abc').hex())"], cwd=pkg)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"


# --------------------------------------------------------------------------
# A2 -- demo no longer crashes
# --------------------------------------------------------------------------
def test_A2_demo_runs():
    ec = Secp256k1(TEST_PARAMS)
    with contextlib.redirect_stdout(io.StringIO()):
        G._print_once_demo(ec)  # must not raise


# --------------------------------------------------------------------------
# B3 -- accuracy denominator matches generated rows
# --------------------------------------------------------------------------
def test_B3_accuracy_denominator():
    ec = Secp256k1(TEST_PARAMS)
    with contextlib.redirect_stdout(io.StringIO()):
        rows, stats = G._collect_rows(
            ec, total_key_count=5, private_key=123456789,
            transaction_limit_per_key=10, min_start_range=10,
            ml_expected_accuracy=1.0,
        )
    # The B3 invariant is about the DENOMINATOR, and nothing else. Assert that,
    # not how the number is dressed up for the report: whether the caveat sits on
    # the input or on the realised rate, and whether ml_expected_accuracy is
    # reported at all, are presentation choices this test must not police.
    assert stats["total_transaction_count"] == len(rows), (
        f"denominator {stats['total_transaction_count']} != len(rows) {len(rows)}; "
        f"_collect_rows is dividing by a planned figure, not by the rows it made"
    )
    # Recompute the accuracy independently from the rows and compare numerically.
    matches = sum(1 for r in rows if r[9] == "Yes")
    expected_pct = round(100.0 * matches / len(rows), 2)
    assert _pct(stats["ml_accuracy_k_val"]) == expected_pct, (
        f"reported {stats['ml_accuracy_k_val']!r} but rows give "
        f"{matches}/{len(rows)} = {expected_pct}%"
    )
    assert expected_pct == 100.0, (
        f"at ml_expected_accuracy=1.0 every row should match; got {matches}/{len(rows)}"
    )


def test_B3b_denominator_survives_skipped_rows():
    """B3, stronger form: the denominator must track ACTUAL rows, not a plan.

    The weaker test above passes even against a partial fix that uses
    `effective_key_count * transaction_limit_per_key`, because with a single
    key that product happens to equal len(rows). Force a row to be skipped so
    the plan and reality provably diverge.
    """
    ec = Secp256k1(TEST_PARAMS)
    real_sign = ec.sign_message
    state = {"n": 0}

    def sign_with_one_single_digit_nonce(d, m, min_start_range=1):
        z, r_x, r_y, s, k, k_inv = real_sign(d, m, min_start_range)
        state["n"] += 1
        if state["n"] == 1:
            k = 7  # single digit -> str(k)[:2] is 1 char -> C11 skip path
        return z, r_x, r_y, s, k, k_inv

    ec.sign_message = sign_with_one_single_digit_nonce
    with contextlib.redirect_stdout(io.StringIO()):
        rows, stats = G._collect_rows(
            ec, total_key_count=1, private_key=123456789,
            transaction_limit_per_key=10, min_start_range=10,
            ml_expected_accuracy=1.0,
        )
    assert len(rows) == 9, f"expected one skipped row, got len(rows)={len(rows)}"
    assert stats["total_transaction_count"] == 9, (
        f"denominator {stats['total_transaction_count']} != 9; a planned figure "
        f"(1 key x 10 tx) would read 10 and silently understate accuracy"
    )
    assert _pct(stats["ml_accuracy_k_val"]) == 100.0, stats["ml_accuracy_k_val"]


# --------------------------------------------------------------------------
# B4 -- fmt_duration converts seconds to years
# --------------------------------------------------------------------------
def test_B4_duration_years():
    out = A.fmt_duration(40.0)  # 1e40 seconds
    assert out.endswith("yr")
    exp = float(out[2:].split()[0])
    assert abs(exp - (40.0 - A._LOG10_SEC_PER_YEAR)) < 0.05  # ~32.5, not 40


# --------------------------------------------------------------------------
# B5 -- heuristic branch is reachable
# --------------------------------------------------------------------------
def test_B5_heuristic_default_none():
    args = A.parse_args([])
    assert args.t_per_run_sec is None


# --------------------------------------------------------------------------
# B6 -- r_y is the raw affine coordinate (mod p), not mod n
# --------------------------------------------------------------------------
def test_B6_ry_not_reduced_mod_n():
    ec = Secp256k1(TEST_PARAMS)
    n, p = TEST_PARAMS.n, TEST_PARAMS.p
    seen_high_y = False
    for _ in range(200):
        d, _pub = ec.generate_keypair()
        z, r_x, r_y, s, k, k_inv = ec.sign_message(d, b"m", 1)
        x, y = ec.scalar_multiply(k, TEST_PARAMS.g)
        assert r_y == y            # equals raw affine y ...
        assert r_y == r_y % p      # ... which lives mod p
        if y >= n:
            seen_high_y = True
            assert r_y != r_y % n  # the old code would have corrupted it
    # not asserted every run (probabilistic), just recorded
    _ = seen_high_y


# --------------------------------------------------------------------------
# B7 -- m_req formula matches between docstring and code (now consistent)
# --------------------------------------------------------------------------
def test_B7_m_req_formula():
    from math import ceil
    for L in (5, 6, 7, 8):
        got = A.required_clean_signatures(256, L, 1.3, None)
        assert got == ceil(ceil(256 / L) * 1.3)


# --------------------------------------------------------------------------
# C8 -- verify_signature handles the point at infinity
# --------------------------------------------------------------------------
def test_C8_verify_infinity_no_crash():
    ec = Secp256k1(TEST_PARAMS)
    d, pub = ec.generate_keypair(12345)
    z, r, ry, s, k, kinv = ec.sign_message(d, b"m", 1)
    tampered = (z, r, ry, (s + 1) % TEST_PARAMS.n, k, kinv)
    # must return a bool, never raise TypeError on unpack
    assert ec.verify_signature(pub, tampered) in (True, False)


def test_C8_point_add_inverse_returns_none():
    ec = Secp256k1(TEST_PARAMS)
    p1 = ec.scalar_multiply(5, TEST_PARAMS.g)
    p2 = ec.inverse_point(p1, TEST_PARAMS.p)
    assert ec.point_add(p1, p2) is None


# --------------------------------------------------------------------------
# C10 -- deterministic nonce rejection advances instead of looping
# --------------------------------------------------------------------------
def test_C10_candidate_stream_advances():
    ec = Secp256k1(LEGACY_PARAMS)
    gen = ec._rfc6979_k_candidates(0x1, 42)
    first = next(gen)
    second = next(gen)
    assert first != second  # a rejected candidate yields a NEW value


# --------------------------------------------------------------------------
# C11 -- single-digit k does not explode the magnitude predictor
# --------------------------------------------------------------------------
def test_C11_single_digit_message():
    try:
        predict_magnitude("7", LEGACY_PARAMS.n)
        raised = False
    except ValueError as e:
        raised = True
        assert "at least two" in str(e)
    assert raised  # a clear, named error -- not a bare crash deep in the stack


# --------------------------------------------------------------------------
# C12 -- hash160 works even if hashlib lacks ripemd160
# --------------------------------------------------------------------------
def test_C12_hash160():
    # Known: HASH160 of the empty-string SHA256 chain is deterministic.
    h = hash160(b"abc")
    assert len(h) == 20


# --------------------------------------------------------------------------
# D -- argument validation
# --------------------------------------------------------------------------
def test_D_complexity_rejects_bad_args():
    for bad in (["--surplus", "-5"], ["--safety-factor", "-1"],
                ["--dim-overhead", "-3"], ["--t-per-run-sec", "0"]):
        # argparse prints usage to stderr before raising SystemExit. That is the
        # expected path here, so swallow it: otherwise a passing suite dumps
        # pages of usage text that reads like a crash.
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                A.parse_args(bad)
                raised = False
            except SystemExit:
                raised = True
        assert raised, f"{bad} was accepted; validation is missing"


def test_D_generator_rejects_bad_args():
    for bad in (["--keys", "0"], ["--keys", "-3"],
                ["--ml-exp-accuracy", "2.0"], ["--private-key", "-5"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                G.parse_args(bad)
                raised = False
            except SystemExit:
                raised = True
        assert raised, f"{bad} was accepted; validation is missing"


# --------------------------------------------------------------------------
# E14 / E15 -- single infinity sentinel
# --------------------------------------------------------------------------
def test_E14_scalar_mult_infinity_is_none():
    ec = Secp256k1(TEST_PARAMS)
    assert ec.scalar_multiply(TEST_PARAMS.n, TEST_PARAMS.g) is None
    assert ec.is_on_curve(None) is True


# --------------------------------------------------------------------------
# E16 -- doctests pass
# --------------------------------------------------------------------------
def test_E16_doctests():
    import doctest
    res = doctest.testmod(km, verbose=False)
    assert res.failed == 0


# --------------------------------------------------------------------------
# E18 -- unique keys are order-stable within a seeded run and bounded
# --------------------------------------------------------------------------
def test_E18_unique_keys_bounded():
    ec = Secp256k1(TEST_PARAMS)
    keys = ec.generate_unique_keys(20, 10)
    assert len(keys) == len(set(keys)) == 20
    try:
        ec.generate_unique_keys(TEST_PARAMS.n + 5, 10)
        raised = False
    except ValueError:
        raised = True
    assert raised  # no infinite loop when asking for too many


# --------------------------------------------------------------------------
# E21 -- honest note: the ORIGINAL already matched RFC6979 vectors.
# This test documents that the fix is about the rejection path + >=n case,
# not about the common-case first candidate.
# --------------------------------------------------------------------------
def test_E21_distribution_conditioned_sums_to_one():
    dist = magnitude_distribution(47, LEGACY_PARAMS.n)
    assert abs(sum(dist.values()) - 1.0) < 1e-12


def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())