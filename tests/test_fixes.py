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
    out = A.fmt_duration(40.0)  # 1e40 seconds, precision defaults to "orders"
    assert out.endswith("yr")
    exp = float(out[2:].split()[0])
    assert abs(exp - (40.0 - A._LOG10_SEC_PER_YEAR)) < 0.05  # ~32.5, not 40


def test_B4b_duration_precision_is_optional():
    """`precision` must stay optional.

    v2 added a precision argument to fmt_duration. Making it mandatory broke
    every existing caller (this suite included) for no benefit: "orders" is the
    only defensible default, since the model cannot support more.
    """
    assert A.fmt_duration(40.0) == A.fmt_duration(40.0, "orders")
    assert A.fmt_duration(40.0, "full") != A.fmt_duration(40.0, "orders")
    assert A.fmt_duration(None) == "--"
    assert A.fmt_duration(-3.0) == "<1 s"      # never render sub-second as "0 s"


# --------------------------------------------------------------------------
# B5 -- timing default is an explicit, flat placeholder.
#        v2 DOES have a dimension-dependent model, but it is opt-in via
#        --t-run-doubling-dims. The flat default is pinned here so it cannot
#        drift silently, and F3 covers the scaling model itself.
# --------------------------------------------------------------------------
def test_B5_t_per_run_sec_default_placeholder():
    args = A.parse_args([])
    assert args.t_per_run_sec == 2.0
    assert args.t_run_doubling_dims is None, (
        "the dimension-scaling model must stay opt-in: silently switching it "
        "on would change every published figure without notice"
    )


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
        got = A.required_clean_signatures(256, L, 1.3)
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
    # HASH160(x) = RIPEMD160(SHA256(x)) must work even when hashlib lacks a
    # native ripemd160 (OpenSSL 3.x): the pure-Python fallback kicks in.
    # Pin the known digest of b"abc", not just the length.
    h = hash160(b"abc")
    assert len(h) == 20
    assert h.hex() == "bb1be98c142444d7a56aa3981c3942a978e4dc33"


# --------------------------------------------------------------------------
# D -- argument validation
# --------------------------------------------------------------------------
def test_D_complexity_rejects_bad_args():
    # NOTE: `--surplus` is NOT in this list any more. v2 removed the flag
    # outright (see F1), so a negative value is rejected because the option no
    # longer exists, not because its range is validated. Asserting it here
    # would be a test that passes for the wrong reason.
    for bad in (["--safety-factor", "-1"], ["--dim-overhead", "-3"],
                ["--t-per-run-sec", "0"], ["--curve-bits", "0"],
                ["--leak-bits", "0"], ["--cores", "0"],
                ["--p-mag", "0"], ["--p-mag", "101"],
                ["--p-ml-step", "0"], ["--correlation", "1.5"],
                ["--p-reduction-success", "0"],
                ["--p-reduction-success", "1.1"],
                ["--t-run-doubling-dims", "0"]):
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
# E21 -- magnitude_distribution is a proper (normalised) conditional
#        distribution: probabilities sum to 1 over the D>=2 conditional space.
#        (The RFC 6979 rejection / >= n behaviour that FIX E21 addressed is
#        covered by test_C10_candidate_stream_advances and
#        test_rfc6979_official_vectors.)
# --------------------------------------------------------------------------
def test_E21_distribution_conditioned_sums_to_one():
    dist = magnitude_distribution(47, LEGACY_PARAMS.n)
    assert abs(sum(dist.values()) - 1.0) < 1e-12


# ==========================================================================
# F-SERIES -- second audit, of the cost model itself (ecdsa_attack_complexity)
#
# v1 was arithmetically sound but structurally wrong in ways that moved the
# headline figure by orders of magnitude. One test per defect.
# ==========================================================================

def _cli(args):
    """Run main() and capture stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A.main(list(args))
    return buf.getvalue()


# --------------------------------------------------------------------------
# F1 -- the supply model is explicit; `surplus` is gone.
#       In v1, `surplus` (default 2.0) silently set the pool size and thereby
#       controlled `runs` across ~11 orders of magnitude, while T_raw -- the
#       only quantity that could have expressed signature scarcity -- fed into
#       nothing at all. The correct i.i.d. limit is q = p_clean^m_req.
# --------------------------------------------------------------------------
def test_F1_surplus_flag_removed_with_explanation():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            A.parse_args(["--surplus", "2.0"])
            raised = False
        except SystemExit:
            raised = True
    assert raised, "--surplus must not be silently accepted"
    assert "removed in v2" in err.getvalue(), (
        "removing a flag without saying why just looks like a typo to the user"
    )


def test_F1_unlimited_supply_is_the_iid_limit():
    from math import log10
    m_req, p = 56, 0.738
    got = A.log10_success_per_draw(None, m_req, None, p, 1.0)
    assert abs(got - m_req * log10(p)) < 1e-12


def test_F1_pool_params_rejected_without_scarce_supply():
    for bad in (["--t-raw", "100"], ["--filter-pass-rate", "0.5"],
                ["--filter-recall", "0.5"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                A.parse_args(bad)
                raised = False
            except SystemExit:
                raised = True
        assert raised, f"{bad} accepted with unlimited supply; it is meaningless there"


def test_F1_scarce_supply_requires_t_raw():
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            A.parse_args(["--supply", "scarce"])
            raised = False
        except SystemExit:
            raised = True
    assert raised


# --------------------------------------------------------------------------
# F2 -- the pre-filter obeys Bayes.
#       v1 charged the cost of filtering (more raw signatures) and credited
#       none of the benefit (a purer pool); filter_pass_rate influenced no
#       result whatsoever.
# --------------------------------------------------------------------------
def test_F2_uninformative_filter_leaves_purity_unchanged():
    post, fpr = A.bayes_post_clean(0.70, 0.5, 0.5)
    assert abs(post - 0.70) < 1e-12, "a blind filter must not change purity"
    assert abs(fpr - 0.5) < 1e-12


def test_F2_informative_filter_raises_purity():
    post, _ = A.bayes_post_clean(0.70, 0.80, 0.95)
    assert 0.70 < post < 1.0, f"recall > pass_rate must enrich the pool; got {post}"


def test_F2_implied_false_pass_rate_obeys_total_probability():
    p_clean, pi, recall = 0.70, 0.80, 0.95
    _, fpr = A.bayes_post_clean(p_clean, pi, recall)
    assert abs(p_clean * recall + (1 - p_clean) * fpr - pi) < 1e-12


def test_F2_inconsistent_filter_is_rejected():
    # pass_rate below p_clean*recall implies P(clean|pass) > 1: impossible.
    # v1 accepted such triples silently.
    try:
        A.bayes_post_clean(0.90, 0.50, 1.0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_F2_filter_pass_rate_actually_changes_the_answer():
    """The v1 inertness regression, pinned.

    In v1 you could vary --filter-pass-rate over a 20x range and every single
    reported figure stayed byte-identical. Sweeping it must now move `runs`.
    """
    common = ["--supply", "scarce", "--p-ml-start", "90", "--p-ml-stop", "90",
              "--no-legend"]
    a = _cli(common + ["--t-raw", "400", "--filter-pass-rate", "1.0"])
    b = _cli(common + ["--t-raw", "400", "--filter-pass-rate", "0.6"])
    assert a != b, "filter_pass_rate is inert again; it must reach `runs`"


# --------------------------------------------------------------------------
# F3 -- reduction cost depends on lattice dimension.
#       v1 computed `dim`, printed it, wrote it to CSV, and used it in no
#       formula: every L was charged a flat 2.0 s. That inverts reality, where
#       small L is expensive precisely because the lattice grows.
# --------------------------------------------------------------------------
def test_F3_t_run_doubles_every_D_dimensions():
    from math import log10
    lo = A.log10_t_run(58, 2.0, 58, 12)
    hi = A.log10_t_run(70, 2.0, 58, 12)
    assert abs((hi - lo) - log10(2.0)) < 1e-12


def test_F3_t_run_flat_without_doubling_dims():
    assert A.log10_t_run(200, 2.0, 58, None) == A.log10_t_run(44, 2.0, 58, None)


def test_F3_dimension_is_load_bearing():
    """`dim` must reach the cost. Different L -> different per-run time."""
    out = _cli(["--leak-bits", "4", "8", "--t-run-doubling-dims", "12",
                "--t-run-dim-ref", "58", "--p-ml-start", "90",
                "--p-ml-stop", "90", "--no-legend"])
    times = re.findall(r"t/run = ([\d.eE+-]+) s", out)
    assert len(times) == 2, out
    assert float(times[0]) != float(times[1]), (
        "L=4 (dim 86) and L=8 (dim 44) were charged the same time; "
        "`dimension` is decorative again"
    )


def test_F3_flat_model_warns_against_cross_L_comparison():
    out = _cli(["--leak-bits", "4", "6", "8", "--p-ml-start", "90",
                "--p-ml-stop", "90", "--no-legend"])
    assert "Do not compare rows across L" in out


# --------------------------------------------------------------------------
# F4 -- the reduction is not assumed infallible.
#       v1 implicitly set P(reduction succeeds | all-clean subset) = 1.
# --------------------------------------------------------------------------
def test_F4_p_reduction_success_enters_the_denominator():
    from math import log10
    full = A.log10_success_per_draw(None, 56, None, 0.738, 1.0)
    half = A.log10_success_per_draw(None, 56, None, 0.738, 0.5)
    assert abs((full - half) - log10(2.0)) < 1e-12, (
        "halving the reduction success rate must double the expected runs"
    )


# --------------------------------------------------------------------------
# F5 -- the p_m parameterisation is no longer over-determined.
#       v1 documented p_m as conditional on correct leading digits AND offered
#       a correlation knob. Holding both double-counts the dependence.
# --------------------------------------------------------------------------
def test_F5_correlation_rejected_in_conditional_mode():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            A.parse_args(["--correlation", "0.2"])
            raised = False
        except SystemExit:
            raised = True
    assert raised
    assert "double-count" in err.getvalue()


def test_F5_marginal_mode_accepts_correlation():
    args = A.parse_args(["--p-mag-mode", "marginal", "--correlation", "0.126"])
    assert args.correlation == 0.126


def test_F5_positive_correlation_raises_p_clean():
    indep = A.joint_clean_probability(0.90, 0.8888, 0.0)
    corr = A.joint_clean_probability(0.90, 0.8888, 0.126)
    assert corr > indep, "rho > 0 means the predictors fail together: easier"


def test_F5_p_clean_always_within_frechet_bounds():
    for p_b in (0.30, 0.50, 0.90, 0.99):
        for p_m in (0.30, 0.50, 0.82, 0.99):
            for rho in (-1.0, -0.5, 0.0, 0.5, 1.0):
                v = A.joint_clean_probability(p_b, p_m, rho)
                assert max(0.0, p_b + p_m - 1.0) - 1e-12 <= v <= min(p_b, p_m) + 1e-12, (
                    f"p_clean={v} escapes Frechet for {p_b=} {p_m=} {rho=}"
                )


# --------------------------------------------------------------------------
# F6 -- the output is a distribution, not a point.
#       v1 froze the pool composition at its mean and called the result
#       "expected runs". It is close to the MEDIAN; the true expectation is
#       dominated by rare bad pools.
# --------------------------------------------------------------------------
def test_F6_quantiles_are_ordered():
    q = A.log10_success_per_draw(None, 56, None, 0.738, 1.0)
    comps = [(1.0, q)]
    p05 = A.runs_quantile(0.05, comps)
    p50 = A.runs_quantile(0.50, comps)
    p95 = A.runs_quantile(0.95, comps)
    assert p05 < p50 < p95


def test_F6_median_matches_closed_form():
    from math import log, log10
    q = A.log10_success_per_draw(None, 56, None, 0.738, 1.0)
    got = A.runs_quantile(0.50, [(1.0, q)])
    want = log10(-log(0.5)) - q          # median of Geometric(q) = ln2 / q
    assert abs(got - want) < 1e-6, f"{got} vs {want}"


def test_F6_geometric_band_width_is_exact():
    from math import log, log10
    q = A.log10_success_per_draw(None, 56, None, 0.738, 1.0)
    comps = [(1.0, q)]
    width = A.runs_quantile(0.95, comps) - A.runs_quantile(0.05, comps)
    assert abs(width - log10(log(0.05) / log(0.95))) < 1e-6


def test_F6_scarce_pool_band_is_wider_than_iid():
    """Pool-composition randomness must actually widen the band.

    This is the Jensen error made visible: v1's single number hid it entirely.
    """
    q = A.log10_success_per_draw(None, 56, None, 0.738, 1.0)
    iid = A.runs_quantile(0.95, [(1.0, q)]) - A.runs_quantile(0.05, [(1.0, q)])
    out = _cli(["--supply", "scarce", "--t-raw", "200", "--p-ml-start", "90",
                "--p-ml-stop", "90", "--no-legend"])
    m = re.search(r"1e([\d.]+)\s+1e([\d.]+)\s+1e([\d.]+)", out)
    assert m, out
    scarce = float(m.group(3)) - float(m.group(1))
    assert scarce > iid, f"scarce band {scarce:.2f} <= iid band {iid:.2f}"


# --------------------------------------------------------------------------
# F7 -- false precision removed.
# --------------------------------------------------------------------------
def test_F7_default_precision_is_orders():
    assert A.parse_args([]).precision == "orders"


def test_F7_unvalidated_assumptions_are_printed():
    out = _cli(["--p-ml-start", "90", "--p-ml-stop", "90"])
    assert "Unvalidated assumptions" in out
    assert "partial ECDLP" in out, (
        "the load-bearing caveat must travel with the table, not sit in a "
        "README nobody opens"
    )


# --------------------------------------------------------------------------
# F8 -- hygiene, plus the two defects the v2 test pass itself uncovered.
# --------------------------------------------------------------------------
def test_F8_accuracy_sweep_grid_does_not_drift():
    for step in (1.0, 0.1, 0.25, 0.3, 0.7):
        got = A.accuracy_sweep(80.0, 99.0, step)
        want = int((99.0 - 80.0) / step + 1e-9) + 1
        assert len(got) == want, f"step={step}: {len(got)} values, expected {want}"


def test_F8_legend_reports_every_swept_L():
    """v1's legend printed only leak_bits[0] above three different tables."""
    out = _cli(["--leak-bits", "4", "6", "8", "--p-ml-start", "90",
                "--p-ml-stop", "90"])
    assert "[4, 6, 8]" in out, "the legend is under-reporting the sweep again"


def test_F8_fully_clean_pool_does_not_raise():
    """Regression: log1p(-q) is a domain error at q = 1.

    A pool that is entirely clean gives q = C(T,m)/C(T,m) = 1 exactly, which
    is reachable whenever p_clean is high and the pool is small.
    """
    out = _cli(["--supply", "scarce", "--t-raw", "60", "--p-mag", "99",
                "--p-ml-start", "99", "--p-ml-stop", "99", "--no-legend"])
    assert "0.9801" in out


def test_F8_filter_recall_defaults_to_uninformative():
    """Regression: a recall default of 1.0 makes most pass rates inconsistent.

    With recall = 1.0 and p_clean = 0.738, any pass_rate < 0.738 implies
    P(clean|pass) > 1 and the run aborts. The neutral default is
    recall = pass_rate, i.e. a filter that only shrinks the pool.
    """
    args = A.parse_args(["--supply", "scarce", "--t-raw", "100",
                         "--filter-pass-rate", "0.5"])
    assert args.filter_recall == 0.5
    out = _cli(["--supply", "scarce", "--t-raw", "400",
                "--filter-pass-rate", "0.5", "--p-ml-start", "90",
                "--p-ml-stop", "90", "--no-legend"])
    assert "inconsistent" not in out


def test_F8_undersized_pool_gives_actionable_advice():
    """v1 printed a bare 'n/a', conflating 'infeasible' with 'below threshold'."""
    out = _cli(["--supply", "scarce", "--t-raw", "40", "--p-ml-start", "90",
                "--p-ml-stop", "90", "--no-legend"])
    assert "need T_raw >=" in out, out


def test_F8_readme_defaults_match_the_parser():
    """The README must not drift from the code again.

    v1 shipped three separate drifts at once: the module docstring claimed
    `--p-mag default 90.0` against a real default of 82.0, and both the
    docstring and the README listed `cores=[1, 8, 64]` against a real
    [1, 8, 64, 124]. Parse the README's declared defaults and diff them.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    block = re.search(r"<!-- defaults:begin -->(.*?)<!-- defaults:end -->",
                      readme, re.S)
    assert block, "README is missing the machine-checked defaults block"

    args = A.parse_args([])
    declared = dict(re.findall(r"`([a-z_]+)\s*=\s*([^`]+)`", block.group(1)))
    assert declared, "defaults block present but empty"
    for name, raw in declared.items():
        assert hasattr(args, name), f"README declares unknown option {name!r}"
        actual = getattr(args, name)
        want = raw.strip()
        got = (f"[{', '.join(str(x) for x in actual)}]"
               if isinstance(actual, list) else str(actual))
        assert got == want, (
            f"README says {name}={want!r}, parser gives {got!r}"
        )


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