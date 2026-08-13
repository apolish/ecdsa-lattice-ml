#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression suite for ecdsa_lattice_ml.

Built on the standard library's ``unittest``, so it runs three ways with no
third-party dependency:

    python3 tests/run_tests.py                # table output (default)
    python3 tests/run_tests.py -k filter      # only tests matching a substring
    python3 tests/run_tests.py --list         # enumerate without running
    python3 tests/run_tests.py --plain        # unittest's own text output
    python3 tests/run_tests.py -v             # + full tracebacks on failure
    python3 -m pytest tests/run_tests.py -q   # if pytest is installed

Note the explicit path in the pytest form. pytest only auto-discovers files
matching ``test_*.py`` or ``*_test.py``, so a bare ``pytest tests/`` collects
nothing here and exits 5. To make the bare form work, add to pytest.ini:

    [pytest]
    python_files = run_tests.py test_*.py

Tests are grouped into ``TestCase`` classes by concern; each class docstring
supplies the section heading in the table. Method names carry the audit ID they
regress (A1, B3, C8, F2, G4, ...); tests without an ID are invariants that were
never tied to a specific defect.

WHY SOME VALUES ARE PINNED AND OTHERS DERIVED
=============================================
Anything fixed by an external specification is pinned as a literal: RFC 6979
vectors, the RIPEMD-160 digest of b"abc", the m_req formula. Anything that is a
*project choice* is derived from the source of truth at runtime -- curve orders,
private keys, digit counts. The previous suite hardcoded the test curve's order
and a private key valid only for it, so swapping TEST_PARAMS broke four tests
that had nothing to do with the change.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import math
import os
import re
import subprocess
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "ecurve")
SEED = 20260810


def _defined_names(path: str) -> set[str]:
    """Top-level names a module defines, read statically (no import)."""
    import ast
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (OSError, SyntaxError):
        return set()
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update(a.asname or a.name.split(".")[0] for a in node.names)
    return out


def _importers_of(name: str) -> list[tuple[str, int]]:
    """Every (file, line) in the project that imports `name` from somewhere."""
    import ast
    hits: list[tuple[str, int]] = []
    for folder in (ROOT, PKG):
        if not os.path.isdir(folder):
            continue
        for entry in sorted(os.listdir(folder)):
            if not entry.endswith(".py"):
                continue
            path = os.path.join(folder, entry)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and any(
                        a.name == name for a in node.names):
                    hits.append((os.path.relpath(path, ROOT), node.lineno))
    return hits


def _diagnose_import_failure(exc: ImportError) -> str:
    """Turn an import-time explosion into an instruction.

    A broken ``ecurve/__init__.py`` takes down the suite before a single test
    can run, so no test can report it -- the traceback is all the user gets.
    That traceback names the symbol but not the fix, and points at the module
    that is innocent (secp256k1.py) rather than the one that is stale
    (__init__.py). Say which file to edit.
    """
    bar = "=" * 78
    out = ["", bar, "  THE SUITE COULD NOT IMPORT THE PROJECT", bar, "",
           f"  {type(exc).__name__}: {exc}", ""]

    m = re.match(r"cannot import name '([^']+)' from '([^']+)'", str(exc))
    if not m:
        out += ["  Check that you are running from the repository root or its",
                "  tests/ directory, and that ecurve/ is importable.", bar, ""]
        return "\n".join(out)

    missing, module = m.group(1), m.group(2)
    mod_path = os.path.join(ROOT, *module.split(".")) + ".py"
    out.append(f"  '{module}' has no top-level name '{missing}'.")

    askers = [(f, ln) for f, ln in _importers_of(missing)
              if os.path.basename(f) != os.path.basename(mod_path)]
    if askers:
        out += ["", "  Asked for by:"]
        out += [f"    {f}:{ln}" for f, ln in askers]

    defined = _defined_names(mod_path)
    close = sorted(n for n in defined
                   if missing.lower().replace("_", "") in n.lower().replace("_", "")
                   or n.lower().replace("_", "") in missing.lower().replace("_", ""))
    out += ["", "  Fix it in one of two ways:", "",
            f"    1. Drop '{missing}' from the import list above, if it is no",
            "       longer used anywhere. This is usually the right answer when",
            "       a helper was deleted during cleanup.", "",
            f"    2. Restore '{missing}' in {os.path.relpath(mod_path, ROOT)},",
            "       if callers still depend on it."]
    if close:
        out += ["", f"  Possibly renamed to: {', '.join(close)}"]
    out += ["", bar, ""]
    return "\n".join(out)


try:
    from ecurve.secp256k1 import (  # noqa: E402
        Secp256k1, TEST_PARAMS, LEGACY_PARAMS, hash160,
    )
    from ecurve.k_magnitude import predict_magnitude, magnitude_distribution  # noqa: E402
    import ecurve  # noqa: E402
    import ecurve.k_magnitude as km  # noqa: E402
    import ecdsa_attack_complexity as A  # noqa: E402
    import generate_transaction_list as G  # noqa: E402
except ImportError as _exc:  # pragma: no cover - only on a broken checkout
    if __name__ == "__main__":
        sys.stderr.write(_diagnose_import_failure(_exc))
        raise SystemExit(2) from None
    raise


# ==========================================================================
# Helpers
# ==========================================================================

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


def _subprocess(cmd, cwd):
    return subprocess.run([sys.executable] + cmd, cwd=cwd,
                          capture_output=True, text=True, timeout=300)


def _cli(args) -> str:
    """Run the cost model's main() and capture stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A.main(list(args))
    return buf.getvalue()


def _quiet(fn, *a, **kw):
    """Call fn with stdout suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def valid_test_key() -> int:
    """A valid, deterministic private key for TEST_PARAMS.

    Derived from n rather than hardcoded. The previous suite used 123456789,
    which was in range only for the 31-bit test curve; the 17-bit secp17k1
    rejects it and two unrelated tests failed with a confusing message.
    """
    return TEST_PARAMS.n // 2 + 1


def is_probable_prime(n: int, rounds: int = 40) -> bool:
    """Miller-Rabin. Trial division cannot reach a 256-bit curve order."""
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % small == 0:
            return n == small
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)[:rounds]:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


ALL_CURVES = (TEST_PARAMS, LEGACY_PARAMS)


# ==========================================================================
# G -- curve parameters and structural invariants
#
# NEW in this revision. README section 3.3 claimed these were "verified
# invariants (all checked in tests/)", but none of them were: the suite tested
# behaviour built ON the curve without ever checking the curve itself. Swapping
# TEST_PARAMS is precisely the event that makes that gap matter.
# ==========================================================================
class TestCurveParameters(unittest.TestCase):
    """Curve parameters and structural invariants"""

    def test_G1_generator_lies_on_curve(self):
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                x, y = c.g
                self.assertEqual((y * y - x**3 - c.a * x - c.b) % c.p, 0)

    def test_G2_order_annihilates_generator(self):
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                self.assertIsNone(Secp256k1(c).scalar_multiply(c.n, c.g))

    def test_G3_order_is_prime(self):
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                self.assertTrue(is_probable_prime(c.n),
                                f"{c.name}: n = {c.n} is composite; the "
                                f"cofactor-1 assumption fails")

    def test_G4_field_is_prime(self):
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                self.assertTrue(is_probable_prime(c.p))

    def test_G5_curve_is_non_singular(self):
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                self.assertNotEqual((4 * c.a**3 + 27 * c.b**2) % c.p, 0,
                                    f"{c.name}: discriminant vanishes")

    def test_G6_hasse_bound_holds(self):
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                t = c.p + 1 - c.n            # Frobenius trace
                self.assertLessEqual(abs(t), 2 * math.isqrt(c.p),
                                     f"{c.name}: |t| exceeds the Hasse bound, "
                                     f"so n cannot be this curve's order")

    def test_G7_generator_inverse_is_consistent(self):
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                ec = Secp256k1(c)
                self.assertEqual(ec.scalar_multiply(c.n - 1, c.g),
                                 (c.g[0], (-c.g[1]) % c.p))

    def test_G8_coordinate_overflow_is_reachable_on_test_curve(self):
        """The p - n gap must be wide enough to actually exercise.

        A y coordinate at or above n is what a `% n` bug would corrupt. On
        secp256k1 that event sits near 2^-128 and is unreachable; the test
        curve exists to make it common. Assert it is, rather than hoping.
        """
        c = TEST_PARAMS
        gap = (c.p - c.n) / c.p
        self.assertGreater(gap, 1e-4,
                           f"p - n = {c.p - c.n} leaves P(y >= n) = {gap:.2e}; "
                           f"too rare to exercise the coordinate path")

    def test_G9_doubling_slope_consumes_the_a_coefficient(self):
        """The `+ a` term in the doubling slope must be exercised by something.

        lambda = (3x^2 + a) / 2y. Both shipped curves set a = 0 -- secp256k1 by
        definition, secp17k1 by choice -- so on them the term is identically
        zero and any defect in it is invisible *no matter how many trials are
        run*. It is a permanent blind spot, not a probabilistic one: mutation
        testing confirmed that deleting `+ a` left the rest of the suite green.

        Rather than introduce a third curve, perturb the `a` field of an
        existing one and require that point_add's arithmetic responds to it,
        matching an independent reference that keeps the term and diverging
        from the same reference with the term dropped. Nothing here claims the
        perturbed parameters form a valid curve, and nothing is signed or
        verified on them -- the assertion is purely differential, about whether
        the implementation reads `curve.a` and uses it in the right place.

        What this does NOT cover, and a real a != 0 curve would: that the group
        law's *invariants* still hold when a != 0 -- that doubling lands back
        on the curve, that it agrees with scalar_multiply, that the order is
        respected. Those need parameters that are genuinely a curve. See the
        limitation noted in the README.
        """
        def ref_double(point, a, p):
            x, y = point
            m = (3 * x * x + a) * pow(2 * y, -1, p) % p
            x3 = (m * m - 2 * x) % p
            return x3, (m * (x - x3) - y) % p

        for c in ALL_CURVES:
            for a in (1, 2, 7):
                with self.subTest(curve=c.name, a=a):
                    base = Secp256k1(c)
                    point = base.scalar_multiply(5, c.g)

                    probed = Secp256k1(c)
                    probed._curve = dataclasses.replace(c, a=a)
                    got = probed.point_add(point, point)

                    self.assertEqual(got, ref_double(point, a, c.p),
                                     "doubling slope does not match "
                                     "(3x^2 + a) / 2y for a != 0")
                    self.assertNotEqual(got, ref_double(point, 0, c.p),
                                        "changing `a` changed nothing, so "
                                        "point_add is ignoring the term")
                    self.assertEqual(base.point_add(point, point),
                                     ref_double(point, c.a, c.p),
                                     "perturbing the probe leaked into the "
                                     "unmodified instance")

    def test_G9b_shipped_curves_leave_a_untested_behaviourally(self):
        """Record the residual gap so it cannot quietly become an assumption.

        G9 pins the formula. It cannot pin the group-law invariants for a != 0,
        because neither shipped curve has a != 0 and this suite deliberately
        does not define one. If TEST_PARAMS ever gains a non-zero `a`, this
        test says so and the gap closes on its own.
        """
        with_a = [c.name for c in ALL_CURVES if c.a % c.p != 0]
        if with_a:
            self.skipTest(f"shipped curves now cover `a` behaviourally: "
                          f"{with_a}; G9's differential probe is no longer the "
                          f"only coverage and this note can be dropped")
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                self.assertEqual(c.a % c.p, 0,
                                 "a shipped curve gained a != 0 but was not "
                                 "reported above; the skip logic is broken")


    def test_G10_is_on_curve_rejects_off_curve_points(self):
        """A membership test that never says "no" is not a membership test.

        Mutation-found gap: replacing the body of ``is_on_curve`` with
        ``return True`` left the whole suite green, because every existing
        caller only ever fed it points that were genuinely on the curve.
        """
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                ec = Secp256k1(c)
                self.assertNotEqual((1 + c.a + c.b) % c.p, 0)   # (1, 0) is off
                self.assertFalse(ec.is_on_curve((1, 0)))
                _d, pub = ec.generate_keypair(2)
                x, y = pub
                self.assertTrue(ec.is_on_curve(pub))
                self.assertFalse(ec.is_on_curve((x, (y + 1) % c.p)))

    def test_G11_verify_validates_key_before_touching_arithmetic(self):
        """An off-curve public key must be rejected by validation, not by luck.

        Mutation-found gap: deleting the ``is_on_curve`` guard from
        ``verify_signature`` still returned False, because (1, 0) degenerates
        to the point at infinity under scalar multiplication. A black-box
        assertion therefore cannot tell the two apart. Pin the ordering
        instead: with the guard present, no scalar multiplication happens at
        all. This is what stops invalid-curve attacks.
        """
        ec = Secp256k1(TEST_PARAMS)
        d, _pub = ec.generate_keypair(valid_test_key())
        sig = ec.sign_message(d, b"m", 1)

        def explode(*_a, **_kw):
            raise AssertionError("arithmetic ran before the key was validated")

        ec.scalar_multiply = explode
        self.assertFalse(ec.verify_signature((1, 0), sig))
        self.assertFalse(ec.verify_signature(None, sig))

    def test_G12_verify_rejects_tampered_signature(self):
        """C8 only asserted that a bool came back; require the right bool."""
        ec = Secp256k1(TEST_PARAMS)
        d, pub = ec.generate_keypair(valid_test_key())
        z, r, ry, s, k, kinv = ec.sign_message(d, b"m", 1)
        self.assertTrue(ec.verify_signature(pub, (z, r, ry, s, k, kinv)))
        for label, bad in (
            ("s+1", (z, r, ry, (s + 1) % TEST_PARAMS.n, k, kinv)),
            ("r+1", (z, (r + 1) % TEST_PARAMS.n, ry, s, k, kinv)),
            ("z+1", ((z + 1) % TEST_PARAMS.n, r, ry, s, k, kinv)),
            ("s=0", (z, r, ry, 0, k, kinv)),
            ("r=0", (z, 0, ry, s, k, kinv)),
        ):
            with self.subTest(tamper=label):
                self.assertFalse(ec.verify_signature(pub, bad))

    def test_G13_equivalent_mutants_are_documented(self):
        """Two plausible defects are provably unreachable on these curves.

        Mutation testing flags them as uncaught; they are equivalent mutants,
        not coverage gaps, and this test pins the property that makes them so.

          * point_add's inverse test. The old form ``x1 == x2 and y1 != y2``
            diverges from ``(y1 + y2) % p == 0`` only at y1 == y2 == 0, i.e. a
            point of order two. A group of odd prime order has none.
          * scalar_multiply's skipped final doubling is an optimisation; the
            discarded point cannot affect the result.
        """
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                self.assertEqual(c.n % 2, 1,
                                 "an even order would admit a point of order "
                                 "two and make point_add's inverse test "
                                 "reachable;")
                self.assertTrue(is_probable_prime(c.n))


# ==========================================================================
# H -- package surface
#
# The import-time half of this problem cannot be a test: a broken
# __init__.py stops the suite from loading at all, which is why
# _diagnose_import_failure() exists above. What IS catchable is everything
# that survives import: a stale __all__, a re-export that no longer resolves,
# a module gaining a public name the package never surfaces.
# ==========================================================================
class TestPackageSurface(unittest.TestCase):
    """Package surface and re-exports"""

    def test_H1_every_name_in_all_resolves(self):
        """__all__ must not promise names the package cannot deliver.

        `from ecurve import *` fails loudly on a stale entry; a plain
        `import ecurve` does not, so this drifts silently until someone uses
        the star form.
        """
        missing = [n for n in ecurve.__all__ if not hasattr(ecurve, n)]
        self.assertEqual(missing, [],
                         f"ecurve.__all__ lists names that do not exist: {missing}")

    def test_H2_reexports_are_the_same_objects(self):
        """A re-export must be the module's object, not a stale copy."""
        import ecurve.secp256k1 as S
        for name in ("Secp256k1", "TEST_PARAMS", "LEGACY_PARAMS",
                     "CurveParams", "hash160"):
            with self.subTest(name=name):
                self.assertIs(getattr(ecurve, name), getattr(S, name))

    def test_H3_init_import_list_matches_the_modules(self):
        """Read __init__.py statically and check every imported name exists.

        This is the check that would have caught `int_to_bytes`: the name was
        removed from secp256k1.py during cleanup (it was dead code, nothing
        called it) but left in the package's import list, so `import ecurve`
        raised ImportError and every consumer went down with it.
        """
        import ast
        init = os.path.join(PKG, "__init__.py")
        if not os.path.exists(init):
            self.skipTest("ecurve/__init__.py not present")
        tree = ast.parse(open(init, encoding="utf-8").read())
        checked = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            target = os.path.join(PKG, f"{node.module}.py")
            self.assertTrue(os.path.exists(target),
                            f"__init__.py imports from missing {node.module}.py")
            defined = _defined_names(target)
            for alias in node.names:
                checked += 1
                with self.subTest(name=alias.name, module=node.module):
                    self.assertIn(alias.name, defined,
                                  f"__init__.py imports '{alias.name}' from "
                                  f"{node.module}.py, which does not define it")
        self.assertGreater(checked, 0, "no relative imports found to check")

    def test_H4_all_entries_are_sorted_into_the_import_list(self):
        """Every non-dunder name in __all__ must actually be imported."""
        import ast
        init = os.path.join(PKG, "__init__.py")
        if not os.path.exists(init):
            self.skipTest("ecurve/__init__.py not present")
        tree = ast.parse(open(init, encoding="utf-8").read())
        imported = {a.asname or a.name
                    for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                    for a in n.names}
        assigned = _defined_names(init)
        for name in ecurve.__all__:
            if name.startswith("__"):
                continue
            with self.subTest(name=name):
                self.assertTrue(name in imported or name in assigned,
                                f"__all__ exports '{name}' but __init__.py "
                                f"neither imports nor defines it")

    def test_H5_version_is_declared(self):
        self.assertRegex(getattr(ecurve, "__version__", ""), r"^\d+\.\d+")


# ==========================================================================
# ECDSA correctness foundations
# ==========================================================================
class TestSigning(unittest.TestCase):
    """ECDSA signing, verification and nonces"""

    def test_rfc6979_official_vectors(self):
        """RFC 6979 + SHA-256 test vectors for secp256k1.

        Pinned literals: these come from the specification, not from us.
        """
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
            with self.subTest(msg=msg):
                z = int.from_bytes(hashlib.sha256(msg).digest(), "big") % n
                self.assertEqual(ec._rfc6979_generate_k(d, z), want)

    def test_sign_verify_roundtrip_legacy(self):
        ec = Secp256k1(LEGACY_PARAMS)
        d, pub = ec.generate_keypair(0x1234567890ABCDEF1234567890ABCDEF)
        for msg in (b"a", b"hello", os.urandom(40)):
            with self.subTest(msg=msg[:8]):
                self.assertTrue(ec.verify_signature(
                    pub, ec.sign_message(d, msg, min_start_range=1)))

    def test_sign_verify_roundtrip_test_curve(self):
        ec = Secp256k1(TEST_PARAMS)
        for _ in range(50):
            d, pub = ec.generate_keypair()
            self.assertTrue(ec.verify_signature(
                pub, ec.sign_message(d, b"msg", min_start_range=1)))

    def test_pubkey_on_curve(self):
        for c in ALL_CURVES:
            ec = Secp256k1(c)
            for _ in range(20):
                with self.subTest(curve=c.name):
                    _, pub = ec.generate_keypair()
                    self.assertTrue(ec.is_on_curve(pub))

    def test_B6_ry_not_reduced_mod_n(self):
        """r_y is the raw affine coordinate (mod p), never reduced mod n.

        Strengthened: the old version merely recorded whether a y >= n ever
        showed up. With p - n = 336 on secp17k1 that event has probability
        3.4e-3, so 5000 signatures miss it with probability ~5e-8. Require it.
        """
        c = TEST_PARAMS
        ec = Secp256k1(c)
        seen_high_y = 0
        for _ in range(5000):
            d, _pub = ec.generate_keypair()
            _z, _rx, r_y, _s, k, _ki = ec.sign_message(d, b"m", 1)
            x, y = ec.scalar_multiply(k, c.g)
            self.assertEqual(r_y, y)               # equals raw affine y ...
            self.assertEqual(r_y, r_y % c.p)       # ... which lives mod p
            if y >= c.n:
                seen_high_y += 1
                self.assertNotEqual(r_y, r_y % c.n)
        self.assertGreater(seen_high_y, 0,
                           "no coordinate reached y >= n; the regression this "
                           "test exists for was never actually exercised")

    def test_C8_verify_infinity_no_crash(self):
        ec = Secp256k1(TEST_PARAMS)
        _d, pub = ec.generate_keypair(valid_test_key())
        z, r, ry, s, k, kinv = ec.sign_message(valid_test_key(), b"m", 1)
        tampered = (z, r, ry, (s + 1) % TEST_PARAMS.n, k, kinv)
        self.assertIn(ec.verify_signature(pub, tampered), (True, False))

    def test_C8_point_add_inverse_returns_none(self):
        ec = Secp256k1(TEST_PARAMS)
        p1 = ec.scalar_multiply(5, TEST_PARAMS.g)
        p2 = ec.inverse_point(p1, TEST_PARAMS.p)
        self.assertIsNone(ec.point_add(p1, p2))

    def test_C10_candidate_stream_advances(self):
        """A rejected deterministic nonce must yield a NEW value, not loop."""
        gen = Secp256k1(LEGACY_PARAMS)._rfc6979_k_candidates(0x1, 42)
        self.assertNotEqual(next(gen), next(gen))

    def test_E14_scalar_mult_infinity_is_none(self):
        ec = Secp256k1(TEST_PARAMS)
        self.assertIsNone(ec.scalar_multiply(TEST_PARAMS.n, TEST_PARAMS.g))
        self.assertIs(ec.is_on_curve(None), True)

    def test_E18_unique_keys_bounded(self):
        ec = Secp256k1(TEST_PARAMS)
        keys = ec.generate_unique_keys(20, 10)
        self.assertEqual(len(keys), 20)
        self.assertEqual(len(set(keys)), 20)
        with self.assertRaises(ValueError):
            ec.generate_unique_keys(TEST_PARAMS.n + 5, 10)

    def test_C12_hash160(self):
        """HASH160(x) = RIPEMD160(SHA256(x)), including the pure-Python path.

        Pinned literal: an external known-answer vector.
        """
        h = hash160(b"abc")
        self.assertEqual(len(h), 20)
        self.assertEqual(h.hex(), "bb1be98c142444d7a56aa3981c3942a978e4dc33")


# ==========================================================================
# A -- import layout and invocation modes
# ==========================================================================
class TestInvocationModes(unittest.TestCase):
    """Import layout and invocation modes"""

    def test_A1_package_imports(self):
        import importlib
        self.assertTrue(hasattr(
            importlib.import_module("generate_transaction_list"), "main"))

    def test_A3_standalone_secp256k1_from_inside_package_dir(self):
        """`python secp256k1.py` from inside ecurve/ must work.

        The expected order is read from TEST_PARAMS instead of being pasted in:
        the old literal "1241630743" belonged to a curve that no longer ships.
        """
        r = _subprocess(["-c", "import secp256k1; print(secp256k1.TEST_PARAMS.n)"],
                        cwd=PKG)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), str(TEST_PARAMS.n))

    def test_A3_standalone_k_magnitude_from_inside_package_dir(self):
        r = _subprocess(["k_magnitude.py", "--curve-mode", "test",
                         "--digits", "47"], cwd=PKG)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("magnitude", r.stdout)

    def test_A3_module_mode_from_root(self):
        r = _subprocess(["-m", "ecurve.k_magnitude", "--curve-mode", "legacy",
                         "--digits", "47"], cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_A3_ripemd160_fallback_importable_both_ways(self):
        r = _subprocess(["-c", "from _ripemd160 import ripemd160; "
                               "print(ripemd160(b'abc').hex())"], cwd=PKG)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(),
                         "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc")


# ==========================================================================
# A2 / B3 / D -- transaction generator and report
# ==========================================================================
class TestGeneratorReport(unittest.TestCase):
    """Transaction generator and report"""

    def test_A2_demo_runs(self):
        _quiet(G._print_once_demo, Secp256k1(TEST_PARAMS))

    def test_B3_accuracy_denominator(self):
        """The accuracy denominator must be the rows produced, not a plan."""
        rows, stats = _quiet(
            G._collect_rows, Secp256k1(TEST_PARAMS), total_key_count=5,
            private_key=valid_test_key(), transaction_limit_per_key=10,
            min_start_range=10, ml_expected_accuracy=1.0,
        )
        self.assertEqual(stats["total_transaction_count"], len(rows),
                         "_collect_rows is dividing by a planned figure")
        matches = sum(1 for r in rows if r[9] == "Yes")
        expected = round(100.0 * matches / len(rows), 2)
        self.assertEqual(_pct(stats["ml_accuracy_k_val"]), expected)
        self.assertEqual(expected, 100.0,
                         "at ml_expected_accuracy=1.0 every row should match")

    def test_B3b_denominator_survives_skipped_rows(self):
        """Stronger form: force a skip so plan and reality provably diverge.

        The weaker test passes even against a partial fix using
        effective_key_count * transaction_limit_per_key, because with one key
        that product happens to equal len(rows).
        """
        ec = Secp256k1(TEST_PARAMS)
        real_sign = ec.sign_message
        state = {"n": 0}

        def sign_with_one_single_digit_nonce(d, m, min_start_range=1):
            z, r_x, r_y, s, k, k_inv = real_sign(d, m, min_start_range)
            state["n"] += 1
            if state["n"] == 1:
                k = 7          # single digit -> str(k)[:2] is 1 char -> C11 skip
            return z, r_x, r_y, s, k, k_inv

        ec.sign_message = sign_with_one_single_digit_nonce
        rows, stats = _quiet(
            G._collect_rows, ec, total_key_count=1,
            private_key=valid_test_key(), transaction_limit_per_key=10,
            min_start_range=10, ml_expected_accuracy=1.0,
        )
        self.assertEqual(len(rows), 9, "expected exactly one skipped row")
        self.assertEqual(stats["total_transaction_count"], 9,
                         "a planned figure would read 10 and understate accuracy")
        self.assertEqual(_pct(stats["ml_accuracy_k_val"]), 100.0)

    def test_D_generator_rejects_bad_args(self):
        for bad in (["--keys", "0"], ["--keys", "-3"],
                    ["--ml-exp-accuracy", "2.0"], ["--private-key", "-5"]):
            with self.subTest(args=bad), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    G.parse_args(bad)


# ==========================================================================
# C11 / E16 / E21 -- magnitude predictor
# ==========================================================================
class TestMagnitudePredictor(unittest.TestCase):
    """Magnitude predictor"""

    def test_C11_single_digit_message(self):
        """A single-digit k must raise a named error, not crash deep down."""
        with self.assertRaises(ValueError) as ctx:
            predict_magnitude("7", LEGACY_PARAMS.n)
        self.assertIn("at least two", str(ctx.exception))

    def test_E16_doctests(self):
        import doctest
        res = doctest.testmod(km, verbose=False)
        self.assertEqual(res.failed, 0,
                         f"{res.failed}/{res.attempted} doctests failed; the "
                         f"TEST_PARAMS examples are curve-specific and must be "
                         f"updated whenever the curve changes")

    def test_E21_distribution_conditioned_sums_to_one(self):
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                dist = magnitude_distribution(47, c.n)
                self.assertAlmostEqual(sum(dist.values()), 1.0, places=12)

    def test_E21b_predictor_is_effectively_constant(self):
        """The documented negative result, stated curve-independently.

        Whatever the curve, one magnitude dominates almost every leading pair:
        the predictor carries the base rate, not information. Asserting the
        shape rather than a magic number keeps this valid across curve swaps.
        """
        from collections import Counter
        for c in ALL_CURVES:
            with self.subTest(curve=c.name):
                counts = Counter(predict_magnitude(lead, c.n).magnitude
                                 for lead in range(10, 100))
                self.assertGreaterEqual(counts.most_common(1)[0][1], 88,
                                        f"{c.name}: predictor is less "
                                        f"degenerate than documented")


# ==========================================================================
# B4 / B5 / B7 / D -- cost model, core formulas
# ==========================================================================
class TestCostModelCore(unittest.TestCase):
    """Cost model: core formulas and validation"""

    def test_B4_duration_years(self):
        out = A.fmt_duration(40.0)                 # 1e40 seconds
        self.assertTrue(out.endswith("yr"))
        exp = float(out[2:].split()[0])
        self.assertAlmostEqual(exp, 40.0 - A._LOG10_SEC_PER_YEAR, delta=0.05)

    def test_B4b_duration_precision_is_optional(self):
        """`precision` must stay optional: "orders" is the only defensible
        default, since the model cannot support more."""
        self.assertEqual(A.fmt_duration(40.0), A.fmt_duration(40.0, "orders"))
        self.assertNotEqual(A.fmt_duration(40.0, "full"),
                            A.fmt_duration(40.0, "orders"))
        self.assertEqual(A.fmt_duration(None), "--")
        self.assertEqual(A.fmt_duration(-3.0), "<1 s")

    def test_B5_t_per_run_sec_default_placeholder(self):
        args = A.parse_args([])
        self.assertEqual(args.t_per_run_sec, 2.0)
        self.assertIsNone(args.t_run_doubling_dims,
                          "dimension scaling must stay opt-in: switching it on "
                          "silently would change every published figure")

    def test_B7_m_req_formula(self):
        """Pinned formula: m_req = ceil(ceil(bits / L) * safety_factor)."""
        for L in (5, 6, 7, 8):
            with self.subTest(L=L):
                self.assertEqual(A.required_clean_signatures(256, L, 1.3),
                                 math.ceil(math.ceil(256 / L) * 1.3))

    def test_D_complexity_rejects_bad_args(self):
        """--surplus is deliberately absent here: v2 removed the flag (F1), so
        a negative value is rejected because the option is gone, not because
        its range is validated. Asserting it would pass for the wrong reason."""
        for bad in (["--safety-factor", "-1"], ["--dim-overhead", "-3"],
                    ["--t-per-run-sec", "0"], ["--curve-bits", "0"],
                    ["--leak-bits", "0"], ["--cores", "0"],
                    ["--p-mag", "0"], ["--p-mag", "101"],
                    ["--p-ml-step", "0"], ["--correlation", "1.5"],
                    ["--p-reduction-success", "0"],
                    ["--p-reduction-success", "1.1"],
                    ["--t-run-doubling-dims", "0"]):
            with self.subTest(args=bad), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    A.parse_args(bad)


# ==========================================================================
# F1 -- signature supply is explicit; `surplus` is gone
# ==========================================================================
class TestCostModelSupply(unittest.TestCase):
    """Cost model: signature supply (F1)"""

    def test_F1_surplus_flag_removed_with_explanation(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            A.parse_args(["--surplus", "2.0"])
        self.assertIn("removed in v2", err.getvalue(),
                      "removing a flag without saying why reads like a typo")

    def test_F1_unlimited_supply_is_the_iid_limit(self):
        m_req, p = 56, 0.738
        self.assertAlmostEqual(
            A.log10_success_per_draw(None, m_req, None, p, 1.0),
            m_req * math.log10(p), places=12)

    def test_F1_pool_params_rejected_without_scarce_supply(self):
        for bad in (["--t-raw", "100"], ["--filter-pass-rate", "0.5"],
                    ["--filter-recall", "0.5"]):
            with self.subTest(args=bad), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    A.parse_args(bad)

    def test_F1_scarce_supply_requires_t_raw(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            A.parse_args(["--supply", "scarce"])


# ==========================================================================
# F2 -- the pre-filter obeys Bayes
# ==========================================================================
class TestCostModelFilter(unittest.TestCase):
    """Cost model: Bayesian pre-filter (F2)"""

    def test_F2_uninformative_filter_leaves_purity_unchanged(self):
        post, fpr = A.bayes_post_clean(0.70, 0.5, 0.5)
        self.assertAlmostEqual(post, 0.70, places=12,
                               msg="a blind filter must not change purity")
        self.assertAlmostEqual(fpr, 0.5, places=12)

    def test_F2_informative_filter_raises_purity(self):
        post, _ = A.bayes_post_clean(0.70, 0.80, 0.95)
        self.assertGreater(post, 0.70)
        self.assertLess(post, 1.0)

    def test_F2_implied_false_pass_rate_obeys_total_probability(self):
        p_clean, pi, recall = 0.70, 0.80, 0.95
        _, fpr = A.bayes_post_clean(p_clean, pi, recall)
        self.assertAlmostEqual(p_clean * recall + (1 - p_clean) * fpr, pi,
                               places=12)

    def test_F2_inconsistent_filter_is_rejected(self):
        """pass_rate below p_clean*recall implies P(clean|pass) > 1."""
        with self.assertRaises(ValueError):
            A.bayes_post_clean(0.90, 0.50, 1.0)

    def test_F2_filter_pass_rate_actually_changes_the_answer(self):
        """The v1 inertness regression, pinned: sweeping --filter-pass-rate
        over a 20x range left every reported figure byte-identical."""
        common = ["--supply", "scarce", "--p-ml-start", "90", "--p-ml-stop",
                  "90", "--no-legend"]
        a = _cli(common + ["--t-raw", "400", "--filter-pass-rate", "1.0"])
        b = _cli(common + ["--t-raw", "400", "--filter-pass-rate", "0.6"])
        self.assertNotEqual(a, b, "filter_pass_rate is inert again")


# ==========================================================================
# F3 / F4 -- lattice dimension and reduction success
# ==========================================================================
class TestCostModelDimension(unittest.TestCase):
    """Cost model: dimension and reduction success (F3, F4)"""

    def test_F3_t_run_doubles_every_D_dimensions(self):
        lo = A.log10_t_run(58, 2.0, 58, 12)
        hi = A.log10_t_run(70, 2.0, 58, 12)
        self.assertAlmostEqual(hi - lo, math.log10(2.0), places=12)

    def test_F3_t_run_flat_without_doubling_dims(self):
        self.assertEqual(A.log10_t_run(200, 2.0, 58, None),
                         A.log10_t_run(44, 2.0, 58, None))

    def test_F3_dimension_is_load_bearing(self):
        """Different L must cost different per-run time once calibrated."""
        out = _cli(["--leak-bits", "4", "8", "--t-run-doubling-dims", "12",
                    "--t-run-dim-ref", "58", "--p-ml-start", "90",
                    "--p-ml-stop", "90", "--no-legend"])
        times = re.findall(r"t/run = ([\d.eE+-]+) s", out)
        self.assertEqual(len(times), 2, out)
        self.assertNotEqual(float(times[0]), float(times[1]),
                            "`dimension` is decorative again")

    def test_F3_flat_model_warns_against_cross_L_comparison(self):
        out = _cli(["--leak-bits", "4", "6", "8", "--p-ml-start", "90",
                    "--p-ml-stop", "90", "--no-legend"])
        self.assertIn("Do not compare rows across L", out)

    def test_F4_p_reduction_success_enters_the_denominator(self):
        full = A.log10_success_per_draw(None, 56, None, 0.738, 1.0)
        half = A.log10_success_per_draw(None, 56, None, 0.738, 0.5)
        self.assertAlmostEqual(full - half, math.log10(2.0), places=12,
                               msg="halving the success rate must double runs")


# ==========================================================================
# F5 -- p_m parameterisation is not over-determined
# ==========================================================================
class TestCostModelParameterisation(unittest.TestCase):
    """Cost model: p_m parameterisation (F5)"""

    def test_F5_correlation_rejected_in_conditional_mode(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            A.parse_args(["--correlation", "0.2"])
        self.assertIn("double-count", err.getvalue())

    def test_F5_marginal_mode_accepts_correlation(self):
        args = A.parse_args(["--p-mag-mode", "marginal", "--correlation", "0.126"])
        self.assertEqual(args.correlation, 0.126)

    def test_F5_positive_correlation_raises_p_clean(self):
        """rho > 0 means the predictors fail together: the attack gets easier."""
        self.assertGreater(A.joint_clean_probability(0.90, 0.8888, 0.126),
                           A.joint_clean_probability(0.90, 0.8888, 0.0))

    def test_F5_p_clean_always_within_frechet_bounds(self):
        for p_b in (0.30, 0.50, 0.90, 0.99):
            for p_m in (0.30, 0.50, 0.82, 0.99):
                for rho in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    with self.subTest(p_b=p_b, p_m=p_m, rho=rho):
                        v = A.joint_clean_probability(p_b, p_m, rho)
                        self.assertGreaterEqual(v, max(0.0, p_b + p_m - 1.0) - 1e-12)
                        self.assertLessEqual(v, min(p_b, p_m) + 1e-12)


# ==========================================================================
# F6 -- the output is a distribution, not a point
# ==========================================================================
class TestCostModelDistribution(unittest.TestCase):
    """Cost model: output distribution (F6)"""

    def setUp(self):
        self.q = A.log10_success_per_draw(None, 56, None, 0.738, 1.0)
        self.comps = [(1.0, self.q)]

    def test_F6_quantiles_are_ordered(self):
        p05, p50, p95 = (A.runs_quantile(t, self.comps) for t in (0.05, 0.5, 0.95))
        self.assertLess(p05, p50)
        self.assertLess(p50, p95)

    def test_F6_median_matches_closed_form(self):
        """Median of Geometric(q) is ln2 / q."""
        got = A.runs_quantile(0.50, self.comps)
        want = math.log10(-math.log(0.5)) - self.q
        self.assertAlmostEqual(got, want, places=6)

    def test_F6_geometric_band_width_is_exact(self):
        """The p05-p95 span is a constant, independent of q."""
        width = (A.runs_quantile(0.95, self.comps)
                 - A.runs_quantile(0.05, self.comps))
        self.assertAlmostEqual(
            width, math.log10(math.log(0.05) / math.log(0.95)), places=6)

    def test_F6_scarce_pool_band_is_wider_than_iid(self):
        """Pool-composition randomness must widen the band: the Jensen error
        made visible, which a single point estimate hid entirely."""
        iid = (A.runs_quantile(0.95, self.comps)
               - A.runs_quantile(0.05, self.comps))
        out = _cli(["--supply", "scarce", "--t-raw", "200", "--p-ml-start",
                    "90", "--p-ml-stop", "90", "--no-legend"])
        m = re.search(r"1e([\d.]+)\s+1e([\d.]+)\s+1e([\d.]+)", out)
        self.assertIsNotNone(m, out)
        self.assertGreater(float(m.group(3)) - float(m.group(1)), iid)


# ==========================================================================
# F7 / F8 -- reporting, hygiene, and the two defects the v2 pass uncovered
# ==========================================================================
class TestCostModelReporting(unittest.TestCase):
    """Cost model: reporting and hygiene (F7, F8)"""

    def test_F7_default_precision_is_orders(self):
        self.assertEqual(A.parse_args([]).precision, "orders")

    def test_F7_unvalidated_assumptions_are_printed(self):
        out = _cli(["--p-ml-start", "90", "--p-ml-stop", "90"])
        self.assertIn("Unvalidated assumptions", out)
        self.assertIn("partial ECDLP", out,
                      "the load-bearing caveat must travel with the table")

    def test_F8_accuracy_sweep_grid_does_not_drift(self):
        for step in (1.0, 0.1, 0.25, 0.3, 0.7):
            with self.subTest(step=step):
                self.assertEqual(len(A.accuracy_sweep(80.0, 99.0, step)),
                                 int((99.0 - 80.0) / step + 1e-9) + 1)

    def test_F8_legend_reports_every_swept_L(self):
        """v1's legend printed only leak_bits[0] above three different tables."""
        out = _cli(["--leak-bits", "4", "6", "8", "--p-ml-start", "90",
                    "--p-ml-stop", "90"])
        self.assertIn("[4, 6, 8]", out)

    def test_F8_fully_clean_pool_does_not_raise(self):
        """Regression: log1p(-q) is a domain error at q = 1, which a wholly
        clean pool reaches exactly."""
        out = _cli(["--supply", "scarce", "--t-raw", "60", "--p-mag", "99",
                    "--p-ml-start", "99", "--p-ml-stop", "99", "--no-legend"])
        self.assertIn("0.9801", out)

    def test_F8_filter_recall_defaults_to_uninformative(self):
        """Regression: a recall default of 1.0 makes most pass rates
        inconsistent. The neutral default is recall = pass_rate."""
        args = A.parse_args(["--supply", "scarce", "--t-raw", "100",
                             "--filter-pass-rate", "0.5"])
        self.assertEqual(args.filter_recall, 0.5)
        out = _cli(["--supply", "scarce", "--t-raw", "400",
                    "--filter-pass-rate", "0.5", "--p-ml-start", "90",
                    "--p-ml-stop", "90", "--no-legend"])
        self.assertNotIn("inconsistent", out)

    def test_F8_undersized_pool_gives_actionable_advice(self):
        """v1 printed a bare 'n/a', conflating 'infeasible' with 'below
        threshold'."""
        out = _cli(["--supply", "scarce", "--t-raw", "40", "--p-ml-start",
                    "90", "--p-ml-stop", "90", "--no-legend"])
        self.assertIn("need T_raw >=", out)

    def test_F8_readme_defaults_match_the_parser(self):
        """The README must not drift from the code again.

        v1 shipped three drifts at once: the docstring claimed --p-mag default
        90.0 against a real 82.0, and both docstring and README listed
        cores=[1, 8, 64] against a real [1, 8, 64, 124].
        """
        path = os.path.join(ROOT, "README.md")
        if not os.path.exists(path):
            self.skipTest("README.md not present next to the suite")
        with open(path, encoding="utf-8") as f:
            block = re.search(r"<!-- defaults:begin -->(.*?)<!-- defaults:end -->",
                              f.read(), re.S)
        self.assertIsNotNone(block, "README lacks the machine-checked defaults block")
        args = A.parse_args([])
        declared = dict(re.findall(r"`([a-z_]+)\s*=\s*([^`]+)`", block.group(1)))
        self.assertTrue(declared, "defaults block present but empty")
        for name, raw in declared.items():
            with self.subTest(option=name):
                self.assertTrue(hasattr(args, name),
                                f"README declares unknown option {name!r}")
                actual = getattr(args, name)
                got = (f"[{', '.join(str(x) for x in actual)}]"
                       if isinstance(actual, list) else str(actual))
                self.assertEqual(got, raw.strip())


# ==========================================================================
# Table-formatting runner
# ==========================================================================

_ANSI = {"PASS": "\033[32m", "FAIL": "\033[31m", "ERROR": "\033[31m",
         "SKIP": "\033[33m", "XFAIL": "\033[33m", "dim": "\033[2m",
         "bold": "\033[1m", "off": "\033[0m"}


def _colour(text: str, key: str, enabled: bool) -> str:
    return f"{_ANSI[key]}{text}{_ANSI['off']}" if enabled and key in _ANSI else text


_ID_RE = re.compile(r"test_([A-Z]\d{0,2}[a-z]?)_")


def _audit_id(method_name: str) -> str:
    m = _ID_RE.match(method_name)
    return m.group(1) if m else "-"


def _pretty(method_name: str) -> str:
    name = re.sub(r"^test_", "", method_name)
    name = re.sub(r"^[A-Z]\d{0,2}[a-z]?_", "", name)
    return name.replace("_", " ")


def _wrap(text: str, width: int, indent: str) -> list[str]:
    """Word-wrap a detail line instead of truncating it mid-sentence."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return [indent + ln for ln in lines]


# Logical reading order for the table; anything unlisted sorts last, by name.
SECTION_ORDER = [
    "TestPackageSurface",
    "TestCurveParameters",
    "TestSigning",
    "TestInvocationModes",
    "TestGeneratorReport",
    "TestMagnitudePredictor",
    "TestCostModelCore",
    "TestCostModelSupply",
    "TestCostModelFilter",
    "TestCostModelDimension",
    "TestCostModelParameterisation",
    "TestCostModelDistribution",
    "TestCostModelReporting",
]


class TableResult(unittest.TestResult):
    """Collects (test, status, duration, detail) instead of printing dots.

    Note on subtests. unittest reports a failing ``self.subTest(...)`` block
    through ``addSubTest`` and then does NOT call ``addSuccess`` for the parent
    test. A result class that only overrides addSuccess/addFailure/addError
    therefore records nothing at all for that test: it vanishes from the table,
    the FAIL count stays at zero, and the summary reports ALL GREEN over a real
    failure. Dozens of tests here use subTest, so this is handled explicitly
    below, and ``print_table`` cross-checks the record count against
    ``testsRun`` so a dropped outcome can never pass silently again.
    """

    def __init__(self):
        super().__init__()
        self.records: list[tuple[unittest.TestCase, str, float, str]] = []
        self._t0 = 0.0
        self._emitted = True
        self._sub_failures: list[tuple[object, tuple]] = []

    def startTest(self, test):
        super().startTest(test)
        self._t0 = time.perf_counter()
        self._emitted = False
        self._sub_failures = []

    def _record(self, test, status, detail=""):
        self._emitted = True
        self.records.append((test, status, time.perf_counter() - self._t0, detail))

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "PASS")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "FAIL", self._headline(err))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "ERROR", self._headline(err))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "SKIP", reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, "XFAIL", self._headline(err))

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._sub_failures.append((subtest, err))

    def stopTest(self, test):
        super().stopTest(test)
        if self._emitted:
            return
        if self._sub_failures:
            sub, err = self._sub_failures[0]
            count = len(self._sub_failures)
            params = re.search(r"\(([^()]+)\)\s*$", str(sub))
            where = f" [{params.group(1)}]" if params else ""
            plural = "s" if count > 1 else ""
            status = ("FAIL" if issubclass(err[0], AssertionError) else "ERROR")
            self._record(test, status,
                         f"{count} subtest{plural} failed; "
                         f"first{where}: {self._headline(err)}")
        else:
            # A test that reports no outcome at all is a defect in this result
            # class, not a pass. Surface it rather than dropping the row.
            self._record(test, "ERROR", "unittest reported no outcome")

    @staticmethod
    def _headline(err) -> str:
        exc_type, exc_value, _ = err
        text = str(exc_value).strip().splitlines()
        first = text[0] if text else ""
        return f"{exc_type.__name__}: {first}" if first else exc_type.__name__


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def print_table(result: TableResult, wall: float, colour: bool) -> None:
    if not result.records:
        print("  no tests selected")
        return

    name_w = max(len(_pretty(r[0]._testMethodName)) for r in result.records)
    name_w = max(38, min(name_w, 60))
    width = 5 + 6 + name_w + 3 + 8 + 11

    def rule(ch="-"):
        print("  " + ch * width)

    title = "ECDSA lattice ML  --  regression suite"
    print()
    print("  " + "=" * width)
    print("  " + _colour(title, "bold", colour))
    print(f"  curves: {TEST_PARAMS.name} ({TEST_PARAMS.n.bit_length()}-bit)"
          f"  |  {LEGACY_PARAMS.name} ({LEGACY_PARAMS.n.bit_length()}-bit)"
          f"  |  python {sys.version.split()[0]}")
    print("  " + "=" * width)

    grouped: dict[str, list] = {}
    titles: dict[str, str] = {}
    for rec in result.records:
        cls = type(rec[0])
        key = cls.__name__
        titles[key] = (cls.__doc__ or key).strip().splitlines()[0]
        grouped.setdefault(key, []).append(rec)

    def order(key: str) -> tuple[int, str]:
        return ((SECTION_ORDER.index(key), "") if key in SECTION_ORDER
                else (len(SECTION_ORDER), key))

    idx = 0
    details: list[tuple[str, str, str]] = []
    for key in sorted(grouped, key=order):
        recs = grouped[key]
        print()
        print("  " + _colour(titles[key], "bold", colour))
        rule()
        for test, status, secs, detail in recs:
            idx += 1
            label = _pretty(test._testMethodName)
            dots = "." * max(1, name_w - len(label) + 1)
            ms = f"{secs * 1000:,.1f} ms" if secs < 1 else f"{secs:,.2f} s"
            print(f"  {idx:>3}  {_audit_id(test._testMethodName):<5} "
                  f"{label} {_colour(dots, 'dim', colour)} "
                  f"{_colour(status.ljust(6), status, colour)} {ms:>10}")
            if detail:
                details.append((f"{idx}", label, detail))
                for i, line in enumerate(_wrap(detail, width - 12, "       ")):
                    prefix = "^ " if i == 0 else "  "
                    print(_colour(f"{line[:7]}{prefix}{line[7:]}", status, colour))

    total = len(result.records)
    counts = {s: sum(1 for r in result.records if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "SKIP", "XFAIL")}
    print()
    print("  " + "=" * width)
    if total != result.testsRun:
        print("  " + _colour(
            f"WARNING: {result.testsRun} tests ran but {total} rows were "
            f"recorded; an outcome was dropped.", "FAIL", colour))
    # One row per test, even when several of its subtests failed, so compare
    # DISTINCT failing test cases rather than raw entry counts.
    failing = {id(getattr(t, "test_case", t))
               for t, _ in result.failures + result.errors}
    if len(failing) != counts["FAIL"] + counts["ERROR"]:
        print("  " + _colour(
            f"WARNING: unittest failed {len(failing)} distinct tests, the "
            f"table shows {counts['FAIL'] + counts['ERROR']}.", "FAIL", colour))
    summary = (f"  total {total}   "
               + "   ".join(f"{_colour(s, s, colour)} {counts[s]}"
                            for s in ("PASS", "FAIL", "ERROR", "SKIP")
                            if counts[s] or s in ("PASS", "FAIL")))
    print(f"{summary}      wall {wall:,.2f} s")
    verdict = ("ALL GREEN" if result.wasSuccessful() and not counts["FAIL"]
               and not counts["ERROR"] else "FAILURES PRESENT")
    print("  " + _colour(verdict, "PASS" if verdict == "ALL GREEN" else "FAIL", colour))
    print("  " + "=" * width)
    print()


def build_suite(pattern: str | None) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    if not pattern:
        return suite
    keep = unittest.TestSuite(
        t for t in _iter_tests(suite)
        if pattern.lower() in f"{type(t).__name__}.{t._testMethodName}".lower()
    )
    return keep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="regression suite with table output")
    ap.add_argument("-k", "--filter", default=None,
                    help="run only tests whose class.method contains this")
    ap.add_argument("--list", action="store_true", help="enumerate, do not run")
    ap.add_argument("--plain", action="store_true",
                    help="use unittest's own text runner instead of the table")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print full tracebacks for failures")
    args = ap.parse_args(argv)

    suite = build_suite(args.filter)
    tests = list(_iter_tests(suite))
    if not tests:
        print(f"no test matches {args.filter!r}")
        return 1

    if args.list:
        for i, t in enumerate(tests, 1):
            print(f"{i:>3}  {_audit_id(t._testMethodName):<5} "
                  f"{type(t).__name__}.{t._testMethodName}")
        return 0

    if args.plain:
        return 0 if unittest.TextTestRunner(
            verbosity=2 if args.verbose else 1).run(suite).wasSuccessful() else 1

    colour = (not args.no_color) and sys.stdout.isatty() \
        and os.environ.get("TERM", "") != "dumb"
    result = TableResult()
    t0 = time.perf_counter()
    suite.run(result)
    wall = time.perf_counter() - t0
    print_table(result, wall, colour)

    if args.verbose:
        for label, tb in (("FAIL", result.failures), ("ERROR", result.errors)):
            for test, trace in tb:
                print(f"\n{'=' * 70}\n{label}: {test}\n{'-' * 70}\n{trace}")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # `run_tests.py --list | head` closes the pipe early; exiting through
        # the normal path would dump a traceback over the user's output.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
