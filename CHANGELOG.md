# CHANGELOG — fixes from the audit

Version 1.2.0. Entries tagged with a bug ID (A1, B3, …) map to the original
audit. Every fix is covered by a regression test in `tests/test_fixes.py`
(30/30 passing).

1.2.0 is a **minor** bump, not a patch: it changes default behaviour and renames
report keys. Anything parsing the statistics block of a report, or relying on
`--curve-mode` defaulting to `test`, will need updating. See §"Behaviour changes".

## Important correction to my own earlier claims

**E21 (RFC 6979).** In the initial analysis I claimed the original code "will
not match libsecp256k1/bitcoin-core". That was overstated. Checking against the
official RFC 6979 test vectors showed the opposite: **the original passes them
(3/3)**. The reason is that the first candidate, `int.from_bytes(v) % n`, equals
`bits2int(v)` whenever the candidate is `< n` (probability ~1 − 2⁻¹²⁸). The real
defects — genuine, but far narrower — that are fixed here:

- the missing rejection path `K = HMAC(K, V‖0x00); V = HMAC(K, V)`, which is
  precisely what made bug C10 (infinite loop) unfixable without rewriting nonce
  generation;
- a deviation from the standard only in the theoretical `candidate ≥ n` case.

This does not change any other conclusion of the audit, but accuracy matters
more than saving face.

---

## Behaviour changes in 1.2.0

These are deliberate design decisions, not bug fixes. They are listed first
because they are the ones that will break something downstream.

### The toy curve is now named for what it is

`TEST_PARAMS.name` changed from `"secp256k1"` to `"toy-p31r1"`. The old value
was a factual misstatement: that curve is `y² = x³ + x + 35` over a 31-bit
prime, and shares nothing with secp256k1 (`y² = x³ + 7`) but a variable name.
Reports generated in test mode now print `name = toy-p31r1`.

The new name follows SEC nomenclature — prime field, 31 bits, `r` for a
verifiably-random (non-Koblitz) curve, first of its kind. The `r` matters: `a=0`
is the Koblitz family secp256k1 belongs to, and this curve's `a=1` is precisely
what lets it catch defects in the `a` term that secp256k1 structurally cannot
(see README §3.3.1).

Nothing branches on `name` — it appears only in report headers, so the rename is
behaviour-preserving for logic. `mode` remains the selector. Verified: 30/30
tests pass across the rename.

### `--curve-mode` now defaults to `legacy`

In `generate_transaction_list.py` and `ecurve/k_magnitude.py`. The toy curve is a
test fixture, not an alternative research target; defaulting to it invited
readers to mistake toy output for secp256k1 output. Explicit `--curve-mode test`
still selects it.

Consequence: default runs are ~70× slower (≈125 sign/s vs ≈8,810).

### Report statistics keys renamed and reduced

- `case_D<n>_count` → `k_mag_<n>_count`. The `_write_report` regex was updated in
  step, so the block stays live (this was the E17 fix; the key name is cosmetic,
  the wiring is what mattered).
- `ml_expected_accuracy` was **removed** from `stats` and from `base_keys`.
- The `(INPUT ASSUMPTION, not measured)` caveat moved onto `ml_accuracy_k_val`.

The caveat's new placement is the better call: it now sits on the number a reader
actually looks at. Previously the warning lived on a separate line while
`Ml Accuracy K Val: 93.33%` read like a measurement. And it is substantively
honest — the realised match rate of a mask *constructed* to hit a target rate
measures nothing about any model; it echoes the input back.

**Known consequence, unresolved.** Dropping `ml_expected_accuracy` means the
report no longer records what was asked for, and the realised rate can diverge
from it because `num_correct = round(accuracy × planned_tx)`:

| `--ml-exp-accuracy` | `--tx-per-key` | report says |
|---|---|---|
| 0.93 | 10 | **90.00%** |
| 0.93 | 7 | **100.00%** |
| 0.95 | 3 | **100.00%** |
| 0.93 | 200 | 93.00% |

At small `--tx-per-key` the input is unrecoverable from the report. If
reproducibility matters, add an unlabelled `ml_expected_accuracy_input` key
recording the CLI value. The test suite does not require this either way.

---

## A — code did not run at all

- **A1** `generate_transaction_list.py` required an `ecurve` package that did
  not exist. Added `ecurve/__init__.py`; the importer now tries the package
  layout → the flat layout → and emits a diagnostic message that describes the
  actual problem.
- **A2** `_print_once_demo` crashed on every invocation: (1) `sign_message`
  called without `min_start_range`; (2) unpacking 5 names from a 6-tuple. Both
  fixed.
- **A3** *(regression introduced by version 1.1.0 — fixed in 1.1.1)*
  `secp256k1.py` was given a hard relative import, `from ._ripemd160 import ...`,
  which broke the `cd ecurve && python secp256k1.py > secp256k1.txt` workflow
  ("attempted relative import with no known parent package"). That is exactly
  how the original `secp256k1.txt` was generated. Added a try/except fallback
  (as `k_magnitude.py` already had) plus 4 tests covering **all** 9 invocation
  modes — the earlier tests only exercised the package import, which is why the
  regression slipped through.

## B — silently wrong numbers

- **B3** The headline accuracy metric was divided by `total_key_count *
  tx_per_key` instead of the number of rows actually generated. With
  `--private-key` (pool of 1) and `--keys > 1` this produced nonsense (100%
  reported as 20%). The denominator is now `len(rows)`; the simulation mask is
  sized the same way; and when a key off the curve is skipped, the mask index
  advances in step.
- **B4** `fmt_duration` printed log10 of **seconds** with a "yr" suffix (a ~7.5
  order-of-magnitude overstatement) for values above 1e30. Added the conversion
  to years.
- **B5** `--t-per-run-sec` had `default=2.0`, which made the heuristic branch
  dead code and made the legend permanently lie with "USER-PROVIDED". Changed to
  `None`.
- **B6** `r_y = y % n` — a meaningless coordinate reduction (y lives mod p). It
  corrupted the coordinate whenever `y ≥ n` (~1 in 20,912 on `toy-p31r1`).
  `r_y` is now returned raw (mod p).
  **Correction in 1.1.1:** in version 1.1.0 this fix was **incomplete**.
  `identify_condition` compares against topology rows, where both coordinates are
  stored mod p, but `r_x` is `x mod n` **by ECDSA definition** and must be
  reduced. So passing `(r_x, r_y)` still mismatched when `x ≥ n`. Added the
  `Secp256k1.last_nonce_point` property, which exposes the raw `k·G` (both
  coordinates mod p), cached during signing; `identify_condition` now receives
  that instead.
- **B7** The `m_req` formula in the docstring (`ceil(cb/L*sf)`) did not match the
  code (`ceil(ceil(cb/L)*sf)`). Reconciled in favour of the code (both copies).

## C — crashes on reachable inputs

- **C8** `verify_signature` unpacked the result of `point_add` before checking
  for `None` → `TypeError` on the point at infinity. Reordered.
- **C9** `count_subsets` raised `ValueError` when `total_adds < 2`. The
  combination size is now clamped to 0 → `C(n, 0) = 1`.
- **C10** RFC 6979 is deterministic, so `continue` in `sign_message` re-derived
  the same `k` forever. Candidates are now supplied by a generator that
  implements the standard's rejection path.
- **C11** A single-digit `k` (`str(k)[:2]` → 1 character) raised an opaque
  ValueError deep in the stack. Added an explicit guard and a clear message.
- **C12** `hashlib.new("ripemd160")` is unavailable on OpenSSL 3.x. Added the
  pure-Python fallback `_ripemd160.py` (verified against the official vectors
  plus fuzzing against hashlib).

## D — missing validation

- Complexity script: `--surplus`, `--safety-factor`, `--dim-overhead`,
  `--m-req`, `--t-per-run-sec`, `--run-base-sec` are now validated.
- Generator: argparse type-validators on `--keys` (>=1), `--tx-per-key` (>=1),
  `--min-start-range` (>=1), `--ml-exp-accuracy` (in [0,1]), `--private-key`
  (>=0 plus range), and a check that the key space is not exhausted.

## E — hidden / minor

- **E14/E15** Removed the second infinity sentinel `(0,0)` (which is not on the
  curve); `None` is now the only representation.
- **E16** The doctests in `k_magnitude.py` (2 of 4 failing) are fixed and pass.
- **E17** The dead `case_D<n>_count` block in the report is now live:
  `_collect_rows` actually counts the distribution of decimal lengths of `k`.
  (Renamed to `k_mag_<n>_count` in 1.2.0 — see "Behaviour changes".)
- **E18** `generate_unique_keys` returned `list(set(...))` (non-deterministic
  ordering) and could loop forever. It now preserves insertion order and checks
  the bound.
- **E19** Removed the redundant final doubling in `scalar_multiply`.
- **E20** `_prepare_topology_of_key` truncated at the *first* match (which could
  hit a doubling row). It now searches backwards for the *last* match.
- **E21** See the correction at the top.
- **E22** `point_add` crashed in `inverse_mod(0)` for an order-2 point
  (`y1 = y2 = 0`). The opposite-point condition is now `(y1 + y2) % p == 0`.

## Test-suite defects fixed in 1.2.0

The suite itself was wrong in four ways. All four were found by a real failure
on a real machine, not by inspection — which is the point.

- **The B3 test policed presentation, not behaviour.** It asserted
  `stats["ml_accuracy_k_val"] == "100.00%"` and, later, that a caveat string sat
  on a specific key. Both encode one particular report layout as if it were a
  correctness property; the 1.2.0 stats redesign broke them
  (`KeyError: 'ml_expected_accuracy'`) while the code was entirely correct. The
  B3 invariant is the **denominator** and nothing else. The test now asserts
  `total_transaction_count == len(rows)` and independently recomputes
  `matches / len(rows)` from the rows, parsing the numeric prefix out of the
  display string via a `_pct()` helper. Labels are no longer the suite's business.
- **The B3 test was weak.** It passed against a partial fix using
  `effective_key_count * transaction_limit_per_key`, because with a single key
  that product coincidentally equals `len(rows)`. Added
  `test_B3b_denominator_survives_skipped_rows`, which forces a single-digit nonce
  to trigger the C11 skip path so the plan (10) and reality (9) provably diverge.
  Mutation-checked: the full bug fails 2 B3 tests, the partial fix fails 1, a
  stubbed-out accuracy value fails 2, correct code fails 0.
- **Assertions had no messages.** Failures surfaced as a bare
  `AssertionError:` with nothing else, which cost two rounds of wrong diagnosis.
  Every assertion in the B3/B3b/D group now states what was expected, what was
  found, and what it implies.
- **Validation tests dumped argparse usage to stderr.** ~14 lines of usage text
  interleaved with a passing run, reading like a crash. `p.error()` writing to
  stderr before `SystemExit` is the expected path here, so it is now swallowed
  with `contextlib.redirect_stderr`. Mutation-checked that suppressing the noise
  did not suppress the check: removing the `--surplus` validation still fails the
  test, now with `['--surplus', '-5'] was accepted; validation is missing`.

Test count: 29 → 30.

## Documentation defects fixed in 1.2.0

- **`ecurve/k_magnitude.py` module docstring stated two things that are false**,
  and they survived the original audit because the README numbers were verified
  against code while the docstring was not:
  1. *"every leading pair in 10..99 predicts D=77 with p=0.90 (except pair 11,
     p=0.839)"* — pairs **10 and 11 predict D=78**, not 77. Correct: pairs 12..99
     → D=77 at p=0.90 (88 pairs); pair 10 → D=78 at p=0.90; pair 11 → D=78 at
     p=0.839.
  2. *"the base rate of 'n has 77 digits'"* — `n` has **78** decimal digits. The
     0.90 figure is the share of values in `[1, n−1]` that have 77 digits, which
     is a property of the interval, not of `n`.

  Neither error changes the module's conclusion — the predictor is still a
  constant function whose accuracy is a base rate rather than information gained
  — but the specific claims were wrong and are now corrected.

## Deliberately NOT changed (to avoid breaking output compatibility)

- The magic `- 2` in `count_subsets` is kept — it compensates for the systematic
  undercount of the first addition (documented in `UNRECORDED_LEADING_ADDS`).
  Changing it would renumber every row of every existing report.
- `hash_message` in test mode still randomises `z` (not bound to the message) —
  intentional for the toy curve.

## What is NOT fixed, because it is not a code bug but a methodological flaw

The "ML predictor" remains a simulation driven by `--ml-exp-accuracy`.
Recovering the leading bits of `k` from the public `(r, s, z, Q)` is equivalent
to solving (a partial) ECDLP; no script does this, and the real hook is still a
`# TODO`. Every complexity figure inherits that assumption. The fixes make the
code correct and reproducible — they do **not** make the attack real.
