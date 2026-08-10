"""ecurve -- elliptic-curve arithmetic, ECDSA, RFC 6979, magnitude analysis.

WHY THIS FILE IS SHORT
======================
The previous ``__init__.py`` carried a hand-maintained list of re-exports that
included ``int_to_bytes``. That helper was dead code even before it was
removed from ``secp256k1.py`` -- nothing in the repository ever called it --
so deleting it was correct cleanup. But the export list was not updated to
match, and the result was worse than a missing helper: ``import ecurve`` began
raising ``ImportError`` at import time, which takes down every consumer of the
package, including the test suite that would otherwise have reported the
problem.

The list below is therefore kept deliberately small: the names a caller
actually reaches for, nothing else. Everything else stays addressable at its
real home (``ecurve.secp256k1.mod_sqrt_all``, and so on) rather than being
duplicated here, because every duplicated name is another opportunity for the
package surface to drift away from the modules.

``tests/run_tests.py`` pins this file against the modules (class
``TestPackageSurface``) so the drift fails as a test, not as an ImportError.
"""

from __future__ import annotations

from .secp256k1 import (
    CurveParams,
    LEGACY_PARAMS,
    Point,
    Secp256k1,
    TEST_PARAMS,
    compress_public_key,
    encode_varint,
    hash160,
    inverse_mod_safe,
    make_bitcoin_legacy_sighash_message,
    make_bitcoin_segwit_sighash_message,
    mod_sqrt_all,
    recover_x_from_s_zk,
)
from .k_magnitude import (
    Prediction,
    magnitude_distribution,
    predict_magnitude,
)

__version__ = "0.3.0"

__all__ = [
    # curve parameters and the group
    "CurveParams",
    "Point",
    "Secp256k1",
    "TEST_PARAMS",
    "LEGACY_PARAMS",
    # public-only ECDSA relations
    "inverse_mod_safe",
    "mod_sqrt_all",
    "recover_x_from_s_zk",
    # Bitcoin encodings
    "compress_public_key",
    "encode_varint",
    "hash160",
    "make_bitcoin_legacy_sighash_message",
    "make_bitcoin_segwit_sighash_message",
    # nonce magnitude analysis
    "Prediction",
    "magnitude_distribution",
    "predict_magnitude",
    # metadata
    "__version__",
]
