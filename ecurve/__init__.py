"""ecurve -- secp256k1 research package.

FIX A1: generate_transaction_list.py imported ``ecurve.secp256k1`` while the
files shipped flat, so the module never loaded and the raised ImportError said
"Make sure the files are in the same directory" -- which was exactly where they
already were. The package now actually exists, and the importer in
generate_transaction_list.py falls back to a flat layout as well.
"""

from .secp256k1 import (
    CurveParams,
    LEGACY_PARAMS,
    TEST_PARAMS,
    Secp256k1,
    compress_public_key,
    encode_varint,
    hash160,
    int_to_bytes,
    make_bitcoin_legacy_sighash_message,
    make_bitcoin_segwit_sighash_message,
)
from .k_magnitude import (
    Prediction,
    magnitude_distribution,
    predict_magnitude,
)

__all__ = [
    "CurveParams",
    "LEGACY_PARAMS",
    "TEST_PARAMS",
    "Secp256k1",
    "Prediction",
    "compress_public_key",
    "encode_varint",
    "hash160",
    "int_to_bytes",
    "magnitude_distribution",
    "make_bitcoin_legacy_sighash_message",
    "make_bitcoin_segwit_sighash_message",
    "predict_magnitude",
]

__version__ = "1.1.1"
