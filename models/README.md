# models/

Reserved for machine-learning work. **Empty, and frozen.**

> 🛑 **STATUS:** This directory is frozen under the project-wide suspension
> (see the top-level [README §0](../README.md#0-why-this-project-is-suspended)).
> No machine-learning model is planned or in progress.

The "predictor" in `generate_transaction_list.py` is a simulation: a shuffled
boolean mask sized to `--ml-exp-accuracy`. It is not a model, and its accuracy is
an input assumption rather than a measurement. See README §1.

## Why no model belongs here

Anything that recovers leading bits of `k` from the public `(r, s, z, Q)` of a
**correctly implemented** signer is equivalent to solving a (partial) ECDLP,
because `r = (k·G).x mod n`. That is precisely the barrier on which the whole
project was suspended: the one model that would matter cannot exist without
breaking the curve, and the only channel that was investigated — the
leading-digit → magnitude predictor (README §3.9) — is a proven negative result,
a constant function returning the prior.

Targets that are **not** ECDLP, and where the literature already shows real
results, are deliberately **out of scope** for this project — it models correct
RFC 6979 nonces with no leak — and are listed here only to mark the boundary:

- **Biased or broken nonce generation** — an implementation flaw, not a curve
  weakness. Cf. the Android `SecureRandom` failure (2013), the PS3 fixed-nonce
  incident, LadderLeak.
- **Side channels** — timing, power, EM. Cf. Minerva, TPM-Fail.
- **Lattice preprocessing** — signature subset selection, block-size prediction
  for BKZ.

Should the project ever resume, any model added here must ship the measurement
that justifies it: the accuracy of bit recovery on held-out data, against a
stated baseline. Until then, `--ml-exp-accuracy` stays an assumption and every
downstream complexity figure inherits it.
