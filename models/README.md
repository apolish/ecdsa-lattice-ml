# models/

Reserved for machine-learning work.

**Currently empty.** The "predictor" in `generate_transaction_list.py` is a
simulation: a shuffled boolean mask sized to `--ml-exp-accuracy`. It is not a
model, and its accuracy is an input assumption rather than a measurement. See
README §1.

## What a model here can and cannot target

Anything that recovers leading bits of `k` from the public `(r, s, z, Q)` of a
**correctly implemented** signer is equivalent to solving a (partial) ECDLP,
because `r = (k·G).x mod n`. No model belongs here for that.

Targets that are not ECDLP, and where the literature shows real results:

- **Biased or broken nonce generation** — an implementation flaw, not a curve
  weakness. Cf. the Android `SecureRandom` failure (2013), the PS3 fixed-nonce
  incident, LadderLeak.
- **Side channels** — timing, power, EM. Cf. Minerva, TPM-Fail.
- **Lattice preprocessing** — signature subset selection, block-size prediction
  for BKZ.

> ⚠️ **NOTICE:** The author is actively researching new ML approaches 
> for partial solutions of ECDLP within the framework of an ongoing project, 
> but these studies are limited to local experiments until the time 
> of official publication.

Any model added here should ship the measurement that justifies it: the
accuracy of bit recovery on held-out data, against a stated baseline. Until
then, `--ml-exp-accuracy` stays an assumption and every downstream complexity
figure inherits it.
