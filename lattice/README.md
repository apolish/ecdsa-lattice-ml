# lattice/

Reserved for the HNP lattice attack implementation. **Empty, and frozen.**

> 🛑 **STATUS:** This directory is frozen under the project-wide suspension
> (see the top-level [README §0](../README.md#0-why-this-project-is-suspended)).
> No lattice-reduction implementation is planned.

`ecdsa_attack_complexity.py` at the repository root only *evaluates a closed-form
cost model* — it performs no lattice reduction. The formula it uses is documented
in README §3.10.

Implementing the reduction here would not change the project's conclusion. The
lattice step is the cheap, already-published half of the attack (Howgrave-Graham
& Smart 2001; Nguyen & Shparlinski 2003). The expensive half — supplying the
leading bits of each nonce — is a (partial) ECDLP, and is exactly where the
project was suspended (README §0). A reduction run over bits no one can obtain
recovers no key.

For the record, such an implementation would need, at minimum:

- construction of the HNP lattice basis from `(r_i, s_i, z_i)` plus assumed
  leading-bit bounds on each `k_i`;
- LLL/BKZ reduction (fpylll or equivalent — note this would be the project's
  first third-party dependency, and CI currently asserts stdlib-only);
- recovery and verification of `d` from the short vector;
- **calibration of `--t-per-run-sec`**, which the cost model presently guesses
  with a heuristic. This is the single most load-bearing number in the model.

The technique is not novel: Howgrave-Graham & Smart (2001), Nguyen &
Shparlinski (2003). See README §9.
