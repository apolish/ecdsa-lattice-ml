# data/

Curated example outputs, committed by hand. **Not** auto-generated on each run —
the scripts write to `out/` (git-ignored) by default.

| File | Produced by |
|---|---|
| `ecdsa_attack_complexity.txt` | `python ecdsa_attack_complexity.py --cores 1` |
| `transaction_list_*.txt` | `python generate_transaction_list.py --curve-mode legacy ...` |

## Read this before quoting any number in these files

`ecdsa_attack_complexity.txt` contains rows such as `3.57 h` against a
`secp256k1` heading. **That figure does not mean secp256k1 can be broken in 3.57
hours.** It is conditional on two assumptions the repository does not
demonstrate:

1. an oracle that supplies the leading bits of every nonce — obtaining one from
   public signature data is equivalent to solving ECDLP;
2. `t_per_run_sec`, the wall-clock of one lattice reduction, which is a
   heuristic guess unless `--t-per-run-sec` was calibrated on real hardware.

Regenerate these examples **with the legend** (i.e. without `--no-legend`) so the
caveats travel with the table. A cost row separated from its assumptions is
misinformation with a decimal point.

See README §1 and §3.12.
