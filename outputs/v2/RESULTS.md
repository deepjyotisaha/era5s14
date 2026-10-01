# Session 14 results — notebook v2, profile full

Dense model 16 × 320, 22.5M parameters. MoE: 32 experts per layer (4 families × 8), 114.4M total, 22.63M active. Device cuda (Tesla T4).

Checks: **14/16**

| run | steps | val at start | final val | tokens/s | peak MiB |
|---|---|---|---|---|---|
| R1 dense | 3,051 | 9.0792 | 4.1230 | 54,666 | 4,052 |
| MoE preview | 450 | 4.1230 | 4.1625 | 17,584 | 9,441 |

G1: fp64 1.3e-14, fp32 2.1e-06 (relative logit difference); converted val 4.12303 vs R1 4.12303.
G2: router gradient min 2.20e-02 (joint), max 0.0e+00 (per-family).
G3: toy MaxVio 0.67 -> 0.05.
Preview: dead experts 1, MaxVio 1.964, clone distance 0.4126.

## Checks

- [PASS] data: 100M tokens, enough windows for R1 and both continuations
- [PASS] R1: finished its step budget
- [PASS] R1: validation loss fell
- [PASS] R1: never decayed (last learning rate = peak)
- [PASS] R1: checkpoint reloads to the same validation loss
- [PASS] conversion: clones hold R1's slices, weights and Adam averages
- [PASS] G1: converted logits match R1's in fp64
- [PASS] G1: converted validation loss matches R1's
- [PASS] G2: every router gets a gradient (joint rescaling)
- [PASS] G2: the per-family rule gives no router gradient (zero up to rounding)
- [PASS] G3: bias balancing evens out a skewed toy layer
- [PASS] G4: parameter counts match the formula
- [PASS] preview: finished its steps
- [FAIL] preview: validation loss fell after conversion
- [FAIL] preview: no dead experts in the last window
- [PASS] preview: the clones have started to separate
