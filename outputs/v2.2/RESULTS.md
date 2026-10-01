# Session 14 results — notebook v2.2, profile full

Dense model 16 × 320, 22.5M parameters. MoE: 32 experts per layer (4 families × 8), 114.4M total, 22.63M active. Device cuda (Tesla T4).

Checks: **15/17**

| run | steps | val at start | final val | tokens/s | peak MiB |
|---|---|---|---|---|---|
| R1 dense | 3,051 | 9.0792 | 4.1237 | 59,674 | 4,052 |
| MoE preview | 450 | 4.1237 | 4.1653 | 17,994 | 9,435 |

G1: fp64 3.5e-15, fp32 2.2e-06 (relative logit difference); converted val 4.12367 vs R1 4.12368.
G2: router gradient min 2.22e-02 (joint), max 0.0e+00 (per-family).
G3: toy MaxVio 0.67 -> 0.05.
Preview: dead experts 1, MaxVio 1.781, clone distance 0.4121.

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
- [PASS] diagnostic v2.2: all six runs finished
