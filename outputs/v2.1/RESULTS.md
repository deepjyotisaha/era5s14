# Session 14 results — notebook v2.1, profile full

Dense model 16 × 320, 22.5M parameters. MoE: 32 experts per layer (4 families × 8), 114.4M total, 22.63M active. Device cuda (Tesla T4).

Checks: **16/17**

| run | steps | val at start | final val | tokens/s | peak MiB |
|---|---|---|---|---|---|
| R1 dense | 3,051 | 9.0792 | 4.1243 | 53,355 | 4,052 |
| MoE preview | 450 | 4.1243 | 4.1654 | 17,691 | 9,437 |

G1: fp64 3.4e-15, fp32 1.5e-06 (relative logit difference); converted val 4.12426 vs R1 4.12427.
G2: router gradient min 1.84e-02 (joint), max 0.0e+00 (per-family).
G3: toy MaxVio 0.67 -> 0.05.
Preview: dead experts 0, MaxVio 1.740, clone distance 0.4129.

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
- [PASS] preview: no dead experts in the last window
- [PASS] preview: the clones have started to separate
- [PASS] diagnostic: B (dense in MoE clothing) tracks A
