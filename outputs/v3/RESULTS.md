# Session 14 results — notebook v3, profile full

Dense model 16 × 320, 22.5M parameters. MoE: 32 experts per layer (4 families × 8), 114.4M total, 22.63M active. Device cuda (Tesla T4).

Checks: **17/17**

| run | steps | val at start | final val | tokens/s | peak MiB |
|---|---|---|---|---|---|
| R1 dense | 3,051 | 9.0792 | 4.1230 | 54,666 | 4,052 |
| R2 dense continues | 3,051 | 4.1230 | 3.7757 | 50,728 | 6,321 |
| R2s dense, FFN at 1/8 speed | 3,051 | 4.1230 | 3.8268 | 49,374 | 6,374 |
| R3 MoE | 3,051 | 4.1230 | 3.7932 | 16,253 | 11,500 |

- the MoE (R3) finishes above the dense model (R2) by 0.0175 (+0.0175)
- the MoE (R3) finishes below the dense model with the same slowed FFN (R2s) by 0.0336 (-0.0336)
- slowing the dense FFN (R2s) finishes above the dense model (R2) by 0.0511 (+0.0511)

**Conclusion:** The MoE beats the dense model given the same slowed FFN, so its extra stored parameters helped, at the same work per token.

R3 balance: MaxVio 0.214 at the end, dead experts 0.

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
- [PASS] v3: R2s starts as R1 (slowed FFN keeps the function)
- [PASS] v3: R2, R2s and R3 finished their steps
- [PASS] v3: R3 started at R1's validation loss
- [PASS] G5: the MoE keeps training (loss below the conversion value by more than the noise)
- [PASS] v3: no dead experts at the end of R3
