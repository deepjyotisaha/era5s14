# Session 14 results — notebook v1, profile full

Model 16 × 320, vocab 8,192, 22.5M parameters (58% in the feed-forward layers). Device cuda (Tesla T4).

V1 checks: **5/5**

| run | steps | tokens | val before | final train | final val | tokens/s | peak MiB |
|---|---|---|---|---|---|---|---|
| R1 dense | 3,051 | 50.0M | 9.0792 | 4.1474 | 4.1230 | 54,666 | 4,052 |

## Samples from R1

- 'The best way to learn a new language is to try to make a new generation of people. The first one is to take a new generation to write a new generation of people who have seen the right method of finding and learning to the world.'
- 'Scientists have found that it was a great day to use the world. It is not an easy method to use. It also comes with a very simple tool that they are very professional and not very important for them. It'
- 'In the middle of the city there is no lap.\nIn the middle of the Southwest, the middle of the Navy is an indication of the airways, whose main characteristic is the first time to test the class'

## Checks

- [PASS] data: 100M tokens, enough windows for R1 and both continuations
- [PASS] R1: finished its step budget
- [PASS] R1: validation loss fell
- [PASS] R1: never decayed (last learning rate = peak)
- [PASS] R1: checkpoint reloads to the same validation loss
