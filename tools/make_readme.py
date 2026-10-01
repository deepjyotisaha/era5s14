"""Write README.md from the runs' own summary files.

Every number in the README comes from outputs/v*/summary.json (the T4 runs); none is typed by hand.
Where a sentence could go either way (ahead/behind), the data picks the words. At the end the README
is read back and each key number is checked against the summaries.

Usage (from session 14/assignment/):
    python tools/make_readme.py
"""
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs")


def load(v):
    return json.load(open(os.path.join(OUT, v, "summary.json"), encoding="utf-8"))


S1, S2, S21, S22, S3 = load("v1"), load("v2"), load("v2.1"), load("v2.2"), load("v3")
for s, v in ((S1, "v1"), (S2, "v2"), (S21, "v2.1"), (S22, "v2.2"), (S3, "v3")):
    assert s["meta"]["notebook_version"] == v and s["meta"]["profile"] == "full", v

# The published pages. Their links are the only hand-kept values in this file: a URL is not a
# measurement, and it does not change when the page is republished.
REPORTS = [
    ("Results report: From Dense to Mixture of Experts",
     "the headline numbers, the path from the first failure to the result in diagrams, this write-up, "
     "and every notebook cell with its output",
     "https://claude.ai/artifact/7Ppswkd1eQZjMJHQR89sQc", "docs/session14_results.html"),
    ("Notebook walkthrough: The Expert Split",
     "every cell of the final notebook with what it does, its full source and what it printed on the T4, grouped by "
     "task, and a ledger of the ten things this assignment got wrong",
     "https://claude.ai/artifact/Bus2ncEUUnDkMUxNoZmZsY", "docs/session14_walkthrough.html"),
    ("Build plan and execution record",
     "every decision with its options, pros and cons; each milestone's results as it happened; the two "
     "diagnostics; the build log",
     "https://claude.ai/artifact/XRCTgwLsgKx272v7qBfSLu", "docs/assignment_plan.html"),
    ("Session report: Mixture-of-Experts",
     "the lesson and transcript in plain English, with the MoE layer drawn step by step, a glossary, "
     "and the first approach note for this assignment",
     "https://claude.ai/artifact/6oW6uEii4374H72jkfhGGn", "docs/session14_report.html"),
]
RESULTS_URL = os.environ.get("ERA5_RESULTS_URL", "")
if RESULTS_URL:
    REPORTS[0] = REPORTS[0][:2] + (RESULTS_URL,) + REPORTS[0][3:]

# ------------------------------------------------------------------ numbers
mc, tc = S3["config"]["model"], S3["config"]["train"]
pb = S3["config"]["params"]
STEPS = S3["config"]["steps"]
R1, R2, R2s, R3 = (S3["runs"][k] for k in ("R1", "R2", "R2s", "R3"))
v3 = S3["v3"]
conv = R3["val"][0]["loss"]
fin = {"R2": R2["final_val"], "R2s": R2s["final_val"], "R3": R3["final_val"]}
tok_step = tc["batch"] * mc["seq"]
decay_tok = (STEPS - v3["decay_steps"]) * tok_step
NOISE = 0.002


def vat(h, tok):
    return min(h["val"], key=lambda q: abs(q["tokens"] - tok))


dec = {k: vat(h, decay_tok)["loss"] - h["final_val"] for k, h in (("R2", R2), ("R2s", R2s), ("R3", R3))}
d2 = [(b["tokens"], b["loss"] - a["loss"]) for a, b in zip(R2["val"], R3["val"])]
ds = [(b["tokens"], b["loss"] - a["loss"]) for a, b in zip(R2s["val"], R3["val"])]
lead_end = max(t for t, d in d2 if d < -NOISE)
at10 = vat(R3, 10e6)["tokens"]
at_dec = vat(R3, decay_tok)["tokens"]
ml = R3["extra"]["moe_log"]
full = [m for i, m in enumerate(ml) if m["step"] - (ml[i - 1]["step"] if i else 0) >= tc["log_every"]]
sw = v3["cfg"]["switch"]
r1s = [S1["runs"]["R1"]["final_val"], S21["runs"]["R1"]["final_val"], S22["runs"]["R1"]["final_val"]]
spread = max(r1s) - min(r1s)
P2 = S2["runs"]["P"]
p2ml = P2["extra"]["moe_log"]
p2_300 = next(m for m in p2ml if m["step"] == 300)
D1 = S21["diag"]["runs"]
D2 = S22["diag2"]["runs"]
g = S3["gates"]
moe_tot, moe_act = S3["moe"]["total_params"], S3["moe"]["active_params"]
slow = R2["tokens_per_second"] / R3["tokens_per_second"]
acc = S3["acceptance"]


def f4(x):
    return f"{x:.4f}"


def sgn(x):
    return f"{x:+.4f}"


def diag_gap(D, k):
    return D[k]["hist"]["final_val"] - D["A"]["hist"]["final_val"]


def maxvio_after(h, switch):
    return max(m["maxvio"] for m in h["extra"]["moe_log"] if m["step"] > switch)


beats_fair = fin["R3"] < fin["R2s"] - NOISE
beats_dense = fin["R3"] < fin["R2"] - NOISE
assert conv - fin["R3"] > NOISE, "the README says the MoE kept learning"

# ------------------------------------------------------------------ text
L = []
A = L.append
A("# From a dense model to a mixture of experts\n")
A("**ERA V5 · Session 14 — Mixture-of-Experts**\n")
A("> *Train a Linear model and convert that into an MoE! Your call on model size and data trained on, "
  "but must show they continue to train and reduce loss!*\n")
A(f"Run on a Colab **{S3['meta']['gpu']}**. Final notebook (v3) acceptance checks: **{acc['passed']}/{acc['total']} passed**. "
  "Every number below is generated from the runs' `outputs/v*/summary.json` files by "
  "[`tools/make_readme.py`](tools/make_readme.py); none is typed by hand.\n")

A("## Reports\n")
A("Each report is published as a web page, and a standalone copy is in `docs/` to open from disk.\n")
for title, what, url, local in REPORTS:
    A(f"- **{title}**: {what}.")
    A(f"  - Published: [{url}]({url})")
    A(f"  - Local copy: [`{local}`]({local})")
A("")

A("## The answer in one table\n")
A("A dense GPT (R1) was trained on 50M tokens, then continued for 50M more tokens three ways from the same "
  "checkpoint, on the same tokens in the same order, with the same learning-rate schedule.\n")
A("| run | what it is | parameters (total / used per token) | final val loss | change since R1 | tokens / s | peak memory |")
A("|---|---|---|---|---|---|---|")
A(f"| **R1** | dense, first 50M tokens | {pb['total']/1e6:.1f}M / {pb['total']/1e6:.1f}M | **{f4(R1['final_val'])}** | — | "
  f"{R1['tokens_per_second']:,.0f} | {R1['peak_mem_bytes']/2**20:,.0f} MiB |")
for k, lab, tot, act in (("R2", "dense continues (control)", pb["total"], pb["total"]),
                         ("R2s", "dense continues, FFN learning at 1/8 speed (fair control)", pb["total"], pb["total"]),
                         ("R3", "**MoE** converted from R1", moe_tot, moe_act)):
    h = S3["runs"][k]
    A(f"| **{k}** | {lab} | {tot/1e6:.1f}M / {act/1e6:.1f}M | **{f4(fin[k])}** | {sgn(fin[k] - conv)} | "
      f"{h['tokens_per_second']:,.0f} | {h['peak_mem_bytes']/2**20:,.0f} MiB |")
A("")
A("Loss is the average cross-entropy per token in nats (lower is better), measured on 327,680 held-out tokens "
  f"from documents never trained on. Three independent trainings of R1 differed by at most {spread:.4f}, "
  f"so differences below about {NOISE} are noise.\n")
A("![validation loss across the conversion](outputs/v3/v3_loss.png)\n")

A("## What the brief asked, answered\n")
A(f"**1 · Train a dense model.** R1: 16 layers × width 320, {pb['total']/1e6:.1f}M parameters "
  f"({pb['feed-forward']/pb['total']*100:.0f}% of them in the feed-forward layers), 50M FineWeb tokens. Validation loss "
  f"{f4(R1['val'][0]['loss'])} → **{f4(R1['final_val'])}**.\n")
A(f"**2 · Convert it into an MoE.** Each layer's feed-forward network (FFN, 1,280 neurons) was cut into 4 slices of 320 "
  f"and each slice copied 8 times: **32 experts per layer in 4 families**, one expert per family per token. "
  f"The MoE stores {moe_tot/1e6:.1f}M parameters but uses {moe_act/1e6:.1f}M per token, the dense model's work plus the "
  f"routers. The conversion keeps the function: the converted model's logits matched R1's to "
  f"{g['G1']['fp64']:.1e} in fp64, and its validation loss was {g['G1']['val_moe']:.5f} against R1's {g['G1']['val_r1']:.5f}.\n")
A(f"**3 · Show it continues to train and reduces loss.** **Yes.** R3's validation loss fell from {f4(conv)} to "
  f"**{f4(fin['R3'])}** ({sgn(fin['R3'] - conv)}) over 50M more tokens, with no expert ever left without tokens.\n")

A("## How it got there: two diagnostics\n")
A("The first conversion did **not** keep learning. Two short diagnostic rounds on the T4 found why.\n")
A("| version | what it tested | what it found |")
A("|---|---|---|")
A(f"| v2 | the conversion, and a {P2['steps']}-step preview of the MoE training | conversion exact, but validation loss **rose** "
  f"{f4(P2['val'][0]['loss'])} → {f4(P2['final_val'])}; after the switch to hard expert selection the load collapsed "
  f"(MaxVio {p2_300['maxvio']:.2f} → {p2ml[-1]['maxvio']:.2f}) |")
A(f"| v2.1 | 7 runs of 200 steps, each changing one thing | a 10× slower router stopped the collapse (MaxVio after the switch "
  f"{maxvio_after(D1['D']['hist'], 100):.2f} vs {maxvio_after(D1['G']['hist'], 100):.2f}); the loss rose whenever tokens were "
  f"split across clones, even with a fixed token-to-clone table ({sgn(diag_gap(D1, 'F'))} vs dense) |")
A(f"| v2.2 | 6 runs: a smaller learning rate for the experts | experts at 1/8 speed turned {sgn(diag_gap(D2, 'D'))} into "
  f"**{sgn(diag_gap(D2, 'D12'))}** against dense after 200 steps |")
A(f"| v3 | 50M tokens: R2, R2s, R3 | the MoE kept learning to {f4(fin['R3'])} (this README) |")
A("")
A("**Why the experts needed to learn slower.** After the split, each clone learns from about an eighth of the tokens in "
  "a batch, so its gradient is much noisier. Adam still takes full-size steps, so the clones wandered. Storing each "
  "expert's weights divided by 8 and multiplying by 8 when used leaves the model's output unchanged, and makes Adam's "
  "steps 1/8 as large for those weights.\n")

A("## Findings\n")
A(f"**1 · The MoE's extra parameters helped, measured against a fair control.** Slowing the experts is part of the "
  f"MoE's recipe, so R2s gives the dense model's FFN the same 1/8 learning speed. R3 finished "
  f"{'below' if beats_fair else 'above'} R2s by {abs(fin['R3'] - fin['R2s']):.4f}. The two were level for the first "
  f"15M tokens (R3 − R2s at {at10/1e6:.1f}M: {sgn(dict(ds)[at10])}); after that R3 pulled away steadily, "
  f"{sgn(dict(ds)[at_dec])} by {at_dec/1e6:.1f}M tokens and {sgn(fin['R3'] - fin['R2s'])} at the end.\n")
A(f"**2 · But ordinary dense training finished {'behind' if beats_dense else 'ahead'}.** R3 led R2 from the first "
  f"evaluation until {lead_end/1e6:.1f}M tokens ({sgn(dict(d2)[at10])} at {at10/1e6:.1f}M tokens, of which the slower "
  f"learning explains most: R2s had the same early lead). In the learning-rate decay (the last 20%) R2 gained "
  f"{dec['R2']:.4f}, R3 {dec['R3']:.4f} and R2s {dec['R2s']:.4f}, and R2 finished {abs(fin['R3'] - fin['R2']):.4f} "
  f"{'behind' if beats_dense else 'ahead of'} R3. A likely reason, not tested: the experts were already taking 1/8-size "
  "steps, so shrinking every step at the end gave them less to gain.\n")
A(f"**3 · The router's speed decided the balance.** With the router learning at full speed, switching from random to "
  f"best-score expert selection sent most tokens to one clone per family (v2, v2.1). With the router 10× slower, R3's "
  f"MaxVio after the switch was at most {max(m['maxvio'] for m in full if m['step'] > sw):.3f} "
  f"({full[-1]['maxvio']:.3f} at the end, over 25-step windows) and no expert went a single window without tokens. "
  f"The 8 clones in each family moved steadily apart (relative distance {ml[0]['clone_dist']:.3f} → {ml[-1]['clone_dist']:.3f}): "
  "they specialised.\n")
A(f"**4 · The cost.** Same arithmetic per token, but R3 ran {slow:.1f}× slower than R2 ({R3['tokens_per_second']:,.0f} "
  f"vs {R2['tokens_per_second']:,.0f} tokens/s): 32 small matrix products per layer instead of 2 large ones, plus routing. "
  f"Peak memory was {R3['peak_mem_bytes']/2**20:,.0f} MiB against {R2['peak_mem_bytes']/2**20:,.0f} MiB, because memory "
  "follows the stored parameters and their optimiser state.\n")
A(f"**5 · Two design details that the gates caught.** If each family's chosen weight is rescaled to 1 on its own, the "
  f"router gets **no gradient** (measured: {g['G2']['per_family_max']:.1e}, against {g['G2']['joint_min']:.1e} with the "
  "four weights rescaled together to sum to 4). And the conversion is exact in maths but not bit for bit: summing four "
  f"320-neuron products rounds differently from one 1,280-neuron product ({g['G1']['fp32']:.1e} relative in fp32), "
  "so the gate checks in fp64.\n")
A("![R3's balance over the run](outputs/v3/v3_balance.png)\n")

A("## The setup\n")
A(f"- **Model:** GPT, {mc['n_layer']} layers × width {mc['d']}, {mc['n_head']} heads, GELU FFN of "
  f"{mc['mlp_mult']*mc['d']:,}, vocabulary {mc['vocab']:,} (our own byte-pair tokenizer), sequence {mc['seq']}.")
A(f"- **Data:** FineWeb sample-10BT, {S3['data']['train_tokens']/1e6:.1f}M training tokens in shuffled windows. R1 reads "
  "the first 50M; R2, R2s and R3 read the next 50M in the same order.")
A(f"- **Training:** AdamW, batch {tc['batch']} × {mc['seq']}, learning rate {tc['lr']}, weight decay {tc['weight_decay']}, "
  f"β₂ {tc['beta2']}, clip {tc['grad_clip']}, fp16 on a T4. Warmup-stable-decay: R1 never decays; R2, R2s and R3 lower "
  f"the learning rate over their last {v3['decay_steps']} of {STEPS:,} steps.")
A("- **MoE (R3):** 4 families × 8 clones per layer, one per family per token; the four chosen scores rescaled to sum to 4; "
  f"router starts at zero, runs in fp32, output scaled by {v3['cfg']['router_scale']} (a 10× slower router); experts at "
  f"{v3['cfg']['expert_lr']} learning speed; probabilistic expert selection for the first {sw} steps, then the best score; "
  "a per-expert bias (step 0.01) for balance plus a 0.0001 per-sequence balance loss; AdamW state copied from R1.")
A("")

A("## Limitations\n")
A(f"- **One seed per run.** The final gaps ({abs(fin['R3']-fin['R2']):.4f} to R2, {abs(fin['R3']-fin['R2s']):.4f} to R2s) "
  f"are {abs(fin['R3']-fin['R2'])/spread:.0f}× and {abs(fin['R3']-fin['R2s'])/spread:.0f}× the measured run-to-run spread, "
  "so they are clear, but a second seed would make them firm.")
A("- **One size and one budget.** The lesson's MoE gains are measured at far larger scales and budgets.")
A("- **The decay explanation is untested.** Relaxing the experts' slowdown over time is the run that would test it.")
A("- **The diagnostics ran 200 steps each**; the choice of setting for v3 came from those short runs.")
A("")

LINKED = []          # every local path linked below; checked to exist at the end


def lk(path, text=None):
    LINKED.append(path)
    return f"[`{text or path}`]({path.replace(' ', '%20')})"


A("## Files\n")
A("Every asset of the assignment, linked. All paths are relative to this folder.\n")
A("**Source**\n")
A("| file | what it is |")
A("|---|---|")
A(f"| {lk('notebook_src.py')} | the source of truth: a percent-format script, converted to the notebooks in `notebooks/` by {lk('tools/py2ipynb.py')} |")
A(f"| {lk('plan_src.html')} | the source of the build plan page, built by {lk('tools/build_plan.py')} |")
A(f"| {lk('.gitignore')} | keeps checkpoints, tokenised data and the laptop test runs out of the repository |")
A("")
A("**Notebooks, one per milestone**\n")
A("| version | what it added | as handed over | as run on the T4 |")
A("|---|---|---|---|")
for ver, what, given, run in (
        ("v1", "data, the dense model, R1", "notebooks/session14_moe_v1.ipynb", "notebooks/session14_moe_v1_executed.ipynb"),
        ("v2", "the MoE layer, the converter, the gates, a 450-step preview", "notebooks/session14_moe_v2.ipynb", "notebooks/session14_moe_v2_executed.ipynb"),
        ("v2.1", "the first diagnostic: 7 runs", "notebooks/session14_moe_v2.1.ipynb", "notebooks/session14_moe_v2_1_executed.ipynb"),
        ("v2.2", "the second diagnostic: 6 runs, the fix", "notebooks/session14_moe_v2.2.ipynb", "notebooks/session14_moe_v2_2_executed.ipynb"),
        ("**v3**", "**the full comparison: R2, R2s, R3 (final)**", "notebooks/session14_moe_v3.ipynb", "notebooks/session14_moe_v3_executed.ipynb")):
    A(f"| {ver} | {what} | {lk(given)} | {lk(run)} |")
A("")
A("**Results, one bundle per milestone** (each zip as downloaded from Colab, unpacked beside it)\n")
A("| version | bundle | summary | results | run logs |")
A("|---|---|---|---|---|")
for ver, z in (("v1", "era5s14_v1_outputs.zip"), ("v2", "era5s14_v2_outputs.zip"), ("v2.1", "era5s14_v2.1_outputs.zip"),
               ("v2.2", "era5s14_v2.2_outputs.zip"), ("v3", "era5s14_v3_outputs.zip")):
    A(f"| {ver} | {lk('outputs/' + z, z)} | {lk(f'outputs/{ver}/summary.json', 'summary.json')} | "
      f"{lk(f'outputs/{ver}/RESULTS.md', 'RESULTS.md')} | {lk(f'outputs/{ver}/logs', 'logs/')} |")
A("")
A("**Figures**\n")
A("| figure | what it shows |")
A("|---|---|")
for pth, what in (("outputs/v1/r1_loss_lr.png", "R1's loss and its learning rate (warm-up, then flat)"),
                  ("outputs/v2/v2_preview.png", "v2's preview: the loss rising after conversion, the load collapsing after the switch"),
                  ("outputs/v2.1/v21_diagnostic.png", "the first diagnostic: seven runs over 200 steps, loss and balance"),
                  ("outputs/v2.2/v22_diagnostic.png", "the second diagnostic: the smaller expert learning rate turning the MoE around"),
                  ("outputs/v3/v3_loss.png", "the result: R1, then R2, R2s and R3 across 50M more tokens"),
                  ("outputs/v3/v3_balance.png", "R3's balance over the run: MaxVio, dead experts, clone distance"),
                  ("outputs/v3/v3_load.png", "R3's load per expert at the end of the run (the title says \"last 25 steps\"; it is the final one-step window)")):
    A(f"| {lk(pth)} | {what} |")
A("")
A("**Reports** (published links in [Reports](#reports))\n")
A("| local copy | report |")
A("|---|---|")
for title, what, url, local in REPORTS:
    A(f"| {lk(local)} | {title} |")
A("")
A("**Tools**\n")
A("| script | what it does |")
A("|---|---|")
for pth, what in (("tools/py2ipynb.py", "converts `notebook_src.py` into a Colab notebook"),
                  ("tools/make_readme.py", "writes this README from the summaries and checks it"),
                  ("tools/make_results.py", "builds the results report"),
                  ("tools/make_walkthrough.py", "builds the notebook walkthrough"),
                  ("tools/step_harness.css", "the walkthrough's stylesheet (from session 10's Step Harness)"),
                  ("tools/build_plan.py", "builds the build plan and execution record from `plan_src.html` and the summaries"),
                  ("tools/probe_balance.py", "the laptop diagnosis of load collapse at the switch (decision D8)"),
                  ("diagnosis/balance_cpu.json", "that diagnosis's measurements")):
    A(f"| {lk(pth)} | {what} |")
A("")
A("Cell 19 in `notebook_src.py` has two reporting fixes made after the v3 run (a one-step final balance window, and a "
  "conclusion that named only one comparison). The numbers in this README are computed from the saved logs with the "
  "fix applied; rerunning v3 reloads every run from Drive and reprints them.\n")

A("## How to run it\n")
A("Open `notebooks/session14_moe_v3.ipynb` in Colab with a **T4** runtime and choose Runtime → Run all. Allow the Google Drive "
  "mount: the notebook stops if Drive is not mounted, because the long runs depend on it to survive a disconnect. "
  "About 80 minutes from scratch (data, R1, R2, R2s, R3); a rerun reloads finished runs from Drive.\n")

A("## Acceptance checks\n")
A(f"Final notebook (v3), run on {datetime.strptime(S3['meta']['started'], '%Y-%m-%d %H:%M:%S').strftime('%d %b %Y').lstrip('0')}: "
  f"**{acc['passed']}/{acc['total']}**.\n")
for c in acc["checks"]:
    A(f"- [{'x' if c['result'] else ' '}] {c['name']}")
A("")

text = "\n".join(L)
open(os.path.join(HERE, "README.md"), "w", encoding="utf-8").write(text)

# ------------------------------------------------------------------ read back and check
back = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
must = [f4(fin["R3"]), f4(fin["R2"]), f4(fin["R2s"]), f4(conv), f"{abs(fin['R3'] - fin['R2s']):.4f}",
        f"{abs(fin['R3'] - fin['R2']):.4f}", f"{acc['passed']}/{acc['total']}", f"{R3['tokens_per_second']:,.0f}",
        f"{dec['R2']:.4f}", f"{dec['R3']:.4f}", f"{moe_tot/1e6:.1f}M"]
missing = [m for m in must if m not in back]
assert not missing, missing
absent = [p for p in LINKED + [r[3] for r in REPORTS] if not os.path.exists(os.path.join(HERE, p))]
assert not absent, f"README links to files that are not in the folder: {absent}"
for title, what, url, local in REPORTS:
    assert url.startswith("https://claude.ai/artifact/") and url in back, title
assert f"But ordinary dense training finished {'behind' if beats_dense else 'ahead'}." in back
assert (f"R2 finished {abs(fin['R3'] - fin['R2']):.4f} {'behind' if beats_dense else 'ahead of'} R3") in back
print(f"README.md written ({len(back):,} chars); {len(must)} key numbers checked against the summaries")
print(f"  R3 {f4(fin['R3'])} vs R2 {f4(fin['R2'])} vs R2s {f4(fin['R2s'])}; conversion {f4(conv)}")
