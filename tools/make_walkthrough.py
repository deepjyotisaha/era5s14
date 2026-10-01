"""Build the Session 14 notebook walkthrough (session14_walkthrough.html), in the style of Session 10's
"The Step Harness": every cell of the final notebook with what it does, what goes in and out, its full
source (collapsed) and what it printed on the T4, grouped by stage, and a closing ledger of the things
this assignment got wrong.

Source and output are extracted from the EXECUTED notebook, not transcribed. Numbers in the prose come
from outputs/v*/summary.json. The stylesheet is tools/step_harness.css (Session 10's, unchanged).

Usage (from session 14/assignment/):
    python tools/make_walkthrough.py
"""
import html
import io
import json
import keyword
import os
import re
import tokenize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "session14_walkthrough.html")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def load(v):
    return json.load(open(os.path.join(HERE, "outputs", v, "summary.json"), encoding="utf-8"))


S1, S2, S21, S22, S3 = load("v1"), load("v2"), load("v2.1"), load("v2.2"), load("v3")
NB = json.load(open(os.path.join(HERE, "notebooks", "session14_moe_v3_executed.ipynb"), encoding="utf-8"))
CSS = open(os.path.join(HERE, "tools", "step_harness.css"), encoding="utf-8").read()

# ------------------------------------------------------------------ numbers used in the prose
R1, R2, R2s, R3 = (S3["runs"][k] for k in ("R1", "R2", "R2s", "R3"))
g = S3["gates"]
conv = R3["val"][0]["loss"]
fin = {k: S3["runs"][k]["final_val"] for k in ("R2", "R2s", "R3")}
ml = R3["extra"]["moe_log"]
full = [m for i, m in enumerate(ml) if m["step"] - (ml[i - 1]["step"] if i else 0) >= 25]
P2 = S2["runs"]["P"]
D1, D2 = S21["diag"]["runs"], S22["diag2"]["runs"]
gap = lambda D, k: D[k]["hist"]["final_val"] - D["A"]["hist"]["final_val"]
mv_after = lambda h, sw: max(m["maxvio"] for m in h["extra"]["moe_log"] if m["step"] > sw)
at398 = min(R3["val"], key=lambda q: abs(q["tokens"] - 39.8e6))
r2_398 = min(R2["val"], key=lambda q: abs(q["tokens"] - 39.8e6))
v = dict(
    r1_0=f"{R1['val'][0]['loss']:.4f}", r1=f"{R1['final_val']:.4f}", r1_tps=f"{R1['tokens_per_second']:,.0f}",
    tot=f"{S3['moe']['total_params']:,}", act=f"{S3['moe']['active_params']:,}",
    g1_64=f"{g['G1']['fp64']:.1e}", g1_32=f"{g['G1']['fp32']:.1e}", g2=f"{g['G2']['joint_min']:.1e}",
    conv=f"{conv:.4f}", R3=f"{fin['R3']:.4f}", R2=f"{fin['R2']:.4f}", R2s=f"{fin['R2s']:.4f}",
    d_R3=f"{fin['R3'] - conv:+.4f}", vs_R2=f"{fin['R3'] - fin['R2']:+.4f}", vs_R2s=f"{fin['R3'] - fin['R2s']:+.4f}",
    p_rise=f"{P2['final_val'] - P2['val'][0]['loss']:+.4f}", p_mv=f"{P2['extra']['moe_log'][-1]['maxvio']:.2f}",
    d1_D=f"{mv_after(D1['D']['hist'], 100):.2f}", d1_G=f"{mv_after(D1['G']['hist'], 100):.2f}",
    d1_F=f"{gap(D1, 'F'):+.4f}", d2_D=f"{gap(D2, 'D'):+.4f}", d2_D12=f"{gap(D2, 'D12'):+.4f}",
    d2_aw=f"{D2['Aw']['hist']['val'][1]['loss'] - D2['A']['hist']['val'][1]['loss']:+.4f}",
    mv_end=f"{full[-1]['maxvio']:.3f}", mv_max=f"{max(m['maxvio'] for m in full if m['step'] > 100):.3f}",
    cd=f"{ml[-1]['clone_dist']:.3f}", tps3=f"{R3['tokens_per_second']:,.0f}", tps2=f"{R2['tokens_per_second']:,.0f}",
    slow=f"{R2['tokens_per_second'] / R3['tokens_per_second']:.1f}",
    lead398=f"{at398['loss'] - r2_398['loss']:+.4f}", passed=f"{S3['acceptance']['passed']}/{S3['acceptance']['total']}",
)

# ------------------------------------------------------------------ the prose, cell by cell
# Each entry: title, paragraphs (HTML), inputs, outputs, note under the printout, lines of output to show.
C = {
1: ("Environment, and which run this is",
    ["Imports, a fixed seed (1337) and a decision about the <b>profile</b>: <code>full</code> on a GPU, <code>smoke</code> "
     "for a quick GPU check, or <code>cpu</code>, a tiny version used to test the notebook on a laptop. A CPU run prints a "
     "loud banner so its numbers can never be mistaken for the experiment.",
     "The T4 has no fast bf16, so the heavy work runs in <b>fp16 with a loss scaler</b>. Everything that must stay exact "
     "(the router, the conversion gates) opts out of fp16 explicitly further down."],
    "none", "<code>DEVICE</code>, <code>PROFILE</code>, <code>AMP</code>, the results dictionary <code>R</code>",
    "Tesla T4, torch 2.11, profile <code>full</code>: this is the real run.", 20),
2: ("One place for every number that shapes the run",
    ["Session 13's model and optimiser settings, reused because you approved them: 16 layers of width 320, a GELU "
     "feed-forward network (FFN) of 1,280 neurons, vocabulary 8,192, batch 32 &times; 512, learning rate 1e-3.",
     "Two things are new. The schedule is <b>warmup-stable-decay</b>: 200 steps up, then flat, and lowered only in the "
     "last 20% of the continuation runs. R1 never decays, so the conversion happens while the model can still learn fast. "
     "And the data is twice as long, 100.5M tokens, so the continuations never repeat a sentence R1 saw.",
     "The breakdown it prints matters for everything after it: <b>58% of the parameters are in the FFN layers</b>, and "
     "that is exactly the part the MoE multiplies."],
    "the profile", "<code>MC</code>, <code>TC</code>, <code>DC</code>, <code>STEPS</code> = 3,051, the parameter breakdown",
    "Every run afterwards uses these numbers; none is set inside a later cell.", 20),
3: ("Storage that survives a disconnect, and refuses to run without it",
    ["Everything that must outlive the runtime goes to Google Drive: the tokenised data, every finished run, the "
     "checkpoints and <code>outputs/</code>. A finished run is recognised by a hash of its settings <b>and of the source "
     "code that produced it</b>, but not of the notebook version, so moving from v1 to v3 replays R1 instead of retraining it.",
     "<b>The cell's hardest lesson.</b> In v2.1 and v2.2 the Drive mount failed with a <code>ValueError</code>, and the "
     "notebook fell back to the runtime's temporary disk. It printed a warning and carried on, so it quietly rebuilt the "
     "data and retrained R1 twice. v3 prints the full error, retries once with a forced remount, and <b>stops</b> "
     "(<code>REQUIRE_DRIVE = True</code>): losing an 80-minute run to a disconnect would be worse."],
    "the Colab runtime", "<code>ROOT</code>, <code>OUT_DIR</code>, <code>CACHE_DIR</code>, <code>BASE_KEY</code>, <code>_src_hash()</code>",
    "\"drive mounted: True\": the mount worked on this run, so everything below reloads from Drive.", 20),
4: ("Web pages to two files of token ids",
    ["Streams FineWeb (sample-10BT), keeps the first documents aside as validation text, trains an 8,192-token byte-pair "
     "tokenizer on the next 40M characters, and tokenises 100.5M training tokens and 1M validation tokens into flat files.",
     "The training file is cut into windows of 513 tokens and shuffled once with a fixed seed. <b>R1 reads the first "
     "97,632 windows; R2, R2s and R3 all read the next 97,632, in the same order.</b> So the three continuations see "
     "exactly the same new text, and none of it was in R1's training data."],
    "FineWeb, streamed", "<code>train.bin</code>, <code>val.bin</code>, <code>TOK</code>, <code>TRAIN_W</code>, <code>VAL_W</code>, <code>CONT_START</code>",
    "\"[cached]\": prepared once in v1 (4.6 minutes) and reloaded from Drive ever since.", 20),
5: ("The dense model, frozen after v1",
    ["A small GPT: token and position embeddings, 16 pre-norm blocks of attention then FFN, a final norm, and an output "
     "layer that shares the embedding matrix. The loss is computed in slices of 8,192 tokens and recomputed in the backward "
     "pass, so the full score table over the vocabulary never sits in memory at once.",
     "The FFN is its own module, <code>block.mlp</code>, with two plain matrices (<code>up</code>, <code>down</code>). "
     "That one design choice is what makes the whole assignment possible without touching this cell again: the "
     "converter later <b>replaces</b> <code>block.mlp</code> with an MoE layer, and the rest of the model never knows."],
    "<code>MC</code>", "<code>GPT</code>, <code>Block</code>, <code>MLP</code>, <code>chunked_ce</code>, <code>make_dense()</code>",
    "22,467,200 parameters, matching the breakdown in Cell 2 to the last one.", 20),
6: ("The training loop, frozen after v1",
    ["One function, <code>train_run</code>, trains any model: AdamW (weight decay on matrices only), gradient clipping, "
     "fp16 with a loss scaler, validation every 2.5M tokens, a checkpoint every 8 minutes, and a <b>final checkpoint</b> "
     "(weights, optimiser moments, scaler) that later runs start from.",
     "It was written in v1 with arguments v1 did not need: <code>init</code> (start from saved weights), "
     "<code>window_start</code> (read later data), <code>warmup</code>/<code>decay</code>, and two hooks, "
     "<code>extra_loss</code> and <code>after_step</code>, which the MoE later uses for its balance loss and bias updates. "
     "Because the cache hashes this cell's source, editing it would have retrained R1; designing the hooks in advance meant "
     "it never had to change."],
    "a model builder, a step count, the schedule, optional hooks", "a history dictionary; a final checkpoint on Drive",
    "", 5),
7: ("R1: the dense model, 50M tokens",
    ["Trains the dense model from random weights: 200 warm-up steps, then a flat learning rate to the end, with no decay "
     "on purpose. On this run it was <b>reloaded from Drive</b>, not retrained, and the log below is the replay of the v1 run."],
    "<code>make_dense</code>, the first 50M tokens", "<code>R1</code>: validation loss "
    + v["r1_0"] + " &rarr; " + v["r1"] + "; its final checkpoint",
    f"Validation {v['r1_0']} &rarr; {v['r1']} at {v['r1_tps']} tokens/s. Predicted beforehand: 4.2&ndash;4.4. The prediction "
    "missed; the plan's Section 6.5 says why that comparison with session 13 cannot isolate the decay.", 40),
8: ("Gate: R1's checkpoint reloads to the same model",
    ["Everything after this point starts from R1's final checkpoint, so the gate loads it into a fresh model and checks "
     "that the validation loss comes out exactly as it did at the end of training."],
    "R1's final checkpoint", "a PASS, or the notebook stops", "Difference 0.0: the checkpoint is exactly R1.", 10),
9: ("What R1 learned: its curve, its schedule, three samples",
    ["Draws R1's loss and learning rate, and lets the model continue three prompts at temperature 0.8, so a number like "
     "4.12 has a picture: English-shaped sentences that hold for a clause and then wander."],
    "<code>R1</code>, its checkpoint", "<code>r1_loss_lr.png</code>, three samples", "", 12),
10: ("v1's checks, re-run in every version",
     ["The five promises v1 made (data long enough, R1 finished, R1's loss fell, R1 never decayed, the checkpoint "
      "reloads). They are checked again here because everything below depends on them."],
     "R1", "<code>CHECKS_V1</code>", "", 10),
11: ("The MoE layer",
     ["<code>FamilyMoE</code> replaces one FFN. It stores <b>32 experts in 4 families of 8</b>; each expert is a small "
      "FFN of 320 neurons. For every token the <b>router</b> (a 320 &times; 32 matrix, run in fp32, starting at zero) scores "
      "all 32; in each family the expert with the highest <code>logit + bias</code> is chosen (or, early in training, one "
      "is drawn at random in proportion to those values); and the four chosen scores are rescaled to <b>sum to 4</b>.",
      "Rescaling the four together, not each family on its own, is the design detail that was caught on paper before any "
      "code ran: with one pick per family, rescaling each pick by itself always gives exactly 1, and the router would "
      "never receive a gradient. The <code>joint=False</code> switch keeps the flawed rule so Cell 13 can prove it.",
      "Balancing follows the lesson's Section 18: a per-expert <b>bias</b> used only for choosing, nudged after every step "
      "towards each expert's fair share (step 0.01, ten times the lesson's value, decision D8), plus a tiny per-sequence "
      "balance loss."],
     "a token vector per position", "the layer's output (same shape), token counts per expert, the balance loss",
     f"{v['tot']} parameters stored, {v['act']} used per token: the dense model's work plus 163,840 of routers.", 10),
12: ("The converter: R1 becomes the MoE",
     ["Cuts each of R1's 16 FFNs into 4 slices of 320 neurons and copies slice <i>f</i> into the 8 experts of family "
      "<i>f</i>. The router, biases and counts start at zero.",
      "It also converts <b>AdamW's memory</b> (decision D6): each expert gets the running averages of the slice it was "
      "copied from, and R1's step count. Without that, Adam's first steps on empty averages would move every expert by "
      "about the full learning rate at once, a jolt the dense control would not have. The check at the end compares the "
      "copied weights and averages with R1's, slice by slice."],
     "R1's final checkpoint", "<code>CONV</code>: the MoE's starting weights, optimiser state and loss scaler",
     "PASS: every clone holds its slice of R1, weights and Adam averages alike.", 10),
13: ("Gates: same function, a router that learns, balancing that works",
     ["<b>G1</b> runs R1 and the converted MoE on two validation sequences on the CPU. In fp64 the logits must agree to "
      "1e-10; the fp32 difference is only reported. Four 320-neuron products add the same numbers as one 1,280-neuron "
      "product in a different order, and computers round after every addition, so fp32 cannot be bit-exact. The plan "
      "first promised bit-exact, and a laptop test corrected it.",
      "<b>G2</b> runs one backward pass: every layer's router must get a gradient with the joint rule, and the flawed "
      "per-family rule must get none. <b>G3</b> gives a toy layer a router that favours one expert per family and checks "
      "that the bias alone evens it out. <b>G4</b> checks the parameter counts against the formula."],
     "<code>CONV</code>, R1", "four gates; the notebook stops if any fails",
     f"fp64 {v['g1_64']} (the maths matches), fp32 {v['g1_32']} (rounding), and the per-family rule's router gradient is "
     "exactly 0.0.", 20),
14: ("The v2 preview (switched off in v3)",
     ["The first MoE run: the converted model trained for 450 steps with the router at full speed and the experts at the "
      f"normal learning rate. It is where the assignment went wrong. The loss <b>rose</b> ({v['p_rise']}) instead of falling, "
      f"and after the switch to hard selection the load collapsed (MaxVio {v['p_mv']}).",
      "The cell also defines the helpers every later MoE run uses: <code>moe_after_step</code> (bias updates, the switch "
      "from random to best-score selection, a load log every 25 steps), the clone distance, and the balance-loss hook. "
      "In v3 the run itself is off (<code>RUN_PREVIEW = False</code>); its results are in the plan's Section 8."],
     "<code>CONV</code>", "the helpers; no run in v3", "", 5),
15: ("The preview's figure (skipped with it)",
     ["Draws the preview's loss across the conversion, its balance over time and a load heatmap, only when the preview ran."],
     "the preview", "<code>v2_preview.png</code> (from v2)", "", 5),
16: ("The first diagnostic, v2.1 (switched off in v3)",
     ["Seven 200-step runs from the conversion point, each changing one thing. <code>DiagMoE</code> can pick experts the "
      "normal way, always use the first clone (the dense model in MoE form), or use a fixed table from token id to clone "
      "(built from token frequencies so each clone gets an eighth of the traffic).",
      f"Its two answers: a 10&times; slower router stops the collapse (MaxVio after the switch {v['d1_D']} against "
      f"{v['d1_G']}), and the loss rises whenever tokens are <b>split</b> across clones, even with the fixed table "
      f"({v['d1_F']} against dense). The plan's Section 10 has the full table."],
     "<code>CONV</code>", "<code>DiagMoE</code>, <code>make_diag</code>, <code>diag_run</code>; no runs in v3", "", 5),
17: ("The second diagnostic, v2.2: the fix (switched off in v3)",
     ["Tested the leading explanation: each clone learns from about an eighth of the tokens, so its gradient is noisier, "
      "but Adam still takes full-size steps. <code>ScaledDiagMoE</code> stores each expert as <i>w</i>&nbsp;/&nbsp;<i>c</i> "
      "and uses <i>c</i>&nbsp;&times;&nbsp;(<i>w</i>&nbsp;/&nbsp;<i>c</i>): the same function, but Adam's fixed-size steps "
      "now move the real weights <i>c</i> times less. The copied Adam averages are rescaled to match (momentum &times;&nbsp;<i>c</i>, "
      "squared average &times;&nbsp;<i>c</i>&sup2;).",
      f"With the experts at 1/8 speed the MoE went from {v['d2_D']} behind dense to <b>{v['d2_D12']}</b> ahead after 200 "
      f"steps. The same run also settled what v2.1's warm-up run had suggested: a dense model with the same warm-up dipped "
      f"too ({v['d2_aw']}), so that early dip was the learning rate, not the MoE."],
     "<code>CONV</code>", "<code>ScaledDiagMoE</code>, <code>scaled_init</code>; an exactness gate; no runs in v3",
     "The w / c trick changes the logits only by rounding.", 5),
18: ("The full comparison: R2, R2s and R3, 50M tokens each",
     ["Three continuations of R1 on the same 50M new tokens, each with its learning rate lowered over the last 610 of "
      "3,051 steps. <b>R2</b> is the dense model continuing. <b>R3</b> is the MoE with both fixes: router output &times;&nbsp;0.1, "
      "experts at 1/8 speed, and the switch to best-score selection at step 100.",
      "<b>R2s</b> is the fair control. Slowing the experts is part of the MoE's recipe, so R2s gives the dense model's FFN "
      "the same 1/8 speed with the same w&nbsp;/&nbsp;c trick (<code>ScaledMLP</code>). R3 against R2s is the test of "
      "whether the MoE's extra parameters helped; R3 against R2 is the practical question."],
     "R1's checkpoint, the continuation data", "<code>R2</code>, <code>R2s</code>, <code>R3</code> histories and checkpoints",
     "R2 and R2s reloaded from Drive; R3 resumed from its step-2,368 checkpoint after a Colab disconnect and finished there.", 70),
19: ("The answer",
     [f"Checks <b>G5</b>, the assignment's requirement: the MoE's loss fell from {v['conv']} to {v['R3']} ({v['d_R3']}). "
      f"Then it compares: R3 finished {v['vs_R2s']} against R2s, so the extra parameters helped, and {v['vs_R2']} against "
      f"R2, so the ordinary dense model still won. R3 was {v['lead398']} against R2 at 39.8M tokens, just before the decay, "
      "and the decay turned that lead around.",
      "<b>Two lines in this printout were corrected afterwards.</b> The conclusion named only the favourable comparison. "
      "And \"MaxVio 0.214 at the end\" came from a final log window only one step long; over the last full 25-step window "
      f"it is {v['mv_end']}. Both are fixed in <code>notebook_src.py</code>; the printout below is what ran."],
     "<code>R2</code>, <code>R2s</code>, <code>R3</code>", "gate G5, the verdicts, three figures, samples",
     f"R3 used all 512 experts for the whole run, and its clones drifted steadily apart (distance {v['cd']}).", 40),
20: ("Checks, summary, bundle",
     ["Collects every check that ran in this version, writes <code>summary.json</code> and <code>RESULTS.md</code>, and zips "
      "the outputs. The README and the reports are generated from those files."],
     "everything above", "<code>summary.json</code>, <code>RESULTS.md</code>, <code>era5s14_v3_outputs.zip</code>",
     f"{v['passed']} passed.", 30),
}
GROUPS = [
    ("Cells 1 &ndash; 3", "Ground the run stands on", (1, 2, 3)),
    ("Cells 4 &ndash; 6", "The data, the dense model, the loop", (4, 5, 6)),
    ("Task 1", "Train a dense model", (7, 8, 9, 10)),
    ("Task 2 &middot; gates", "Convert it into an MoE", (11, 12, 13)),
    ("Cells 14 &ndash; 17", "The two diagnostics that found the fix", (14, 15, 16, 17)),
    ("Task 3", "Show it keeps training and reduces loss", (18, 19, 20)),
]

LEDGER = [
    ("The conversion is bit-exact in fp32",
     f"Four small products round differently from one big one ({v['g1_32']} in fp32). Exact in maths; the gate moved to fp64 ({v['g1_64']})."),
    ("Rescale each family's chosen weight to 1",
     "The router would get no gradient at all. Caught on paper; Cell 13 measures it at exactly 0.0."),
    ("R1 will end at 4.2&ndash;4.4 without decay",
     f"It ended at {v['r1']}, within 0.009 of session 13's decayed run. That comparison could not isolate the decay anyway."),
    ("A bias step of 0.01 fixes the collapse",
     f"It did on the laptop's tiny model. On the T4 the load still collapsed (MaxVio {v['p_mv']}). The fix was a 10&times; slower router."),
    ("The loss rises because clones are picked at random, or because the router moves",
     f"A fixed token-to-clone table, with no randomness and no router choice, rose the most ({v['d1_F']}). The cause was the experts' learning rate."),
    ("The warm-up's early dip is the MoE",
     f"A dense model with the same warm-up dipped too ({v['d2_aw']}): it was the learning rate."),
    ("A 300-step preview is enough",
     "The switch to best-score selection was at step 300, exactly where it ended. It was lengthened to 450."),
    ("Falling back to the runtime disk is harmless",
     "In two runs it printed a warning and carried on, rebuilding the data and retraining R1. The notebook now stops if Drive does not mount."),
    ("MaxVio 0.214 at the end of R3",
     f"A one-step log window. Over 25-step windows: {v['mv_end']}."),
    ("The MoE's extra parameters helped, so it won",
     f"It beat the fair control ({v['vs_R2s']}), but the ordinary dense model finished ahead ({v['vs_R2']})."),
]

# ------------------------------------------------------------------ helpers
def highlight(src):
    out, last = [], (1, 0)
    lines = src.splitlines(keepends=True)
    def text_between(a, b):
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:
            return lines[r1 - 1][c1:c2] if r1 - 1 < len(lines) else ""
        s = lines[r1 - 1][c1:]
        for r in range(r1, r2 - 1):
            s += lines[r]
        return s + (lines[r2 - 1][:c2] if r2 - 1 < len(lines) else "")
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            out.append(html.escape(text_between(last, tok.start)))
            t = tok.string
            e = html.escape(t)
            if tok.type == tokenize.COMMENT:
                out.append(f'<span class="c">{e}</span>')
            elif tok.type == tokenize.STRING:
                out.append(f'<span class="s">{e}</span>')
            elif tok.type == tokenize.NAME and (keyword.iskeyword(t) or t in ("True", "False", "None")):
                out.append(f'<span class="k">{e}</span>')
            else:
                out.append(e)
            last = tok.end
        return "".join(out)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return html.escape(src)

def cell_pairs():
    pairs, cur = {}, None
    for c in NB["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            m = re.search(r"## Cell (\d+)", src)
            cur = int(m.group(1)) if m else None
        elif cur is not None:
            outs = []
            for o in c.get("outputs", []):
                t = o.get("text") or o.get("data", {}).get("text/plain") or ""
                t = ANSI.sub("", "".join(t))
                if t and "Resolving data files" not in t and "README.md:" not in t and "it/s]" not in t \
                        and "unauthenticated requests" not in t:
                    outs.append(t)
            pairs[cur] = (src, "".join(outs).rstrip("\n"))
    return pairs

PAIRS = cell_pairs()
assert set(PAIRS) == set(C), (sorted(PAIRS), sorted(C))

def cell_html(n):
    title, paras, inp, outp, note, show = C[n]
    src, out = PAIRS[n]
    nlines = src.count("\n") + 1
    h = [f'<article class="cell"><h3>Cell {n} &middot; {title}</h3><div class="body">']
    h += [f"<p>{p}</p>" for p in paras]
    h.append(f'</div><p class="io"><b>Inputs:</b> {inp}. <b>Outputs:</b> {outp}.</p>')
    h.append(f'<details class="src"><summary>Show the full cell source &mdash; {nlines} lines</summary>'
             f'<pre class="code">{highlight(src)}</pre></details>')
    if out:
        ol = out.split("\n")
        shown = "\n".join(ol[:show])
        more = f"\n&hellip; {len(ol) - show} more lines in the executed notebook" if len(ol) > show else ""
        h.append(f'<h4>What it printed</h4><div class="outwrap"><pre class="out">{html.escape(shown)}{more}</pre></div>')
    if note:
        h.append(f'<p class="outnote">{note}</p>')
    h.append("</article>")
    return "".join(h)

facts = [("Device", "Tesla T4 &middot; fp16"), ("Dense model", "16L &times; 320 &middot; 22.5M"),
         ("MoE", "32 experts/layer &middot; 114.4M / 22.6M"), ("Tokens", "50M + 50M"), ("Checks", f"{v['passed']} pass")]
tasks = [("T1", "train dense", f"R1: validation {v['r1_0']} &rarr; {v['r1']} on 50M FineWeb tokens"),
         ("T2", "convert", f"32 experts per layer from R1's FFNs; same function (fp64 {v['g1_64']}); same work per token"),
         ("T3", "keep training", f"MoE {v['conv']} &rarr; <b>{v['R3']}</b> over 50M more tokens, no dead experts"),
         ("vs", "dense", f"fair control R2s {v['R2s']} ({v['vs_R2s']}); ordinary dense R2 {v['R2']} ({v['vs_R2']})")]

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Expert Split</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style></head><body>
<div class="wrap">
<header class="mast">
  <p class="eyebrow">ERA V5 &middot; Session 14 &middot; Mixture-of-Experts</p>
  <h1>Split a working model into experts, and keep it learning</h1>
  <p class="standfirst">A dense GPT, cut into 32 experts per layer without changing what it computes, then trained on.
  The conversion was the easy part. Keeping the experts learning took a router slowed tenfold and experts slowed eightfold,
  and two diagnostics to find out why.</p>
  <div class="facts">{"".join(f'<div class="fact"><span class="fk">{k}</span><span class="fv">{val}</span></div>' for k, val in facts)}</div>
</header>

<section class="grp">
  <h2><span class="gtag">The task</span><span class="gname">What we were asked to do</span></h2>
  <div class="body">
    <p>Train a dense ("linear") model, convert it into a mixture of experts, and show that the MoE continues to train and
    reduce its loss. Model size and data were our call.</p>
    <p>The design choice that shapes every cell below: the MoE does <b>the same work per token</b> as the dense model
    (4 experts of 320 neurons = one FFN of 1,280) while storing <b>eight times</b> the FFN parameters. So any gain it shows
    comes from the extra parameters, which is the lesson's central claim about MoE.</p>
  </div>
  <div class="tablewrap"><table><tbody>{"".join(f'<tr><td class="tk">{a}</td><td class="tn">{b}</td><td>{c}</td></tr>' for a, b, c in tasks)}</tbody></table></div>
  <div class="callout">
    <h4>How to read this</h4>
    <p>Each cell of the final notebook (v3) is below: what it does, what goes in and comes out, its full source in a
    collapsed block, and what it printed on the T4. Every code block is the code that ran and every output block is what it
    printed; nothing is retyped. Cells 14&ndash;17 are the experiments that found the fix; they are switched off in v3 and
    their results are summarised in the text.</p>
  </div>
</section>
{"".join(f'<section class="grp"><h2><span class="gtag">{tag}</span><span class="gname">{name}</span></h2>' + "".join(cell_html(n) for n in ns) + "</section>" for tag, name, ns in GROUPS)}
<section class="grp">
  <h2><span class="gtag">The ledger</span><span class="gname">Ten things this assignment got wrong</span></h2>
  <div class="body"><p>Kept on purpose. Each was caught by a gate, a diagnostic or a second look at a number, not by reading
  the code. Most produced no error message, only a plausible number.</p></div>
  <div class="tablewrap"><table><thead><tr><th>The claim</th><th>What actually happened</th></tr></thead><tbody>
  {"".join(f'<tr><td class="claim">{a}</td><td>{b}</td></tr>' for a, b in LEDGER)}
  </tbody></table></div>
  <div class="callout warn">
    <h4>The lead the decay took away</h4>
    <p>For most of the run the MoE was <b>ahead</b> of the ordinary dense model: {v['lead398']} at 39.8M tokens. Then the
    learning rate was lowered for the last 20%, and the dense model gained far more from it. It finished {v['vs_R2']} in
    the dense model's favour.</p>
    <p>The likely reason (not tested): the MoE's experts had been taking 1/8-size steps all along, so shrinking every step
    at the end gave them less to gain. Quoting the MoE's mid-run lead would have been true, flattering and misleading.</p>
  </div>
</section>
<footer class="foot">Generated from <code>notebooks/session14_moe_v3_executed.ipynb</code> and the runs' <code>outputs/v*/summary.json</code>:
source and output are extracted from the executed notebook, not transcribed. Run: {S3['meta']['gpu']}, torch {S3['meta']['torch']},
seed 1337, {v['passed']} acceptance checks passing. Companion reports: the results report, the build plan and execution
record, and the session report (links in the README).</footer>
</div></body></html>"""
open(OUT, "w", encoding="utf-8").write(page)
print(f"wrote {OUT} ({len(page)/1024:.0f} KB), {len(C)} cells, {len(LEDGER)} ledger rows")
