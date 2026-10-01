"""Builds assignment_plan.html from plan_src.html.

Every measured number on the page comes from the run's own files, never typed in:
  outputs/v1/summary.json             v1 on the T4
  notebooks/session14_moe_v1_executed.ipynb   lines the summary does not hold (data-preparation time)
  ../../session 13/assignment/outputs/summary.json   session 13's run A, for comparison
The rounding test in Section 7.1 is re-run here on every build.
Usage (from session 14/assignment/): python tools/build_plan.py"""
import html
import json
import math
import re
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

src = open("plan_src.html", encoding="utf-8").read()
rep = open("../assignment_research/session14_report.html", encoding="utf-8").read()
S = json.load(open("outputs/v1/summary.json", encoding="utf-8"))
S13 = json.load(open("../../session 13/assignment/outputs/summary.json", encoding="utf-8"))
NB = json.load(open("notebooks/session14_moe_v1_executed.ipynb", encoding="utf-8"))
F_ = {}

# ---------------------------------------------------------------- measured values
r1, a13 = S["runs"]["R1"], S13["runs"]["A"]
val = r1["val"]
nb_text = "".join("".join(o.get("text", "")) for c in NB["cells"] if c["cell_type"] == "code"
                  for o in c.get("outputs", []))
data_min = float(re.search(r"data prepared in ([\d.]+) min", nb_text).group(1))
base_key = re.search(r"base key\s+:\s+(\w+)", nb_text).group(1)
n_params = int(re.search(r"parameters: ([\d,]+)", nb_text).group(1).replace(",", ""))

def val_at(tokens):
    return min(val, key=lambda p: abs(p["tokens"] - tokens))

v0, vend = val[0]["loss"], r1["final_val"]
v10, v40 = val_at(10e6), val_at(40e6)
last10 = v40["loss"] - vend
rate = last10 / ((val[-1]["tokens"] - v40["tokens"]) / 1e6)
share10 = (v0 - v10["loss"]) / (v0 - vend) * 100
t0 = datetime.strptime(S["meta"]["started"], "%Y-%m-%d %H:%M:%S")
t1 = datetime.strptime(S["meta"]["finished"], "%Y-%m-%d %H:%M:%S")
pb = S["config"]["params"]
mc = S["config"]["model"]
steps = S["config"]["steps"]
L, d, H = mc["n_layer"], mc["d"], mc["d"] * mc["mlp_mult"]
moe_ffn = L * 32 * 2 * d * (H // 4)          # 32 experts of 320 neurons per layer
router = L * d * 32
moe_total = pb["total"] - pb["feed-forward"] + moe_ffn + router
moe_active = pb["total"] + router            # 4 experts of 320 = the dense FFN's work
acc = S["acceptance"]
assert acc["passed"] == acc["total"]
tps_diff = (r1["tokens_per_second"] / a13["tokens_per_second"] - 1) * 100

# rounding test for Section 7.1: four slices vs one FFN, our shapes
rt = {}
for name, dt in (("fp32", torch.float32), ("fp64", torch.float64)):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, d, dtype=dt, generator=g)
    up = torch.randn(H, d, dtype=dt, generator=g) * 0.05
    down = torch.randn(d, H, dtype=dt, generator=g) * 0.02
    dense = F.gelu(x @ up.t()) @ down.t()
    q = H // 4
    parts = sum(F.gelu(x @ up[i*q:(i+1)*q].t()) @ down[:, i*q:(i+1)*q].t() for i in range(4))
    rt[name] = ((dense - parts).abs().max() / dense.abs().max()).item()
assert rt["fp64"] < 1e-12 < rt["fp32"]

V = {
    "v1_pass": f"{acc['passed']}/{acc['total']}",
    "run_date": t0.strftime("%d %b %Y").lstrip("0"),
    "gpu": S["meta"]["gpu"], "torch": S["meta"]["torch"],
    "wall_min": f"{(t1 - t0).total_seconds()/60:.0f}",
    "r1_train_min": f"{r1['train_seconds']/60:.1f}",
    "r1_tps": f"{r1['tokens_per_second']:,.0f}", "s13_tps": f"{a13['tokens_per_second']:,.0f}",
    "tps_diff": f"{tps_diff:+.1f}%",
    "r1_peak": f"{r1['peak_mem_bytes']/2**20:,.0f}", "s13_peak": f"{a13['peak_mem_bytes']/2**20:,.0f}",
    "train_tokens": f"{S['data']['train_tokens']:,}", "val_tokens": f"{S['data']['val_tokens']:,}",
    "train_windows": f"{(S['data']['train_tokens']-1)//mc['seq']:,}",
    "r1_last_win": f"{steps*32-1:,}", "cont_first": f"{steps*32:,}", "cont_last": f"{2*steps*32-1:,}",
    "cpt": f"{S['data']['chars_per_token']:.2f}", "s13_cpt": f"{S13['data']['chars_per_token']:.2f}",
    "data_min": f"{data_min:.1f}",
    "r1_val0": f"{v0:.4f}", "r1_val10": f"{v10['loss']:.4f}", "share10": f"{share10:.0f}",
    "r1_final": f"{vend:.4f}", "s13_final": f"{a13['final_val']:.4f}",
    "gap_s13": f"{vend - a13['final_val']:.4f}",
    "last10_drop": f"{last10:.4f}", "rate_per_m": f"{rate:.4f}",
    "r1_final_train": f"{r1['final_train']:.4f}", "s13_final_train": f"{a13['final_train']:.4f}",
    "ppl": f"{math.exp(vend):.0f}",
    "ffn_share": f"{pb['feed-forward']/pb['total']*100:.1f}", "ffn_m": f"{pb['feed-forward']/1e6:.2f}",
    "total_m": f"{pb['total']/1e6:.2f}", "steps": f"{steps:,}", "base_key": base_key,
    "n_params": f"{n_params:,}", "reload_diff": f"{S['gates']['r1_reload']['diff']:.1e}",
    "moe_total": f"{moe_total/1e6:.1f}", "moe_active": f"{moe_active/1e6:.2f}",
    "rt_fp32": f"{rt['fp32']:.1e}", "rt_fp64": f"{rt['fp64']:.1e}",
}
assert n_params == pb["total"]

# ---------------------------------------------------------------- figures
F_["<!--CSS-->"] = re.search(r"<style>.*?</style>", rep, re.S).group(0)
a = rep.index('<h3 id="a5why">')
F_["<!--EFIG-->"] = re.search(r"<svg.*?</svg>", rep[a:], re.S).group(0)

# milestone strip
ms = [("v1", "the dense model", "done 29 Sep: " + V["v1_pass"] + " checks", "done"),
      ("v2", "the conversion", "exact; two diagnostics found the fixes", "done"),
      ("v3", "the comparison", "done: the MoE kept learning", "done")]
col = {"done": ("var(--ok-bg)", "var(--ok)"), "open": ("var(--open-bg)", "var(--open)"),
       "grey": ("var(--panel2)", "var(--line2)"), "flag": ("var(--flag-bg)", "var(--flag)")}
c = ['<defs><marker id="mm" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
     '<path d="M0 0 L10 5 L0 10 z" fill="var(--ink3)"/></marker></defs>']
for i, (k, t, s, kind) in enumerate(ms):
    x = 10 + 232 * i
    fill, stroke = col[kind]
    c.append(f'<rect x="{x}" y="12" width="210" height="72" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    c.append(f'<text x="{x+14}" y="36" font-size="12" font-weight="700" fill="{stroke if kind != "grey" else "var(--ink3)"}">{k}</text>')
    c.append(f'<text x="{x+14}" y="55" font-size="13" font-weight="700" fill="var(--ink)">{t}</text>')
    c.append(f'<text x="{x+14}" y="73" font-size="10.5" fill="var(--ink2)">{s}</text>')
    if i < 2:
        c.append(f'<path d="M{x+212} 48 L{x+229} 48" stroke="var(--ink3)" stroke-width="1.5" marker-end="url(#mm)"/>')
F_["<!--MILESTONES-->"] = "\n".join(c)

# stage strip
stages = [("S0", "Decisions", "done 28 Sep", "done"), ("S1", "Data + dense", "done 29 Sep (v1)", "done"),
          ("S2", "MoE layer", "v2: router, experts", "open"), ("S3", "GATES", "v2: G1 to G4", "flag"),
          ("S4", "Runs", "v3: R2 and R3", "grey"), ("S5", "Analysis", "v3: plots + README", "grey")]
c = ['<defs><marker id="pm" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
     '<path d="M0 0 L10 5 L0 10 z" fill="var(--ink3)"/></marker></defs>']
for i, (k, t, s, kind) in enumerate(stages):
    x = 6 + 116 * i
    fill, stroke = col[kind]
    lab = stroke if kind != "grey" else "var(--ink3)"
    c.append(f'<rect x="{x}" y="16" width="106" height="74" rx="7" fill="{fill}" stroke="{stroke}"/>')
    c.append(f'<text x="{x+10}" y="36" font-size="10.5" font-weight="700" fill="{lab}">{k}</text>')
    c.append(f'<text x="{x+10}" y="56" font-size="12" font-weight="700" fill="var(--ink)">{t}</text>')
    c.append(f'<text x="{x+10}" y="74" font-size="9" fill="var(--ink2)">{s}</text>')
    if i < 5:
        c.append(f'<path d="M{x+107} 53 L{x+114} 53" stroke="var(--ink3)" stroke-width="1.4" marker-end="url(#pm)"/>')
F_["<!--STAGES-->"] = "\n".join(c)

# weights example (Section 4)
s = [0.050, 0.020, 0.040, 0.030]
joint = [4 * v / sum(s) for v in s]
assert abs(sum(joint) - 4) < 1e-12
c = []
for x0, w, good in ((80, [1.0] * 4, False), (435, joint, True)):
    for j in range(4):
        x = x0 + 62 * j
        c.append(f'<rect x="{x}" y="36" width="54" height="26" rx="3" fill="var(--panel2)" stroke="var(--line2)"/>')
        c.append(f'<text x="{x+27}" y="54" text-anchor="middle" font-family="var(--mono)" fill="var(--ink)">{s[j]:.3f}</text>')
        fill, stroke = ("var(--ok-bg)", "var(--ok)") if good else ("var(--flag-bg)", "var(--flag)")
        c.append(f'<rect x="{x}" y="74" width="54" height="26" rx="3" fill="{fill}" stroke="{stroke}"/>')
        c.append(f'<text x="{x+27}" y="92" text-anchor="middle" font-family="var(--mono)" font-weight="700" fill="var(--ink)">{w[j]:.2f}</text>')
        c.append(f'<text x="{x+27}" y="116" text-anchor="middle" font-size="10" fill="var(--ink3)">fam {"abcd"[j]}</text>')
F_["<!--WEIGHTS-->"] = "\n".join(c)

# R1 learning curve, drawn from the logged points: the whole run, and a zoom on its second half
def panel(X0, X1, Y0, Y1, xlo, xhi, ylo, yhi, yticks, xticks, full):
    px = lambda t: X0 + (X1 - X0) * (t / 1e6 - xlo) / (xhi - xlo)
    py = lambda l: Y1 - (Y1 - Y0) * (l - ylo) / (yhi - ylo)
    inx = lambda t: xlo * 1e6 <= t <= xhi * 1e6 + 1
    o = [f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="var(--line2)"/>',
         f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="var(--line2)"/>']
    for yv in yticks:
        o.append(f'<line x1="{X0}" y1="{py(yv):.1f}" x2="{X1}" y2="{py(yv):.1f}" stroke="var(--line)" stroke-dasharray="2 3"/>')
        o.append(f'<text x="{X0-6}" y="{py(yv)+4:.1f}" text-anchor="end" font-size="10" fill="var(--ink3)">{yv:g}</text>')
    for xv in xticks:
        o.append(f'<text x="{px(xv*1e6):.1f}" y="{Y1+15}" text-anchor="middle" font-size="10" fill="var(--ink3)">{xv}M</text>')
    tr = " ".join(f"{px(q['tokens']):.1f},{py(min(max(q['loss'], ylo), yhi)):.1f}" for q in r1["train"] if inx(q["tokens"]))
    o.append(f'<polyline points="{tr}" fill="none" stroke="var(--grey)" stroke-width="1" opacity="0.8"/>')
    pts = [q for q in val if inx(q["tokens"])]
    o.append('<polyline points="' + " ".join(f"{px(q['tokens']):.1f},{py(q['loss']):.1f}" for q in pts)
             + '" fill="none" stroke="var(--c1)" stroke-width="2"/>')
    for q in pts:
        o.append(f'<circle cx="{px(q["tokens"]):.1f}" cy="{py(q["loss"]):.1f}" r="2.6" fill="var(--c1)"/>')
    ys = py(a13["final_val"])
    o.append(f'<line x1="{X0}" y1="{ys:.1f}" x2="{X1}" y2="{ys:.1f}" stroke="var(--c2)" stroke-dasharray="5 4"/>')
    return o, px, py

xmax = val[-1]["tokens"] / 1e6
c = []
o, px, py = panel(45, 325, 30, 250, 0, xmax, 3.8, 9.4, (4, 5, 6, 7, 8, 9), range(0, 51, 10), True)
c += o
c.append(f'<text x="185" y="16" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--ink)">the whole run</text>')
c.append(f'<text x="{px(0)+8:.1f}" y="{py(v0)+4:.1f}" font-size="10" fill="var(--c1)">{v0:.2f} before training</text>')
c.append(f'<rect x="{px(20e6):.1f}" y="{py(4.8):.1f}" width="{px(xmax*1e6)-px(20e6):.1f}" height="{py(4.05)-py(4.8):.1f}" fill="none" stroke="var(--c3)" stroke-dasharray="3 2"/>')
c.append(f'<text x="{px(20e6):.1f}" y="{py(4.8)-5:.1f}" font-size="9.5" fill="var(--c3)">zoomed on the right</text>')
o, px, py = panel(400, 680, 30, 250, 20, xmax, 4.05, 4.8, (4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7), range(20, 51, 10), False)
c += o
c.append(f'<text x="540" y="16" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--ink)">zoom: 20M to 50M tokens</text>')
c.append(f'<text x="676" y="{py(a13["final_val"])+14:.1f}" text-anchor="end" font-size="10" fill="var(--c2)">session 13 run A final {a13["final_val"]:.4f} (decayed)</text>')
c.append(f'<text x="{px(31e6):.1f}" y="{py(4.16):.1f}" font-size="10.5" font-weight="700" fill="var(--c1)">R1 final {vend:.4f} &#8594;</text>')
c.append(f'<text x="{px(val_at(40e6)["tokens"]):.1f}" y="{py(4.55):.1f}" text-anchor="middle" font-size="10" fill="var(--ink2)">still falling: {last10:.3f} in the last 10M</text>')
c.append('<text x="362" y="285" text-anchor="middle" font-size="11" fill="var(--ink2)">training tokens &#183; orange dots: validation loss &#183; thin grey: training loss (25-step means)</text>')
F_["<!--R1CURVE-->"] = "\n".join(c)

# samples, with the prompt in bold
prompts = ["The best way to learn a new language is", "Scientists have found that",
           "In the middle of the city there is"]
c = []
for p, smp in zip(prompts, S["samples_r1"]):
    assert smp.startswith(p)
    c.append(f'<div class="sample"><b>{html.escape(p)}</b>{html.escape(smp[len(p):]).replace(chr(10), " &#8629; ")}</div>')
F_["<!--SAMPLES-->"] = "\n".join(c)

# ---------------------------------------------------------------- v2 laptop test and the balance diagnosis
import zipfile as _zf
# the v2-era laptop summary, frozen inside the v2.2 laptop bundle (later laptop runs overwrite outputs/cpu/summary.json)
C2 = json.loads(_zf.ZipFile("outputs/cpu/era5s14_v2.2_outputs.zip").read("summary.json"))
assert C2["meta"]["notebook_version"] in ("v2", "v2.1", "v2.2", "v3")   # later versions replay v2's gates and preview unchanged
_v2checks = [c for c in C2["acceptance"]["checks"] if not c["name"].startswith("diagnostic")]
cg, cml = C2["gates"], C2["runs"]["P"]["extra"]["moe_log"][-1]
V.update({
    "cpu_pass": f"{sum(1 for c in _v2checks if c['result'])}/{len(_v2checks)}",
    "cpu_g1_64": f"{cg['G1']['fp64']:.1e}", "cpu_g1_32": f"{cg['G1']['fp32']:.1e}",
    "cpu_val_moe": f"{cg['G1']['val_moe']:.5f}", "cpu_val_r1": f"{cg['G1']['val_r1']:.5f}",
    "cpu_g2_joint": f"{cg['G2']['joint_min']:.1e}", "cpu_g2_flawed": f"{cg['G2']['per_family_max']:.1e}",
    "cpu_g3_start": f"{cg['G3']['start']:.2f}", "cpu_g3_end": f"{cg['G3']['end']:.2f}",
    "cpu_p_mv": f"{cml['maxvio']:.2f}", "cpu_p_dead": f"{cml['dead']}",
    "cpu_p_steps": f"{C2['runs']['P']['steps']}",
    "cpu_p_cd0": f"{C2['runs']['P']['extra']['moe_log'][0]['clone_dist']:.3f}", "cpu_p_cd1": f"{cml['clone_dist']:.3f}",
    "cpu_p_v0": f"{C2['runs']['P']['val'][0]['loss']:.4f}", "cpu_p_v1": f"{C2['runs']['P']['final_val']:.4f}",
})
DG = json.load(open("diagnosis/balance_cpu.json", encoding="utf-8"))
V["diag_steps"] = f"{DG['steps']}"
variants = {}
for r in DG["runs"]:
    variants.setdefault(r["label"], []).append(r)
names = {"gamma 1e-3, switch at 80": ("D. plan: bias step 0.001", "var(--flag)"),
         "gamma 1e-3, probabilistic always": ("C. sampling throughout", "var(--c3)"),
         "gamma 1e-2, switch at 80": ("A. bias step 0.01", "var(--ok)"),
         "gamma 1e-3, router lr x0.1, sw 80": ("B. router 10x slower", "var(--c2)")}
assert set(names) == set(variants)
steps_axis = [p[0] for p in variants[next(iter(variants))][0]["log"]]
X0, X1, Y0, Y1 = 60, 520, 20, 230
ytop = 1.5
px = lambda st: X0 + (X1 - X0) * st / DG["steps"]
py = lambda mv: Y1 - (Y1 - Y0) * min(mv, ytop) / ytop
c = [f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="var(--line2)"/>',
     f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="var(--line2)"/>']
for yv in (0, 0.5, 1.0, 1.5):
    c.append(f'<line x1="{X0}" y1="{py(yv):.1f}" x2="{X1}" y2="{py(yv):.1f}" stroke="var(--line)" stroke-dasharray="2 3"/>')
    c.append(f'<text x="{X0-6}" y="{py(yv)+4:.1f}" text-anchor="end" font-size="10" fill="var(--ink3)">{yv:g}</text>')
for st in steps_axis:
    c.append(f'<text x="{px(st):.1f}" y="{Y1+15}" text-anchor="middle" font-size="10" fill="var(--ink3)">{st}</text>')
c.append(f'<text x="{(X0+X1)/2}" y="{Y1+32}" text-anchor="middle" font-size="11" fill="var(--ink2)">training steps after conversion</text>')
c.append(f'<text x="16" y="{(Y0+Y1)/2}" font-size="11" fill="var(--ink2)" transform="rotate(-90 16 {(Y0+Y1)/2})" text-anchor="middle">MaxVio</text>')
c.append(f'<line x1="{px(80):.1f}" y1="{Y0}" x2="{px(80):.1f}" y2="{Y1}" stroke="var(--ink3)" stroke-dasharray="5 4"/>')
c.append(f'<text x="{px(80)+4:.1f}" y="{Y0+10}" font-size="10" fill="var(--ink3)">switch to hard selection</text>')
rows = []
for i, (lab, (nm, colr)) in enumerate(names.items()):
    runs = variants[lab]
    mv = [float(np.mean([r["log"][j][1] for r in runs])) for j in range(len(steps_axis))]
    pts = " ".join(f"{px(st):.1f},{py(m):.1f}" for st, m in zip(steps_axis, mv))
    c.append(f'<polyline points="{px(0):.1f},{py(0):.1f} {pts}" fill="none" stroke="{colr}" stroke-width="2.2"/>')
    for st, m in zip(steps_axis, mv):
        c.append(f'<circle cx="{px(st):.1f}" cy="{py(m):.1f}" r="2.6" fill="{colr}"/>')
    c.append(f'<rect x="538" y="{40 + 24*i}" width="14" height="4" fill="{colr}"/>')
    c.append(f'<text x="558" y="{45 + 24*i}" font-size="10.5" fill="var(--ink)">{nm}</text>')
    vals = sorted(runs, key=lambda r: r["seed"])
    rows.append((nm, [r["val"] for r in vals], max(r["log"][j][1] for r in vals for j in range(len(steps_axis)) if steps_axis[j] > 80),
                 float(np.mean([r["log"][-1][1] for r in vals])), sum(r["log"][j][2] for r in vals for j in range(len(steps_axis)))))
F_["<!--BALFIG-->"] = "\n".join(c)
t = ['<div class="tblwrap"><table>',
     '<tr><th>setting</th><th class="num">val loss (seeds 0 / 1)</th><th class="num">worst MaxVio</th>'
     f'<th class="num">MaxVio at end</th><th class="num">dead</th></tr>']
for nm, vl, worst, last, dead in rows:
    hl = ' class="hl"' if nm.startswith("A.") else ""
    t.append(f'<tr{hl}><td>{nm}</td><td class="num">{vl[0]:.4f} / {vl[1]:.4f}</td><td class="num">{worst:.2f}</td>'
             f'<td class="num">{last:.2f}</td><td class="num">{dead}</td></tr>')
t.append('</table></div>')
F_["<!--BALTABLE-->"] = "\n".join(t)


# ---------------------------------------------------------------- v2 on the T4
S2 = json.load(open("outputs/v2/summary.json", encoding="utf-8"))
assert S2["meta"]["notebook_version"] == "v2" and S2["meta"]["profile"] == "full"
P2, g2 = S2["runs"]["P"], S2["gates"]
ml = P2["extra"]["moe_log"]
by_step = {m["step"]: m for m in ml}
pv = [q["loss"] for q in P2["val"]]
tr2 = {q["step"]: q["loss"] for q in P2["train"]}
share_last = np.array(ml[-1]["share"])
V.update({
    "v2_pass": f"{S2['acceptance']['passed']}/{S2['acceptance']['total']}",
    "v2_date": datetime.strptime(S2["meta"]["started"], "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y").lstrip("0"),
    "t4_g1_64": f"{g2['G1']['fp64']:.1e}", "t4_g1_32": f"{g2['G1']['fp32']:.1e}",
    "t4_val_moe": f"{g2['G1']['val_moe']:.5f}", "t4_val_r1": f"{g2['G1']['val_r1']:.5f}",
    "t4_g2_joint": f"{g2['G2']['joint_min']:.1e}", "t4_g2_flawed": f"{g2['G2']['per_family_max']:.1e}",
    "t4_total": f"{S2['moe']['total_params']/1e6:.1f}M", "t4_active": f"{S2['moe']['active_params']/1e6:.2f}M",
    "p_v0": f"{pv[0]:.4f}", "p_v1": f"{pv[1]:.4f}", "p_v2": f"{pv[2]:.4f}", "p_v3": f"{pv[3]:.4f}",
    "p_rise": f"{pv[-1] - pv[0]:.4f}",
    "p_t25": f"{tr2[25]:.3f}", "p_t300": f"{tr2[300]:.3f}",
    "p_mv300": f"{by_step[300]['maxvio']:.2f}", "p_mv450": f"{ml[-1]['maxvio']:.2f}",
    "p_b300": f"{by_step[300]['bias_absmax']:.2f}", "p_b450": f"{ml[-1]['bias_absmax']:.2f}",
    "p_worst": f"{share_last.max()*100:.0f}", "p_dead": f"{ml[-1]['dead']}",
    "p_cd0": f"{ml[0]['clone_dist']:.2f}", "p_cd300": f"{by_step[300]['clone_dist']:.2f}", "p_cd450": f"{ml[-1]['clone_dist']:.2f}",
    "p_tps": f"{P2['tokens_per_second']:,.0f}", "p_peak": f"{P2['peak_mem_bytes']/2**20:,.0f}",
    "p_slow": f"{r1['tokens_per_second']/P2['tokens_per_second']:.1f}",
})
assert pv[-1] > pv[0], "the text says the loss rose; the data must agree"
assert by_step[300]["maxvio"] < 0.1 < ml[-1]["maxvio"]
c = []
# left panel: validation loss across the conversion
X0, X1, Y0, Y1 = 55, 320, 30, 240
tail = [q for q in val if q["tokens"] >= val[-1]["tokens"] - 12.6e6]
off = val[-1]["tokens"]
pts = [(q["tokens"], q["loss"]) for q in tail] + [(off + q["tokens"], q["loss"]) for q in P2["val"]]
xlo, xhi = pts[0][0] / 1e6, pts[-1][0] / 1e6
ylo, yhi = 4.10, 4.27
qx = lambda t: X0 + (X1 - X0) * (t / 1e6 - xlo) / (xhi - xlo)
qy = lambda l: Y1 - (Y1 - Y0) * (l - ylo) / (yhi - ylo)
c.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="var(--line2)"/><line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="var(--line2)"/>')
for yv in (4.10, 4.14, 4.18, 4.22, 4.26):
    c.append(f'<line x1="{X0}" y1="{qy(yv):.1f}" x2="{X1}" y2="{qy(yv):.1f}" stroke="var(--line)" stroke-dasharray="2 3"/>')
    c.append(f'<text x="{X0-6}" y="{qy(yv)+4:.1f}" text-anchor="end" font-size="10" fill="var(--ink3)">{yv:.2f}</text>')
for xv in (35, 40, 45, 50, 55):
    if xlo <= xv <= xhi:
        c.append(f'<text x="{qx(xv*1e6):.1f}" y="{Y1+15}" text-anchor="middle" font-size="10" fill="var(--ink3)">{xv}M</text>')
c.append(f'<line x1="{qx(off):.1f}" y1="{Y0}" x2="{qx(off):.1f}" y2="{Y1}" stroke="var(--c3)" stroke-dasharray="5 4"/>')
c.append(f'<text x="{qx(off)+4:.1f}" y="{Y0+10}" font-size="10" fill="var(--c3)">conversion</text>')
d_pts = " ".join(f"{qx(t):.1f},{qy(l):.1f}" for t, l in pts[:len(tail)])
m_pts = " ".join(f"{qx(t):.1f},{qy(l):.1f}" for t, l in pts[len(tail) - 1:])
c.append(f'<polyline points="{d_pts}" fill="none" stroke="var(--c1)" stroke-width="2"/>')
c.append(f'<polyline points="{m_pts}" fill="none" stroke="var(--c2)" stroke-width="2.2"/>')
for i, (t, l) in enumerate(pts):
    col_ = "var(--c1)" if i < len(tail) else "var(--c2)"
    c.append(f'<circle cx="{qx(t):.1f}" cy="{qy(l):.1f}" r="2.6" fill="{col_}"/>')
c.append(f'<text x="{qx(41e6):.1f}" y="{qy(4.25):.1f}" font-size="10.5" fill="var(--c1)">R1 dense</text>')
c.append(f'<text x="{qx(off)+6:.1f}" y="{qy(pv[2])-8:.1f}" font-size="10.5" font-weight="700" fill="var(--c2)">MoE rises</text>')
c.append(f'<text x="{(X0+X1)/2}" y="{Y1+32}" text-anchor="middle" font-size="10.5" fill="var(--ink2)">training tokens &#183; validation loss</text>')
# right panel: MaxVio and clone distance over preview steps
X0, X1 = 400, 680
yhi2 = 2.1
sx = lambda st: X0 + (X1 - X0) * st / P2["steps"]
sy = lambda v: Y1 - (Y1 - Y0) * v / yhi2
c.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="var(--line2)"/><line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="var(--line2)"/>')
for yv in (0, 0.5, 1.0, 1.5, 2.0):
    c.append(f'<line x1="{X0}" y1="{sy(yv):.1f}" x2="{X1}" y2="{sy(yv):.1f}" stroke="var(--line)" stroke-dasharray="2 3"/>')
    c.append(f'<text x="{X0-6}" y="{sy(yv)+4:.1f}" text-anchor="end" font-size="10" fill="var(--ink3)">{yv:g}</text>')
for st in (0, 150, 300, 450):
    c.append(f'<text x="{sx(st):.1f}" y="{Y1+15}" text-anchor="middle" font-size="10" fill="var(--ink3)">{st}</text>')
c.append(f'<line x1="{sx(300):.1f}" y1="{Y0}" x2="{sx(300):.1f}" y2="{Y1}" stroke="var(--ink3)" stroke-dasharray="2 3"/>')
c.append(f'<text x="{sx(300)-4:.1f}" y="{Y0+10}" text-anchor="end" font-size="10" fill="var(--ink3)">switch to hard</text>')
mv_pts = " ".join(f"{sx(m['step']):.1f},{sy(m['maxvio']):.1f}" for m in ml)
cd_pts = " ".join(f"{sx(m['step']):.1f},{sy(m['clone_dist']):.1f}" for m in ml)
c.append(f'<polyline points="{mv_pts}" fill="none" stroke="var(--c1)" stroke-width="2.2"/>')
c.append(f'<polyline points="{cd_pts}" fill="none" stroke="var(--ok)" stroke-width="2.2"/>')
c.append(f'<text x="{sx(440):.1f}" y="{sy(ml[-1]["maxvio"])-8:.1f}" text-anchor="end" font-size="10.5" font-weight="700" fill="var(--c1)">MaxVio</text>')
c.append(f'<text x="{sx(200):.1f}" y="{sy(by_step[200]["clone_dist"])-8:.1f}" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--ok)">clone distance</text>')
c.append(f'<text x="{(X0+X1)/2}" y="{Y1+32}" text-anchor="middle" font-size="10.5" fill="var(--ink2)">preview step</text>')
F_["<!--V2FIG-->"] = "\n".join(c)


# ---------------------------------------------------------------- v2.1 diagnostic on the T4
S3 = json.load(open("outputs/v2.1/summary.json", encoding="utf-8"))
assert S3["meta"]["notebook_version"] == "v2.1" and S3["meta"]["profile"] == "full"
DR = S3["diag"]["runs"]
sw = S3["diag"]["cfg"]["switch"]
def dv(k, st):
    return next(q["loss"] for q in DR[k]["hist"]["val"] if q["step"] == st)
def after(k, key):
    return [m[key] for m in DR[k]["hist"]["extra"].get("moe_log", []) if m["step"] > sw]
dA = DR["A"]["hist"]["final_val"]
gaps = {k: DR[k]["hist"]["final_val"] - dA for k in DR}
noise = abs(S3["runs"]["R1"]["final_val"] - r1["final_val"])
design_gaps = [gaps[k] for k in ("G", "D", "E", "F")]
V.update({
    "d_date": datetime.strptime(S3["meta"]["started"], "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y").lstrip("0"),
    "d_A": f"{dA:.4f}", "d_B": f"{DR['B']['hist']['final_val']:.4f}",
    "r1b_final": f"{S3['runs']['R1']['final_val']:.4f}", "r1_repeat": f"{noise:.4f}",
    "p2_rise": f"{S3['runs']['P']['final_val'] - S3['runs']['P']['val'][0]['loss']:+.4f}",
    "d_gap_B": f"{gaps['B']:+.4f}", "d_gap_C": f"{gaps['C']:+.4f}", "d_gap_G": f"{gaps['G']:+.4f}",
    "d_C150": f"{dv('C', 150):.4f}", "d_A150": f"{dv('A', 150):.4f}",
    "d_mv_D": f"{max(after('D', 'maxvio')):.2f}", "d_mv_G": f"{max(after('G', 'maxvio')):.2f}",
    "d_dead_G": f"{max(after('G', 'dead'))}", "d_mv_F": f"{max(m['maxvio'] for m in DR['F']['hist']['extra']['moe_log']):.2f}",
    "d_F50": f"{dv('F', 50):.4f}", "d_E50": f"{dv('E', 50):.4f}", "d_A50": f"{dv('A', 50):.4f}", "d_E200": f"{dv('E', 200):.4f}",
    "d_cd_F25": f"{DR['F']['hist']['extra']['moe_log'][0]['clone_dist']*100:.0f}%",
    "d_gap_lo": f"{min(design_gaps):.3f}", "d_gap_hi": f"{max(design_gaps):.3f}",
    "d_x_lo": f"{min(design_gaps)/noise:.0f}", "d_x_hi": f"{max(design_gaps)/noise:.0f}",
})
assert abs(gaps["B"]) < 0.002 and min(design_gaps) > 20 * noise
assert dv("E", 50) < dv("A", 50) and max(after("D", "maxvio")) < 0.3 < max(after("G", "maxvio"))
cols = {"A": "var(--c1)", "B": "var(--grey)", "C": "var(--c3)", "G": "var(--c2)", "D": "var(--ok)", "E": "#c9a227", "F": "var(--flag)"}
c = []
X0, X1, Y0, Y1 = 55, 400, 25, 250
ylo, yhi = 4.05, 4.22
steps_d = [q["step"] for q in DR["A"]["hist"]["val"]]
gx = lambda st: X0 + (X1 - X0) * st / steps_d[-1]
gy = lambda l: Y1 - (Y1 - Y0) * (l - ylo) / (yhi - ylo)
c.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="var(--line2)"/><line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="var(--line2)"/>')
for yv in (4.06, 4.10, 4.14, 4.18, 4.22):
    c.append(f'<line x1="{X0}" y1="{gy(yv):.1f}" x2="{X1}" y2="{gy(yv):.1f}" stroke="var(--line)" stroke-dasharray="2 3"/>')
    c.append(f'<text x="{X0-6}" y="{gy(yv)+4:.1f}" text-anchor="end" font-size="10" fill="var(--ink3)">{yv:.2f}</text>')
for st in steps_d:
    c.append(f'<text x="{gx(st):.1f}" y="{Y1+15}" text-anchor="middle" font-size="10" fill="var(--ink3)">{st}</text>')
c.append(f'<line x1="{gx(sw):.1f}" y1="{Y0}" x2="{gx(sw):.1f}" y2="{Y1}" stroke="var(--ink3)" stroke-dasharray="2 3"/>')
c.append(f'<text x="{(X0+X1)/2}" y="{Y1+32}" text-anchor="middle" font-size="10.5" fill="var(--ink2)">steps after conversion &#183; validation loss</text>')
for k in ("A", "B", "C", "G", "D", "E", "F"):
    pts = " ".join(f"{gx(q['step']):.1f},{gy(q['loss']):.1f}" for q in DR[k]["hist"]["val"])
    dash = ' stroke-dasharray="5 4"' if k == "B" else ""
    c.append(f'<polyline points="{pts}" fill="none" stroke="{cols[k]}" stroke-width="2"{dash}/>')
    last = DR[k]["hist"]["val"][-1]
for i, k in enumerate(("A", "B", "C", "G", "D", "E", "F")):
    y = 40 + 17 * i
    c.append(f'<rect x="412" y="{y-5}" width="12" height="3" fill="{cols[k]}"/>')
    c.append(f'<text x="428" y="{y}" font-size="10" fill="var(--ink)">{k} {DR[k]["label"]}</text>')
# MaxVio inset
X0, X1, Y0b, Y1b = 440, 680, 170, 250
mx = 2.4
hx = lambda st: X0 + (X1 - X0) * st / steps_d[-1]
hy = lambda v: Y1b - (Y1b - Y0b) * min(v, mx) / mx
c.append(f'<line x1="{X0}" y1="{Y1b}" x2="{X1}" y2="{Y1b}" stroke="var(--line2)"/><line x1="{X0}" y1="{Y0b}" x2="{X0}" y2="{Y1b}" stroke="var(--line2)"/>')
for yv in (0, 1, 2):
    c.append(f'<text x="{X0-5}" y="{hy(yv)+4:.1f}" text-anchor="end" font-size="9.5" fill="var(--ink3)">{yv}</text>')
c.append(f'<line x1="{hx(sw):.1f}" y1="{Y0b}" x2="{hx(sw):.1f}" y2="{Y1b}" stroke="var(--ink3)" stroke-dasharray="2 3"/>')
for k in ("G", "D", "E", "F"):
    ml_ = DR[k]["hist"]["extra"]["moe_log"]
    pts = " ".join(f"{hx(m['step']):.1f},{hy(m['maxvio']):.1f}" for m in ml_)
    c.append(f'<polyline points="{pts}" fill="none" stroke="{cols[k]}" stroke-width="2"/>')
c.append(f'<text x="{X0}" y="{Y0b-6}" font-size="10.5" font-weight="700" fill="var(--ink2)">MaxVio (G, D, E, F)</text>')
c.append(f'<text x="{(X0+X1)/2}" y="{Y1b+15}" text-anchor="middle" font-size="9.5" fill="var(--ink3)">steps</text>')
F_["<!--D21FIG-->"] = "\n".join(c)
t = ['<div class="tblwrap"><table>',
     '<tr><th>run</th><th>what differs</th>' + "".join(f'<th class="num">val @{st}</th>' for st in steps_d)
     + '<th class="num">vs A</th><th class="num">MaxVio after switch</th><th class="num">tok/s</th></tr>']
for k in ("A", "B", "C", "G", "D", "E", "F"):
    h = DR[k]["hist"]
    mvs = after(k, "maxvio")
    mv = "n/a" if k in ("A", "B", "C") or not mvs else f"{max(mvs):.2f}"
    hl = ' class="hl"' if k == "D" else ""
    t.append(f'<tr{hl}><td><b>{k}</b></td><td>{DR[k]["label"]}</td>'
             + "".join(f'<td class="num">{dv(k, st):.4f}</td>' for st in steps_d)
             + f'<td class="num">{gaps[k]:+.4f}</td><td class="num">{mv}</td><td class="num">{h["tokens_per_second"]:,.0f}</td></tr>')
t.append('</table></div>')
F_["<!--D21TABLE-->"] = "\n".join(t)


# ---------------------------------------------------------------- v2.2 diagnostic on the T4
S4 = json.load(open("outputs/v2.2/summary.json", encoding="utf-8"))
assert S4["meta"]["notebook_version"] == "v2.2" and S4["meta"]["profile"] == "full"
ER = S4["diag2"]["runs"]
def ev(k, st):
    return next(q["loss"] for q in ER[k]["hist"]["val"] if q["step"] == st)
def eafter(k, key):
    return [m[key] for m in ER[k]["hist"]["extra"].get("moe_log", []) if m["step"] > sw]
eA = ER["A"]["hist"]["final_val"]
eg = {k: ER[k]["hist"]["final_val"] - eA for k in ER}
r1s = [r1["final_val"], S3["runs"]["R1"]["final_val"], S4["runs"]["R1"]["final_val"]]
spread = max(r1s) - min(r1s)
leads = [-eg[k] for k in ("D35", "D12", "F35")]
V.update({
    "e_date": datetime.strptime(S4["meta"]["started"], "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y").lstrip("0"),
    "e_exact": f"{S4['diag2']['exact_rel']:.1e}",
    "r1c_final": f"{r1s[2]:.4f}", "r1_spread": f"{spread:.4f}",
    "p3_rise": f"{S4['runs']['P']['final_val'] - S4['runs']['P']['val'][0]['loss']:+.4f}",
    "e_gap_D": f"{eg['D']:+.4f}", "e_gap_D35": f"{eg['D35']:+.4f}", "e_gap_D12": f"{eg['D12']:+.4f}",
    "e_mv_D35": f"{max(eafter('D35', 'maxvio')):.2f}", "e_mv_D12": f"{max(eafter('D12', 'maxvio')):.2f}",
    "e_mv_F35": f"{max(m['maxvio'] for m in ER['F35']['hist']['extra']['moe_log']):.2f}",
    "e_aw_dip": f"{ev('Aw', 50) - ev('A', 50):+.4f}", "e_e_dip": f"{dv('E', 50) - dv('A', 50):+.4f}",
    "e_sl_A": f"{ev('A', 200) - ev('A', 150):+.4f}", "e_sl_F35": f"{ev('F35', 200) - ev('F35', 150):+.4f}",
    "e_sl_D35": f"{ev('D35', 200) - ev('D35', 150):+.4f}", "e_sl_D12": f"{ev('D12', 200) - ev('D12', 150):+.4f}",
    "e_lead_lo": f"{min(leads):.3f}", "e_lead_hi": f"{max(leads):.3f}",
    "e_x_lo": f"{min(leads)/spread:.0f}", "e_x_hi": f"{max(leads)/spread:.0f}",
})
assert eg["D"] > 0 > max(eg[k] for k in ("D35", "D12", "F35")), "the text says the fix turned the MoE around"
assert all(max(m["dead"] for m in ER[k]["hist"]["extra"]["moe_log"]) == 0 for k in ("D35", "D12", "F35"))
assert ev("Aw", 50) - ev("A", 50) < dv("E", 50) - dv("A", 50) < 0
cols2 = {"A": "var(--c1)", "Aw": "var(--c1)", "D": "var(--c2)", "D35": "var(--ok)", "D12": "var(--c4)", "F35": "var(--c3)"}
c = []
X0, X1, Y0, Y1 = 55, 400, 25, 250
ylo, yhi = 4.04, 4.18
steps_e = [q["step"] for q in ER["A"]["hist"]["val"]]
gx = lambda st: X0 + (X1 - X0) * st / steps_e[-1]
gy = lambda l: Y1 - (Y1 - Y0) * (l - ylo) / (yhi - ylo)
c.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="var(--line2)"/><line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="var(--line2)"/>')
for yv in (4.04, 4.08, 4.12, 4.16):
    c.append(f'<line x1="{X0}" y1="{gy(yv):.1f}" x2="{X1}" y2="{gy(yv):.1f}" stroke="var(--line)" stroke-dasharray="2 3"/>')
    c.append(f'<text x="{X0-6}" y="{gy(yv)+4:.1f}" text-anchor="end" font-size="10" fill="var(--ink3)">{yv:.2f}</text>')
for st in steps_e:
    c.append(f'<text x="{gx(st):.1f}" y="{Y1+15}" text-anchor="middle" font-size="10" fill="var(--ink3)">{st}</text>')
c.append(f'<line x1="{gx(sw):.1f}" y1="{Y0}" x2="{gx(sw):.1f}" y2="{Y1}" stroke="var(--ink3)" stroke-dasharray="2 3"/>')
c.append(f'<text x="{(X0+X1)/2}" y="{Y1+32}" text-anchor="middle" font-size="10.5" fill="var(--ink2)">steps after conversion &#183; validation loss</text>')
order = ("A", "Aw", "D", "D35", "D12", "F35")
for k in order:
    pts = " ".join(f"{gx(q['step']):.1f},{gy(q['loss']):.1f}" for q in ER[k]["hist"]["val"])
    dash = ' stroke-dasharray="5 4"' if k == "Aw" else ""
    wid = "2.8" if k in ("A", "D12") else "2"
    c.append(f'<polyline points="{pts}" fill="none" stroke="{cols2[k]}" stroke-width="{wid}"{dash}/>')
for i, k in enumerate(order):
    y = 45 + 20 * i
    dash = ' stroke-dasharray="4 3"' if k == "Aw" else ""
    c.append(f'<line x1="420" y1="{y-4}" x2="440" y2="{y-4}" stroke="{cols2[k]}" stroke-width="2.4"{dash}/>')
    c.append(f'<text x="446" y="{y}" font-size="10.5" fill="var(--ink)">{k}: {ER[k]["label"]}</text>')
    c.append(f'<text x="690" y="{y}" text-anchor="end" font-size="10.5" font-family="var(--mono)" fill="var(--ink2)">{eg[k]:+.4f}</text>')
c.append('<text x="690" y="30" text-anchor="end" font-size="10" fill="var(--ink3)">vs A at 200</text>')
F_["<!--D22FIG-->"] = "\n".join(c)
t = ['<div class="tblwrap"><table>',
     '<tr><th>run</th><th>what differs</th>' + "".join(f'<th class="num">val @{st}</th>' for st in steps_e)
     + '<th class="num">vs A</th><th class="num">MaxVio after switch</th><th class="num">dead</th></tr>']
for k in order:
    h = ER[k]["hist"]
    ml_ = h["extra"].get("moe_log", [])
    mv = "n/a" if not ml_ else f"{max(eafter(k, 'maxvio') or [m['maxvio'] for m in ml_]):.2f}"
    dd = "n/a" if not ml_ else f"{max(m['dead'] for m in ml_)}"
    hl = ' class="hl"' if k == "D12" else ""
    t.append(f'<tr{hl}><td><b>{k}</b></td><td>{ER[k]["label"]}</td>'
             + "".join(f'<td class="num">{ev(k, st):.4f}</td>' for st in steps_e)
             + f'<td class="num">{eg[k]:+.4f}</td><td class="num">{mv}</td><td class="num">{dd}</td></tr>')
t.append('</table></div>')
F_["<!--D22TABLE-->"] = "\n".join(t)


# ---------------------------------------------------------------- v3 on the T4
S5 = json.load(open("outputs/v3/summary.json", encoding="utf-8"))
NB5 = json.load(open("notebooks/session14_moe_v3_executed.ipynb", encoding="utf-8"))
assert S5["meta"]["notebook_version"] == "v3" and S5["meta"]["profile"] == "full"
nb5 = "".join("".join(o.get("text", "")) for c in NB5["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []))
assert "drive mounted: True" in nb5
resume = re.search(r"\[resume\] 'R3_moe_sw\d+' from step ([\d,]+)", nb5)
RR = {k: S5["runs"][k] for k in ("R2", "R2s", "R3")}
conv = RR["R3"]["val"][0]["loss"]
fin = {k: h["final_val"] for k, h in RR.items()}
steps_all = S5["config"]["steps"]
decay_steps = S5["v3"]["decay_steps"]
tok_step = 32 * S5["config"]["model"]["seq"]
decay_tok = (steps_all - decay_steps) * tok_step
def vat(k, tok):
    return min(RR[k]["val"], key=lambda q: abs(q["tokens"] - tok))
dec = {k: vat(k, decay_tok)["loss"] - fin[k] for k in RR}
diff2 = [(q3["tokens"], q3["loss"] - q2["loss"]) for q2, q3 in zip(RR["R2"]["val"], RR["R3"]["val"])]
diffs = [(q3["tokens"], q3["loss"] - qs["loss"]) for qs, q3 in zip(RR["R2s"]["val"], RR["R3"]["val"])]
NOISE_ = 0.002
lead_end = max(t for t, d in diff2 if d < -NOISE_)
ml5 = RR["R3"]["extra"]["moe_log"]
full5 = [m for i, m in enumerate(ml5) if m["step"] - (ml5[i - 1]["step"] if i else 0) >= 25]
p2tok = vat("R3", decay_tok)["tokens"]
V.update({
    "f_date": datetime.strptime(S5["meta"]["started"], "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y").lstrip("0"),
    "f_pass": f"{S5['acceptance']['passed']}/{S5['acceptance']['total']}",
    "f_resume": resume.group(1) if resume else "n/a",
    "f_conv": f"{conv:.4f}", "f_R3": f"{fin['R3']:.4f}", "f_d_R3": f"{fin['R3'] - conv:+.4f}",
    "f_gap_R2": f"{fin['R3'] - fin['R2']:+.4f}", "f_gap_R2s": f"{fin['R3'] - fin['R2s']:+.4f}",
    "f_gap_R2s_abs": f"{abs(fin['R3'] - fin['R2s']):.4f}",
    "f_gap_R2_abs": f"{abs(fin['R3'] - fin['R2']):.4f}",
    "f_p1_r2": f"{-dict(diff2)[vat('R3', 10e6)['tokens']]:.4f}", "f_p1_r2s": f"{dict(diffs)[vat('R3', 10e6)['tokens']]:+.4f}",
    "f_p2_r2s": f"{-dict(diffs)[p2tok]:.4f}", "f_p2_r2": f"{-dict(diff2)[p2tok]:.4f}", "f_p2_tok": f"{p2tok/1e6:.1f}",
    "f_dec_R2": f"{dec['R2']:.4f}", "f_dec_R3": f"{dec['R3']:.4f}", "f_dec_R2s": f"{dec['R2s']:.4f}",
    "f_lead_end": f"{lead_end/1e6:.1f}",
    "f_cd0": f"{ml5[0]['clone_dist']:.3f}", "f_cd": f"{ml5[-1]['clone_dist']:.3f}",
    "f_dead": f"{max(m['dead'] for m in full5)}",
    "f_mv_max": f"{max(m['maxvio'] for m in full5 if m['step'] > S5['v3']['cfg']['switch']):.3f}",
    "f_mv_end": f"{full5[-1]['maxvio']:.3f}",
    "f_tps_R3": f"{RR['R3']['tokens_per_second']:,.0f}",
    "f_slow": f"{RR['R2']['tokens_per_second']/RR['R3']['tokens_per_second']:.1f}",
    "f_mem_R3": f"{RR['R3']['peak_mem_bytes']/2**20:,.0f}", "f_mem_R2": f"{RR['R2']['peak_mem_bytes']/2**20:,.0f}",
    "noise": f"{NOISE_}",
    "f_x_r2": f"{abs(fin['R3'] - fin['R2'])/spread:.0f}", "f_x_r2s": f"{abs(fin['R3'] - fin['R2s'])/spread:.0f}",
})
# the text's claims, checked against the data
assert conv - fin["R3"] > NOISE_ and fin["R3"] > fin["R2"] + NOISE_ and fin["R3"] < fin["R2s"] - NOISE_
assert abs(dict(diffs)[vat("R3", 10e6)["tokens"]]) < 0.01 and dict(diff2)[vat("R3", 10e6)["tokens"]] < -0.02
assert dec["R2"] > dec["R3"] and dec["R2"] > dec["R2s"] and max(m["dead"] for m in full5) == 0
assert all(d > NOISE_ or t <= lead_end for t, d in diff2 if t > lead_end) and lead_end >= decay_tok

# figure: loss across the conversion
c = []
X0, X1, Y0, Y1 = 55, 520, 20, 250
tail = [q for q in val if q["tokens"] >= val[-1]["tokens"] - 20e6]
off = val[-1]["tokens"]
xlo, xhi = tail[0]["tokens"] / 1e6, (off + RR["R2"]["val"][-1]["tokens"]) / 1e6
ylo, yhi = 3.74, 4.36
fx = lambda t: X0 + (X1 - X0) * (t / 1e6 - xlo) / (xhi - xlo)
fy = lambda l: Y1 - (Y1 - Y0) * (l - ylo) / (yhi - ylo)
c.append(f'<rect x="{fx(off + decay_tok):.1f}" y="{Y0}" width="{fx(xhi*1e6) - fx(off + decay_tok):.1f}" height="{Y1-Y0}" fill="var(--panel2)"/>')
c.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="var(--line2)"/><line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="var(--line2)"/>')
for yv in (3.8, 3.9, 4.0, 4.1, 4.2, 4.3):
    c.append(f'<line x1="{X0}" y1="{fy(yv):.1f}" x2="{X1}" y2="{fy(yv):.1f}" stroke="var(--line)" stroke-dasharray="2 3"/>')
    c.append(f'<text x="{X0-6}" y="{fy(yv)+4:.1f}" text-anchor="end" font-size="10" fill="var(--ink3)">{yv:.1f}</text>')
for xv in range(40, 101, 10):
    if xlo <= xv <= xhi:
        c.append(f'<text x="{fx(xv*1e6):.1f}" y="{Y1+15}" text-anchor="middle" font-size="10" fill="var(--ink3)">{xv}M</text>')
c.append(f'<line x1="{fx(off):.1f}" y1="{Y0}" x2="{fx(off):.1f}" y2="{Y1}" stroke="var(--c3)" stroke-dasharray="5 4"/>')
c.append(f'<text x="{fx(off)+4:.1f}" y="{Y0+12}" font-size="10" fill="var(--c3)">conversion</text>')
c.append(f'<text x="{fx(off + decay_tok)+4:.1f}" y="{Y0+12}" font-size="10" fill="var(--ink3)">decay</text>')
c.append('<polyline points="' + " ".join(f"{fx(q['tokens']):.1f},{fy(q['loss']):.1f}" for q in tail) + '" fill="none" stroke="var(--grey)" stroke-width="2"/>')
cols3 = {"R2": "var(--c1)", "R2s": "var(--c1)", "R3": "var(--c2)"}
for k in ("R2", "R2s", "R3"):
    dash = ' stroke-dasharray="5 4"' if k == "R2s" else ""
    c.append('<polyline points="' + " ".join(f"{fx(off + q['tokens']):.1f},{fy(q['loss']):.1f}" for q in RR[k]["val"])
             + f'" fill="none" stroke="{cols3[k]}" stroke-width="{2.6 if k == "R3" else 2}"{dash}/>')
labels = (("R1 dense (first 50M)", "var(--grey)", ""), ("R2 dense continues", "var(--c1)", ""),
          ("R2s dense, FFN at 1/8 speed", "var(--c1)", ' stroke-dasharray="4 3"'), ("R3 MoE", "var(--c2)", ""))
for i, (lab, col_, dash) in enumerate(labels):
    y = 60 + 22 * i
    c.append(f'<line x1="540" y1="{y-4}" x2="562" y2="{y-4}" stroke="{col_}" stroke-width="2.4"{dash}/>')
    c.append(f'<text x="568" y="{y}" font-size="10.5" fill="var(--ink)">{lab}</text>')
for i, k in enumerate(("R2", "R2s", "R3")):
    c.append(f'<text x="568" y="{170 + 18*i}" font-size="10.5" font-family="var(--mono)" fill="var(--ink2)">{k:<3} final {fin[k]:.4f}</text>')
c.append(f'<text x="{(X0+X1)/2}" y="{Y1+32}" text-anchor="middle" font-size="10.5" fill="var(--ink2)">training tokens &#183; validation loss</text>')
F_["<!--V3FIG-->"] = "\n".join(c)

# figure: R3's lead over each control
c = []
X0, X1, Y0, Y1 = 55, 560, 20, 215
dlo, dhi = -0.06, 0.03
tmax = RR["R3"]["val"][-1]["tokens"]
dx = lambda t: X0 + (X1 - X0) * t / tmax
dy = lambda d: Y1 - (Y1 - Y0) * (d - dlo) / (dhi - dlo)
c.append(f'<rect x="{dx(decay_tok):.1f}" y="{Y0}" width="{X1 - dx(decay_tok):.1f}" height="{Y1-Y0}" fill="var(--panel2)"/>')
c.append(f'<rect x="{X0}" y="{dy(NOISE_):.1f}" width="{X1-X0}" height="{dy(-NOISE_)-dy(NOISE_):.1f}" fill="var(--grey)" opacity="0.25"/>')
c.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="var(--line2)"/><line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="var(--line2)"/>')
for yv in (-0.06, -0.04, -0.02, 0.0, 0.02):
    c.append(f'<line x1="{X0}" y1="{dy(yv):.1f}" x2="{X1}" y2="{dy(yv):.1f}" stroke="var(--line)" stroke-dasharray="2 3"/>')
    c.append(f'<text x="{X0-6}" y="{dy(yv)+4:.1f}" text-anchor="end" font-size="10" fill="var(--ink3)">{yv:+.2f}</text>')
for xv in range(0, 51, 10):
    c.append(f'<text x="{dx(xv*1e6):.1f}" y="{Y1+15}" text-anchor="middle" font-size="10" fill="var(--ink3)">{xv}M</text>')
c.append(f'<text x="{(X0+X1)/2}" y="{Y1+32}" text-anchor="middle" font-size="10.5" fill="var(--ink2)">tokens after conversion</text>')
for series, col_, lab in ((diff2, "var(--c1)", "R3 &#8722; R2 (vs dense)"), (diffs, "var(--ok)", "R3 &#8722; R2s (vs fair control)")):
    c.append('<polyline points="' + " ".join(f"{dx(t):.1f},{dy(d):.1f}" for t, d in series) + f'" fill="none" stroke="{col_}" stroke-width="2.4"/>')
    for t, d in series:
        c.append(f'<circle cx="{dx(t):.1f}" cy="{dy(d):.1f}" r="2.4" fill="{col_}"/>')
c.append(f'<line x1="575" y1="56" x2="597" y2="56" stroke="var(--c1)" stroke-width="2.4"/><text x="603" y="60" font-size="10.5" fill="var(--ink)">vs dense R2</text>')
c.append(f'<line x1="575" y1="80" x2="597" y2="80" stroke="var(--ok)" stroke-width="2.4"/><text x="603" y="84" font-size="10.5" fill="var(--ink)">vs fair R2s</text>')
c.append('<text x="575" y="112" font-size="10" fill="var(--ink3)">below 0: MoE ahead</text>')
c.append('<text x="575" y="128" font-size="10" fill="var(--ink3)">above 0: MoE behind</text>')
F_["<!--V3DIFF-->"] = "\n".join(c)

# table
rows = (("R1 dense (start)", None), ("R2 dense continues", "R2"), ("R2s dense, FFN at 1/8 speed", "R2s"), ("R3 MoE", "R3"))
t = ['<div class="tblwrap"><table>',
     '<tr><th>run</th><th class="num">params total / active</th><th class="num">final val</th><th class="num">since conversion</th>'
     '<th class="num">in the decay</th><th class="num">tok/s</th><th class="num">peak MiB</th></tr>',
     f'<tr><td>R1 dense (the conversion point)</td><td class="num">{(pb['total']/1e6):.1f}M / {(pb['total']/1e6):.1f}M</td><td class="num">{conv:.4f}</td>'
     '<td class="num">&#8212;</td><td class="num">&#8212;</td><td class="num">&#8212;</td><td class="num">&#8212;</td></tr>']
for lab, k in rows[1:]:
    h = RR[k]
    ptxt = f"{S5['moe']['total_params']/1e6:.1f}M / {S5['moe']['active_params']/1e6:.1f}M" if k == "R3" else f"{(pb['total']/1e6):.1f}M / {(pb['total']/1e6):.1f}M"
    hl = ' class="hl"' if k == "R3" else ""
    t.append(f'<tr{hl}><td>{lab}</td><td class="num">{ptxt}</td><td class="num"><b>{fin[k]:.4f}</b></td><td class="num">{fin[k]-conv:+.4f}</td>'
             f'<td class="num">{-dec[k]:+.4f}</td><td class="num">{h["tokens_per_second"]:,.0f}</td><td class="num">{h["peak_mem_bytes"]/2**20:,.0f}</td></tr>')
t.append('</table></div>')
F_["<!--V3TABLE-->"] = "\n".join(t)

# samples
c = []
for i, p_ in enumerate(prompts):
    for k in ("R2", "R3"):
        smp = S5["samples_v3"][k][i]
        assert smp.startswith(p_)
        c.append(f'<div class="sample"><b>{k}: {html.escape(p_)}</b>{html.escape(smp[len(p_):]).replace(chr(10), " &#8629; ")}</div>')
F_["<!--V3SAMPLES-->"] = "\n".join(c)

# ---------------------------------------------------------------- fill
for k, v in F_.items():
    assert k in src, k
    src = src.replace(k, v)
for k, v in V.items():
    src = src.replace("{{" + k + "}}", v)
left = re.findall(r"\{\{\w+\}\}", src)
assert not left, left
assert "<!--" not in src.replace("<!-- ====", "")
open("assignment_plan.html", "w", encoding="utf-8").write(src)
print("ok", len(src), "bytes")
for k in ("r1_final", "s13_final", "gap_s13", "last10_drop", "data_min", "wall_min", "moe_total", "moe_active", "rt_fp32", "rt_fp64"):
    print(f"  {k:<12} {V[k]}")
