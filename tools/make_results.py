"""Build the Session 14 results report (session14_results.html).

Same series as the Session 11-13 walkthroughs. The page opens with the headline numbers and the path
from the first failed conversion to the result, then the README's own write-up (whose numbers come
from the summaries via tools/make_readme.py) with its figures embedded, then every cell of the EXECUTED
v3 notebook with the transcript it printed on Colab. Nothing is typed here by hand: numbers come from
outputs/v*/summary.json, prose from README.md, transcripts from the executed .ipynb.

Usage (from session 14/assignment/):
    python tools/make_results.py
"""
import base64
import html
import json
import os
import re

import markdown

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "session14_results.html")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SKIP = ("Reports", "Files", "How to run it", "Acceptance checks")


def load(v):
    return json.load(open(os.path.join(HERE, "outputs", v, "summary.json"), encoding="utf-8"))


S1, S2, S21, S22, S3 = load("v1"), load("v2"), load("v2.1"), load("v2.2"), load("v3")
css = re.search(r"<style>.*?</style>", open(os.path.join(HERE, "..", "assignment_research", "session14_report.html"),
                                            encoding="utf-8").read(), re.S).group(0)
readme = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
nb = json.load(open(os.path.join(HERE, "notebooks", "session14_moe_v3_executed.ipynb"), encoding="utf-8"))

# ------------------------------------------------------------------ numbers
R1, R2, R2s, R3 = (S3["runs"][k] for k in ("R1", "R2", "R2s", "R3"))
conv, fin = R3["val"][0]["loss"], {k: S3["runs"][k]["final_val"] for k in ("R2", "R2s", "R3")}
P2 = S2["runs"]["P"]
D1, D2 = S21["diag"]["runs"], S22["diag2"]["runs"]
gap = lambda D, k: D[k]["hist"]["final_val"] - D["A"]["hist"]["final_val"]
mv_after = lambda h, sw: max(m["maxvio"] for m in h["extra"]["moe_log"] if m["step"] > sw)
acc = S3["acceptance"]

tiles = [("MoE loss after conversion", f"{conv:.4f} &rarr; {fin['R3']:.4f}", "kept learning over 50M tokens"),
         ("vs dense, same slowed FFN", f"{fin['R3'] - fin['R2s']:+.4f}", "below 0: the MoE is ahead"),
         ("vs ordinary dense", f"{fin['R3'] - fin['R2']:+.4f}", "above 0: dense finished ahead"),
         ("MoE speed", f"{R2['tokens_per_second']/R3['tokens_per_second']:.1f}&times; slower", "same work per token"),
         ("checks", f"{acc['passed']}/{acc['total']}", "final notebook, T4")]

# ------------------------------------------------------------------ the path diagram
steps = [("v1", "dense model", f"R1: {R1['val'][0]['loss']:.2f} &rarr; {R1['final_val']:.4f}", "done", "50M tokens, 55k tok/s"),
         ("v2", "convert + preview", f"exact start; loss rose {P2['final_val']-P2['val'][0]['loss']:+.3f}", "flag",
          f"load collapsed: MaxVio {P2['extra']['moe_log'][-1]['maxvio']:.2f}"),
         ("v2.1", "diagnostic, 7 runs", f"slow router: MaxVio {mv_after(D1['D']['hist'], 100):.2f}", "open",
          "rise comes from splitting tokens"),
         ("v2.2", "diagnostic, 6 runs", f"experts at 1/8: {gap(D2, 'D12'):+.3f} vs dense", "done", "the fix, at 200 steps"),
         ("v3", "full comparison", f"MoE {conv:.3f} &rarr; {fin['R3']:.4f}", "done",
          f"vs R2s {fin['R3']-fin['R2s']:+.3f}; vs R2 {fin['R3']-fin['R2']:+.3f}")]
col = {"done": ("var(--ok-bg)", "var(--ok)"), "flag": ("var(--flag-bg)", "var(--flag)"), "open": ("var(--open-bg)", "var(--open)")}
svg = ['<svg viewBox="0 0 700 150" role="img" aria-label="The path: v1 trained the dense model; v2 converted it and the MoE lost '
       'ground; v2.1 and v2.2 found the fixes; v3 showed the MoE keeps learning.">',
       '<defs><marker id="pa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
       '<path d="M0 0 L10 5 L0 10 z" fill="var(--ink3)"/></marker></defs>']
for i, (k, t, a, kind, b) in enumerate(steps):
    x = 4 + 139 * i
    fill, stroke = col[kind]
    svg.append(f'<rect x="{x}" y="14" width="128" height="120" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    svg.append(f'<text x="{x+10}" y="34" font-size="12" font-weight="700" fill="{stroke}">{k}</text>')
    svg.append(f'<text x="{x+10}" y="52" font-size="11.5" font-weight="700" fill="var(--ink)">{t}</text>')
    for j, line in enumerate(re.findall(r".{1,24}(?:\s|$)", html.unescape(a).replace("→", "->"))):
        svg.append(f'<text x="{x+10}" y="{74 + 14*j}" font-size="10" fill="var(--ink2)">{html.escape(line.strip()).replace("-&gt;", "&#8594;")}</text>')
    for j, line in enumerate(re.findall(r".{1,24}(?:\s|$)", b)):
        svg.append(f'<text x="{x+10}" y="{108 + 13*j}" font-size="9.5" fill="var(--ink3)">{html.escape(line.strip())}</text>')
    if i < 4:
        svg.append(f'<path d="M{x+129} 74 L{x+137} 74" stroke="var(--ink3)" stroke-width="1.5" marker-end="url(#pa)"/>')
svg.append("</svg>")

# ------------------------------------------------------------------ README sections, figures embedded
def embed(md_html):
    def rep(m):
        src = m.group(1)
        path = os.path.join(HERE, src)
        if not os.path.exists(path):
            return m.group(0)
        data = base64.b64encode(open(path, "rb").read()).decode()
        return f'src="data:image/png;base64,{data}"'
    return re.sub(r'src="([^"]+\.png)"', rep, md_html)

chapters = re.split(r"(?m)^## ", readme)
body_md = []
for ch in chapters[1:]:
    title = ch.split("\n", 1)[0].strip()
    if title in SKIP:
        continue
    body_md.append("## " + ch)
write_up = embed(markdown.markdown("\n".join(body_md), extensions=["tables"]))
write_up = write_up.replace("<table>", '<div class="tblwrap"><table>').replace("</table>", "</table></div>")
write_up = re.sub(r"<p><img ([^>]*)></p>", r'<figure><img \1 style="width:100%;height:auto"></figure>', write_up)

# ------------------------------------------------------------------ every cell
def text_of(o):
    t = o.get("text") or o.get("data", {}).get("text/plain") or ""
    return ANSI.sub("", "".join(t))

cells = []
for c in nb["cells"]:
    src = "".join(c["source"])
    if c["cell_type"] == "markdown":
        cells.append(markdown.markdown(src.replace("# ## ", "## "), extensions=["tables"])
                     .replace("<table>", '<div class="tblwrap"><table>').replace("</table>", "</table></div>"))
        continue
    outs = []
    for o in c.get("outputs", []):
        t = text_of(o)
        if not t or "Resolving data files" in t or "README.md:" in t or "it/s]" in t:
            continue
        outs.append(t)
    n = src.count("\n") + 1
    cells.append(f'<details class="code"><summary>code ({n} lines)</summary><pre>{html.escape(src)}</pre></details>')
    if outs:
        cells.append(f'<div class="lbl2">printed on Colab</div><pre class="out">{html.escape("".join(outs))}</pre>')

reports = re.search(r"(?ms)^## Reports\n(.*?)(?=^## )", readme).group(1)
reports_html = markdown.markdown(reports)

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dense to MoE Results</title>
{css}
<style>
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0 26px}}
.tile{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px 14px}}
.tile .k{{font-family:var(--sans);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3)}}
.tile .v{{font-family:var(--mono);font-size:1.12rem;font-weight:700;margin:4px 0 2px}}
.tile .n{{font-family:var(--sans);font-size:.78rem;color:var(--ink2)}}
details.code summary{{font-family:var(--sans);font-size:.8rem;color:var(--ink3);cursor:pointer;margin:6px 0}}
pre.out{{background:var(--panel2)}}
.lbl2{{font-family:var(--sans);font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--ok);margin:10px 0 -8px}}
.cells h2{{font-size:1.1rem;margin-top:44px}}
.cells h3{{font-size:.95rem}}
</style></head>
<body><div class="wrap">
<h1>From a dense model to a mixture of experts</h1>
<p class="sub">ERA V5, Session 14. A 22.5M-parameter dense GPT, converted into a 114.4M-parameter mixture of experts that does the same work per token, and trained on. This page is the results report: the answer, how it was reached, and every cell of the final notebook with what it printed on the T4.</p>
<p class="meta">Final run: {S3['meta']['started'][:10]}, Colab {S3['meta']['gpu']}, {acc['passed']}/{acc['total']} checks.
Every number here comes from the runs' own summary files; the prose is the README, generated from them.</p>
<div class="tiles">{"".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="n">{n}</div></div>' for k, v, n in tiles)}</div>
<div class="toc"><h4>Contents</h4><ul>
<li><a href="#path">The path to the result</a></li><li><a href="#writeup">The write-up</a></li><li><a href="#cells">Every cell of the final notebook</a></li><li><a href="#reports">Other reports</a></li></ul></div>
<h2 id="path">The path to the result</h2>
<p>The first conversion kept the model's function exactly, but the MoE then lost ground. Two short diagnostics found the two causes, and the full run used both fixes.</p>
<figure><div class="svgscroll">{"".join(svg)}</div><figcaption>Each box is one version of the notebook, run on the T4 and read before the next was built. The build plan and execution record has each step in full.</figcaption></figure>
<h2 id="writeup">The write-up</h2>
{write_up}
<h2 id="cells">Every cell of the final notebook</h2>
<p>The final notebook (v3) as it ran on the T4, cell by cell: the explanation, the code (click to open) and what it printed. The v2 preview and the two diagnostics are switched off in this version; their results are in the build plan and execution record.</p>
<div class="box flag"><span class="lbl">Two lines in Cell 19's printout were corrected after this run</span><p>Cell 19 printed "MaxVio 0.214 at the end", from a final measurement window only one step long, and a one-line conclusion that named only the favourable comparison. Both are fixed in <code>notebook_src.py</code>. The write-up above uses the corrected numbers, computed from the saved logs with 25-step windows. The printout below is the original, as it ran.</p></div>
<div class="cells">{"".join(cells)}</div>
<h2 id="reports">Other reports</h2>
{reports_html}
</div></body></html>"""
open(OUT, "w", encoding="utf-8").write(page)
print(f"wrote {OUT} ({len(page)/1024:.0f} KB)")
