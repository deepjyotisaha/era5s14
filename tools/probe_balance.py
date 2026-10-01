# Scratch diagnosis: why does the preview concentrate load? Runs the notebook up to Cell 13, then
# short MoE runs (tiny CPU profile) under different settings, logging MaxVio / dead / router logits.
import os, sys, io, contextlib, json
os.chdir(sys.argv[1])
src = open("notebook_src.py", encoding="utf-8").read()
src = src[:src.index("# %% [markdown]\n# ## Cell 14")]
g = {"__name__": "nb"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "nb", "exec"), g)
import torch, numpy as np
from dataclasses import replace
STEPS = int(sys.argv[2])
def run(label, gamma, prob_steps, router_lr_mult=1.0, seed=0):
    torch.manual_seed(1000 + seed)
    g["XC"] = replace(g["XC"], bias_gamma=gamma, prob_steps=prob_steps)
    m = g["make_moe"](); m.load_state_dict(g["CONV"]["model"])
    for b in m.blocks: b.mlp.probabilistic = prob_steps > 0
    opt = g["make_opt"](m, 1e-3); opt.load_state_dict(g["CONV"]["opt"])
    if router_lr_mult != 1.0:
        # put routers in their own group
        rp = [b.mlp.router.weight for b in m.blocks]
        for grp in opt.param_groups:
            grp["params"] = [p for p in grp["params"] if all(p is not r for r in rp)]
        opt.add_param_group({"params": rp, "weight_decay": 0.1, "lr": 1e-3 * router_lr_mult})
    B = g["TC"].batch; out = []
    win = torch.zeros(len(m.blocks), g["N_EXP"])
    for s in range(STEPS):
        x, y = g["TRAIN_W"].get(g["CONT_START"] + s * B, B)
        loss = m(x, y) + 1e-4 * sum(b.mlp.aux for b in m.blocks) / len(m.blocks)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
        for li, b in enumerate(m.blocks):
            win[li] += b.mlp.counts; g["update_bias"](b.mlp, gamma); b.mlp.probabilistic = s + 1 < prob_steps
        if (s + 1) % 40 == 0:
            mv = np.mean([g["maxvio"](win[li], 4, 8) for li in range(len(m.blocks))])
            zs = max(b.mlp.router.weight.norm(dim=1).max().item() for b in m.blocks)
            out.append((s + 1, round(float(mv), 2), int((win == 0).sum()), round(zs, 3)))
            win.zero_()
    v = g["evaluate"](m)
    print(f"{label:<34} val {v:.4f}  (step, MaxVio, dead, max router-row norm): {out}")
    RES.append({"label": label, "seed": seed, "gamma": gamma, "switch": prob_steps, "router_lr_mult": router_lr_mult, "val": v, "log": out})
RES = []
for sd in (0, 1):
    print(f"--- seed {sd}")
    run("gamma 1e-3, switch at 80", 1e-3, 80, seed=sd)
    run("gamma 1e-3, probabilistic always", 1e-3, 10**9, seed=sd)
    run("gamma 1e-2, switch at 80", 1e-2, 80, seed=sd)
    run("gamma 1e-3, router lr x0.1, sw 80", 1e-3, 80, 0.1, seed=sd)

json.dump({"note": "tiny CPU profile (4 x 64 model); MoE continued from the converted R1; windows of 40 steps", "steps": STEPS, "runs": RES}, open("diagnosis/balance_cpu.json", "w"), indent=1)
print("saved diagnosis/balance_cpu.json")
