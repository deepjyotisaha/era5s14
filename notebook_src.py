# %% [markdown]
# # Session 14 — From a dense model to a mixture of experts
#
# **The brief.** Train a dense ("linear") model and convert it into an MoE. Model size and data
# are our call, but the MoE must be shown to keep training and keep reducing the loss.
#
# **The decisions behind this notebook** are recorded, with their reasons, in
# `assignment_plan.html`. In one line each:
#
# | decision | choice |
# |---|---|
# | model | session 13's 16 layers × width 320, 8K-token tokenizer (22.4M parameters) |
# | data | FineWeb sample-10BT; R1 reads the first 50M tokens, R2 and R3 the next 50M |
# | conversion | option E: each FFN cut into 4 slices of 320, each copied 8× → 32 experts in 4 families, one per family per token |
# | clones | probabilistic selection inside each family for the first ~300 steps |
# | control | the dense model continues for the same 50M tokens (R2) |
# | schedule | warmup-stable-decay: R1 never decays, R2 and R3 decay in their last 20% |
# | fixed | Colab T4, fp16, sequence 512, batch 32, lr 1e-3 |
#
# ### This notebook grows in three versions
#
# ```
# v1  (done 29 Sep)  data, the dense model, run R1 ............ R1 reloads from Drive
# v2  (done 29 Sep)  + the MoE layer, the converter, the gates, a 450-step preview
# v2.1 (done 29 Sep) + a diagnostic: 7 short runs to find why the MoE lost ground
# v2.2 (done 29 Sep) + a second diagnostic: 6 runs testing a smaller learning rate for the experts
# v3   (this one)    + R2, R2s and R3 (50M tokens each), analysis and figures: the full comparison
# ```
#
# Each version only **adds cells at the end**. The finished R1 is cached on your Drive, so v2
# and v3 load it instead of training it again. For that reason the dense model and the training
# loop (Cells 5 and 6) are frozen after v1.
#
# ### Running this
#
# Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all. Everything needed for the
# write-up lands in `outputs/` on your Drive and is zipped by the last cell. If Colab
# disconnects, run all again: finished stages reload from the cache, and an interrupted run
# resumes from its last checkpoint.

# %% [markdown]
# ## Cell 1 — environment
#
# ### What this block does
# Imports, fixes the seed, and decides the **profile**: `full` on a GPU (the real experiment),
# `smoke` on a GPU (real model and data, a few million tokens, to test every code path), or
# `cpu`, a tiny version of everything used to test the notebook on a laptop.
#
# ### What you should see
# A version box, the GPU name, and `profile: full`. If it says `cpu`, you are not on a GPU
# runtime and none of the numbers are the real experiment.

# %%
import sys, subprocess

def _pip(mod, pkg=None):
    """Install only if the import actually fails - Colab already has most of this."""
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg or mod], check=False)

_pip("tokenizers")
_pip("datasets")

import os, math, time, json, zipfile, random, platform, hashlib, inspect, io
from contextlib import redirect_stdout, nullcontext
from dataclasses import dataclass, asdict, replace

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTEBOOK_VERSION = "v3"
SEED = 1337
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ┌──────────────────────────────────────────────────────────────────────────────────────┐
# │ MODE: "full" is the experiment. "smoke" runs the real model on ~2M tokens per run to   │
# │ check every GPU code path in a few minutes (plus the one-time data preparation).       │
# └──────────────────────────────────────────────────────────────────────────────────────┘
MODE = "full"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROFILE = os.environ.get("ERA5_PROFILE") or (MODE if DEVICE == "cuda" else "cpu")
AMP = DEVICE == "cuda"                  # fp16 autocast + loss scaling (a T4 has no bf16)
AMP_DTYPE = torch.float16

if DEVICE == "cuda":
    _p = torch.cuda.get_device_properties(0)
    GPU_NAME, VRAM_GB = _p.name, _p.total_memory / 1e9
else:
    GPU_NAME, VRAM_GB = None, 0.0

_BANNER = [f"NOTEBOOK {NOTEBOOK_VERSION}  -  Session 14, dense model to mixture of experts",
           f"profile: {PROFILE}   device: {DEVICE}" + (f"   gpu: {GPU_NAME}" if GPU_NAME else "")]
if PROFILE == "cpu":
    _BANNER += ["", "!! REDUCED PROFILE: tiny model, tiny data.",
                "!! Use it to test the notebook. None of these numbers are the experiment."]
elif PROFILE == "smoke":
    _BANNER += ["", "SMOKE TEST: real model and data, ~2M tokens per run.",
                "Checks every GPU code path. Set MODE = \"full\" afterwards for the experiment."]
_w = max(len(x) for x in _BANNER) + 4
print("+" + "-" * _w + "+")
for _line in _BANNER:
    print("|  " + _line.ljust(_w - 2) + "|")
print("+" + "-" * _w + "+\n")
print(f"python : {platform.python_version()}")
print(f"torch  : {torch.__version__}")
if GPU_NAME:
    print(f"gpu    : {GPU_NAME}, {VRAM_GB:.1f} GB")

R = {"meta": {"notebook_version": NOTEBOOK_VERSION, "torch": torch.__version__,
              "python": platform.python_version(), "seed": SEED, "device": DEVICE,
              "profile": PROFILE, "gpu": GPU_NAME, "vram_gb": VRAM_GB,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")},
     "gates": {}}

# %% [markdown]
# ## Cell 2 — configuration
#
# ### What this block does
# Every number that shapes the experiment, in one place, printed.
#
# ### Why these values
# The model and the optimiser settings are session 13's, which you approved reusing ("similar
# to last assignment"). Two things are new:
#
# - **The schedule is warmup-stable-decay (WSD).** The learning rate rises for about 200
#   steps, then stays flat. It is only lowered in the last 20% of R2 and R3. R1 never decays,
#   so the conversion in v2 happens while the model can still learn quickly. Session 13's cosine
#   schedule would have shrunk the learning rate to 10% by then.
# - **Twice the data.** R1 reads the first 50M tokens. R2 and R3 continue on the *next* 50M,
#   so the continuations never see a repeated sentence.

# %%
@dataclass(frozen=True)
class ModelCfg:
    vocab: int = 8192
    d: int = 320            # width: how many numbers carry one token between layers
    n_layer: int = 16       # depth
    n_head: int = 5         # 320 = 5 heads x 64
    seq: int = 512
    mlp_mult: int = 4       # feed-forward middle = 4 x width = 1,280 neurons

@dataclass(frozen=True)
class TrainCfg:
    tokens: int = 50_000_000        # per run: R1, R2 and R3 each read 50M tokens
    batch: int = 32
    lr: float = 1e-3
    min_lr_frac: float = 0.1        # where the decay ends, as a fraction of lr
    warmup_steps: int = 200
    decay_frac: float = 0.2         # R2 and R3 decay over their last 20%; R1 never decays
    weight_decay: float = 0.1
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every_tokens: int = 2_500_000
    eval_batches: int = 20          # 20 x 32 sequences of 512 = 327k validation tokens
    log_every: int = 25
    loss_chunk: int = 8192          # tokens per slice of the chunked loss
    ckpt_minutes: float = 8.0

@dataclass(frozen=True)
class DataCfg:
    train_tokens: int = 100_500_000  # R1's 50M, then the continuations' 50M, plus a margin
    val_tokens: int = 1_000_000
    tokenizer_chars: int = 40_000_000
    fineweb_name: str = "sample-10BT"

MC, TC, DC = ModelCfg(), TrainCfg(), DataCfg()
if PROFILE == "cpu":
    MC = replace(MC, vocab=1024, d=64, n_layer=4, n_head=4, seq=128)
    TC = replace(TC, tokens=300_000, batch=8, warmup_steps=20, eval_every_tokens=75_000,
                 eval_batches=4, log_every=10, loss_chunk=512)
    DC = replace(DC, train_tokens=650_000, val_tokens=40_000, tokenizer_chars=1_000_000)
elif PROFILE == "smoke":
    TC = replace(TC, tokens=2_000_000, warmup_steps=20, eval_every_tokens=500_000, eval_batches=5)
TC = replace(TC, ckpt_minutes=float(os.environ.get("ERA5_CKPT_MIN", TC.ckpt_minutes)))

STEPS = TC.tokens // (TC.batch * MC.seq)          # optimiser steps per run

def param_breakdown(mc=MC):
    """Where the dense model's weights live (the output layer shares the embedding)."""
    emb = mc.vocab * mc.d + mc.seq * mc.d
    attn = mc.n_layer * 4 * mc.d * mc.d
    ffn = mc.n_layer * 2 * mc.d * (mc.mlp_mult * mc.d)
    norms = mc.n_layer * 4 * mc.d + 2 * mc.d
    return {"embeddings": emb, "attention": attn, "feed-forward": ffn, "layer norms": norms,
            "total": emb + attn + ffn + norms}

PB = param_breakdown()
print(f"model   : {MC.n_layer} layers x width {MC.d}, {MC.n_head} heads, FFN {MC.mlp_mult*MC.d} neurons, "
      f"vocab {MC.vocab:,}, sequence {MC.seq}")
for k, v in PB.items():
    print(f"          {k:<13} {v/1e6:6.2f}M" + ("" if k == "total" else f"  ({v/PB['total']*100:4.1f}%)"))
print(f"training: {TC.tokens/1e6:.1f}M tokens per run, batch {TC.batch} x {MC.seq} "
      f"-> {STEPS:,} steps; lr {TC.lr}, warm-up {TC.warmup_steps} steps, "
      f"decay over the last {TC.decay_frac:.0%} of R2/R3 only")
R["config"] = {"model": asdict(MC), "train": asdict(TC), "data": asdict(DC), "params": PB, "steps": STEPS}

# %% [markdown]
# ## Cell 3 — storage and caching
#
# ### What this block does
# Mounts Google Drive and sets up two folders. **outputs/** holds what you download: logs,
# plots, `summary.json`. **cache/** holds machinery: the tokenised data, finished runs,
# mid-run checkpoints and R1's final checkpoint, which v2 and v3 start from.
#
# ### How it works
# A finished run is saved together with a hash of its settings **and of the code that produced
# it**. Change a setting or edit a function, and the old result is recomputed instead of
# silently replayed. The version number is deliberately *not* part of the hash, so moving from
# v1 to v2 does not retrain R1.

# %%
RESET_CACHE = False        # True forces every run to recompute

try:
    from google.colab import drive as _colab_drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REQUIRE_DRIVE = True       # v3: stop here if Drive does not mount (a disconnect would lose the long runs)

DRIVE_OK = False
if IN_COLAB:
    _err = None
    for _force in (False, True):             # one retry with a forced remount
        try:
            _colab_drive.mount("/content/drive", force_remount=_force)
            ROOT, DRIVE_OK = "/content/drive/MyDrive/era5_session14", True
            break
        except Exception as e:
            _err = e
            print(f"Drive mount attempt {'2 (forced)' if _force else '1'} failed: {type(e).__name__}: {e}")
    if not DRIVE_OK:
        ROOT = "/content/era5_session14"
        print("!" * 74)
        print(f"!! DRIVE MOUNT FAILED ({type(_err).__name__}: {_err}).")
        print("!! Without Drive, R1 is retrained and a disconnect loses every finished run.")
        print("!! Fix: Runtime -> Disconnect and delete runtime, then Run all again, and in the")
        print("!! Google pop-up choose the SAME account as before and allow access.")
        print("!" * 74)
        if REQUIRE_DRIVE:
            raise RuntimeError("Drive is not mounted (REQUIRE_DRIVE = True). See the message above; "
                               "set REQUIRE_DRIVE = False only if you accept losing the runs on a disconnect.")
    OUT_DIR, CACHE_DIR = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "cache")
else:
    ROOT, OUT_DIR, CACHE_DIR = ".", "outputs", "checkpoints"
DATA_DIR = os.path.join(CACHE_DIR, "data_" + hashlib.sha1(
    json.dumps({"d": asdict(DC), "v": MC.vocab}, sort_keys=True).encode()).hexdigest()[:8])
if PROFILE != "full":                     # smoke and CPU results never mix with the real ones
    OUT_DIR = os.path.join(OUT_DIR, PROFILE)
    CACHE_DIR = os.path.join(CACHE_DIR, PROFILE)
LOG_DIR = os.path.join(OUT_DIR, "logs")
for _d in (OUT_DIR, CACHE_DIR, DATA_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

def out(name):   return os.path.join(OUT_DIR, name)
def cache(name): return os.path.join(CACHE_DIR, name)

def _h(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]

BASE_KEY = _h({"model": asdict(MC), "train": {k: v for k, v in asdict(TC).items() if k != "ckpt_minutes"},
               "data": asdict(DC), "seed": SEED, "device": DEVICE, "profile": PROFILE})

def _src_hash(*objs):
    """Hash of the SOURCE of the functions and classes a cached result depends on."""
    parts = []
    for o in objs:
        try:
            parts.append(inspect.getsource(o))
        except (OSError, TypeError):
            parts.append("")
    return hashlib.sha1("".join(parts).encode()).hexdigest()[:10]

def _atomic_save(obj, path, as_json=False):
    tmp = path + ".tmp"
    if as_json:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=float)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)

print(f"in colab   : {IN_COLAB}  (drive mounted: {DRIVE_OK})")
print(f"outputs    : {OUT_DIR}   <- download this folder")
print(f"cache      : {CACHE_DIR}   <- R1's checkpoint lives here; keep it for v2 and v3")
print(f"base key   : {BASE_KEY}")

# %% [markdown]
# ## Cell 4 — data: FineWeb, a tokenizer of our own, tokenised once
#
# ### What this block does
# Streams FineWeb (sample-10BT) from Hugging Face, trains an 8,192-token byte-pair tokenizer on
# the start of it, and turns the text into two flat files of token ids: 100.5M training tokens
# and 1M validation tokens from **separate documents**. Everything is saved to Drive, so this
# runs once (it is the slowest part of v1, roughly 10–20 minutes).
#
# ### How the runs split the data
# The training file is cut into windows of 513 tokens (512 inputs plus the next token as the
# target), and their order is shuffled once with a fixed seed. R1 reads the first 97,632
# windows of that order (50M tokens). R2 and R3 both read the next 97,632, in the same order, so
# the control and the MoE see exactly the same new text.
#
# ### What you should see
# The token counts, a sample sentence round-tripped through the tokenizer, and about 3.7
# characters per token (session 13 measured 3.73).

# %%
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

EOT = "<|eot|>"

def _fineweb_texts():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name=DC.fineweb_name, split="train",
                      streaming=True)
    for row in ds:
        yield row["text"]

def _fallback_texts():
    """Only used by the CPU profile when the FineWeb stream is unreachable."""
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    text = open(path, encoding="utf-8").read()
    for _ in range(40):                      # small corpus: repeat it (CPU profile only)
        for block in text.split("\n\n"):
            if len(block) > 64:
                yield block

def build_data():
    meta_path = os.path.join(DATA_DIR, "meta.json")
    want = {"data": asdict(DC), "vocab": MC.vocab}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        if meta.get("want") == want:
            print(f"[cached] tokenised data found in {DATA_DIR}")
            return meta
    t0 = time.time()
    source = "fineweb"
    try:
        stream = _fineweb_texts()
        first = next(stream)
    except Exception as e:
        if PROFILE != "cpu":
            raise
        print(f"FineWeb unreachable ({type(e).__name__}); CPU profile falls back to Tiny Shakespeare")
        source, stream = "tinyshakespeare", _fallback_texts()
        first = next(stream)

    def docs():
        yield first
        yield from stream

    it = docs()
    # 1) validation documents first, kept aside as raw text (6 chars per token is generous)
    val_texts, n = [], 0
    for t in it:
        val_texts.append(t); n += len(t)
        if n >= DC.val_tokens * 6:
            break
    # 2) the tokenizer is trained on the next documents, which also start the training set
    tok_texts, n = [], 0
    for t in it:
        tok_texts.append(t); n += len(t)
        if n >= DC.tokenizer_chars:
            break
    print(f"source {source}: {len(val_texts):,} validation docs, {len(tok_texts):,} docs "
          f"({sum(len(x) for x in tok_texts)/1e6:.1f}M chars) for the tokenizer")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=MC.vocab, special_tokens=[EOT], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(tok_texts, trainer=trainer)
    tok.save(os.path.join(DATA_DIR, "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    print(f"tokenizer trained: vocabulary {tok.get_vocab_size():,} in {time.time()-t0:.0f}s")

    def encode_to(texts_iter, limit, path):
        chunks, n, chars, batch = [], 0, 0, []
        def flush(batch):
            nonlocal n, chars
            enc = tok.encode_batch(batch)
            for e, t in zip(enc, batch):
                ids = np.asarray(e.ids + [eot], dtype=np.uint16)
                chunks.append(ids); n += len(ids); chars += len(t)
        for t in texts_iter:
            batch.append(t)
            if len(batch) == 512:
                flush(batch); batch = []
                if n >= limit:
                    break
        if batch and n < limit:
            flush(batch)
        arr = np.concatenate(chunks)[:limit]
        if len(arr) < limit:
            raise RuntimeError(f"only {len(arr):,} tokens available for {path}, wanted {limit:,}")
        arr.tofile(path)
        return len(arr), chars / max(n, 1)

    n_val, _ = encode_to(iter(val_texts), DC.val_tokens, os.path.join(DATA_DIR, "val.bin"))
    def train_iter():
        yield from tok_texts
        yield from it
    n_tr, cpt = encode_to(train_iter(), DC.train_tokens, os.path.join(DATA_DIR, "train.bin"))
    meta = {"want": want, "source": source, "train_tokens": n_tr, "val_tokens": n_val,
            "chars_per_token": cpt, "eot_id": eot, "vocab_size": tok.get_vocab_size(),
            "seconds": time.time() - t0}
    _atomic_save(meta, meta_path, as_json=True)
    print(f"data prepared in {meta['seconds']/60:.1f} min")
    return meta

DATA_META = build_data()
TOK = Tokenizer.from_file(os.path.join(DATA_DIR, "tokenizer.json"))
TRAIN = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
VAL = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
assert DATA_META["vocab_size"] <= MC.vocab

class Windows:
    """Fixed windows of seq+1 tokens in a fixed shuffled order. Window k is always the same."""
    def __init__(self, data, seq, shuffle, seed=SEED):
        self.data, self.seq = data, seq
        n = (len(data) - 1) // seq
        self.order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    def __len__(self):
        return len(self.order)
    def get(self, start, count, device=DEVICE):
        idx = self.order[start:start + count]
        x = np.stack([self.data[i * self.seq: i * self.seq + self.seq + 1] for i in idx]).astype(np.int64)
        x = torch.from_numpy(x)
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        return x[:, :-1], x[:, 1:]

TRAIN_W = Windows(TRAIN, MC.seq, shuffle=True)
VAL_W = Windows(VAL, MC.seq, shuffle=False)
R1_WINDOWS = STEPS * TC.batch                    # R1 reads windows [0, R1_WINDOWS)
CONT_START = R1_WINDOWS                          # R2 and R3 read [R1_WINDOWS, 2 x R1_WINDOWS)
assert 2 * R1_WINDOWS <= len(TRAIN_W), "not enough training windows for R1 plus the continuations"
_sample = "A mixture of experts runs only a few small networks for each token."
_enc = TOK.encode(_sample)
print(f"source           : {DATA_META['source']}")
print(f"training tokens  : {len(TRAIN):,}  ({len(TRAIN_W):,} windows of {MC.seq}+1)")
print(f"  R1 reads       : windows 0 .. {R1_WINDOWS-1:,}  ({R1_WINDOWS*MC.seq/1e6:.1f}M tokens)")
print(f"  R2, R3 read    : windows {CONT_START:,} .. {2*R1_WINDOWS-1:,}  (the next {R1_WINDOWS*MC.seq/1e6:.1f}M)")
print(f"validation tokens: {len(VAL):,}  (separate documents)")
print(f"chars per token  : {DATA_META['chars_per_token']:.2f}")
print(f"round trip       : {len(_enc.ids)} tokens -> {TOK.decode(_enc.ids)!r}")
R["data"] = {k: DATA_META[k] for k in ("source", "train_tokens", "val_tokens", "chars_per_token", "vocab_size")}

# %% [markdown]
# ## Cell 5 — the dense model  (frozen after v1)
#
# ### What this block does
# A small GPT: token and position embeddings, 16 pre-norm blocks, a final norm, and an output
# layer that shares the embedding matrix. Each block is attention followed by the
# **feed-forward network (FFN)**: `up` widens 320 → 1,280, GELU, `down` narrows 1,280 → 320.
#
# ### Why the FFN is written this way
# The FFN is the part v2 turns into experts. It is kept as its own module, `block.mlp`, with
# two plain matrices, so v2 can swap it for an MoE layer **without changing any code in this
# cell**. That matters: the cache recognises R1 by the source of this cell, and editing it
# would retrain R1.
#
# The loss is computed in slices of 8,192 tokens and recomputed in the backward pass, as in
# session 13, so the full table of scores over the vocabulary never sits in memory at once.

# %%
class Attention(nn.Module):
    def __init__(self, mc):
        super().__init__()
        self.n_head = mc.n_head
        self.qkv = nn.Linear(mc.d, 3 * mc.d, bias=False)
        self.proj = nn.Linear(mc.d, mc.d, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, mc):
        super().__init__()
        self.up = nn.Linear(mc.d, mc.mlp_mult * mc.d, bias=False)
        self.down = nn.Linear(mc.mlp_mult * mc.d, mc.d, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class Block(nn.Module):
    def __init__(self, mc):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(mc.d), Attention(mc)
        self.ln2, self.mlp = nn.LayerNorm(mc.d), MLP(mc)
    def forward(self, x):
        x = x + self.attn(self.ln1(x)).to(x.dtype)
        return x + self.mlp(self.ln2(x)).to(x.dtype)

def _ce_slice(h, w, y):
    logits = h @ w.t()
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return F.cross_entropy(logits, y, reduction="sum")

def chunked_ce(h, w, y, chunk):
    """Mean cross-entropy, computed slice by slice and recomputed in the backward pass."""
    h, y = h.reshape(-1, h.size(-1)), y.reshape(-1)
    total = h.new_zeros((), dtype=torch.float64 if h.dtype == torch.float64 else torch.float32)
    for i in range(0, h.size(0), chunk):
        if torch.is_grad_enabled():
            total = total + checkpoint(_ce_slice, h[i:i + chunk], w, y[i:i + chunk], use_reentrant=False)
        else:
            total = total + _ce_slice(h[i:i + chunk], w, y[i:i + chunk])
    return total / y.numel()

class GPT(nn.Module):
    def __init__(self, mc):
        super().__init__()
        self.mc = mc
        self.wte = nn.Embedding(mc.vocab, mc.d)
        self.wpe = nn.Embedding(mc.seq, mc.d)
        self.blocks = nn.ModuleList(Block(mc) for _ in range(mc.n_layer))
        self.ln_f = nn.LayerNorm(mc.d)
        self.apply(self._init)
        for b in self.blocks:        # GPT-2 style: shrink the layers that write into the stream
            for w in (b.attn.proj.weight, b.mlp.down.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * mc.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, idx):
        T = idx.size(1)
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, idx, targets, chunk=None):
        return chunked_ce(self.hidden(idx), self.wte.weight, targets, chunk or TC.loss_chunk)

    @torch.no_grad()
    def logits(self, idx):
        return self.hidden(idx) @ self.wte.weight.t()

def make_dense():
    torch.manual_seed(SEED)
    return GPT(MC).to(DEVICE)

_m = make_dense()
_n = sum(p.numel() for p in _m.parameters())
assert _n == PB["total"], (_n, PB["total"])
print(f"parameters: {_n:,}  (matches the breakdown in Cell 2)")
del _m

# %% [markdown]
# ## Cell 6 — the training loop  (frozen after v1)
#
# ### What this block does
# One function trains any model for a given number of steps and returns its history. R1 uses
# it now; R2 and R3 will use the same function in v3, which is why it takes a few arguments
# that R1 does not need yet:
#
# | argument | what R1 uses | what v3 will use |
# |---|---|---|
# | `init` | nothing: random weights | R1's final weights (and optimiser state) |
# | `window_start` | 0 | the first window after R1's data |
# | `warmup`, `decay` | 200 steps up, no decay | no warm-up (already warm), decay over the last 20% |
# | `extra_loss`, `after_step` | not used | the MoE's small balancing loss and bias update |
#
# ### How it works
# - AdamW, weight decay 0.1 on the weight matrices only, gradient clipping at 1.0.
# - fp16 autocast with loss scaling on the GPU.
# - Validation loss on the same 327k held-out tokens every 2.5M training tokens.
# - **Speed** is measured on the GPU clock over training steps only (the first 10 steps and
#   every validation pass are excluded). **Peak memory** is `torch.cuda.max_memory_allocated()`.
# - A checkpoint is saved every 8 minutes; after a disconnect the run resumes from it.
# - At the end, the final weights, optimiser moments and loss scaler are saved as the run's
#   **final checkpoint**. R1's final checkpoint is the starting point for everything in v2 and v3.

# %%
def make_opt(model, lr):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": TC.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = {"fused": True} if DEVICE == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, TC.beta2), eps=1e-8, **kw)

def lr_at(step, steps, lr, warmup, decay):
    """Warmup-stable-decay: linear up over `warmup` steps, flat, then linear down to
    min_lr_frac x lr over the last `decay` steps (decay = 0 means it never comes down)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    if decay and step >= steps - decay:
        t = (step - (steps - decay) + 1) / decay
        return lr * (1 - (1 - TC.min_lr_frac) * t)
    return lr

def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def evaluate(model):
    model.eval()
    ctx = torch.autocast("cuda", dtype=AMP_DTYPE) if AMP else nullcontext()
    tot = 0.0
    for i in range(TC.eval_batches):
        x, y = VAL_W.get(i * TC.batch, TC.batch)
        with ctx:
            tot += model(x, y).item()
    model.train()
    return tot / TC.eval_batches

def _ckpt_path(key):  return cache(f"ckpt_{BASE_KEY}_{key}.pt")
def _final_path(key): return cache(f"final_{BASE_KEY}_{key}.pt")
def _run_path(key):   return cache(f"run_{BASE_KEY}_{key}.json")

def train_run(key, make_model, steps, warmup, decay, window_start=0, init=None,
              extra_loss=None, after_step=None, lr=None):
    batch, lr = TC.batch, lr or TC.lr
    if window_start + steps * batch > len(TRAIN_W):
        raise ValueError(f"{key}: needs windows up to {window_start + steps*batch:,}, only {len(TRAIN_W):,} exist")
    model = make_model()
    opt = make_opt(model, lr)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    if init is not None:
        model.load_state_dict(init["model"])
        if init.get("opt") is not None:
            opt.load_state_dict(init["opt"])
        if init.get("scaler") is not None and AMP:
            scaler.load_state_dict(init["scaler"])
    ctx = torch.autocast("cuda", dtype=AMP_DTYPE) if AMP else nullcontext()
    eval_every = max(1, TC.eval_every_tokens // (batch * MC.seq))
    hist = {"key": key, "steps": steps, "batch": batch, "lr": lr, "warmup": warmup, "decay": decay,
            "window_start": window_start, "tokens_per_step": batch * MC.seq,
            "train": [], "val": [], "lr_trace": [], "train_seconds": 0.0, "timed_tokens": 0, "extra": {}}
    start = 0
    ck = _ckpt_path(key)
    if not RESET_CACHE and os.path.exists(ck):
        try:
            blob = torch.load(ck, map_location=DEVICE, weights_only=False)
            model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"])
            scaler.load_state_dict(blob["scaler"]); hist = blob["hist"]; start = blob["step"] + 1
            print(f"    [resume] '{key}' from step {start:,} of {steps:,}")
        except Exception as e:
            print(f"    [resume failed] '{key}': {type(e).__name__}, starting over")
    if start == 0:                               # loss before any step: where this run begins
        hist["val"].append({"step": 0, "tokens": 0, "loss": evaluate(model), "train_seconds": 0.0})
        print(f"    {key:<14} step      0/{steps:,}  val {hist['val'][0]['loss']:.4f}  (before training)")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    last_ck, win_t0, win_tok, run_loss = time.time(), None, 0, []
    for step in range(start, steps):
        cur_lr = lr_at(step, steps, lr, warmup, decay)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = TRAIN_W.get(window_start + step * batch, batch)
        if step == start + 10:
            _sync(); win_t0, win_tok = time.time(), 0
        with ctx:
            loss = model(x, y)
            total = loss + extra_loss(model) if extra_loss else loss
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TC.grad_clip)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        if after_step:
            after_step(model, step, hist)
        run_loss.append(loss.item())
        if win_t0 is not None:
            win_tok += batch * MC.seq
        last = step == steps - 1
        is_eval = (step + 1) % eval_every == 0 or last
        if ((step + 1) % TC.log_every == 0 or is_eval) and run_loss:
            hist["train"].append({"step": step + 1, "tokens": (step + 1) * batch * MC.seq,
                                  "loss": float(np.mean(run_loss))})
            hist["lr_trace"].append({"step": step + 1, "lr": cur_lr})
            run_loss = []
        if is_eval:
            if win_t0 is not None:
                _sync(); hist["train_seconds"] += time.time() - win_t0; hist["timed_tokens"] += win_tok
            v = evaluate(model)
            hist["val"].append({"step": step + 1, "tokens": (step + 1) * batch * MC.seq, "loss": v,
                                "train_seconds": hist["train_seconds"]})
            tps = hist["timed_tokens"] / hist["train_seconds"] if hist["train_seconds"] else float("nan")
            print(f"    {key:<14} step {step+1:>6,}/{steps:,}  tokens {(step+1)*batch*MC.seq/1e6:5.1f}M  "
                  f"lr {cur_lr:.2e}  train {hist['train'][-1]['loss']:.4f}  val {v:.4f}  {tps:,.0f} tok/s")
            if win_t0 is not None:
                _sync(); win_t0, win_tok = time.time(), 0
        if time.time() - last_ck > TC.ckpt_minutes * 60 and not last:
            if win_t0 is not None:
                _sync(); hist["train_seconds"] += time.time() - win_t0; hist["timed_tokens"] += win_tok
                win_t0, win_tok = time.time(), 0
            _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                          "scaler": scaler.state_dict(), "hist": hist, "step": step}, ck)
            last_ck = time.time()
    hist["tokens_per_second"] = hist["timed_tokens"] / hist["train_seconds"] if hist["train_seconds"] else None
    hist["peak_mem_bytes"] = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else None
    hist["final_val"] = hist["val"][-1]["loss"]
    hist["final_train"] = hist["train"][-1]["loss"]
    _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict()},
                 _final_path(key))
    del model, opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return hist

def _replay(hist):
    """Print a reloaded run's progress from its saved log."""
    key, tr = hist["key"], {p["step"]: p["loss"] for p in hist["train"]}
    print(f"    [cached run] '{key}' reloaded, not retrained; progress replayed from its saved log:")
    for p in hist["val"]:
        earlier = [s for s in tr if s <= p["step"]]
        t = f"{tr[max(earlier)]:.4f}" if earlier else "  -   "
        print(f"    {key:<14} step {p['step']:>6,}/{hist['steps']:,}  tokens {p['tokens']/1e6:5.1f}M  "
              f"train {t}  val {p['loss']:.4f}  {p['train_seconds']/60:5.1f} train-min")
    tps, pk = hist.get("tokens_per_second"), hist.get("peak_mem_bytes")
    print(f"    {key:<14} measured: " + (f"{tps:,.0f} tokens/s" if tps else "speed not timed")
          + (f", peak memory {pk/2**20:,.0f} MiB" if pk else ""))

DENSE_CODE = (Attention, MLP, Block, GPT, chunked_ce, _ce_slice, make_dense, train_run, lr_at, make_opt, evaluate, Windows)

def cached_run(key, code, **kw):
    """Train once per configuration and code version; a finished run reloads instantly.
    `code` lists the functions and classes whose source defines this run."""
    path = _run_path(key)
    want = {"src": _src_hash(*code), "kw": {k: (v.__name__ if callable(v) else v) for k, v in kw.items()
                                             if k != "init"}}
    want = json.loads(json.dumps(want, default=str))
    if not RESET_CACHE and os.path.exists(path) and os.path.exists(_final_path(key)):
        blob = json.load(open(path, encoding="utf-8"))
        if blob.get("want") == want:
            _replay(blob["hist"])
            return blob["hist"]
        print(f"    [stale] '{key}': settings or code changed since it was trained, training again")
    hist = train_run(key, **kw)
    _atomic_save({"want": want, "hist": hist}, path, as_json=True)
    _atomic_save(hist, os.path.join(LOG_DIR, f"{key}.json"), as_json=True)
    if os.path.exists(_ckpt_path(key)):
        os.remove(_ckpt_path(key))
    return hist

def load_final(key):
    return torch.load(_final_path(key), map_location=DEVICE, weights_only=False)

print(f"eval every {TC.eval_every_tokens/1e6:.1f}M tokens on {TC.eval_batches*TC.batch*MC.seq:,} "
      f"validation tokens; checkpoint every {TC.ckpt_minutes:g} min")

# %% [markdown]
# ## Cell 7 — R1: train the dense model on the first 50M tokens
#
# ### What this block does
# Trains the dense model from random weights: 200 warm-up steps, then a flat learning rate of
# 1e-3 to the end. There is no decay, on purpose (see Cell 2). The final weights, optimiser state
# and loss scaler are saved as R1's final checkpoint.
#
# ### What you should see
# A validation line every 2.5M tokens: about 9.0 before training (ln 8,192 = 9.01, a model that
# knows nothing), dropping fast at first and then more slowly. Session 13's comparable run
# ended at 4.11, but its learning rate had decayed to 10%. Without the decay, expect R1 to end a
# little higher, around 4.2–4.4. The speed should be close to session 13's 53,000 tokens/s.

# %%
RUN_R1 = "R1_dense"
R1 = cached_run(RUN_R1, DENSE_CODE, make_model=make_dense, steps=STEPS,
                warmup=TC.warmup_steps, decay=0, window_start=0)
_first, _last = R1["val"][0]["loss"], R1["final_val"]
print(f"\nR1: validation loss {_first:.4f} before training -> {_last:.4f} after {STEPS:,} steps "
      f"({(_first - _last):.3f} lower)")
if R1.get("tokens_per_second"):
    print(f"    speed {R1['tokens_per_second']:,.0f} tokens/s"
          + (f", peak memory {R1['peak_mem_bytes']/2**20:,.0f} MiB" if R1.get("peak_mem_bytes") else ""))

# %% [markdown]
# ## Cell 8 — Gate: R1's checkpoint reloads to exactly the same model
#
# ### What this block does
# v2 and v3 build everything from R1's final checkpoint. This gate reloads it into a fresh
# model and checks that the validation loss comes out **exactly** as it did at the end of
# training. If this ever fails, the conversion in v2 would start from the wrong model.
#
# ### What you should see
# `[PASS]` and a difference of 0 (or a few units in the last decimal place, from GPU kernels).

# %%
class GateError(AssertionError):
    pass

def gate(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        raise GateError(msg)

_blob = load_final(RUN_R1)
_m = make_dense(); _m.load_state_dict(_blob["model"])
_v = evaluate(_m)
_diff = abs(_v - R1["final_val"])
gate(_diff < 1e-5, f"R1 checkpoint reloads: validation loss {_v:.6f} vs {R1['final_val']:.6f} "
                   f"at the end of training (difference {_diff:.1e})")
_opt_steps = [s["step"] for s in _blob["opt"]["state"].values()][:1]
gate(len(_blob["opt"]["state"]) > 0, f"optimiser moments saved too (Adam step count {int(_opt_steps[0]) if _opt_steps else 0:,})")
R["gates"]["r1_reload"] = {"val_reloaded": _v, "val_recorded": R1["final_val"], "diff": _diff}

# %% [markdown]
# ## Cell 9 — what R1 learned: its curve, its schedule, and a sample
#
# ### What this block does
# Draws R1's loss and learning rate against tokens, and lets the model continue a few prompts,
# so you can see what "loss 4.3" looks like as text.
#
# ### What you should see
# The loss falls steeply in the first few million tokens, then keeps falling slowly. The
# learning-rate panel shows the short warm-up ramp and the long flat line. The samples should
# be grammatical-looking English that wanders off topic: a 22M model on 50M tokens has learned
# the local shape of text, not facts.

# %%
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

def _save(fig, name):
    fig.tight_layout()
    fig.savefig(out(name), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot([p["tokens"]/1e6 for p in R1["train"]], [p["loss"] for p in R1["train"]],
           color="#cbc5bb", lw=1, label="train (25-step means)")
ax[0].plot([p["tokens"]/1e6 for p in R1["val"]], [p["loss"] for p in R1["val"]],
           color="#b1521f", marker="o", ms=3, label="validation")
ax[0].set_xlabel("training tokens (millions)"); ax[0].set_ylabel("loss"); ax[0].legend(frameon=False)
ax[0].set_ylim(top=min(10, R1["val"][0]["loss"] + 0.3))
ax[1].plot([p["step"] for p in R1["lr_trace"]], [p["lr"] for p in R1["lr_trace"]], color="#1f6f8b")
ax[1].set_xlabel("step"); ax[1].set_ylabel("learning rate"); ax[1].set_ylim(0, TC.lr * 1.15)
ax[1].set_title("warm-up, then flat: no decay in R1", fontsize=9)
_save(fig, "r1_loss_lr.png")

@torch.no_grad()
def sample(model, prompt, n=40, temperature=0.8, top_k=40, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = TOK.encode(prompt).ids
    model.eval()
    for _ in range(n):
        x = torch.tensor([ids[-MC.seq:]], device=DEVICE)
        logits = model.logits(x)[0, -1].float().cpu() / temperature
        v, i = torch.topk(logits, top_k)
        nxt = i[torch.multinomial(torch.softmax(v, -1), 1, generator=g)].item()
        ids.append(nxt)
    return TOK.decode(ids)

PROMPTS = ["The best way to learn a new language is", "Scientists have found that",
           "In the middle of the city there is"]
R["samples_r1"] = []
for p in PROMPTS:
    s = sample(_m, p)
    R["samples_r1"].append(s)
    print(f"\n  > {s!r}")
del _m


# %% [markdown]
# ## Cell 10 — v1 checks
#
# ### What this block does
# Re-checks v1's five promises. In v2 they still must hold, because everything below starts
# from R1. (The summary and the zip are now written by the last cell, for whichever version
# is running.)

# %%
CHECKS_V1 = [
    ("data: 100M tokens, enough windows for R1 and both continuations", 2 * R1_WINDOWS <= len(TRAIN_W)),
    ("R1: finished its step budget", R1["train"][-1]["step"] == STEPS),
    ("R1: validation loss fell", R1["final_val"] < R1["val"][0]["loss"]),
    ("R1: never decayed (last learning rate = peak)", abs(R1["lr_trace"][-1]["lr"] - TC.lr) < 1e-12),
    ("R1: checkpoint reloads to the same validation loss", R["gates"]["r1_reload"]["diff"] < 1e-5),
]
print("V1 CHECKS")
for name, ok in CHECKS_V1:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
R["runs"] = {"R1": R1}

# %% [markdown]
# ## Cell 11 — the MoE layer (option E)
#
# ### What this block does
# Defines `FamilyMoE`, the layer that replaces each FFN. It holds **32 experts in 4 families of
# 8**. Every expert is a small FFN of 320 neurons (320 → 320 → 320). For each token:
#
# 1. The **router** (a 320 × 32 matrix, run in fp32, starting at zero) gives one score per
#    expert: softmax over all 32.
# 2. In each family it **picks one expert**. It picks the highest `logit + bias`, or, during the
#    first 300 training steps, draws one at random in proportion to those values (decision D3).
# 3. The four chosen scores are **rescaled together so they sum to 4** (Section 4 of the plan;
#    the per-family rule would give the router no gradient, which Cell 13 demonstrates).
# 4. Each chosen expert runs on its tokens, and the four outputs are added with their weights.
#
# ### Balancing (lesson Section 18)
# Each expert has a **bias**, used only for choosing. After every step, an expert that got more
# than its family's average share of tokens has its bias lowered by 0.01, and a quieter one has
# it raised. The bias never enters the loss. A very small per-sequence balance loss (0.0001) is
# added as a safety net, as in DeepSeek-V3.
#
# **Why 0.01 and not the lesson's 0.001 (decision D8).** When selection switches from random to
# "take the highest", the 8 clones in a family are still nearly identical, and almost every token
# goes to whichever is a hair ahead. On the laptop, a bias step of 0.001 left the load badly
# uneven for 200+ steps after the switch (MaxVio above 1, one dead expert); 0.01 settled it
# within about 40 steps on both seeds, with no loss penalty. See the plan, Section 7.4.
#
# ### Why the start is exact
# With the router at zero, every score is 1/32. Each family's pick then gets weight
# 4 × (1/32) ÷ (4/32) = 1, so the layer adds up the four slices of the old FFN, which is exactly
# what the dense FFN computed.

# %%
@dataclass(frozen=True)
class MoECfg:
    families: int = 4
    clones: int = 8
    scale: float = 4.0          # the 4 chosen weights sum to 4 (routed scaling factor)
    bias_gamma: float = 1e-2    # bias step per training step: 10x the lesson's 1e-3 (decision D8)
    seq_aux: float = 1e-4       # small per-sequence balance loss
    prob_steps: int = 300       # probabilistic selection for the first 300 steps (D3)
    preview_steps: int = 450    # the v2 preview: 300 probabilistic + 150 hard (D7, revised)

@dataclass(frozen=True)
class Thresholds:
    g1_fp64_rel: float = 1e-10  # converted vs dense logits, fp64: the maths must match
    g1_loss_abs: float = 2e-3   # converted model's validation loss vs R1's, GPU fp16 rounding
    g3_maxvio: float = 0.3      # toy balancing must end below this ...
    g3_ratio: float = 0.5       # ... and below half of where it started

XC, TH = MoECfg(), Thresholds()
if PROFILE == "cpu":
    XC = replace(XC, prob_steps=20, preview_steps=60)
elif PROFILE == "smoke":
    XC = replace(XC, prob_steps=40, preview_steps=60)
Q = MC.mlp_mult * MC.d // XC.families      # neurons per expert: 1,280 / 4 = 320
N_EXP = XC.families * XC.clones             # 32

class FamilyMoE(nn.Module):
    def __init__(self, d, q, fam, clones, scale, joint=True):
        super().__init__()
        self.d, self.q, self.fam, self.clones, self.E = d, q, fam, clones, fam * clones
        self.scale, self.joint = scale, joint
        self.w_up = nn.Parameter(torch.empty(self.E, q, d))
        self.w_down = nn.Parameter(torch.empty(self.E, d, q))
        nn.init.normal_(self.w_up, std=0.02); nn.init.normal_(self.w_down, std=0.02)
        self.router = nn.Linear(d, self.E, bias=False)
        nn.init.zeros_(self.router.weight)
        self.register_buffer("bias", torch.zeros(self.E))       # choosing only, never in the loss
        self.register_buffer("counts", torch.zeros(self.E))     # tokens per expert this step
        self.probabilistic = False
        self.aux = None

    def route(self, x2):
        """x2 [N, d] -> logits [N, E], global expert id per family [N, fam], weights [N, fam]."""
        hi = torch.promote_types(x2.dtype, torch.float32)       # fp32, or fp64 in the gate
        with torch.autocast(x2.device.type, enabled=False):
            z = x2.to(hi) @ self.router.weight.to(hi).t()
        s = torch.softmax(z, dim=-1)
        key = (z + self.bias.to(hi)).view(-1, self.fam, self.clones)
        if self.training and self.probabilistic:
            p = torch.softmax(key.reshape(-1, self.clones), -1)
            pick = torch.multinomial(p, 1).view(-1, self.fam)
        else:
            pick = key.argmax(-1)
        eid = pick + torch.arange(self.fam, device=x2.device) * self.clones
        chosen = s.gather(1, eid)
        if self.joint:
            w = self.scale * chosen / chosen.sum(-1, keepdim=True)
        else:                       # the flawed rule: each family's pick divided by itself
            w = chosen / chosen
        return z, eid, w

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.d)
        z, eid, w = self.route(x2)
        cnt = torch.bincount(eid.reshape(-1), minlength=self.E)
        if self.training:
            self.counts += cnt.to(self.counts.dtype)
        cnt_l = cnt.tolist()
        out = torch.zeros_like(x2)
        for f in range(self.fam):
            order = torch.argsort(eid[:, f])
            xs, ys, start = x2[order], [], 0
            for c in range(self.clones):
                e, n = f * self.clones + c, cnt_l[f * self.clones + c]
                if n == 0:
                    continue
                h = F.gelu(xs[start:start + n] @ self.w_up[e].t())
                ys.append(h @ self.w_down[e].t())
                start += n
            y = torch.cat(ys)
            y = y * w[order, f].unsqueeze(1).to(y.dtype)
            out = out.index_add(0, order, y.to(out.dtype))
        if self.training and len(shp) == 3:          # per-sequence balance loss (safety net)
            B, T = shp[0], shp[1]
            local = (eid - torch.arange(self.fam, device=x.device) * self.clones).view(B, T, self.fam)
            frac = F.one_hot(local, self.clones).float().mean(1)                        # [B, fam, C]
            prob = torch.softmax(z.view(B, T, self.fam, self.clones).float(), -1).mean(1)
            self.aux = (self.clones * (frac * prob).sum(-1)).mean()
        return out.view(shp)

def make_moe(joint=True, device=None):
    torch.manual_seed(SEED)
    m = GPT(MC)
    for b in m.blocks:
        b.mlp = FamilyMoE(MC.d, Q, XC.families, XC.clones, XC.scale, joint)
    return m.to(device or DEVICE)

def make_moe_prob():
    """The model used for training: probabilistic selection switched on at the start."""
    m = make_moe()
    for b in m.blocks:
        b.mlp.probabilistic = XC.prob_steps > 0
    return m

def moe_params(model):
    tot = sum(p.numel() for p in model.parameters())
    per_exp = 2 * MC.d * Q
    idle = MC.n_layer * (N_EXP - XC.families) * per_exp       # experts a token does not use
    return tot, tot - idle

_m = make_moe()
_tot, _act = moe_params(_m)
_want_tot = PB["total"] - PB["feed-forward"] + MC.n_layer * (N_EXP * 2 * MC.d * Q + MC.d * N_EXP)
print(f"MoE model: {MC.n_layer} layers x {N_EXP} experts ({XC.families} families x {XC.clones}), "
      f"{Q} neurons each, {XC.families} per token")
print(f"  total parameters : {_tot:,}  (dense model: {PB['total']:,})")
print(f"  active per token : {_act:,}  (the dense work plus the routers)")
R["moe"] = {"cfg": asdict(XC), "total_params": _tot, "active_params": _act}
del _m

# %% [markdown]
# ## Cell 12 — the converter: R1's weights and optimiser state become the MoE
#
# ### What this block does
# Builds the MoE's starting point from R1's final checkpoint:
#
# - **Weights:** each FFN's `up` (1,280 × 320) and `down` (320 × 1,280) are cut into 4 slices of
#   320 neurons; slice *f* is copied into the 8 experts of family *f*.
# - **Optimiser state (decision D6):** AdamW's two running averages for each FFN weight are cut
#   and copied the same way, and each expert keeps R1's step count, so the first MoE step moves
#   the experts exactly as the next dense step would have moved those slices. The router starts
#   with empty averages. Everything else (attention, embeddings, norms) keeps R1's state as is.
# - The loss scaler is carried over too.
#
# ### What you should see
# The converted parameter count, and `[PASS]` for a check that the copied optimiser state
# matches R1's, slice by slice.

# %%
def _expand_up(v):      # [H, d] -> [E, q, d]
    return torch.cat([v[f * Q:(f + 1) * Q].unsqueeze(0).expand(XC.clones, -1, -1)
                      for f in range(XC.families)]).contiguous()

def _expand_down(v):    # [d, H] -> [E, d, q]
    return torch.cat([v[:, f * Q:(f + 1) * Q].unsqueeze(0).expand(XC.clones, -1, -1)
                      for f in range(XC.families)]).contiguous()

def convert_weights(dense_sd):
    sd = {}
    for k, v in dense_sd.items():
        if k.endswith("mlp.up.weight"):
            sd[k.replace("mlp.up.weight", "mlp.w_up")] = _expand_up(v)
        elif k.endswith("mlp.down.weight"):
            sd[k.replace("mlp.down.weight", "mlp.w_down")] = _expand_down(v)
        else:
            sd[k] = v
    ref = v
    for i in range(MC.n_layer):
        sd[f"blocks.{i}.mlp.router.weight"] = torch.zeros(N_EXP, MC.d, dtype=ref.dtype, device=ref.device)
        sd[f"blocks.{i}.mlp.bias"] = torch.zeros(N_EXP, dtype=ref.dtype, device=ref.device)
        sd[f"blocks.{i}.mlp.counts"] = torch.zeros(N_EXP, dtype=ref.dtype, device=ref.device)
    return sd

def convert_optimizer(dense_opt_sd, moe_model):
    dense = make_dense()
    dopt = make_opt(dense, TC.lr)
    dopt.load_state_dict(dense_opt_sd)
    by_name = {n: dopt.state[p] for n, p in dense.named_parameters() if p in dopt.state}
    mopt = make_opt(moe_model, TC.lr)
    for n, p in moe_model.named_parameters():
        if n.endswith("mlp.router.weight"):
            continue                                        # new: starts with empty averages
        if n.endswith("mlp.w_up") or n.endswith("mlp.w_down"):
            up = n.endswith("mlp.w_up")
            src = by_name[n.replace("mlp.w_up", "mlp.up.weight").replace("mlp.w_down", "mlp.down.weight")]
            ex = _expand_up if up else _expand_down
            mopt.state[p] = {"step": src["step"].clone(), "exp_avg": ex(src["exp_avg"]),
                             "exp_avg_sq": ex(src["exp_avg_sq"])}
        else:
            mopt.state[p] = {k: v.clone() for k, v in by_name[n].items()}
    return mopt.state_dict()

def build_conversion():
    blob = load_final(RUN_R1)
    moe = make_moe()
    msd = convert_weights(blob["model"])
    moe.load_state_dict(msd)
    return {"model": msd, "opt": convert_optimizer(blob["opt"], moe), "scaler": blob["scaler"]}

CONV = build_conversion()
# check: family f's clones hold R1's slice f, for weights and for both Adam averages
_r1 = load_final(RUN_R1)
_dense = make_dense(); _dopt = make_opt(_dense, TC.lr); _dopt.load_state_dict(_r1["opt"])
_moe = make_moe(); _mopt = make_opt(_moe, TC.lr); _mopt.load_state_dict(CONV["opt"])
_dn = dict(_dense.named_parameters()); _mn = dict(_moe.named_parameters())
_ok = True
for i in (0, MC.n_layer - 1):
    for f in range(XC.families):
        for c in (0, XC.clones - 1):
            e = f * XC.clones + c
            _ok &= torch.equal(CONV["model"][f"blocks.{i}.mlp.w_up"][e], _r1["model"][f"blocks.{i}.mlp.up.weight"][f*Q:(f+1)*Q])
            _ok &= torch.equal(_mopt.state[_mn[f"blocks.{i}.mlp.w_down"]]["exp_avg_sq"][e],
                               _dopt.state[_dn[f"blocks.{i}.mlp.down.weight"]]["exp_avg_sq"][:, f*Q:(f+1)*Q])
    _ok &= _mn[f"blocks.{i}.mlp.router.weight"] not in _mopt.state
del _r1, _dense, _dopt, _moe, _mopt

print(f"converted R1 -> {N_EXP} experts per layer; router at zero, biases at zero")
gate(bool(_ok), "each clone holds its family's slice of R1: weights and Adam averages copied; router state empty")
R["gates"]["conversion_state"] = bool(_ok)

# %% [markdown]
# ## Cell 13 — Gates: the conversion keeps the function, the router learns, balancing works
#
# ### What this block does
# Four checks, each of which stops the notebook if it fails:
#
# | gate | question | how |
# |---|---|---|
# | **G1** | does the MoE compute what R1 computed? | fp64 on the CPU: logits must agree to 1e-10 (relative); the fp32 difference is reported as the rounding floor; on the GPU, the validation loss must match R1's |
# | **G2** | does the router get a gradient? | one backward pass: every layer's router gradient must be non-zero; and the flawed per-family rule must give none (zero apart from rounding), the reason for the joint rescaling |
# | **G3** | does the bias balancing work? | a toy layer whose router favours one expert in each family; the bias alone must bring MaxVio down |
# | **G4** | are the parameter counts right? | the model's own count against the formula |
#
# ### Why "to 1e-10" and not "exactly"
# Splitting one 1,280-neuron product into four 320-neuron products adds the same numbers in a
# different order, and computers round after every addition. In fp64 the difference is about
# 1e-15; in fp32 about 1e-7. Both are rounding, not a mismatch (plan, Section 7.1).

# %%
def _val_batch(n=2):
    x, y = VAL_W.get(0, n, device="cpu")
    return x, y

def _g1():
    blob = load_final(RUN_R1)
    cpu_sd = {k: v.detach().cpu() for k, v in blob["model"].items()}
    x, _ = _val_batch()
    res = {}
    for name, dt in (("fp64", torch.float64), ("fp32", torch.float32)):
        torch.manual_seed(SEED)
        dense = GPT(MC).to(dt); dense.load_state_dict({k: v.to(dt) for k, v in cpu_sd.items()})
        moe = make_moe(device="cpu").to(dt)
        moe.load_state_dict({k: v.to(dt) for k, v in convert_weights(cpu_sd).items()})
        dense.eval(); moe.eval()
        a, b = dense.logits(x), moe.logits(x)
        res[name] = ((a - b).abs().max() / a.abs().max()).item()
    return res

G1 = _g1()
print("G1 - the conversion keeps the function (2 validation sequences, CPU)")
gate(G1["fp64"] < TH.g1_fp64_rel, f"fp64 logits: largest relative difference {G1['fp64']:.1e} (limit {TH.g1_fp64_rel:.0e})")
print(f"  [info] fp32 logits: largest relative difference {G1['fp32']:.1e}  <- the rounding floor, not a mismatch")
_moe = make_moe(); _moe.load_state_dict(CONV["model"])
G1_val = evaluate(_moe)
gate(abs(G1_val - R1["final_val"]) < TH.g1_loss_abs,
     f"{DEVICE} validation loss: converted MoE {G1_val:.5f} vs R1 {R1['final_val']:.5f} "
     f"(difference {abs(G1_val - R1['final_val']):.1e}, limit {TH.g1_loss_abs:.0e})")
R["gates"]["G1"] = {**G1, "val_moe": G1_val, "val_r1": R1["final_val"]}

def _router_grads(joint):
    m = make_moe(joint=joint); m.load_state_dict(CONV["model"]); m.train()
    x, y = TRAIN_W.get(CONT_START, 4)
    m(x, y).backward()
    return [0.0 if b.mlp.router.weight.grad is None else b.mlp.router.weight.grad.norm().item() for b in m.blocks]

print("G2 - the router gets a gradient")
_gj, _gf = _router_grads(True), _router_grads(False)
gate(min(_gj) > 0, f"joint rescaling (ours): smallest router gradient over {MC.n_layer} layers {min(_gj):.2e} > 0")
gate(max(_gf) < 1e-6 * min(_gj), f"per-family rescaling (the flawed rule): largest router gradient {max(_gf):.1e}, "
     f"over a million times smaller than ours - zero apart from rounding, which is why it was rejected")
R["gates"]["G2"] = {"joint_min": min(_gj), "joint_max": max(_gj), "per_family_max": max(_gf)}

@torch.no_grad()
def update_bias(layer, gamma):
    """Lesson Section 18: busier than the family average -> bias down; quieter -> bias up."""
    load = layer.counts.view(layer.fam, layer.clones)
    layer.bias += gamma * torch.sign(load.mean(-1, keepdim=True) - load).view(-1)
    layer.counts.zero_()

def maxvio(counts, fam, clones):
    load = counts.view(fam, clones).float()
    mean = load.mean(-1)
    return ((load.max(-1).values - mean) / mean.clamp(min=1e-9)).mean().item()

def _g3(steps=600, n=2048):
    g = torch.Generator().manual_seed(SEED)
    lay = FamilyMoE(16, 8, 2, 4, 4.0)
    with torch.no_grad():
        lay.router.weight.normal_(0, 0.3, generator=g)
        lay.router.weight[0::4, 0] += 0.3          # a constant pull towards clone 0 of each family
    lay.train()
    trace = []
    for t in range(steps):
        x = torch.randn(n, 16, generator=g); x[:, 0] = 1.0
        with torch.no_grad():
            lay(x)
        trace.append(maxvio(lay.counts, 2, 4))
        update_bias(lay, XC.bias_gamma)
    return trace

print("G3 - bias balancing on a skewed toy layer (2 families x 4 clones, no training, bias only)")
_tr = _g3()
_start, _end = float(np.mean(_tr[:10])), float(np.mean(_tr[-50:]))
gate(_end < TH.g3_maxvio and _end < TH.g3_ratio * _start,
     f"MaxVio {_start:.2f} at the start -> {_end:.2f} after {len(_tr)} steps (limits: < {TH.g3_maxvio}, < half the start)")
R["gates"]["G3"] = {"start": _start, "end": _end, "trace": _tr[::10]}

print("G4 - parameter counts")
_tot, _act = moe_params(_moe)
gate(_tot == _want_tot, f"MoE total {_tot:,} = dense {PB['total']:,} - FFN {PB['feed-forward']:,} + experts and routers")
gate(_act - PB["total"] == MC.n_layer * MC.d * N_EXP,
     f"active {_act:,} = the dense model's {PB['total']:,} + routers {MC.n_layer * MC.d * N_EXP:,}: same work per token")
R["gates"]["G4"] = {"total": _tot, "active": _act}
del _moe

# %% [markdown]
# ## Cell 14 — the preview: 450 steps of the converted MoE
#
# ### What this block does
# Trains the converted model for 450 steps (about 7.4M tokens, decision D7 as revised) on the
# first windows after R1's data, the same text R3 will read in v3. The learning rate stays at
# R1's 1e-3 (no warm-up: the optimiser state came across), and there is no decay. Selection is
# probabilistic for the first 300 steps (D3), then hard for the last 150, so the preview shows
# the switch that the laptop test found fragile. Evaluation always uses hard selection.
#
# After every step the bias of every expert is nudged (Cell 11). Every 25 steps the notebook
# records, per layer and expert, how many tokens each expert received; from that come
# **MaxVio** (how uneven the load is) and the **dead-expert** count (experts that got no tokens
# in the last 25 steps). It also records how far apart the 8 clones in each family have drifted:
# the average distance between their weights, relative to the weights' size. At conversion this
# is exactly 0.
#
# This is a throwaway look. v3's R3 starts again from the conversion point.
#
# ### What you should see
# The line "before training" reads R1's loss, then the loss keeps falling. Dead experts should
# be 0, MaxVio should stay small, perhaps with a short bump right after step 300, and the clone
# distance should grow from 0. The final checkpoint of this throwaway run is about 1.4 GB on
# Drive (114M weights plus Adam's two averages); it can be deleted afterwards.

# %%
_WIN = {}

@torch.no_grad()
def clone_distance(model):
    """Mean over layers and families of the average pairwise distance between the clones'
    up-weights, relative to their size. 0 at conversion."""
    out = []
    for b in model.blocks:
        w = b.mlp.w_up.detach().float().view(XC.families, XC.clones, -1)
        dist = torch.cdist(w, w)                                    # [fam, C, C]
        scale = w.norm(dim=-1).mean(-1, keepdim=True)               # [fam, 1]
        iu = torch.triu_indices(XC.clones, XC.clones, 1)
        out.append((dist[:, iu[0], iu[1]] / scale).mean().item())
    return float(np.mean(out))

def moe_after_step(model, step, hist):
    layers = [b.mlp for b in model.blocks]
    if "win" not in _WIN:
        _WIN["win"] = torch.zeros(len(layers), N_EXP, device=layers[0].counts.device)
    ex = hist["extra"]
    for li, lay in enumerate(layers):
        _WIN["win"][li] += lay.counts
        update_bias(lay, XC.bias_gamma)
        lay.probabilistic = step + 1 < XC.prob_steps
    if (step + 1) % TC.log_every == 0 or step + 1 == hist["steps"]:
        win = _WIN["win"]
        mv = float(np.mean([maxvio(win[li], XC.families, XC.clones) for li in range(len(layers))]))
        ex.setdefault("moe_log", []).append({
            "step": step + 1, "maxvio": mv, "dead": int((win == 0).sum().item()),
            "clone_dist": clone_distance(model),
            "bias_absmax": max(l.bias.abs().max().item() for l in layers),
            "share": (win.view(len(layers), XC.families, XC.clones) /
                      win.view(len(layers), XC.families, XC.clones).sum(-1, keepdim=True)).cpu().tolist()})
        _WIN["win"].zero_()

def moe_extra_loss(model):
    return XC.seq_aux * sum(b.mlp.aux for b in model.blocks) / MC.n_layer

MOE_CODE = (FamilyMoE, make_moe, make_moe_prob, convert_weights, convert_optimizer, _expand_up, _expand_down,
            build_conversion, update_bias, maxvio, clone_distance, moe_after_step, moe_extra_loss)
RUN_PREVIEW = False   # the v2 preview: results are in the plan (Section 8); True reruns it (~7 min)
PREVIEW, _ml = None, None
if RUN_PREVIEW:
    RUN_P = "P_moe_preview"
    _WIN.clear()
    PREVIEW = cached_run(RUN_P, DENSE_CODE + MOE_CODE, make_model=make_moe_prob, steps=XC.preview_steps,
                         warmup=0, decay=0, window_start=CONT_START, init=CONV,
                         extra_loss=moe_extra_loss, after_step=moe_after_step)
    _ml = PREVIEW["extra"]["moe_log"]
    _v0, _v1 = PREVIEW["val"][0]["loss"], PREVIEW["final_val"]
    print(f"\npreview: validation {_v0:.4f} at conversion -> {_v1:.4f} after {XC.preview_steps} steps "
          f"({'lower' if _v1 < _v0 else 'HIGHER'} by {abs(_v0 - _v1):.4f})")
    print(f"         dead experts in the last window: {_ml[-1]['dead']} of {MC.n_layer * N_EXP}; "
          f"MaxVio {_ml[-1]['maxvio']:.3f}; clone distance {_ml[0]['clone_dist']:.4f} -> {_ml[-1]['clone_dist']:.4f}")
    if PREVIEW.get("tokens_per_second"):
        print(f"         speed {PREVIEW['tokens_per_second']:,.0f} tokens/s (R1 dense: {R1['tokens_per_second'] or 0:,.0f})"
              + (f", peak memory {PREVIEW['peak_mem_bytes']/2**20:,.0f} MiB" if PREVIEW.get("peak_mem_bytes") else ""))
    R["runs"]["P"] = PREVIEW
else:
    print("v2 preview skipped (RUN_PREVIEW = False): its results are recorded in the plan, Section 8.")

# %% [markdown]
# ## Cell 15 — what the preview shows: continuity, balance, and clones coming apart
#
# ### What this block does
# Three figures. **Left:** R1's last 20M tokens and the preview's 7.4M, on one token axis, so
# you can see the loss carry on across the conversion. **Middle:** MaxVio, dead experts and
# clone distance over the 450 steps (switch at 300). **Right:** the share of each family's tokens that each of
# its 8 experts received in the last 25 steps, for all 16 layers. An even share is 1/8 = 12.5%.

# %%
if PREVIEW:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    _r1v = [p for p in R1["val"] if p["tokens"] >= R1["val"][-1]["tokens"] - 20e6]
    ax[0].plot([p["tokens"]/1e6 for p in _r1v], [p["loss"] for p in _r1v], color="#b1521f", marker="o", ms=3, label="R1 dense")
    _off = R1["val"][-1]["tokens"]
    ax[0].plot([(p["tokens"] + _off)/1e6 for p in PREVIEW["val"]], [p["loss"] for p in PREVIEW["val"]],
               color="#1f6f8b", marker="o", ms=3, label="MoE preview")
    ax[0].axvline(_off/1e6, color="#7a4b9e", ls="--", lw=0.8)
    ax[0].set_xlabel("training tokens (millions)"); ax[0].set_ylabel("validation loss"); ax[0].legend(frameon=False)
    ax[0].set_title("conversion at the dashed line", fontsize=9)
    _st = [m["step"] for m in _ml]
    ax[1].plot(_st, [m["maxvio"] for m in _ml], color="#b1521f", label="MaxVio")
    ax[1].plot(_st, [m["clone_dist"] for m in _ml], color="#2c6b45", label="clone distance")
    ax[1].set_xlabel("preview step"); ax[1].legend(frameon=False)
    ax2 = ax[1].twinx(); ax2.bar(_st, [m["dead"] for m in _ml], width=8, color="#9a968e", alpha=0.5)
    ax2.set_ylabel("dead experts (grey bars)")
    _share = np.array(_ml[-1]["share"]).reshape(MC.n_layer, N_EXP)
    im = ax[2].imshow(_share, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.25, _share.max()))
    for f in range(1, XC.families):
        ax[2].axvline(f * XC.clones - 0.5, color="white", lw=1)
    ax[2].set_xlabel("expert (4 families of 8)"); ax[2].set_ylabel("layer")
    ax[2].set_title("share of the family's tokens, last 25 steps", fontsize=9)
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    _save(fig, "v2_preview.png")

# %% [markdown]
# ## Cell 16 — v2.1 diagnostic: why did the MoE lose ground?
#
# ### Why this cell exists
# In v2 the conversion was exact, but over the 450-step preview the validation loss **rose**
# (4.1230 → 4.1625), starting while the load was still perfectly even, and after the switch to
# hard selection the load collapsed. Three explanations fit, and one run cannot separate them
# (plan, Sections 8–9). This cell runs short experiments from the same conversion point, on the
# same data, each changing one thing.
#
# | run | what differs | question |
# |---|---|---|
# | **A** | the dense model keeps training (no MoE) | how fast should the loss fall here? the yardstick |
# | **B** | MoE, always clone 1 of each family, router frozen: the dense model in MoE clothing | does the MoE code itself change training? must track A |
# | **C** | as B, but the router learns the four family weights | does a moving router alone push the loss up? |
# | **G** | the v2 design, switch to hard selection at step 100 | the reference for D and E, in this shorter run |
# | **D** | as G, router output scaled by 0.1 (a 10× slower router under Adam) | is the router too fast? does slowing it stop the collapse? |
# | **E** | as G, learning rate warmed up from 0 over 100 steps | was the restart too abrupt? |
# | **F** | each token id always goes to the same clone in each family (a fixed, frequency-balanced table); the router learns only the family weights | is the random clone choice the problem? (collapse is impossible by construction) |
#
# Each run is 200 steps (about 3.3M tokens) with validation every 50 steps. The switch is at
# step 100 so that both phases fit. The large throwaway checkpoints are deleted after each run.
# A finished run is cached in a small file, so a reconnect does not repeat it.
#
# ### What you should see
# A table and a figure. **B must match A** within 0.01 (a gate: otherwise the MoE code has a
# bug). The rest is evidence, read in the plan's Section 9.

# %%
@dataclass(frozen=True)
class DiagCfg:
    steps: int = 200
    switch: int = 100
    warmup_e: int = 100
    router_scale_d: float = 0.1
    eval_every_steps: int = 50
    b_vs_a_tol: float = 0.01        # gate: B (dense in MoE clothing) must track A to this

DC_ = DiagCfg()
if PROFILE == "cpu":
    DC_ = replace(DC_, steps=40, switch=20, warmup_e=20, eval_every_steps=10)
elif PROFILE == "smoke":
    DC_ = replace(DC_, steps=60, switch=30, warmup_e=30, eval_every_steps=20)

_CUR = {}

class DiagMoE(FamilyMoE):
    """FamilyMoE with switchable clone choice, for the diagnostic only.
    mode 'design' = as v2; 'fixed0' = always clone 1; 'hash' = a fixed table by token id."""
    mode, zscale, table = "design", 1.0, None

    def route(self, x2):
        hi = torch.promote_types(x2.dtype, torch.float32)
        with torch.autocast(x2.device.type, enabled=False):
            z = self.zscale * (x2.to(hi) @ self.router.weight.to(hi).t())
        s = torch.softmax(z, dim=-1)
        if self.mode == "fixed0":
            pick = torch.zeros(x2.size(0), self.fam, dtype=torch.long, device=x2.device)
        elif self.mode == "hash":
            pick = self.table[_CUR["ids"].reshape(-1)]
        else:
            key = (z + self.bias.to(hi)).view(-1, self.fam, self.clones)
            if self.training and self.probabilistic:
                pick = torch.multinomial(torch.softmax(key.reshape(-1, self.clones), -1), 1).view(-1, self.fam)
            else:
                pick = key.argmax(-1)
        eid = pick + torch.arange(self.fam, device=x2.device) * self.clones
        chosen = s.gather(1, eid)
        return z, eid, self.scale * chosen / chosen.sum(-1, keepdim=True)

def _hash_table():
    """Token id -> clone, per family. Greedy: frequent tokens first, each to the least-loaded
    clone, so every clone gets about 1/8 of the token *occurrences*. Families use different
    random tie-breaks, so they split the vocabulary differently."""
    freq = np.bincount(np.asarray(TRAIN[:min(len(TRAIN), 20_000_000)]), minlength=MC.vocab).astype(np.float64)
    order = np.argsort(-freq, kind="stable")
    rng = np.random.default_rng(SEED)
    tab = np.zeros((MC.vocab, XC.families), dtype=np.int64)
    for f in range(XC.families):
        load = rng.random(XC.clones) * 1e-9
        for t in order:
            c = int(np.argmin(load)); tab[t, f] = c; load[c] += freq[t]
    return torch.from_numpy(tab)

HASH_TABLE = _hash_table()

def make_diag(mode="design", zscale=1.0, freeze_router=False, prob=True):
    m = make_moe()
    for b in m.blocks:
        old = b.mlp
        new = DiagMoE(MC.d, Q, XC.families, XC.clones, XC.scale)
        new.mode, new.zscale = mode, zscale
        new.table = HASH_TABLE.to(DEVICE) if mode == "hash" else None
        new.probabilistic = prob and mode == "design" and XC.prob_steps > 0
        if freeze_router:
            new.router.weight.requires_grad_(False)
        b.mlp = new.to(DEVICE)
        del old
    m.register_forward_pre_hook(lambda mod, args: _CUR.__setitem__("ids", args[0]))
    return m

def diag_run(key, make_model, init, warmup=0, extra_loss=None, after_step=None, prob_steps=None):
    """Train a short diagnostic run; cache only its small log, delete the big checkpoints."""
    global TC, XC
    path = cache(f"diag_{BASE_KEY}_{key}.json")
    src = _src_hash(DiagMoE, make_diag, diag_run, _hash_table, train_run, FamilyMoE, moe_after_step,
                    update_bias, convert_weights, convert_optimizer) + json.dumps(asdict(DC_))
    if not RESET_CACHE and os.path.exists(path):
        blob = json.load(open(path, encoding="utf-8"))
        if blob.get("src") == src:
            print(f"    [cached diag] '{key}' reloaded")
            return blob["hist"]
    old_tc, old_xc = TC, XC
    TC = replace(TC, eval_every_tokens=DC_.eval_every_steps * TC.batch * MC.seq)
    if prob_steps is not None:
        XC = replace(XC, prob_steps=prob_steps)
    _WIN.clear()
    try:
        hist = train_run(key, make_model, steps=DC_.steps, warmup=warmup, decay=0,
                         window_start=CONT_START, init=init, extra_loss=extra_loss, after_step=after_step)
    finally:
        TC, XC = old_tc, old_xc
    for pth in (_final_path(key), _ckpt_path(key)):
        if os.path.exists(pth):
            os.remove(pth)
    _atomic_save({"src": src, "hist": hist}, path, as_json=True)
    return hist

_R1_BLOB = load_final(RUN_R1)
DIAG = {}
_plan = [
    ("A", "dense continues", lambda: diag_run("D_A_dense", make_dense, _R1_BLOB)),
    ("B", "one clone, router frozen", lambda: diag_run("D_B_fixed_frozen", lambda: make_diag("fixed0", freeze_router=True), CONV,
                                                      after_step=moe_after_step)),
    ("C", "one clone, router learns", lambda: diag_run("D_C_fixed_router", lambda: make_diag("fixed0"), CONV,
                                                      after_step=moe_after_step)),
    ("G", "design, switch at step 100", lambda: diag_run("D_G_design", lambda: make_diag("design"), CONV,
                                                        extra_loss=moe_extra_loss, after_step=moe_after_step, prob_steps=DC_.switch)),
    ("D", "design, router 10x slower", lambda: diag_run("D_D_slow_router", lambda: make_diag("design", zscale=DC_.router_scale_d), CONV,
                                                       extra_loss=moe_extra_loss, after_step=moe_after_step, prob_steps=DC_.switch)),
    ("E", "design, 100-step warm-up", lambda: diag_run("D_E_warmup", lambda: make_diag("design"), CONV, warmup=DC_.warmup_e,
                                                      extra_loss=moe_extra_loss, after_step=moe_after_step, prob_steps=DC_.switch)),
    ("F", "clone by token id (hash)", lambda: diag_run("D_F_hash", lambda: make_diag("hash"), CONV,
                                                      after_step=moe_after_step)),
]
RUN_V21 = False   # v2.1's seven runs are recorded in the plan (Section 10); True reruns them (~20 min on a T4)

def _v(h, step):
    return next(p["loss"] for p in h["val"] if p["step"] == step)

_COLS = {"A": "#b1521f", "B": "#9a968e", "C": "#7a4b9e", "G": "#1f6f8b", "D": "#2c6b45", "E": "#c9a227",
         "F": "#d1495b", "Aw": "#e08e45", "D35": "#6aa84f", "D12": "#274e13", "F35": "#8e44ad"}

def report_diag(runs, png, switch, one_clone=("B", "C")):
    """Print the table of a diagnostic round and draw its figure. Returns run A's final loss."""
    steps = [p["step"] for p in runs["A"]["hist"]["val"]]
    a_end = runs["A"]["hist"]["final_val"]
    print(f"\n{'run':<5}{'what differs':<34}" + "".join(f"{'val@'+str(s):>11}" for s in steps)
          + f"{'vs A':>9}{'MaxVio':>8}{'dead':>6}{'tok/s':>9}")
    for name, d in runs.items():
        h = d["hist"]
        ml = h["extra"].get("moe_log", [])
        skip = name in one_clone or not ml
        mv = "n/a" if skip else f"{ml[-1]['maxvio']:.2f}"
        dead = "n/a" if skip else f"{ml[-1]['dead']}"
        tps = f"{h['tokens_per_second']:,.0f}" if h.get("tokens_per_second") else "-"
        print(f"{name:<5}{d['label']:<34}" + "".join(f"{_v(h, s):>11.4f}" for s in steps)
              + f"{h['final_val'] - a_end:>+9.4f}{mv:>8}{dead:>6}{tps:>9}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for name, d in runs.items():
        h = d["hist"]
        ax[0].plot([p["step"] for p in h["val"]], [p["loss"] for p in h["val"]], marker="o", ms=3,
                   color=_COLS.get(name, "#333333"), ls="--" if name in ("B", "Aw") else "-", label=f"{name} {d['label']}")
        ml = h["extra"].get("moe_log", [])
        if ml and name not in one_clone:
            ax[1].plot([m["step"] for m in ml], [m["maxvio"] for m in ml], color=_COLS.get(name, "#333333"), label=name)
    for x in ax:
        x.axvline(switch, color="#9a968e", ls=":", lw=0.8)
    ax[0].set_xlabel("steps after conversion"); ax[0].set_ylabel("validation loss"); ax[0].legend(frameon=False, fontsize=7)
    ax[1].set_xlabel("steps after conversion"); ax[1].set_ylabel("MaxVio"); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title(f"dotted: switch to hard selection (step {switch})", fontsize=9)
    _save(fig, png)
    return a_end

if RUN_V21:
    for name, label, fn in _plan:
        print(f"\n--- run {name}: {label}")
        DIAG[name] = {"label": label, "hist": fn()}
    _a_end = report_diag(DIAG, "v21_diagnostic.png", DC_.switch)
    gate(abs(DIAG["B"]["hist"]["final_val"] - _a_end) < DC_.b_vs_a_tol,
         f"B (the dense model in MoE clothing) tracks A: {DIAG['B']['hist']['final_val']:.4f} vs {_a_end:.4f} "
         f"(limit {DC_.b_vs_a_tol})")
    R["diag"] = {"cfg": asdict(DC_), "runs": {k: {"label": d["label"], "hist": d["hist"]} for k, d in DIAG.items()}}
else:
    print("v2.1 diagnostic skipped (RUN_V21 = False): its results are recorded in the plan, Section 10.")

# %% [markdown]
# ## Cell 17 — v2.2 diagnostic: is the experts' learning rate too high?
#
# ### Why this cell exists
# The v2.1 diagnostic (plan, Section 10) found that the router being too fast causes the
# collapse (fixed by a 10× slower router, run D), and that the loss rises whenever tokens are
# **split** between clones. Each clone then learns from about 2,000 tokens a step instead of
# 16,384, so its gradient is noisier, while Adam still takes full-size steps. This cell tests
# whether a **smaller learning rate for the experts** removes the rise.
#
# ### How the experts get a smaller learning rate without touching Cell 6
# Each expert's weights are stored divided by a constant *c* and multiplied by *c* in the forward
# pass, so the model computes exactly the same thing. Adam moves every stored number by about
# the learning rate, so the real weights move by *c* × the learning rate: the experts learn *c*
# times more slowly, and nothing else changes. The copied Adam averages are rescaled to match
# (momentum × *c*, squared average × *c*²), so the first step is the same as it would have
# been, and the start is still exact.
#
# | run | what it is | question |
# |---|---|---|
# | **A** | dense continues | the yardstick |
# | **Aw** | dense continues, warm-up 0 → 1e-3 over 100 steps | how much of v2.1's E dip was just a lower learning rate? |
# | **D** | the design with the 10× slower router | the reference (it fixed the collapse) |
# | **D35** | D, experts' learning rate × 0.35 (1/√8) | does a smaller expert step stop the rise? |
# | **D12** | D, experts' learning rate × 0.125 (1/8) | even smaller: better, or too slow? |
# | **F35** | token-id routing, experts' learning rate × 0.35 | the same fix on the design balanced by construction |
#
# ### What you should see
# A table and a figure as in v2.1. If D35 or D12 track or beat A, the hypothesis holds and v3's
# R3 uses that setting.

# %%
@dataclass(frozen=True)
class Diag2Cfg:
    expert_lr: tuple = (0.35, 0.125)

D2C = Diag2Cfg()

class ScaledDiagMoE(DiagMoE):
    """DiagMoE whose experts are stored divided by c and used multiplied by c:
    the same function, but under Adam the experts learn c times more slowly."""
    c = 1.0

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.d)
        z, eid, w = self.route(x2)
        cnt = torch.bincount(eid.reshape(-1), minlength=self.E)
        if self.training:
            self.counts += cnt.to(self.counts.dtype)
        cnt_l = cnt.tolist()
        out = torch.zeros_like(x2)
        c = self.c
        for f in range(self.fam):
            order = torch.argsort(eid[:, f])
            xs, ys, start = x2[order], [], 0
            for k in range(self.clones):
                e, n = f * self.clones + k, cnt_l[f * self.clones + k]
                if n == 0:
                    continue
                h = F.gelu(xs[start:start + n] @ (c * self.w_up[e]).t())
                ys.append(h @ (c * self.w_down[e]).t())
                start += n
            y = torch.cat(ys) * w[order, f].unsqueeze(1).to(ys[0].dtype)
            out = out.index_add(0, order, y.to(out.dtype))
        if self.training and len(shp) == 3:
            B, T = shp[0], shp[1]
            local = (eid - torch.arange(self.fam, device=x.device) * self.clones).view(B, T, self.fam)
            frac = F.one_hot(local, self.clones).float().mean(1)
            prob = torch.softmax(z.view(B, T, self.fam, self.clones).float(), -1).mean(1)
            self.aux = (self.clones * (frac * prob).sum(-1)).mean()
        return out.view(shp)

def make_scaled(mode, zscale, c):
    m = make_diag(mode, zscale=zscale)
    for b in m.blocks:
        new = ScaledDiagMoE(MC.d, Q, XC.families, XC.clones, XC.scale)
        new.mode, new.zscale, new.table, new.c = b.mlp.mode, b.mlp.zscale, b.mlp.table, c
        new.probabilistic = b.mlp.probabilistic
        b.mlp = new.to(DEVICE)
    return m

def scaled_init(c, mode="design", zscale=1.0):
    """CONV rescaled for experts stored as w / c: weights / c, momentum x c, squared average x c^2."""
    sd = {k: (v / c if k.endswith(("mlp.w_up", "mlp.w_down")) else v) for k, v in CONV["model"].items()}
    m = make_scaled(mode, zscale, c)
    m.load_state_dict(sd)
    opt = make_opt(m, TC.lr)
    opt.load_state_dict(CONV["opt"])
    for n, p in m.named_parameters():
        if n.endswith(("mlp.w_up", "mlp.w_down")) and p in opt.state:
            st = opt.state[p]
            st["exp_avg"].mul_(c)
            st["exp_avg_sq"].mul_(c * c)
    return {"model": sd, "opt": opt.state_dict(), "scaler": CONV["scaler"]}

# exactness check: stored w / c, used c x (w / c): the same function as the unscaled design
_m1 = make_diag("design"); _m1.load_state_dict(CONV["model"])
_m2 = make_scaled("design", 1.0, D2C.expert_lr[0]); _m2.load_state_dict(scaled_init(D2C.expert_lr[0])["model"])
_x, _ = VAL_W.get(0, 2)
_l1, _l2 = _m1.logits(_x), _m2.logits(_x)
_rel = ((_l1 - _l2).abs().max() / _l1.abs().max()).item()
gate(_rel < 1e-4, f"storing the experts as w / c keeps the function: largest relative logit difference {_rel:.1e}")
del _m1, _m2

DIAG2 = {}
RUN_V22 = False   # v2.2's six runs: results are in the plan (Section 12); True reruns them (~20 min)
if RUN_V22:
    _plan2 = [
        ("A", "dense continues", lambda: diag_run("D2_A_dense", make_dense, _R1_BLOB)),
        ("Aw", "dense, 100-step warm-up", lambda: diag_run("D2_Aw_dense_warm", make_dense, _R1_BLOB, warmup=DC_.warmup_e)),
        ("D", "design, router 10x slower", lambda: diag_run("D2_D_slow_router", lambda: make_diag("design", zscale=DC_.router_scale_d), CONV,
                                                           extra_loss=moe_extra_loss, after_step=moe_after_step, prob_steps=DC_.switch)),
    ]
    for _c, _tag in zip(D2C.expert_lr, ("35", "12")):
        _plan2.append((f"D{_tag}", f"as D, expert lr x {_c}",
                       (lambda c=_c, tag=_tag: diag_run(f"D2_D{tag}_expert_lr", lambda: make_scaled("design", DC_.router_scale_d, c),
                                                        scaled_init(c, "design", DC_.router_scale_d),
                                                        extra_loss=moe_extra_loss, after_step=moe_after_step, prob_steps=DC_.switch))))
    _plan2.append(("F35", f"token-id routing, expert lr x {D2C.expert_lr[0]}",
                   lambda: diag_run("D2_F35_hash_expert_lr", lambda: make_scaled("hash", 1.0, D2C.expert_lr[0]),
                                    scaled_init(D2C.expert_lr[0], "hash"), after_step=moe_after_step)))
    for name, label, fn in _plan2:
        print(f"\n--- run {name}: {label}")
        DIAG2[name] = {"label": label, "hist": fn()}
    _a2_end = report_diag(DIAG2, "v22_diagnostic.png", DC_.switch)
    _best = min((k for k in DIAG2 if k not in ("A", "Aw")), key=lambda k: DIAG2[k]["hist"]["final_val"])
    _gap = DIAG2[_best]["hist"]["final_val"] - _a2_end
    print(f"\nbest MoE run: {_best} ({DIAG2[_best]['label']}), {_gap:+.4f} vs dense A at step {DC_.steps}"
          + ("  -> keeps pace with dense" if abs(_gap) < 0.005 else ("  -> ahead of dense" if _gap < 0 else "  -> still behind dense")))
    _e_like = DIAG2["Aw"]["hist"]["val"][1]["loss"] - DIAG2["A"]["hist"]["val"][1]["loss"]
    print(f"dense with the warm-up vs without, at step {DIAG2['A']['hist']['val'][1]['step']}: {_e_like:+.4f} "
          "(v2.1's MoE run E was 0.0618 below dense at this point)")
    R["diag2"] = {"cfg": asdict(D2C), "exact_rel": _rel, "best": _best,
                  "runs": {k: {"label": d["label"], "hist": d["hist"]} for k, d in DIAG2.items()}}
else:
    print("v2.2 diagnostic skipped (RUN_V22 = False): its results are recorded in the plan, Section 12.")

# %% [markdown]
# ## Cell 18 — v3: the full comparison (R2, R2s, R3), 50M tokens each
#
# ### What this block does
# Three long runs from R1's checkpoint, all on the same 50M new tokens in the same order, all
# with the learning rate held at 1e-3 and then lowered over the last 20% (warmup-stable-decay):
#
# | run | what it is | why |
# |---|---|---|
# | **R2** | the dense model continues | the control |
# | **R2s** | the dense model continues, with its FFN learning at 1/8 speed | the fair control: the MoE's experts learn at 1/8 speed, so this separates "more parameters" from "slower learning" |
# | **R3** | the MoE: 32 experts per layer, router 10× slower, experts at 1/8 speed, switch to hard selection at step 100 | the assignment's answer (decisions V3-1 to V3-4 in the plan) |
#
# The settings come from the diagnostics: the slow router stops the collapse after the switch
# (v2.1), and the slower experts stop the loss rising after conversion (v2.2).
#
# ### How "1/8 speed" is done
# The weights concerned are stored divided by 8 and multiplied by 8 when used, so the model
# computes exactly the same thing, but Adam's fixed-size steps move them 1/8 as far. The copied
# Adam averages are rescaled to match, so the first step is unchanged. A check below confirms the
# function is unchanged for the dense case too.
#
# ### What you should see
# Three progress logs (R2 and R2s about 15 minutes each, R3 about 50 minutes). Each is saved to
# Drive when it finishes, and a disconnect resumes from the last checkpoint (every 8 minutes).

# %%
@dataclass(frozen=True)
class V3Cfg:
    expert_lr: float = 0.125     # V3-1: D12, the best setting in v2.2
    router_scale: float = 0.1    # the 10x slower router (v2.1, run D)
    switch: int = 100            # V3-3: switch to hard selection at step 100
    decay_frac: float = 0.2      # R2, R2s, R3: lower the learning rate over the last 20%

V3 = V3Cfg()
if PROFILE == "cpu":
    V3 = replace(V3, switch=20)
elif PROFILE == "smoke":
    V3 = replace(V3, switch=30)
DECAY = int(V3.decay_frac * STEPS)

class ScaledMLP(MLP):
    """The dense FFN stored as w / c and used as c x (w / c): same function, c-times-slower learning under Adam."""
    c = 1.0
    def forward(self, x):
        return F.linear(F.gelu(F.linear(x, self.c * self.up.weight)), self.c * self.down.weight)

def make_dense_scaled(c=V3.expert_lr):
    m = make_dense()
    for b in m.blocks:
        new = ScaledMLP(MC)
        new.c = c
        b.mlp = new.to(DEVICE)
    return m

def dense_scaled_init(c=V3.expert_lr):
    sd = {k: (v / c if ".mlp." in k else v) for k, v in _R1_BLOB["model"].items()}
    m = make_dense_scaled(c)
    m.load_state_dict(sd)
    opt = make_opt(m, TC.lr)
    opt.load_state_dict(_R1_BLOB["opt"])
    for n, p in m.named_parameters():
        if ".mlp." in n and p in opt.state:
            opt.state[p]["exp_avg"].mul_(c)
            opt.state[p]["exp_avg_sq"].mul_(c * c)
    return {"model": sd, "opt": opt.state_dict(), "scaler": _R1_BLOB["scaler"]}

def make_r3():
    return make_scaled("design", V3.router_scale, V3.expert_lr)

# check: the slowed dense model computes what R1 computes
_m1 = make_dense(); _m1.load_state_dict(_R1_BLOB["model"])
_m2 = make_dense_scaled(); _m2.load_state_dict(dense_scaled_init()["model"])
_x, _ = VAL_W.get(0, 2)
_l1, _l2 = _m1.logits(_x), _m2.logits(_x)
_rel_d = ((_l1 - _l2).abs().max() / _l1.abs().max()).item()
gate(_rel_d < 1e-4, f"R2s starts as R1: largest relative logit difference {_rel_d:.1e}")
del _m1, _m2

V3_CODE = DENSE_CODE + MOE_CODE + (DiagMoE, make_diag, ScaledDiagMoE, make_scaled, scaled_init, ScaledMLP,
                                   make_dense_scaled, dense_scaled_init, make_r3)

print(f"\n--- R2: the dense model continues ({STEPS:,} steps, decay over the last {DECAY:,})")
R2 = cached_run("R2_dense", V3_CODE, make_model=make_dense, steps=STEPS, warmup=0, decay=DECAY,
                window_start=CONT_START, init=_R1_BLOB)
print(f"\n--- R2s: the dense model continues, FFN at {V3.expert_lr}x learning speed")
R2S = cached_run("R2s_dense_slow_ffn", V3_CODE, make_model=make_dense_scaled, steps=STEPS, warmup=0, decay=DECAY,
                 window_start=CONT_START, init=dense_scaled_init())
print(f"\n--- R3: the MoE (router {V3.router_scale}x, experts {V3.expert_lr}x, switch at step {V3.switch})")
XC = replace(XC, prob_steps=V3.switch)       # the switch for R3 (moe_after_step and the model builder read it)
_WIN.clear()
R3 = cached_run(f"R3_moe_sw{V3.switch}", V3_CODE, make_model=make_r3, steps=STEPS, warmup=0, decay=DECAY,
                window_start=CONT_START, init=scaled_init(V3.expert_lr, "design", V3.router_scale),
                extra_loss=moe_extra_loss, after_step=moe_after_step)
R["runs"].update({"R2": R2, "R2s": R2S, "R3": R3})
R["v3"] = {"cfg": asdict(V3), "decay_steps": DECAY, "r2s_exact_rel": _rel_d}

# %% [markdown]
# ## Cell 19 — v3 results: did the MoE keep learning, and did it help?
#
# ### What this block does
# Answers the assignment's question and the comparison question from the measured numbers,
# and draws three figures:
#
# - **`v3_loss.png`**: validation loss from R1's last 20M tokens through the 50M tokens of
#   R2, R2s and R3, with the decay phase shaded.
# - **`v3_balance.png`**: R3's MaxVio, dead experts and clone distance over the run.
# - **`v3_load.png`**: the share of each family's tokens per expert at the end, for all layers.
#
# ### What you should see
# R3's validation loss falls from R1's value (the assignment's requirement, gate G5). Then
# R3 against R2 (did the MoE beat the dense model?) and R3 against R2s (was it the extra
# parameters, or just the slower learning?).

# %%
NOISE = 0.002                      # run-to-run spread of R1 measured across three T4 runs (plan, Section 12)

def _at(h, tokens):
    return min(h["val"], key=lambda p: abs(p["tokens"] - tokens))["loss"]

v_conv = R3["val"][0]["loss"]
res = {k: h["final_val"] for k, h in (("R2", R2), ("R2s", R2S), ("R3", R3))}
decay_tok = (STEPS - DECAY) * TC.batch * MC.seq
dec = {k: _at(h, decay_tok) - h["final_val"] for k, h in (("R2", R2), ("R2s", R2S), ("R3", R3))}
print(f"validation loss at conversion (R1's checkpoint): {v_conv:.4f}")
for k, lab in (("R2", "dense continues"), ("R2s", "dense, FFN at 1/8 speed"), ("R3", "MoE")):
    h = R["runs"][k]
    print(f"  {k:<4} {lab:<26} final {res[k]:.4f}  ({res[k] - v_conv:+.4f} since conversion; "
          f"{dec[k]:.4f} of it in the decay phase)  {(h.get('tokens_per_second') or 0):,.0f} tok/s")
g5 = v_conv - res["R3"] > NOISE
gate(g5, f"G5: the MoE keeps training: {v_conv:.4f} -> {res['R3']:.4f} ({res['R3'] - v_conv:+.4f}; noise floor {NOISE})")

def _verdict(a, b, name_a, name_b):
    d = res[a] - res[b]
    if abs(d) <= NOISE:
        return f"{name_a} and {name_b} finish level ({d:+.4f}, within the noise floor of {NOISE})"
    return f"{name_a} finishes {'below' if d < 0 else 'above'} {name_b} by {abs(d):.4f} ({d:+.4f})"

v_moe_dense = _verdict("R3", "R2", "the MoE (R3)", "the dense model (R2)")
v_moe_fair = _verdict("R3", "R2s", "the MoE (R3)", "the dense model with the same slowed FFN (R2s)")
v_slow = _verdict("R2s", "R2", "slowing the dense FFN (R2s)", "the dense model (R2)")
print("\n" + "\n".join(["- " + v for v in (v_moe_dense, v_moe_fair, v_slow)]))
beats_fair = res["R3"] < res["R2s"] - NOISE
beats_dense = res["R3"] < res["R2"] - NOISE
concl = ("The MoE kept learning after conversion (" + f"{res['R3'] - v_conv:+.4f}" + "). "
         + ("Against the dense model given the same slowed FFN (R2s), it finished ahead, so its extra stored parameters helped "
            "at the same work per token. " if beats_fair else "It did not beat the dense model given the same slowed FFN (R2s). ")
         + ("It also finished ahead of the ordinary dense model (R2)." if beats_dense else
            "But the ordinary dense model (R2) finished ahead of it, so at this size and budget the MoE did not beat plain dense training."))
_d = [(p3["tokens"], p3["loss"] - p2["loss"]) for p2, p3 in zip(R2["val"], R3["val"])]
_ahead = [t for t, d in _d if d < -NOISE]
if _ahead and not beats_dense:
    _lost = next((t for t, d in _d if t > _ahead[-1]), None)
    concl += (f" The MoE was ahead of R2 until {_ahead[-1]/1e6:.1f}M tokens"
              + (f" and fell behind during the learning-rate decay (from {(STEPS - DECAY) * TC.batch * MC.seq/1e6:.1f}M tokens)." if _lost else "."))
R["v3"]["r3_minus_r2"] = _d
print("\nCONCLUSION: " + concl)
R["v3"].update({"v_conv": v_conv, "final": res, "decay_drop": dec, "g5": g5, "noise": NOISE,
                "verdicts": [v_moe_dense, v_moe_fair, v_slow], "conclusion": concl})

# figure 1: loss across the conversion
fig, ax = plt.subplots(figsize=(9, 4))
_r1v = [p for p in R1["val"] if p["tokens"] >= R1["val"][-1]["tokens"] - 20e6]
ax.plot([p["tokens"]/1e6 for p in _r1v], [p["loss"] for p in _r1v], color="#7d7972", marker="o", ms=3, label="R1 dense (first 50M)")
_off = R1["val"][-1]["tokens"]
for k, col, lab in (("R2", "#b1521f", "R2 dense continues"), ("R2s", "#e08e45", "R2s dense, FFN at 1/8 speed"),
                    ("R3", "#1f6f8b", "R3 MoE")):
    h = R["runs"][k]
    ax.plot([(p["tokens"] + _off)/1e6 for p in h["val"]], [p["loss"] for p in h["val"]], color=col, marker="o", ms=3,
            ls="--" if k == "R2s" else "-", label=lab)
ax.axvline(_off/1e6, color="#7a4b9e", ls="--", lw=0.8)
ax.axvspan((_off + decay_tok)/1e6, (_off + STEPS * TC.batch * MC.seq)/1e6, color="#cbc5bb", alpha=0.3, lw=0)
ax.set_xlabel("training tokens (millions)"); ax.set_ylabel("validation loss"); ax.legend(frameon=False)
ax.set_title("conversion at the dashed line; shaded: learning-rate decay", fontsize=9)
_save(fig, "v3_loss.png")

# figure 2: balance over R3
_ml3 = R3["extra"]["moe_log"]
_full = [m for i, m in enumerate(_ml3) if m["step"] - (_ml3[i - 1]["step"] if i else 0) >= TC.log_every]   # drop short windows
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot([m["step"] for m in _ml3], [m["maxvio"] for m in _ml3], color="#b1521f", label="MaxVio")
ax.plot([m["step"] for m in _ml3], [m["clone_dist"] for m in _ml3], color="#2c6b45", label="clone distance")
ax2 = ax.twinx(); ax2.bar([m["step"] for m in _ml3], [m["dead"] for m in _ml3], width=15, color="#9a968e", alpha=0.5)
ax2.set_ylabel("dead experts (grey bars)")
ax.axvline(V3.switch, color="#9a968e", ls=":", lw=0.8)
ax.set_xlabel("R3 step"); ax.legend(frameon=False)
_save(fig, "v3_balance.png")

# figure 3: final load per expert
_share3 = np.array(_full[-1]["share"]).reshape(MC.n_layer, N_EXP)
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(_share3, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.25, _share3.max()))
for f in range(1, XC.families):
    ax.axvline(f * XC.clones - 0.5, color="white", lw=1)
ax.set_xlabel("expert (4 families of 8)"); ax.set_ylabel("layer")
ax.set_title(f"R3, steps {_full[-1]['step'] - TC.log_every + 1}-{_full[-1]['step']}: share of each family's tokens (even = 0.125)", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046)
_save(fig, "v3_load.png")
R["v3"].update({"maxvio_end": _full[-1]["maxvio"], "maxvio_max_after_switch": max(m["maxvio"] for m in _full if m["step"] > V3.switch),
                "dead_max": max(m["dead"] for m in _full), "dead_end": _full[-1]["dead"], "clone_dist_end": _ml3[-1]["clone_dist"],
                "balance_window_end": _full[-1]["step"]})
print(f"\nR3 balance (25-step windows): MaxVio {R['v3']['maxvio_end']:.3f} at the end (at most {R['v3']['maxvio_max_after_switch']:.3f} "
      f"after the switch); dead experts at the end {R['v3']['dead_end']}, at most {R['v3']['dead_max']}; "
      f"clone distance {R['v3']['clone_dist_end']:.3f}")

# the same three prompts, from R2 and R3
R["samples_v3"] = {}
for k, mk in (("R2", make_dense), ("R3", make_r3)):
    m = mk(); m.load_state_dict(load_final({"R2": "R2_dense", "R3": f"R3_moe_sw{V3.switch}"}[k])["model"])
    R["samples_v3"][k] = [sample(m, p) for p in PROMPTS]
    del m
for i, p in enumerate(PROMPTS):
    print(f"\n  R2 > {R['samples_v3']['R2'][i]!r}\n  R3 > {R['samples_v3']['R3'][i]!r}")

# %% [markdown]
# ## Cell 20 — checks, summary.json, bundle
#
# ### What this block does
# Collects every check that ran in this version into one list and prints `N/N`, writes
# `summary.json` and `RESULTS.md`, and zips `outputs/` into `era5s14_<version>_outputs.zip`.
# Download it with the executed notebook and hand both back.

# %%
CHECKS = CHECKS_V1 + [
    ("conversion: clones hold R1's slices, weights and Adam averages", R["gates"]["conversion_state"]),
    ("G1: converted logits match R1's in fp64", G1["fp64"] < TH.g1_fp64_rel),
    ("G1: converted validation loss matches R1's", abs(G1_val - R1["final_val"]) < TH.g1_loss_abs),
    ("G2: every router gets a gradient (joint rescaling)", R["gates"]["G2"]["joint_min"] > 0),
    ("G2: the per-family rule gives no router gradient (zero up to rounding)", R["gates"]["G2"]["per_family_max"] < 1e-6 * R["gates"]["G2"]["joint_min"]),
    ("G3: bias balancing evens out a skewed toy layer", R["gates"]["G3"]["end"] < TH.g3_maxvio),
    ("G4: parameter counts match the formula", R["gates"]["G4"]["total"] == _want_tot),
]
if PREVIEW:
    CHECKS += [("preview: finished its steps", PREVIEW["train"][-1]["step"] == XC.preview_steps),
               ("preview: validation loss fell after conversion", PREVIEW["final_val"] < PREVIEW["val"][0]["loss"]),
               ("preview: no dead experts in the last window", _ml[-1]["dead"] == 0),
               ("preview: the clones have started to separate", _ml[-1]["clone_dist"] > 0)]
if DIAG:
    CHECKS.append(("diagnostic v2.1: B (dense in MoE clothing) tracks A", abs(DIAG["B"]["hist"]["final_val"] - _a_end) < DC_.b_vs_a_tol))
if DIAG2:
    CHECKS.append(("diagnostic v2.2: all six runs finished", len(DIAG2) == 6 and all(
        d["hist"]["train"][-1]["step"] == DC_.steps for d in DIAG2.values())))
CHECKS += [
    ("v3: R2s starts as R1 (slowed FFN keeps the function)", R["v3"]["r2s_exact_rel"] < 1e-4),
    ("v3: R2, R2s and R3 finished their steps", all(R["runs"][k]["train"][-1]["step"] == STEPS for k in ("R2", "R2s", "R3"))),
    ("v3: R3 started at R1's validation loss", abs(R3["val"][0]["loss"] - R1["final_val"]) < TH.g1_loss_abs),
    ("G5: the MoE keeps training (loss below the conversion value by more than the noise)", R["v3"]["g5"]),
    ("v3: no dead experts at the end of R3", R["v3"]["dead_end"] == 0),
]
print("CHECKS")
for name, ok in CHECKS:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
_npass = sum(1 for _, ok in CHECKS if ok)
print(f"\n{NOTEBOOK_VERSION.upper()}: {_npass}/{len(CHECKS)} passed")
R["acceptance"] = {"passed": _npass, "total": len(CHECKS),
                   "checks": [{"name": n, "result": bool(ok)} for n, ok in CHECKS]}
R["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
_atomic_save(R, out("summary.json"), as_json=True)

def _fmt(x, f="{:,.0f}"):
    return f.format(x) if x is not None else "n/a"

_rows = [("R1 dense", R1), ("R2 dense continues", R2), ("R2s dense, FFN at 1/8 speed", R2S), ("R3 MoE", R3)]
L_ = [f"# Session 14 results — notebook {NOTEBOOK_VERSION}, profile {PROFILE}\n",
      f"Dense model {MC.n_layer} × {MC.d}, {PB['total']/1e6:.1f}M parameters. MoE: {N_EXP} experts per layer "
      f"({XC.families} families × {XC.clones}), {R['moe']['total_params']/1e6:.1f}M total, "
      f"{R['moe']['active_params']/1e6:.2f}M active. Device {DEVICE}{' (' + GPU_NAME + ')' if GPU_NAME else ''}.\n",
      f"Checks: **{_npass}/{len(CHECKS)}**\n",
      "| run | steps | val at start | final val | tokens/s | peak MiB |", "|---|---|---|---|---|---|"]
for lab, h in _rows:
    L_.append(f"| {lab} | {h['steps']:,} | {h['val'][0]['loss']:.4f} | {h['final_val']:.4f} | {_fmt(h.get('tokens_per_second'))} | "
              f"{_fmt((h.get('peak_mem_bytes') or 0)/2**20 or None)} |")
L_ += ["", *("- " + v for v in R["v3"]["verdicts"]), "", "**Conclusion:** " + R["v3"]["conclusion"],
       f"\nR3 balance: MaxVio {R['v3']['maxvio_end']:.3f} at the end, dead experts {R['v3']['dead_end']}.",
       "\n## Checks\n"] + [f"- [{'PASS' if c['result'] else 'FAIL'}] {c['name']}" for c in R["acceptance"]["checks"]]
with open(out("RESULTS.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L_) + "\n")

_zip = out(f"era5s14_{NOTEBOOK_VERSION}_outputs.zip")
with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _dirs, files in os.walk(OUT_DIR):
        for name in files:
            if name.endswith((".png", ".json", ".md")):
                p = os.path.join(root, name)
                z.write(p, arcname=os.path.relpath(p, OUT_DIR))
print(f"\nbundle: {_zip}  ({os.path.getsize(_zip)/1024:.0f} KB)")
print("Download it together with the executed .ipynb.")
