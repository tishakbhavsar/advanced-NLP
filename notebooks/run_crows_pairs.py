"""
Tasks 2 & 3: CrowS-Pairs SPS + Bolukbasi Geometric Analysis
=============================================================
Dataset: CrowS-Pairs (Nangia et al. 2020), gender subset (262 pairs)
         Loaded directly from GitHub CSV.

Task 2 — SPS (Stereotype Preference Score):
  For each (sent_more, sent_less) pair:
    score = mean log P(token_i | context) per sentence  [causal LM adaptation]
    model prefers stereotype if score(sent_more) > score(sent_less)
  SPS = 100 * fraction_preferred  |  Ideal unbiased = 50%

Task 3 — Bolukbasi Geometric SPS:
  Reuses gender direction g from run_bolukbasi.py results.
  For each pair, diff the changed words between sent_more and sent_less.
  Project those word representations onto g.
  Geometric SPS = % of pairs where changed words in sent_more project
  more strongly onto g than those in sent_less.
  Measures whether the model's geometry encodes the same stereotypes.
"""

import os, sys, re, json, csv, io
import urllib.request
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "facebook/opt-1.3b"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load CrowS-Pairs ─────────────────────────────────────────────────────────
CROWS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
print("Downloading CrowS-Pairs...")
with urllib.request.urlopen(CROWS_URL) as r:
    content = r.read().decode("utf-8")
all_rows  = list(csv.DictReader(io.StringIO(content)))
crows_gender = [r for r in all_rows if r["bias_type"] == "gender"]
print(f"Gender subset: {len(crows_gender)} pairs")


# ── Gender pairs (same as run_bolukbasi.py) ──────────────────────────────────
GENDER_PAIRS = [
    ("he","she"), ("him","her"), ("his","hers"), ("man","woman"),
    ("men","women"), ("boy","girl"), ("male","female"),
    ("father","mother"), ("son","daughter"), ("brother","sister"),
    ("husband","wife"), ("king","queen"),
]


# ── Core helpers ─────────────────────────────────────────────────────────────
def score_sentence(model, tokenizer, sentence):
    """Mean log-likelihood per token (higher = more probable)."""
    inputs = tokenizer(sentence, return_tensors="pt",
                       truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return -loss.item()   # negate NLL -> log-likelihood


def get_repr(model, tokenizer, word):
    """Last-layer hidden state averaged over token positions."""
    inputs = tokenizer(" " + word, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0].mean(0).cpu().float().numpy()


def compute_gender_direction(model, tokenizer):
    diffs = []
    for m_word, f_word in GENDER_PAIRS:
        diffs.append(get_repr(model, tokenizer, m_word) -
                     get_repr(model, tokenizer, f_word))
    pca = PCA(n_components=1)
    pca.fit(np.array(diffs))
    return pca.components_[0]


def get_diff_words(sent_more, sent_less):
    """Words present in one sentence but not the other (punctuation stripped)."""
    def clean(s):
        return set(re.sub(r"[^\w\s]", "", s.lower()).split())
    w_more, w_less = clean(sent_more), clean(sent_less)
    return list(w_more - w_less), list(w_less - w_more)


def compute_proj(model, tokenizer, words, g):
    """Mean |cos(word_repr, g)| for a list of words. Returns 0 if empty."""
    if not words:
        return 0.0
    g_n = g / (np.linalg.norm(g) + 1e-8)
    scores = []
    for w in words:
        v = get_repr(model, tokenizer, w)
        v_n = v / (np.linalg.norm(v) + 1e-8)
        scores.append(abs(float(np.dot(v_n, g_n))))
    return float(np.mean(scores))


def analyse_model(model, tokenizer, label):
    model.eval()
    print(f"\n[{label}] Computing gender direction for geometric analysis...")
    g = compute_gender_direction(model, tokenizer)

    # ── Task 2: SPS ──────────────────────────────────────────────────────────
    print(f"[{label}] Computing SPS on {len(crows_gender)} gender pairs...")
    stereo_wins = 0
    for row in tqdm(crows_gender, desc="  SPS", leave=False):
        s_more = score_sentence(model, tokenizer, row["sent_more"])
        s_less = score_sentence(model, tokenizer, row["sent_less"])
        if s_more > s_less:
            stereo_wins += 1
    sps = 100.0 * stereo_wins / len(crows_gender)

    # ── Task 3: Geometric SPS ────────────────────────────────────────────────
    print(f"[{label}] Computing Bolukbasi Geometric SPS...")
    geo_wins     = 0
    geo_valid    = 0   # pairs with non-empty diffs on both sides
    proj_more_all = []
    proj_less_all = []

    for row in tqdm(crows_gender, desc="  Geometric SPS", leave=False):
        diff_more, diff_less = get_diff_words(row["sent_more"], row["sent_less"])
        if not diff_more or not diff_less:
            continue
        p_more = compute_proj(model, tokenizer, diff_more, g)
        p_less = compute_proj(model, tokenizer, diff_less, g)
        proj_more_all.append(p_more)
        proj_less_all.append(p_less)
        geo_valid += 1
        if p_more > p_less:
            geo_wins += 1

    geo_sps = 100.0 * geo_wins / geo_valid if geo_valid > 0 else 0.0
    avg_proj_more = float(np.mean(proj_more_all)) if proj_more_all else 0.0
    avg_proj_less = float(np.mean(proj_less_all)) if proj_less_all else 0.0

    result = {
        "sps":             round(sps,          2),
        "geo_sps":         round(geo_sps,       2),
        "stereo_wins":     stereo_wins,
        "total_pairs":     len(crows_gender),
        "geo_valid_pairs": geo_valid,
        "avg_proj_more":   round(avg_proj_more, 4),
        "avg_proj_less":   round(avg_proj_less, 4),
    }
    print(f"  SPS = {sps:.1f}%  (ideal = 50%)")
    print(f"  Geometric SPS = {geo_sps:.1f}%  (valid pairs: {geo_valid})")
    print(f"  Avg projection — stereo words: {avg_proj_more:.4f}  |  anti-stereo: {avg_proj_less:.4f}")
    return result


# ── Load helpers (same pattern as run_bolukbasi.py) ──────────────────────────
def load_baseline():
    print("\nLoading baseline OPT-1.3B (fp16)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    return m, tok


def load_lora(adapter_path):
    print("\nLoading post-LoRA (merged)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    m = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    return m, tok


def load_qlora(adapter_path):
    print("\nLoading post-QLoRA (4-bit + adapter)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto")
    m = PeftModel.from_pretrained(base, adapter_path)
    return m, tok


def free(model):
    del model
    torch.cuda.empty_cache()


# ── Run ──────────────────────────────────────────────────────────────────────
results = {}

m, tok = load_baseline()
results["baseline"] = analyse_model(m, tok, "Baseline")
free(m)

lora_path  = os.path.join(RESULTS_DIR, "lora_adapter")
m, tok     = load_lora(lora_path)
results["post_lora"] = analyse_model(m, tok, "Post-LoRA")
free(m)

qlora_path = os.path.join(RESULTS_DIR, "qlora_adapter")
m, tok     = load_qlora(qlora_path)
results["post_qlora"] = analyse_model(m, tok, "Post-QLoRA")
free(m)

# ── Save JSON ────────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS_DIR, "crows_pairs_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {out_path}")

# ── Plots ────────────────────────────────────────────────────────────────────
labels = ["Baseline", "Post-LoRA", "Post-QLoRA"]
keys   = ["baseline", "post_lora", "post_qlora"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("CrowS-Pairs Gender Bias — OPT-1.3B", fontsize=13)

# Panel 1: SPS
ax = axes[0]
sps_vals = [results[k]["sps"] for k in keys]
bars = ax.bar(labels, sps_vals, color=colors, width=0.45)
ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Ideal (50%)")
ax.set_ylim(0, 100)
ax.set_ylabel("SPS (%)")
ax.set_title("Stereotype Preference Score\n(SPS, lower->50% = less biased)")
ax.legend(fontsize=8)
for b, v in zip(bars, sps_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

# Panel 2: Geometric SPS
ax = axes[1]
geo_vals = [results[k]["geo_sps"] for k in keys]
bars = ax.bar(labels, geo_vals, color=colors, width=0.45)
ax.axhline(50, color="red", linestyle="--", linewidth=1, label="Ideal (50%)")
ax.set_ylim(0, 100)
ax.set_ylabel("Geometric SPS (%)")
ax.set_title("Bolukbasi Geometric SPS\n(% pairs: stereo words more gendered)")
ax.legend(fontsize=8)
for b, v in zip(bars, geo_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

# Panel 3: Average gender-direction projection
ax = axes[2]
x = np.arange(len(labels))
w = 0.3
more_vals = [results[k]["avg_proj_more"] for k in keys]
less_vals  = [results[k]["avg_proj_less"] for k in keys]
ax.bar(x - w/2, more_vals, w, label="Stereo words (sent_more)", color="#E15759")
ax.bar(x + w/2, less_vals, w, label="Anti-stereo words (sent_less)", color="#76B7B2")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("|cos(w, g)|")
ax.set_title("Avg Projection onto Gender Direction\n(Bolukbasi, changed words only)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "crows_pairs_results.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved.")

print("\n=== SUMMARY ===")
print(f"{'':15s}  {'SPS':>7}  {'Geo-SPS':>9}")
for k, lbl in zip(keys, labels):
    r = results[k]
    print(f"{lbl:15s}  {r['sps']:>6.1f}%  {r['geo_sps']:>8.1f}%")
