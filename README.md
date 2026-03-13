# Gender Bias Drift Across LoRA & QLoRA Fine-Tuning in LLMs

**Authors:** Aryan Daga, Aman Jaiswal, Lipi Singhal, Tisha Bhavsar

---

## Overview

This project investigates whether fine-tuning a large language model on a neutral downstream task causes measurable **gender bias drift** — a shift in how the model expresses gender stereotypes, even though the fine-tuning data contains no explicit gender content.

We use `facebook/opt-1.3b` as our base model and apply two parameter-efficient fine-tuning (PEFT) techniques:

- **LoRA** — Low-Rank Adaptation on a full-precision (fp16) model
- **QLoRA** — LoRA applied on top of a 4-bit NF4 quantized model

Both are fine-tuned on **SST-2** (sentiment analysis) and evaluated for gender bias using the **BBQ benchmark** (gender_identity subset) before and after fine-tuning.

---

## Research Question

> Does adapting a language model to a sentiment task — with no gender-related content — shift its expression of gender stereotypes?

---

## Datasets

| Role | Dataset | Size used |
|------|---------|-----------|
| Fine-tuning (task) | [SST-2](https://huggingface.co/datasets/nyu-mll/glue) (`glue/sst2`) | 1,000 train / 200 eval |
| Bias evaluation | [BBQ Gender Identity](https://huggingface.co/datasets/lighteval/bbq) (`lighteval/bbq`) | 200 ambiguous + 200 disambiguated |

**SST-2** provides binary sentiment labels (positive/negative) for movie review sentences. It contains no gender-related content, making it ideal for testing whether bias drift occurs as a side-effect of task fine-tuning.

**BBQ (Bias Benchmark for QA)** measures stereotypical reasoning in question-answering. Each example presents a context with two people and asks a question where:
- In **ambiguous** contexts — the correct answer is always *"Unknown"*; picking any named person indicates bias
- In **disambiguated** contexts — one person is factually identified; accuracy tests factual reasoning

---

## Method

### Evaluation (Log-Probability Scoring)
No text generation is used. For each multiple-choice question, we compute the log-probability the model assigns to each answer token given the prompt and select the highest-scoring option. This is fast, deterministic, and avoids sampling noise.

### LoRA Configuration
```
r = 8  |  alpha = 16  |  target: q_proj, v_proj  |  dropout = 0.05
Trainable params: ~4M / 1.3B  (<0.3% of total)
Training: 2 epochs, lr = 2e-4, batch = 16 (8 × 2 grad accum), cosine schedule
```

### QLoRA Additions
```
Base model: 4-bit NF4 quantization + double quantization
Compute dtype: fp16
Optimizer: paged_adamw_8bit  (prevents OOM from optimizer state spikes)
VRAM (4-bit model): ~0.7 GB  vs  ~2.6 GB for fp16 LoRA
```

---

## Results

### SST-2 Sentiment Accuracy

| Condition | Accuracy |
|-----------|----------|
| Baseline (fp16) | 0.785 |
| **Post-LoRA** | **0.945** (+16.0%) |
| Baseline (4-bit) | 0.680 |
| **Post-QLoRA** | **0.920** (+24.0%) |

Fine-tuning substantially improves sentiment accuracy in both cases, confirming the training signal is effective.

---

### BBQ Gender Bias Metrics

#### LoRA (fp16 baseline → post-LoRA)

| Metric | Baseline | Post-LoRA | Change |
|--------|----------|-----------|--------|
| Disambig Accuracy | 0.340 | 0.325 | -0.015 |
| Ambig Accuracy *(↑ = less biased)* | 0.265 | 0.405 | **+0.140** |
| Stereotype Rate *(↓ = less biased)* | 0.360 | 0.280 | **-0.080** |
| Bias Score *(0 = neutral)* | -0.280 | -0.440 | -0.160 |

#### QLoRA (4-bit baseline → post-QLoRA)

| Metric | Baseline | Post-QLoRA | Change |
|--------|----------|------------|--------|
| Disambig Accuracy | 0.325 | 0.300 | -0.025 |
| Ambig Accuracy *(↑ = less biased)* | 0.375 | 0.440 | **+0.065** |
| Stereotype Rate *(↓ = less biased)* | 0.265 | 0.270 | +0.005 |
| Bias Score *(0 = neutral)* | -0.470 | -0.460 | +0.010 |

---

### Visualizations

**LoRA — before vs after:**
![LoRA bias plot](results/lora_bias_plot.png)

**LoRA vs QLoRA — full comparison:**
![Comparison plot](results/lora_vs_qlora_comparison.png)

---

## Key Findings

1. **LoRA fine-tuning reduces stereotypical bias.** After fine-tuning on SST-2, the model picks the "Unknown" answer in ambiguous BBQ contexts significantly more often (+14.0 pp), and its stereotype rate drops from 0.36 to 0.28. The bias score becomes more negative (more counter-stereotypical), suggesting the LoRA update shifted the model's internal gender representations even though the training task was entirely unrelated to gender.

2. **QLoRA shows a smaller but similar trend.** Ambiguous accuracy also improves (+6.5 pp) post-QLoRA, but the stereotype rate barely changes (+0.5 pp). The overall bias score remains nearly flat (-0.470 → -0.460), suggesting QLoRA's 4-bit quantization dampens the degree to which fine-tuning reshapes gender associations.

3. **Quantization itself shifts the baseline.** The 4-bit baseline (QLoRA's starting point) shows notably different bias metrics than the fp16 baseline — most strikingly, a lower stereotype rate (0.265 vs 0.360) and a more negative bias score (-0.470 vs -0.280). This suggests NF4 quantization itself perturbs the model's stereotype encoding before any fine-tuning occurs.

4. **Task performance and bias drift are decoupled.** LoRA gains +16pp on SST-2 while also reducing bias; QLoRA gains +24pp on SST-2 with minimal bias change. This shows that task performance improvement does not necessarily correlate with bias amplification — and in fact may accompany bias reduction under PEFT.

---

## Repo Structure

```
├── notebooks/
│   ├── 01_lora_experiment.ipynb     # LoRA: baseline eval → fine-tune → re-eval + write-up
│   ├── 02_qlora_experiment.ipynb    # QLoRA: same pipeline with 4-bit model
│   ├── run_lora.py                  # Standalone script for LoRA experiment
│   └── run_qlora.py                 # Standalone script for QLoRA experiment
├── results/
│   ├── lora_results.json / .csv     # LoRA metrics
│   ├── qlora_results.json / .csv    # QLoRA metrics
│   ├── lora_bias_plot.png           # LoRA before/after chart
│   ├── lora_vs_qlora_comparison.png # 3-way comparison plot
│   ├── lora_adapter/                # Saved LoRA adapter weights
│   └── qlora_adapter/               # Saved QLoRA adapter weights
├── requirements.txt
└── README.md
```

---

## Setup & Reproduction

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run experiments (or open notebooks in Jupyter)
python notebooks/run_lora.py
python notebooks/run_qlora.py
```

**Requirements:** Python 3.11+, NVIDIA GPU with CUDA, ~4 GB VRAM minimum (OPT-1.3B fp16 needs ~2.6 GB; 4-bit QLoRA needs ~0.7 GB).

---

## References

- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
- Parrish et al. (2022). *BBQ: A Hand-Built Bias Benchmark for Question Answering.* ACL Findings 2022.
- Zhang et al. (2022). *OPT: Open Pre-trained Transformer Language Models.* arXiv:2205.01068.
- Socher et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP 2013. (SST-2)
