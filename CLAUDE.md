# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository hosts a scientific NLP research project: **Multi-Class Hate Speech Detection on Vietnamese Social Media Texts** using **PhoBERT-base** combined with **Class Weighting** on the **ViHSD** benchmark dataset (Luu et al., EMNLP 2021).

- **Objective:** Controlled A/B experimentation comparing Model A (Standard Cross-Entropy Baseline) vs. Model B (Weighted Cross-Entropy Proposed) to solve severe class imbalance.
- **Dataset:** ViHSD (33,400 Facebook comments). Strict benchmark split: Train (24,048 - 72%) : Dev (2,672 - 8%) : Test (6,680 - 20%).
- **Classes:** `0: CLEAN` (~82.71%), `1: OFFENSIVE` (~6.77%), `2: HATE` (~10.52%).

## Architecture & Methodology

```
ViHSD (Raw 33.4k) ──► Stratified Split (72:8:20) ──► Preprocessing & Cleaning
                                                              │
                                                              ▼
               PhoBERT Tokenizer (BPE, max_length=128) ◄── Word Segmentation (RDRSegmenter/PyVi)
                               │
               ┌───────────────┴───────────────┐
               ▼ (Standard CE)                 ▼ (Balanced Class Weights)
         Model A (Baseline)              Model B (Proposed)
               │                               │
               └───────────────┬───────────────┘
                               ▼ (4 Seeds: 42, 123, 456, 789)
                   Dev Set Model Selection (Best Macro F1)
                               │
                               ▼
            Test Set Evaluation (6,680 samples)
            - Primary: Macro F1, Weighted F1, Per-class F1
            - Trade-off analysis: Per-class Precision & Recall
            - Visualization: Side-by-side Confusion Matrices (3x3)
            - Statistical report: Mean ± Standard Deviation (μ ± σ)
```

### Key Technical Decisions
1. **Backbone:** `vinai/phobert-base-v2` (fine-tuning all parameters, FP16 enabled).
2. **Text Processing Pipeline:**
   - Unicode normalization (NFC).
   - Emoji removal (prevents JVM crashes in `RDRSegmenter` and subword fragmentation).
   - URL & email removal.
   - Teencode & slang normalization via dictionary (Tran et al., 2022).
   - **No stopword removal** (Luu et al., 2024 proved stopword removal degrades Transformer self-attention).
   - Vietnamese word segmentation (words connected with `_`) via `py_vncorenlp` (RDRSegmenter) or `pyvi`.
   - Dynamic batch padding with `max_length = 128`.
3. **Class Weighting:**
   - Balanced class weight formula: $w_c = \frac{N}{3 \times N_c}$ computed solely from cleaned training set.
   - Values: $w_{\text{CLEAN}} \approx 0.40$, $w_{\text{OFFENSIVE}} \approx 4.92$, $w_{\text{HATE}} \approx 3.17$.
4. **Training Hyperparameters:** `lr = 2e-5`, `batch_size = 16`, `epochs = 4`, `weight_decay = 0.01`, linear warmup (10%), gradient clipping `max_grad_norm = 1.0`.
5. **Execution Strategy:** 4 independent seeds (`42, 123, 456, 789`) executed by 4 teammates in parallel on Google Colab GPU T4 to measure standard deviation ($\mu \pm \sigma$).

## Common Commands & Environment Setup

### Environment Dependencies
Install required packages for local development or Colab:
```bash
pip install -q pyvi emoji "transformers>=4.38.0" "accelerate>=0.33.0"
```

### Inspecting & Validating Pipeline Diagrams and Documentation
- Render or test pipeline diagram locally:
  ```bash
  python3 -m http.server 8000
  # Open http://localhost:8000/pipeline/pipeline_diagram.html
  ```

### Evaluating Generated Predictions
Verify test output format and calculate metrics:
```python
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("pipeline/outputs/predictions.csv")
assert len(df) == 6680, "Test set size must be exactly 6,680 rows"
assert list(df.columns) == ['id', 'text', 'ground_truth', 'predicted_label']

print(classification_report(df['ground_truth'], df['predicted_label'], digits=4))
print(confusion_matrix(df['ground_truth'], df['predicted_label'], labels=['CLEAN', 'OFFENSIVE', 'HATE']))
```

## Repository Structure

- `pipeline/`:
  - `NLP.md`: Formal technical specification and experiment protocol.
  - `pipeline.md`: In-depth narrative breakdown of each step, parameters, and rationale.
  - `pipeline_diagram.html`: Monochrome linear 6-step architecture diagram in standalone HTML/SVG.
  - `ViHSD_monolingual_Transformers.ipynb`: End-to-end training and evaluation notebook for Colab (modified from original author Luu et al. 2021).
- `docs/`:
  - `01_tong_quan_de_tai.md` to `05_mau_bao_cao_so_sanh.md`: Technical documentation series.
  - `docs/papers/`: Analyses of literature baselines (Luu 2021, Tran 2022, Luu 2024, Hoang 2023).
- `paper/`: Original academic PDF papers.
- `search/`: Literature deep research notes.

## Deliverables Checklist
- `predictions.csv`: Model B (Class Weight) predictions on 6,680 Test samples (`id, text, ground_truth, predicted_label`).
- `predictions_A_seed{s}.csv` & `predictions_B_seed{s}.csv` for seeds `42, 123, 456, 789`.
- `aggregated_metrics.csv`: Mean ± Std ($\mu \pm \sigma$) summary across all 4 seeds.
- Model checkpoints saved to Google Drive.
- A/B comparative evaluation report.
