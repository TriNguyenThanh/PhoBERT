# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository hosts a scientific NLP research project: **Multi-Class Hate Speech Detection on Vietnamese Social Media Texts** using **PhoBERT-base** combined with **Class Weighting** on the **ViHSD** benchmark dataset (Luu et al., EMNLP 2021).

- **Objective:** Controlled A/B experimentation comparing Model A (Standard Cross-Entropy Baseline) vs. Model B (Weighted Cross-Entropy Proposed) to solve severe class imbalance.
- **Dataset:** ViHSD (33,400 Facebook comments). Strict benchmark split: Train (24,048 - 72%) : Dev (2,672 - 8%) : Test (6,680 - 20%).
- **Cleaned Train Set:** 21,098 samples (1,130 duplicates and train-dev/test leakage samples removed).
- **Classes:** `0: CLEAN` (~82.71%), `1: OFFENSIVE` (~6.77%), `2: HATE` (~10.52%).

## Architecture & Methodology

```
ViHSD (Raw 33.4k) ──► Stratified Split (72:8:20) ──► Preprocessing & Cleaning (21.1k Train)
                                                              │
                                                              ▼
               PhoBERT Tokenizer (BPE, max_length=128) ◄── Word Segmentation (PyVi)
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
2. **Text Processing Pipeline (Data-centric):**
   - Unicode normalization (NFC).
   - Emoji removal (prevents subword fragmentation and token crashes).
   - URL & email removal.
   - Teencode & slang normalization via 414-word dictionary (`data/teencode_414.json`).
   - **No stopword removal** (Luu et al., 2024 proved stopword removal degrades Transformer self-attention).
   - Vietnamese word segmentation (words connected with `_`) via `pyvi` (`ViTokenizer`).
   - Dynamic batch padding with `max_length = 128` via `DataCollatorWithPadding`.
3. **Class Weighting (Model-centric):**
   - Balanced class weight formula: $w_c = \frac{N}{3 \times N_c}$ computed solely from cleaned training set ($N = 21,098$).
   - Values: $w_{\text{CLEAN}} \approx 0.4075$ (~0.41), $w_{\text{OFFENSIVE}} \approx 4.8568$ (~4.86), $w_{\text{HATE}} \approx 2.9401$ (~2.94).
4. **Training Hyperparameters:** `lr = 2e-5`, `batch_size = 16`, `epochs = 4`, `weight_decay = 0.01`, linear warmup (10%), gradient clipping `max_grad_norm = 1.0`.
5. **Execution Strategy:** 4 independent seeds executed in parallel on Google Colab GPU T4 to measure standard deviation ($\mu \pm \sigma$):
   - `Seed 42`: Trí (Baseline reference)
   - `Seed 123`: Ngân (Ablation study & stability)
   - `Seed 456`: Hiếu (Independent verification)
   - `Seed 789`: Tài (Independent verification)

## Common Commands & Environment Setup

### Environment Dependencies
Install required packages for local development or Colab:
```bash
pip install -q pyvi emoji "transformers>=4.38.0" "accelerate>=0.33.0"
```

### Inspecting & Validating Pipeline Diagrams
- Render pipeline diagram locally:
  ```bash
  python3 -m http.server 8000
  # Open http://localhost:8000/pipeline/pipeline_diagram.html
  ```

### Evaluating Generated Predictions
Verify test output format (6,680 rows) and calculate metrics:
```python
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("outputs/seed42_teencode_414/predictions.csv")
assert len(df) == 6680, "Test set size must be exactly 6,680 rows"
assert list(df.columns) == ['id', 'text', 'ground_truth', 'predicted_label']

print(classification_report(df['ground_truth'], df['predicted_label'], digits=4))
print(confusion_matrix(df['ground_truth'], df['predicted_label'], labels=['CLEAN', 'OFFENSIVE', 'HATE']))
```

## Repository Structure

- `PhoBERT_ViHSD_ClassWeight.ipynb`: End-to-end training and evaluation notebook with teencode normalization (414 words), class weighting, and automated Google Drive backup.
- `original_paper_notebooks/`: Original baseline notebooks from Luu et al. (2021).
- `data/`: ViHSD dataset (`train.csv`, `dev.csv`, `test.csv`) and teencode dictionaries (`teencode_414.json`, `teencode_35.json`).
- `outputs/`: Output deliverables grouped by seed (`seed42_teencode_414`, `seed42_teencode_35`, `seed123_teencode_414`, `seed456_teencode_414`, `seed789_teencode_414`).
- `docs/`: Technical documentation series:
  - `01_tong_quan_de_tai.md`: Project overview & ViHSD dataset.
  - `02_kien_truc_va_phuong_phap.md`: PhoBERT architecture, word segmentation & class weighting theory.
  - `03_thiet_ke_thuc_nghiem_va_danh_gia.md`: Experimental design, hyperparameters & metrics.
  - `04_huong_dan_trien_khai_colab.md`: Step-by-step Colab Web & VSCode Colab Extension guide.
  - `05_mau_bao_cao_so_sanh.md`: Academic comparative evaluation report (Baseline vs. Class Weight, ablation, multi-seed).
  - `06_noi_dung_bao_cao.md`: Quick presentation data sheet & Q&A defense cheat sheet.
  - `docs/papers/`: Literature baseline reviews (Luu 2021, Tran 2022, Luu 2024, Hoang 2023).
- `pipeline/`: Pipeline diagrams and visual assets.

## Deliverables Checklist
- `predictions.csv`: Model B (Class Weight) predictions on 6,680 Test samples (`id, text, ground_truth, predicted_label`).
- `aggregated_metrics.csv`: Comparative quantitative report between Model A and Model B.
- `confusion_matrix.png`: Side-by-side confusion matrix visualization.
- Model checkpoints saved to Google Drive under `MyDrive/PhoBERT_ViHSD/`.
