# Part 1.A — Language Modeling from scratch (GPT-2)

## Overview
Training a GPT-2 style decoder-only Transformer from scratch on the Penn TreeBank dataset.
Target: **Test Perplexity < 250**.

## Files
- `main.py` — runs all experiments (hyperparameter search, training, evaluation)
- `model.py` — GPT-2 architecture (MultiHeadAttention, FeedForward, TransformerBlock, GPT2)
- `functions.py` — training loop, eval loop, early stopping, weight init
- `utils.py` — dataset loading, tokenizer setup, DataLoader creation
- `dataset/PennTreeBank/` — PTB corpus (train/valid/test)
- `bin/` — saved model checkpoints (.pt files, git-ignored)

## How to run
```bash
conda activate nlu26
cd LM/A
python main.py
```
Results are saved to `results_1A.csv`.

## Hyperparameter search strategy
- Step 0: learning rate (lr ∈ {0.1, 0.01, 0.001, 0.0001}) — small model
- Step 1: architecture (d_model, n_heads, num_layers, ff_dim) — greedy one-at-a-time
- Step 2: dropout (0.1, 0.2) — with best architecture
- Step 3: weight tying — with best dropout
