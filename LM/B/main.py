"""
main.py — Part 1.B: Fine-tuning GPT-2 with LoRA
=================================================

OVERVIEW
--------
Load pre-trained GPT-2, inject LoRA adapters, fine-tune on Penn TreeBank.
Only the LoRA parameters (A and B matrices) are trained; original weights are frozen.

TODO: implement experiments for different LoRA ranks (r) and alpha values.
"""

# TODO: device setup (CUDA / MPS / CPU)

# TODO: load tokenizer and dataset

# TODO: load pre-trained GPT-2 from HuggingFace

# TODO: apply LoRA adapters

# TODO: freeze original parameters, keep only LoRA trainable

# TODO: training loop with experiments (different r values)

# TODO: evaluate on test set, save results to results_1B.csv
