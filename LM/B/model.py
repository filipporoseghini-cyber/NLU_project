"""
model.py — Part 1.B: Fine-tuning GPT-2 with LoRA
==================================================

OVERVIEW
--------
In this part we fine-tune the pre-trained GPT-2 model (from HuggingFace) on
the Penn TreeBank dataset using LoRA (Low-Rank Adaptation).

LoRA freezes the original pre-trained weights and injects trainable low-rank
decomposition matrices into the attention layers:
    W' = W + BA   where B ∈ R^{d×r}, A ∈ R^{r×k}, rank r << min(d, k)

This drastically reduces the number of trainable parameters.

TODO: implement LoRA adapter classes and modified GPT-2 with LoRA.
"""

# TODO: import statements

# TODO: LoRALinear class (wraps nn.Linear with trainable A, B matrices)

# TODO: apply_lora(model, r, lora_alpha) — inject LoRA into GPT-2 attention layers

# TODO: count_trainable_parameters(model)
