"""
model.py — Part 2.A: NLU model from scratch (BiLSTM or similar)
================================================================

OVERVIEW
--------
Joint model for Intent Classification and Slot Filling.
Common architecture: BiLSTM encoder shared between:
  - Intent classifier (pooled representation → softmax over intents)
  - Slot tagger (per-token representation → softmax over slot labels, BIO scheme)

TODO: implement the joint NLU model.
"""

# TODO: import statements

# TODO: NLUModel class (joint intent + slot)
#   - Embedding layer (word embeddings)
#   - BiLSTM encoder
#   - Intent classification head (linear + softmax)
#   - Slot filling head (linear + CRF or just softmax per token)

# TODO: count_parameters()
