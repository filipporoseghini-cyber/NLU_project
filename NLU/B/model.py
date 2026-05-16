"""
model.py — Part 2.B: NLU with pre-trained BERT
================================================

OVERVIEW
--------
Use pre-trained BERT as encoder, add task-specific heads on top:
  - [CLS] token representation → Intent Classification head
  - Per-token representations → Slot Filling head

Fine-tuning strategy: either full fine-tuning or freeze BERT layers.

TODO: implement BERT-based NLU model.
"""

# TODO: import statements (transformers BertModel)

# TODO: BERTNLUModel class
#   - self.bert = BertModel.from_pretrained("bert-base-uncased")
#   - self.intent_head = nn.Linear(hidden_size, n_intents)
#   - self.slot_head = nn.Linear(hidden_size, n_slots)
#   - forward(): returns (intent_logits, slot_logits)

# TODO: count_trainable_parameters()
