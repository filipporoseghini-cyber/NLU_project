"""
utils.py — Part 2.B: NLU with pre-trained BERT
================================================

OVERVIEW
--------
Fine-tune BERT (or similar encoder-only model) on ATIS for joint
Intent Classification and Slot Filling.

BERT tokenizes at subword level (WordPiece). For slot filling, we need to
align the BIO labels with the subword tokens (only the first subword of
each word gets its label; others are ignored with label=-100).

TODO: implement ATIS loading with BERT tokenizer alignment.
"""

# TODO: import statements (transformers BertTokenizer)

# TODO: load_atis(path)

# TODO: ATISDataset with BERT tokenizer and label alignment

# TODO: collate_fn() for BERT (attention_mask, token_type_ids)

# TODO: get_dataloaders()
