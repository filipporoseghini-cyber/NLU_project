"""
model.py — Part 2.B: Fine-tuning BERT e GPT-2 per Intent Classification + Slot Filling
========================================================================================

ARCHITETTURA GENERALE
----------------------
Sia BERT che GPT-2 condividono la stessa struttura ad alto livello:

  1. Backbone pre-addestrato (BERT encoder / GPT-2 decoder)  ← pesi caricati da HuggingFace
  2. Dropout layer                                           ← regolarizzazione
  3. slot_out:   Linear(hidden_size, slots_size)             ← etichetta per ogni token
  4. intent_out: Linear(hidden_size, n_intents)              ← classificazione intent

La differenza chiave è come si estrae il vettore per l'intent:
  - BERT:  hidden state del token [CLS] alla posizione 0  (encoder bidirezionale)
  - GPT-2: hidden state del token EOS appeso alla fine    (decoder causale)

DIMENSIONI HIDDEN
-----------------
- bert-base-uncased:   hidden_size = 768
- bert-large-uncased:  hidden_size = 1024
- gpt2 (base):         hidden_size = 768  (config.n_embd)
- gpt2-medium:         hidden_size = 1024

Usiamo model.config.hidden_size e model.config.n_embd per essere indipendenti
dalla variante specifica e non codificare il valore a mano.

FINE-TUNING
-----------
Tutti i parametri del backbone sono addestrabili (full fine-tuning).
Le teste di output (slot_out, intent_out) sono inizializzate casualmente
come di consueto con nn.Linear (inizializzazione Kaiming di default).

Se si volessero congelare i layer del backbone (feature extraction invece di
fine-tuning), basterebbe aggiungere:
    for param in self.bert.parameters():
        param.requires_grad = False
"""

import torch
import torch.nn as nn
from transformers import AutoModel


# ===========================================================================
# 1. BERT PER NLU
# ===========================================================================

class BERTforNLU(nn.Module):
    """
    BERT fine-tuned per Intent Classification + Slot Filling.

    FORWARD:
      1. BERT encoder → last_hidden_state (B, L, hidden_size)
      2. Intent:  hidden state [CLS] alla posizione 0   → dropout → Linear → logit (B, n_intents)
      3. Slot:    hidden state di ogni token             → dropout → Linear → logit (B, L, slots_size)

    PERCHÉ [CLS] A POSIZIONE 0?
    BERT è un encoder bidirezionale: ogni token vede tutti gli altri token.
    Il token [CLS] è stato pre-addestrato da BERT specificamente per aggregare
    informazioni sull'intera sequenza → è il vettore naturale per la classificazione.
    """

    def __init__(self, slots_size, n_intents,
                 model_name="bert-base-uncased", dropout=0.1):
        """
        Args:
            slots_size : numero di etichette slot (len(lang.slot2id))
            n_intents  : numero di intent (len(lang.intent2id))
            model_name : variante di BERT ("bert-base-uncased" o "bert-large-uncased")
            dropout    : probabilità di dropout prima delle teste di output
        """
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        hidden_size  = self.bert.config.hidden_size  # 768 per base, 1024 per large

        self.dropout    = nn.Dropout(dropout)
        self.slot_out   = nn.Linear(hidden_size, slots_size)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Args:
            input_ids       : (B, L)  — token ids (include [CLS] e [SEP])
            attention_mask  : (B, L)  — 1 per token reali, 0 per padding
            token_type_ids  : (B, L)  — tutti 0 (singola frase, segmento A)

        Returns:
            slots  : (B, L, slots_size)  — logit slot per ogni token
            intent : (B, n_intents)      — logit intent per l'intera sequenza
        """
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = output.last_hidden_state   # (B, L, hidden_size)

        # [CLS] è sempre alla posizione 0 in BERT (aggiunto dal tokenizer)
        cls_repr = last_hidden[:, 0, :]          # (B, hidden_size)

        slots  = self.slot_out(self.dropout(last_hidden))  # (B, L, slots_size)
        intent = self.intent_out(self.dropout(cls_repr))   # (B, n_intents)

        return slots, intent


# ===========================================================================
# 2. GPT-2 PER NLU
# ===========================================================================

class GPT2forNLU(nn.Module):
    """
    GPT-2 fine-tuned per Intent Classification + Slot Filling.

    FORWARD:
      1. GPT-2 decoder → last_hidden_state (B, L, hidden_size)
      2. Intent:  hidden state EOS/CLS alla posizione seq_lens[i]-1 → dropout → Linear
      3. Slot:    hidden state di ogni token                         → dropout → Linear

    PERCHÉ EOS ALLA FINE INVECE CHE ALL'INIZIO?
    GPT-2 è causale: il token alla posizione i ha accesso solo ai token 0..i.
    Se mettessimo [CLS] all'inizio (posizione 0), non avrebbe visto nessun token
    della frase al momento del forward → rappresentazione inutile.
    Mettendolo alla FINE (come EOS), ha già elaborato l'intera frase e la sua
    rappresentazione aggrega tutto il contesto disponibile.
    """

    def __init__(self, slots_size, n_intents,
                 model_name="openai-community/gpt2", dropout=0.1):
        """
        Args:
            slots_size : numero di etichette slot
            n_intents  : numero di intent
            model_name : variante GPT-2 ("openai-community/gpt2" o "openai-community/gpt2-medium")
            dropout    : probabilità di dropout
        """
        super().__init__()
        self.gpt2    = AutoModel.from_pretrained(model_name)
        hidden_size  = self.gpt2.config.n_embd   # 768 per gpt2, 1024 per gpt2-medium

        self.dropout    = nn.Dropout(dropout)
        self.slot_out   = nn.Linear(hidden_size, slots_size)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask, seq_lens):
        """
        Args:
            input_ids      : (B, L)  — token ids (include EOS/CLS alla fine)
            attention_mask : (B, L)  — 1 per token reali + EOS, 0 per padding
            seq_lens       : (B,)    — lunghezza reale di ogni sequenza incluso EOS

        Returns:
            slots  : (B, L, slots_size)  — logit slot per ogni token
            intent : (B, n_intents)      — logit intent (dal token EOS/CLS)
        """
        output      = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state   # (B, L, hidden_size)

        # Estrae il vettore EOS/CLS: per la sequenza i si trova alla posizione seq_lens[i]-1
        # (seq_lens conta da 1, gli indici contano da 0 → -1)
        cls_repr = torch.stack([
            last_hidden[i, seq_lens[i] - 1]
            for i in range(last_hidden.size(0))
        ])  # (B, hidden_size)

        slots  = self.slot_out(self.dropout(last_hidden))  # (B, L, slots_size)
        intent = self.intent_out(self.dropout(cls_repr))   # (B, n_intents)

        return slots, intent


# ===========================================================================
# 3. UTILITY
# ===========================================================================

def count_parameters(model):
    """Conta i parametri addestrabili del modello (utile per confrontare le varianti)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
