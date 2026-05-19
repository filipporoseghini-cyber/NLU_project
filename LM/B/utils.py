"""
utils.py — Part 1.B: Caricamento dati per il fine-tuning LoRA di GPT-2
=======================================================================

QUADRO GENERALE
---------------
Questo file gestisce la lettura del corpus Penn TreeBank e la creazione dei
DataLoader per il fine-tuning di GPT-2 con LoRA.

Rispetto alla Parte 1.A le differenze sono minime:
  - Il tokenizer usa AutoTokenizer.from_pretrained("openai-community/gpt2")
    invece di GPT2Tokenizer.from_pretrained("gpt2"). Sono equivalenti; il
    notebook usa la forma più generica "openai-community/gpt2".
  - La collate_fn è IDENTICA a quella del notebook (cell 25): restituisce
    (input_ids, labels, n_tokens). Tuttavia, nella Parte 1.B il training loop
    IGNORA le labels provenienti dal dataloader (usa '_') e le ricalcola
    internamente dal campo input_ids. Questo perché GPT2LMHeadModel gestisce
    internamente lo shift dei label quando si passa 'labels' al forward().

DETTAGLIO COLLATE_FN E SHIFT
------------------------------
La collate_fn taglia il PRIMO e l'ULTIMO token per realizzare lo shift:
  - input_ids = token[0..T-2]  (tutti tranne l'ultimo)
  - labels    = token[1..T-1]  (tutti tranne il primo — usate in Parte 1.A)

Nella Parte 1.B, invece di usare labels dal dataloader, il train_loop fa:
  - labels = input_ids.clone()     # copia dell'input (già senza l'ultimo token)
  - labels[pad_pos] = -100         # oscura il padding
  - model(input_ids, labels=labels) # il modello fa internamente lo shift

Il modello HuggingFace shift internamente:
  - shift_logits = logits[..., :-1, :]  # predizioni alle posizioni 0..L-2
  - shift_labels = labels[..., 1:]      # target alle posizioni 1..L-1
Questo equivale al next-token prediction standard.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from functools import partial


# ===========================================================================
# 1. LETTURA DEL FILE RAW
# ===========================================================================

def read_file(path, eos_token="<eos>"):
    """
    Legge il corpus riga per riga e aggiunge <eos> a fine di ogni frase.

    Identico alla Parte 1.A e al notebook (cell 19): nessuna modifica.

    Args:
        path      : percorso al file .txt del Penn TreeBank
        eos_token : stringa da aggiungere a fine di ogni frase

    Returns:
        output : lista di stringhe, una per frase, con <eos> in fondo
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            # strip() rimuove spazi e newline iniziali/finali,
            # poi aggiungiamo spazio + eos token
            output.append(line.strip() + " " + eos_token)
    return output


# ===========================================================================
# 2. DATASET CLASS
# ===========================================================================

class PennTreeBank(Dataset):
    """
    Dataset per il Penn TreeBank.

    Identico alla Parte 1.A e al notebook (cell 22).
    Le frasi rimangono come stringhe raw; la tokenizzazione avviene nella
    collate_fn dove possiamo fare padding a livello di batch intero.
    """

    def __init__(self, corpus):
        """
        Args:
            corpus : lista di stringhe (output di read_file)
        """
        self.sents = corpus

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


# ===========================================================================
# 3. COLLATE FUNCTION
# ===========================================================================

def collate_fn(batch, tokenizer, device):
    """
    Funzione di collating personalizzata.

    Da notebook cell 25 — IDENTICA all'originale.

    Tokenizza il batch, applica padding, e restituisce:
      - input_ids : token[0..T-2] — input al modello
      - labels    : token[1..T-1] — target (usate in Parte 1.A; IGNORATE in 1.B)
      - n_tokens  : numero di token non-padding in input_ids

    PERCHÉ CONSERVIAMO labels ANCHE IN PARTE 1.B?
    La firma del dataloader è la stessa per entrambe le parti, per coerenza.
    Nella Parte 1.B il training loop usa '_' per ignorare questo campo e
    ricalcola le label dal campo input_ids (con -100 al posto dei pad).

    Args:
        batch     : lista di stringhe (frasi del batch)
        tokenizer : tokenizer GPT-2 configurato (pad = eos)
        device    : dispositivo target (cuda / mps / cpu)

    Returns:
        input_ids : (batch_size, seq_len-1)
        labels    : (batch_size, seq_len-1) — shifted di 1 (usate in Parte 1.A)
        n_tokens  : scalare int, token non-padding nel batch
    """
    # Tokenizziamo l'intero batch con padding automatico
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    # input_ids: tutti i token tranne l'ultimo
    # (l'ultimo viene rimosso perché il modello farà shift interno)
    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)

    # labels: tutti i token tranne il primo (shifted di 1 → next-token target)
    # Nota: queste labels sono usate SOLO nella Parte 1.A (train_loop con criterion esterno)
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

    # Contiamo i token NON-padding in input_ids.
    # Questo sarà usato per normalizzare correttamente la loss.
    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)

    return input_ids, labels, n_tokens


# ===========================================================================
# 4. CREAZIONE DEI DATALOADER
# ===========================================================================

def get_dataloaders(train_raw, dev_raw, test_raw, tokenizer, device,
                    batch_size_train=8, batch_size_eval=16):
    """
    Crea i tre DataLoader (train, dev, test).

    Struttura identica alla Parte 1.A. Usa partial() per passare tokenizer
    e device alla collate_fn (il DataLoader chiama collate_fn con un solo arg).

    Args:
        train_raw        : lista di stringhe per il training
        dev_raw          : lista di stringhe per la validation
        test_raw         : lista di stringhe per il test
        tokenizer        : tokenizer GPT-2 configurato
        device           : dispositivo target
        batch_size_train : batch size per il training (più piccolo per memoria GPU)
        batch_size_eval  : batch size per eval (più grande: no backward pass)

    Returns:
        train_loader, dev_loader, test_loader
    """
    train_dataset = PennTreeBank(train_raw)
    dev_dataset   = PennTreeBank(dev_raw)
    test_dataset  = PennTreeBank(test_raw)

    # partial() lega tokenizer e device alla collate_fn,
    # così il DataLoader può chiamarla con solo il batch come argomento
    collate = partial(collate_fn, tokenizer=tokenizer, device=device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,       # mescola i dati a ogni epoca (solo per il training)
        collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size_eval,
        shuffle=False,      # no shuffle per eval (riproducibilità)
        collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=collate
    )

    return train_loader, dev_loader, test_loader


# ===========================================================================
# 5. SETUP TOKENIZER
# ===========================================================================

def get_tokenizer():
    """
    Carica e configura il tokenizer BPE di GPT-2 da HuggingFace.

    MODIFICA RISPETTO A PARTE 1.A:
      Usa AutoTokenizer.from_pretrained("openai-community/gpt2") invece di
      GPT2Tokenizer.from_pretrained("gpt2"). Sono equivalenti; il notebook
      (cell 24 e 35) usa questa forma. AutoTokenizer è l'interfaccia generica
      che sceglie automaticamente la classe giusta in base al modello.

    Il tokenizer BPE ha ~50.257 token nel vocabolario.
    Il pad token viene impostato uguale all'eos token (pratica standard con GPT-2,
    che non ha un pad token di default).

    Returns:
        tokenizer : tokenizer configurato e pronto all'uso
    """
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # GPT-2 non ha un pad token dedicato; usiamo eos come pad.
    # Le posizioni di padding verranno poi ignorate nella loss (rimpiazzate con -100).
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
