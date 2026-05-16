"""
utils.py — Part 1.A: Language Modeling from scratch (GPT-2 style)
==================================================================

QUADRO GENERALE
---------------
L'obiettivo di questa parte è addestrare un modello linguistico (LM) basato
sull'architettura GPT-2 *from scratch* sul dataset Penn TreeBank (PTB).

Un Language Model stima la probabilità di una sequenza di token:
    P(w_1, w_2, ..., w_T) = ∏ P(w_t | w_1, ..., w_{t-1})

Questo è esattamente ciò che fa GPT-2: dato un contesto (i token precedenti),
predice il token successivo. La perdita è la Cross-Entropy tra il token predetto
e il token reale, mediata su tutti i token della sequenza.

La metrica principale è la PERPLEXITY (PPL):
    PPL = exp(cross-entropy per token)
Una PPL bassa indica che il modello assegna alta probabilità alle sequenze reali.
Il target per questa parte è PPL < 250 sul test set.

TOKENIZER
---------
Usiamo il tokenizer BPE (Byte-Pair Encoding) di GPT-2 già pronto da HuggingFace.
Questo converte le stringhe in sequenze di interi (token IDs) che il modello
può elaborare. Il vocabolario ha ~50.257 token.

DATASET: Penn TreeBank (PTB)
-----------------------------
Il PTB è un corpus di testo in lingua inglese ricavato dagli articoli del Wall
Street Journal. È uno dei benchmark standard per il Language Modeling.
Tre split: train (929k token), valid (73k), test (82k).

Ogni riga del file è una frase. Aggiungiamo un token speciale <eos> (end-of-sentence)
alla fine di ogni frase, così il modello impara anche dove finisce una frase.

BATCHING E PADDING
------------------
Poiché le frasi hanno lunghezze diverse, usiamo il *padding*: le frasi più corte
vengono allungate con un token speciale di padding fino alla lunghezza della frase
più lunga nel batch. Il padding NON contribuisce alla perdita (viene ignorato).

Il tokenizer GPT-2 non ha un pad token di default, quindi usiamo l'eos token come pad.

INPUT/LABELS (teacher forcing)
-------------------------------
Per il Language Modeling auto-regressivo (next-token prediction), lo shift è fondamentale:
  - input_ids   = token[0 .. T-2]   (tutti i token tranne l'ultimo)
  - labels      = token[1 .. T-1]   (tutti i token tranne il primo)
In questo modo, per ogni posizione t, il modello vede input[t] e deve predire labels[t]
che è esattamente il token successivo nella sequenza originale.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


# ===========================================================================
# 1. LETTURA DEL FILE RAW
# ===========================================================================

def read_file(path, eos_token="<eos>"):
    """
    Legge un file di testo riga per riga e aggiunge il token <eos> alla fine
    di ogni riga (ogni frase del corpus PTB è su una riga separata).

    Args:
        path      : percorso al file .txt (ptb.train.txt, ptb.valid.txt, ...)
        eos_token : stringa da aggiungere in fondo a ogni frase

    Returns:
        output : lista di stringhe, una per frase, con <eos> alla fine
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            # strip() rimuove spazi e newline iniziali/finali
            # poi aggiungiamo lo spazio + eos token
            output.append(line.strip() + " " + eos_token)
    return output


# ===========================================================================
# 2. DATASET CLASS
# ===========================================================================

class PennTreeBank(Dataset):
    """
    Dataset class per il Penn TreeBank.

    Eredita da torch.utils.data.Dataset: ci basta implementare __len__ e
    __getitem__. Le frasi rimangono come stringhe "raw"; la tokenizzazione
    avviene nella collate_fn, così da poter fare il padding a livello di batch.
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
        # Restituisce la frase raw (stringa) all'indice idx
        return self.sents[idx]


# ===========================================================================
# 3. COLLATE FUNCTION
# ===========================================================================

def collate_fn(batch, tokenizer, device):
    """
    Funzione di collating personalizzata: viene chiamata dal DataLoader per
    ogni batch di frasi raw (lista di stringhe) e restituisce tensori pronti
    per il modello.

    Passi interni:
      1. Tokenizza il batch con padding a destra fino alla frase più lunga
      2. Estrae input_ids (tutti i token tranne l'ultimo)
      3. Estrae labels (tutti i token tranne il primo) — shift di 1
      4. Conta i token NON-padding per il calcolo corretto della loss

    Args:
        batch     : lista di stringhe (frasi del batch)
        tokenizer : GPT2Tokenizer già configurato (pad = eos)
        device    : torch.device su cui mettere i tensori

    Returns:
        input_ids  : (batch_size, seq_len-1) — sequenza di input
        labels     : (batch_size, seq_len-1) — target da predire
        n_tokens   : scalare int, numero di token non-padding nel batch
    """

    # Tokenizza: aggiunge padding automaticamente, restituisce tensori PyTorch
    tokenized = tokenizer(
        batch,
        padding=True,          # pad fino alla lunghezza massima del batch
        return_tensors="pt"    # restituisce torch.Tensor
    )

    # input_ids ha shape (batch_size, seq_len)
    # Facciamo lo shift: input = tutto tranne l'ultimo token
    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)

    # labels = tutto tranne il primo token (shifted di 1 a sinistra)
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

    # Conta i token non-padding: servirà per normalizzare la loss per token
    # (importante: non vogliamo contare il contributo dei token di padding)
    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)

    return input_ids, labels, n_tokens


# ===========================================================================
# 4. CREAZIONE DEI DATALOADER
# ===========================================================================

def get_dataloaders(train_raw, dev_raw, test_raw, tokenizer, device,
                    batch_size_train=8, batch_size_eval=16):
    """
    Crea i tre DataLoader (train, dev, test) a partire dalle liste di frasi raw.

    Il DataLoader gestisce automaticamente:
      - suddivisione in batch
      - shuffling (solo per il training)
      - chiamata alla collate_fn per ogni batch

    Args:
        train_raw        : lista di stringhe per il training
        dev_raw          : lista di stringhe per la validation
        test_raw         : lista di stringhe per il test
        tokenizer        : GPT2Tokenizer configurato
        device           : dispositivo target (cuda / mps / cpu)
        batch_size_train : batch size per il training (più piccolo per memoria)
        batch_size_eval  : batch size per eval (più grande perché no gradients)

    Returns:
        train_loader, dev_loader, test_loader : tre DataLoader pronti
    """

    # Creiamo i Dataset
    train_dataset = PennTreeBank(train_raw)
    dev_dataset   = PennTreeBank(dev_raw)
    test_dataset  = PennTreeBank(test_raw)

    # Usiamo una lambda per passare tokenizer e device alla collate_fn
    # (il DataLoader chiama collate_fn(batch) con un solo argomento)
    collate = lambda batch: collate_fn(batch, tokenizer, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,           # mescola i dati a ogni epoca
        collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size_eval,
        shuffle=False,          # no shuffle per eval (riproducibilità)
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

    GPT-2 usa Byte-Pair Encoding: suddivide le parole in subword units
    (es. "running" -> ["run", "ning"]) per gestire vocaboli sconosciuti.
    Il vocabolario ha 50.257 token (50.256 BPE + 1 speciale).

    Il tokenizer non ha un pad token di default, quindi lo impostiamo
    uguale all'eos token (come da pratica comune con GPT-2).

    Returns:
        tokenizer : GPT2Tokenizer configurato e pronto all'uso
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Imposta il pad token uguale all'eos token.
    # Nota: questo significa che nelle labels, il token di padding avrà
    # lo stesso ID dell'eos token — ma lo gestiremo nella loss function
    # ignorando le posizioni di padding.
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
