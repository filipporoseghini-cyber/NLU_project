"""
utils.py — Part 2.B: Dati per il fine-tuning di BERT e GPT-2 su ATIS
======================================================================

DIFFERENZE PRINCIPALI RISPETTO ALLA PARTE 2.A
----------------------------------------------
Nella Parte 2.A usavamo:
  - Un vocabolario word-level costruito dal training set (Lang.word2id)
  - Un token CLS aggiunto alla fine come stringa nel vocabolario
  - PAD_TOKEN = 0 come ignore_index nella CrossEntropyLoss degli slot

Nella Parte 2.B usiamo tokenizer sub-word pre-addestrati (WordPiece per BERT,
BPE per GPT-2), il che introduce il problema della sub-tokenizzazione.

IL PROBLEMA DELLA SUB-TOKENIZZAZIONE
--------------------------------------
Il tokenizer BPE/WordPiece divide le parole in sotto-unità (subtoken):
  "flying" → ["fly", "##ing"]  (BERT WordPiece)
  "flying" → ["fly", "ing"]    (GPT-2 BPE)
  "New"    → ["New"]           (entrambi, se presente nel vocabolario)

Ma nel dataset ATIS ogni PAROLA ha esattamente UNA etichetta slot:
  "flying" → "O"

Dopo la sub-tokenizzazione, "flying" diventa due token ma ha ancora una
sola etichetta. Come allineiamo?

SOLUZIONE: prima subtoken = etichetta reale, altri = IGNORE_INDEX (-100)
  "fly"  → "O"   (prima subword: prende l'etichetta della parola)
  "##ing" → -100  (subword successive: -100 → ignorate dalla loss)

-100 è la convenzione standard di HuggingFace per indicare "ignora questa
posizione nel calcolo della loss". CrossEntropyLoss la ignora automaticamente.

DIFFERENZE BERT vs GPT-2
-------------------------
BERT (encoder bidirezionale):
  - Aggiunge [CLS] all'INIZIO e [SEP] alla FINE della sequenza
  - [CLS] viene usato per la classificazione dell'intent (posizione 0)
  - Entrambi ricevono -100 per gli slot (posizioni ignorate)
  - Tokenizer: AutoTokenizer("bert-base-uncased") → WordPiece

GPT-2 (decoder causale):
  - NON ha token speciali automatici come BERT
  - Aggiungiamo manualmente EOS alla FINE come token CLS (come in Parte 2.A)
  - L'EOS alla fine "vede" tutta la frase (posizione più a destra = max contesto)
  - Riceve -100 per gli slot (non è una parola con slot label)
  - Tokenizer: AutoTokenizer("openai-community/gpt2") → BPE
  - pad_token = eos_token (GPT-2 non ha un pad token nativo)

VOCABOLARIO (classe Lang)
--------------------------
A differenza della Parte 2.A, NON costruiamo word2id: le parole vengono
gestite direttamente dal tokenizer pre-addestrato. Costruiamo solo:
  - slot2id / id2slot: mapping etichette slot ↔ interi
  - intent2id / id2intent: mapping etichette intent ↔ interi

IGNORE_INDEX = -100 (standard HuggingFace, usato in 2.B)
PAD_TOKEN = 0       (mantenuto per compatibilità, usato in slot2id come pad)
"""

import json
from functools import partial
from collections import Counter

from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PAD_TOKEN    = 0    # id del token 'pad' in slot2id (mai usato come ignore_index in 2.B)
IGNORE_INDEX = -100  # standard HuggingFace: CrossEntropyLoss ignora queste posizioni


# ===========================================================================
# 1. CARICAMENTO DATI (identico a Parte 2.A)
# ===========================================================================

def load_data(path):
    """
    Carica il dataset ATIS da un file JSON.
    Identico alla Parte 2.A — nessuna modifica.
    """
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


# ===========================================================================
# 2. SPLIT STRATIFICATA DEL DEV SET (identico a Parte 2.A)
# ===========================================================================

def create_dev_split(train_raw, dev_size=0.10, random_state=42):
    """
    Crea un dev set stratificato sull'intent.
    Identico alla Parte 2.A — nessuna modifica.
    """
    intents = [x['intent'] for x in train_raw]
    count_y = Counter(intents)

    inputs     = []
    labels     = []
    mini_train = []

    for example in train_raw:
        if count_y[example['intent']] > 1:
            inputs.append(example)
            labels.append(example['intent'])
        else:
            mini_train.append(example)

    X_train, X_dev, _, _ = train_test_split(
        inputs, labels,
        test_size=dev_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels
    )
    X_train.extend(mini_train)
    return X_train, X_dev


# ===========================================================================
# 3. VOCABOLARIO (solo slot e intent — no word2id)
# ===========================================================================

class Lang:
    """
    Vocabolario per slot labels e intent labels.

    Rispetto alla Parte 2.A:
      - NON costruiamo word2id (il tokenizer pretrained gestisce le parole)
      - NON aggiungiamo 'cls' a slot2id (usiamo -100 per tutte le posizioni
        da ignorare, incluso il token CLS)
      - slot2id ha solo 'pad' (id=0) + slot labels reali (id ≥ 1)
      - intent2id non ha 'pad' (ogni esempio ha esattamente un intent)
    """

    def __init__(self, intents, slots):
        """
        Args:
            intents : insieme di tutti gli intent (da train+dev+test)
            slots   : insieme di tutte le etichette slot (da train+dev+test)
        """
        self.slot2id   = self._build_slot_vocab(slots)
        self.intent2id = self._build_intent_vocab(intents)
        self.id2slot   = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def _build_slot_vocab(self, slots):
        """
        Costruisce slot2id: {'pad': 0, 'B-...': 1, ..., 'O': N}

        'pad' è mantenuto a id=0 per simmetria con la Parte 2.A, ma nel
        training di 2.B le posizioni da ignorare usano -100 (non 0).
        Le etichette BIO reali hanno id ≥ 1 e non includono mai 'pad'.
        sorted() garantisce che il mapping sia riproducibile tra run.
        """
        vocab = {'pad': PAD_TOKEN}
        for slot in sorted(slots):
            if slot not in vocab:
                vocab[slot] = len(vocab)
        return vocab

    def _build_intent_vocab(self, intents):
        """
        Costruisce intent2id: {'abbreviation': 0, 'aircraft': 1, ...}
        sorted() garantisce riproducibilità.
        """
        vocab = {}
        for intent in sorted(intents):
            vocab[intent] = len(vocab)
        return vocab


def build_lang(train_raw, dev_raw, test_raw):
    """
    Costruisce il vocabolario Lang da tutti gli split del dataset.

    Le etichette (slot e intent) vengono raccolte da train + dev + test
    per evitare etichette sconosciute a test time.
    """
    corpus  = train_raw + dev_raw + test_raw
    slots   = set(sum([x['slots'].split()   for x in corpus], []))
    intents = set(x['intent'] for x in corpus)
    return Lang(intents, slots)


# ===========================================================================
# 4. DATASET PER BERT
# ===========================================================================

class BERTIntentsAndSlots(data.Dataset):
    """
    Dataset ATIS con tokenizzazione WordPiece (BERT) e allineamento slot labels.

    STRUTTURA DI UN ESEMPIO TOKENIZZATO (sequenza nel modello):
    ─────────────────────────────────────────────────────────
    Input:   [CLS]  fly   ##ing  from  New   York   [SEP]
    input_ids: cls_id  f_id  i_id  fr_id  n_id  y_id  sep_id
    y_slots:   -100    O_id  -100   O_id   B_id  I_id  -100
                ↑                                          ↑
             token speciale                          token speciale
             ignorato dalla loss                   ignorato dalla loss

    Il [CLS] in BERT è il token di classificazione progettato per rappresentare
    l'intera sequenza. Il modello usa il suo hidden state per predire l'intent.
    Il [SEP] segna la fine della sequenza (necessario per BERT).

    DETTAGLIO DELL'ALLINEAMENTO:
    Per ogni parola e i suoi subtoken:
      1. Primo subtoken  → prende l'etichetta slot della parola (id reale ≥ 1)
      2. Subtoken successivi → -100 (ignorati dalla loss)

    Questo garantisce che ogni parola contribuisca esattamente una volta
    al calcolo della loss dello slot filling.
    """

    def __init__(self, dataset, lang, tokenizer, max_len=128):
        """
        Args:
            dataset   : lista di dizionari ATIS (output di load_data)
            lang      : istanza di Lang con slot2id e intent2id
            tokenizer : tokenizer BERT (WordPiece, da AutoTokenizer)
            max_len   : lunghezza massima della sequenza tokenizzata
                        (include [CLS] e [SEP]). 128 è più che sufficiente
                        per ATIS (le frasi sono corte, ~15 parole).
        """
        self.samples = []

        for example in dataset:
            words  = example['utterance'].split()
            slots  = example['slots'].split()
            intent = lang.intent2id[example['intent']]

            # ---------------------------------------------------------------
            # TOKENIZZAZIONE E ALLINEAMENTO
            # ---------------------------------------------------------------
            # Partiamo con i token speciali di BERT: [CLS] all'inizio
            input_ids   = [tokenizer.cls_token_id]
            slot_labels = [IGNORE_INDEX]  # [CLS] → ignora nella loss slot

            for word, slot in zip(words, slots):
                # tokenize() senza aggiungere token speciali → solo subtokens
                word_tokens = tokenizer.tokenize(word)

                if len(word_tokens) == 0:
                    # Parola non tokenizzabile (es. caratteri speciali rari)
                    # → usiamo token UNK
                    word_tokens = [tokenizer.unk_token]

                token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
                slot_id   = lang.slot2id.get(slot, PAD_TOKEN)

                input_ids.extend(token_ids)
                # Primo subtoken → etichetta reale; altri → -100
                slot_labels.extend([slot_id] + [IGNORE_INDEX] * (len(token_ids) - 1))

            # [SEP] alla fine
            input_ids.append(tokenizer.sep_token_id)
            slot_labels.append(IGNORE_INDEX)

            # ---------------------------------------------------------------
            # TRUNCATION (se la sequenza supera max_len)
            # ---------------------------------------------------------------
            # Caso raro su ATIS, ma gestiamo per robustezza.
            # Tronchiamo a max_len-1 e ri-aggiungiamo [SEP].
            if len(input_ids) > max_len:
                input_ids   = input_ids[:max_len - 1] + [tokenizer.sep_token_id]
                slot_labels = slot_labels[:max_len - 1] + [IGNORE_INDEX]

            self.samples.append({
                'input_ids':   input_ids,
                'slot_labels': slot_labels,
                'intent':      intent,
                'words':       words,   # parole originali (per conll eval)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'input_ids':   torch.tensor(s['input_ids'],   dtype=torch.long),
            'slot_labels': torch.tensor(s['slot_labels'], dtype=torch.long),
            'intent':      s['intent'],
            'words':       s['words'],   # lista di stringhe, non tensor
        }


# ===========================================================================
# 5. DATASET PER GPT-2
# ===========================================================================

class GPT2IntentsAndSlots(data.Dataset):
    """
    Dataset ATIS con tokenizzazione BPE (GPT-2) e allineamento slot labels.

    STRUTTURA DI UN ESEMPIO TOKENIZZATO (sequenza nel modello):
    ─────────────────────────────────────────────────────────
    Input:   fly  ing  from  New  York  <eos>     (EOS = token CLS)
    input_ids: f_id i_id fr_id n_id y_id eos_id
    y_slots:   O_id -100  O_id B_id I_id  -100    ← EOS/CLS → -100

    A differenza di BERT che ha [CLS] all'inizio per design, GPT-2 è causale:
    ogni token vede solo i token precedenti. Quindi mettiamo il token di
    classificazione (EOS) alla FINE: in quella posizione il modello ha già
    elaborato tutta la frase e la sua rappresentazione riassume il contenuto.

    L'EOS token viene usato come surrogate CLS per l'intent classification.
    Nel ground truth degli slot, riceve -100 (non è una parola con uno slot).

    NOTA: GPT-2 usa EOS come pad token (tokenizer.pad_token = tokenizer.eos_token).
    Le posizioni di padding REALE hanno attention_mask=0, che le distingue
    dall'EOS/CLS che ha attention_mask=1.
    """

    def __init__(self, dataset, lang, tokenizer, max_len=128):
        """
        Args:
            dataset   : lista di dizionari ATIS
            lang      : istanza di Lang
            tokenizer : tokenizer GPT-2 (BPE), con pad_token = eos_token
            max_len   : lunghezza massima sequenza
        """
        self.samples = []

        for example in dataset:
            words  = example['utterance'].split()
            slots  = example['slots'].split()
            intent = lang.intent2id[example['intent']]

            # ---------------------------------------------------------------
            # TOKENIZZAZIONE E ALLINEAMENTO (senza token speciali automatici)
            # ---------------------------------------------------------------
            # GPT-2 non aggiunge [CLS] o [SEP] automaticamente.
            # Tokenizziamo parola per parola per il corretto allineamento.
            input_ids   = []
            slot_labels = []

            for word, slot in zip(words, slots):
                # encode con add_special_tokens=False → solo i subtoken BPE
                token_ids = tokenizer.encode(word, add_special_tokens=False)

                if len(token_ids) == 0:
                    token_ids = [tokenizer.unk_token_id or tokenizer.eos_token_id]

                slot_id = lang.slot2id.get(slot, PAD_TOKEN)

                input_ids.extend(token_ids)
                # Primo subtoken → etichetta reale; altri → -100
                slot_labels.extend([slot_id] + [IGNORE_INDEX] * (len(token_ids) - 1))

            # ---------------------------------------------------------------
            # TOKEN CLS (EOS) alla FINE
            # ---------------------------------------------------------------
            # Appende il token EOS come surrogate CLS per l'intent classification.
            # Il modello estrarrà il vettore a questa posizione per predire l'intent.
            input_ids.append(tokenizer.eos_token_id)
            slot_labels.append(IGNORE_INDEX)  # CLS non è una parola → -100

            # Truncation
            if len(input_ids) > max_len:
                # Manteniamo EOS/CLS alla fine anche dopo truncation
                input_ids   = input_ids[:max_len - 1] + [tokenizer.eos_token_id]
                slot_labels = slot_labels[:max_len - 1] + [IGNORE_INDEX]

            self.samples.append({
                'input_ids':   input_ids,
                'slot_labels': slot_labels,
                'intent':      intent,
                'words':       words,
                'seq_len':     len(input_ids),  # lunghezza reale (prima del padding)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'input_ids':   torch.tensor(s['input_ids'],   dtype=torch.long),
            'slot_labels': torch.tensor(s['slot_labels'], dtype=torch.long),
            'intent':      s['intent'],
            'words':       s['words'],
            'seq_len':     s['seq_len'],
        }


# ===========================================================================
# 6. COLLATE FUNCTIONS
# ===========================================================================

def collate_fn_bert(batch):
    """
    Collate per BERT: padding + attention_mask + token_type_ids.

    SCHEMA DI UN BATCH (max_len=8):
      [CLS fly ##ing from New York [SEP] PAD]    attention: [1 1 1 1 1 1 1 0]
      [CLS I   want  a   flight   [SEP] PAD PAD] attention: [1 1 1 1 1 1 0 0]

    input_ids:      paddati con 0 (BERT pad_token_id = [PAD] = 0)
    y_slots:        paddati con -100 (IGNORE_INDEX → CrossEntropyLoss ignora)
    attention_mask: 1 per token reali, 0 per padding
    token_type_ids: tutti 0 (una sola frase, nessuna coppia)
    words:          lista di liste di stringhe (non tensor, per eval conll)
    intents:        LongTensor(B,)
    """
    input_ids_list   = [item['input_ids']   for item in batch]
    slot_labels_list = [item['slot_labels'] for item in batch]
    intents          = torch.tensor([item['intent'] for item in batch], dtype=torch.long)
    words_list       = [item['words'] for item in batch]

    max_len = max(len(ids) for ids in input_ids_list)

    padded_input_ids   = []
    padded_slot_labels = []
    attention_masks    = []

    for input_ids, slot_labels in zip(input_ids_list, slot_labels_list):
        pad_len = max_len - len(input_ids)
        padded_input_ids.append(
            torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_slot_labels.append(
            torch.cat([slot_labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
        )
        attention_masks.append(torch.cat([
            torch.ones(len(input_ids), dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ]))

    return {
        'input_ids':      torch.stack(padded_input_ids),
        'attention_mask': torch.stack(attention_masks),
        'token_type_ids': torch.zeros(len(batch), max_len, dtype=torch.long),
        'y_slots':        torch.stack(padded_slot_labels),
        'intents':        intents,
        'words':          words_list,
    }


def collate_fn_gpt2(batch):
    """
    Collate per GPT-2: padding con EOS token + attention_mask + seq_lens.

    SCHEMA DI UN BATCH (max_len=8):
      [fly ing from New York EOS PAD PAD]  attention: [1 1 1 1 1 1 0 0]
      [I  want a  flight     EOS PAD PAD]  attention: [1 1 1 1 1 0 0 0]
                                                                ↑ CLS/EOS

    seq_lens: lunghezza reale di ogni sequenza (incluso EOS/CLS, escluso PAD).
    Usato nel forward del modello per estrarre il vettore del token CLS.

    Padding con EOS token id: non crea ambiguità perché i token di padding
    hanno attention_mask=0, distinguendoli dal vero EOS/CLS che ha mask=1.
    """
    input_ids_list   = [item['input_ids']   for item in batch]
    slot_labels_list = [item['slot_labels'] for item in batch]
    intents          = torch.tensor([item['intent']  for item in batch], dtype=torch.long)
    seq_lens         = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)
    words_list       = [item['words'] for item in batch]

    max_len = max(len(ids) for ids in input_ids_list)
    # EOS token id: ultimo token di ogni sequenza (sempre EOS/CLS che abbiamo appeso)
    pad_id  = input_ids_list[0][-1].item()

    padded_input_ids   = []
    padded_slot_labels = []
    attention_masks    = []

    for input_ids, slot_labels in zip(input_ids_list, slot_labels_list):
        pad_len = max_len - len(input_ids)
        padded_input_ids.append(
            torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        padded_slot_labels.append(
            torch.cat([slot_labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
        )
        attention_masks.append(torch.cat([
            torch.ones(len(input_ids), dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ]))

    return {
        'input_ids':      torch.stack(padded_input_ids),
        'attention_mask': torch.stack(attention_masks),
        'y_slots':        torch.stack(padded_slot_labels),
        'intents':        intents,
        'seq_lens':       seq_lens,
        'words':          words_list,
    }


def get_dataloaders(train_dataset, dev_dataset, test_dataset,
                    collate_fn, batch_size_train=32, batch_size_eval=64):
    """
    Crea i tre DataLoader usando la collate_fn appropriata (BERT o GPT-2).

    Batch size per il fine-tuning di modelli pre-addestrati:
      - train: 32 (più piccolo rispetto alla 2.A perché BERT/GPT-2 occupano
                   molto più memoria di un modello from-scratch piccolo)
      - eval: 64 (no backward pass → meno memoria)
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train,
        collate_fn=collate_fn, shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size_eval,
        collate_fn=collate_fn, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_eval,
        collate_fn=collate_fn, shuffle=False
    )
    return train_loader, dev_loader, test_loader


# ===========================================================================
# 7. SETUP TOKENIZER
# ===========================================================================

def get_bert_tokenizer(model_name="bert-base-uncased"):
    """
    Carica il tokenizer WordPiece di BERT.

    BERT usa WordPiece: le parole non nel vocabolario vengono divise in
    sotto-unità con il prefisso '##' (es. "flying" → ["fly", "##ing"]).
    Il tokenizer ha ~30.000 token nel vocabolario.

    Caratteristiche:
      - cls_token: [CLS] (id=101 per bert-base-uncased)
      - sep_token: [SEP] (id=102)
      - pad_token: [PAD] (id=0)
      - unk_token: [UNK] (id=100)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_gpt2_tokenizer(model_name="openai-community/gpt2"):
    """
    Carica il tokenizer BPE di GPT-2.

    GPT-2 usa BPE (Byte Pair Encoding) con ~50.257 token. A differenza di
    BERT, non ha un pad token dedicato. Impostiamo pad_token = eos_token
    (pratica standard per GPT-2, come fatto in Parte 1.B).

    Caratteristiche:
      - eos_token: '<|endoftext|>' (id=50256) → usato anche come CLS e pad
      - bos_token: '<|endoftext|>' (stesso di eos)
      - NO cls_token, NO sep_token nativi
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # necessario per il padding in batch
    return tokenizer
