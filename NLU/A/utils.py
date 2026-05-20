"""
utils.py — Part 2.A: NLU con GPT-2 from scratch su ATIS
=========================================================

QUADRO GENERALE
---------------
Questo file gestisce il caricamento e la preparazione dei dati per il task NLU
(Natural Language Understanding) sul dataset ATIS.

A differenza della Parte 1 (Language Modeling), qui lavoriamo con:
  - Un vocabolario costruito a partire dalle parole del training set (NON un tokenizer BPE)
  - Etichette di slot (sequence labeling) e di intent (classificazione)
  - Un token speciale CLS aggiunto alla FINE di ogni frase (diverso da BERT che lo mette all'inizio)

PERCHÉ UN VOCABOLARIO CUSTOM E NON IL TOKENIZER GPT-2?
-------------------------------------------------------
Nella Parte 2.A costruiamo il modello da zero, partendo da un vocabolario
basato sulle parole del corpus (word-level, non subword). Questo è più
semplice da gestire con le etichette slot: ogni parola ha esattamente
una sola etichetta, senza il problema della sub-tokenizzazione.

Nella Parte 2.B (con GPT-2 e BERT preaddestrati) useremo invece il tokenizer
BPE, che richiede di allineare le etichette ai subtoken.

DATASET: ATIS (Airline Travel Information Systems)
--------------------------------------------------
ATIS contiene trascrizioni di richieste di informazioni su voli aerei.
Il formato JSON di ogni esempio è:
  {
    "utterance": "on april first i need a flight ...",
    "slots":     "O B-depart_date.month_name B-depart_date.day_number O O O O ...",
    "intent":    "flight"
  }

Statistiche:
  - Training: ~4478 frasi (dopo la split stratificata)
  - Dev:       ~500 frasi (10% del training originale, split stratificata)
  - Test:      891 frasi (split originale del dataset)
  - Vocabolario: ~938 parole
  - Slot labels: ~130 etichette BIO
  - Intent labels: 18 intenti

SPLIT STRATIFICATA DEL DEV SET
--------------------------------
Il dataset ATIS originale non ha un dev set separato. Lo creiamo noi
prendendo il 10% del training set con una split STRATIFICATA sull'intent.
"Stratificata" significa che ogni intent è rappresentato nel dev set
nella stessa proporzione in cui appare nel training set.
Questo è importante perché ATIS è sbilanciato: l'intent "flight" è molto
più frequente degli altri.

TOKEN SPECIALE CLS
------------------
GPT-2 è un modello auto-regressivo (causal): il token alla posizione t
vede solo i token 0..t-1. Per fare intent classification (che richiede
di "vedere" l'intera frase) aggiungiamo un token CLS alla FINE della frase.
Il token CLS, essendo l'ultimo, avrà nel suo vettore hidden state
un riassunto dell'intera sequenza — perfetto per classificare l'intent.

Nella slot filling, invece, usiamo le rappresentazioni di ogni singolo
token della frase (escludendo il CLS).
"""

import json
import random
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

# PAD_TOKEN = 0: le sequenze vengono paddizzate con 0 fino alla lunghezza massima del batch
PAD_TOKEN = 0


# ===========================================================================
# 1. CARICAMENTO DEL DATASET RAW
# ===========================================================================

def load_data(path):
    """
    Carica il dataset ATIS da un file JSON.

    Il file JSON è una lista di dizionari, uno per esempio, con le chiavi:
      - "utterance": la frase dell'utente (stringa)
      - "slots":     le etichette BIO per ogni token (stringa, spazi come separatori)
      - "intent":    l'intento dell'intera frase (stringa)

    Args:
        path : percorso al file .json (train.json o test.json)

    Returns:
        dataset : lista di dizionari, uno per esempio
    """
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


# ===========================================================================
# 2. SPLIT STRATIFICATA DEL DEV SET
# ===========================================================================

def create_dev_split(train_raw, dev_size=0.10, random_state=42):
    """
    Crea un dev set stratificato sull'intent a partire dal training set.

    Perché stratificata?
    Il dataset ATIS è sbilanciato: "flight" rappresenta ~70% degli esempi.
    Una split casuale potrebbe creare un dev set senza esempi di intent rari.
    La split stratificata garantisce che ogni intent sia rappresentato nel
    dev set nella stessa proporzione del training set.

    Gestione degli intent con un solo esempio:
    Se un intent compare una sola volta nel training, non può essere splittato
    (almeno un esempio deve restare nel training). Questi esempi vengono
    aggiunti direttamente al training set.

    Args:
        train_raw    : lista di dizionari (output di load_data per train.json)
        dev_size     : proporzione del training da usare come dev (default 10%)
        random_state : seed per la riproducibilità

    Returns:
        train_data : lista di dizionari per il training
        dev_data   : lista di dizionari per la validation
    """
    intents = [x['intent'] for x in train_raw]
    count_y = Counter(intents)

    # Separiamo gli esempi con intent che compaiono più di una volta
    # (possono essere splittati) da quelli che compaiono una sola volta
    inputs    = []  # esempi che possono finire nel dev
    labels    = []  # loro etichette di intent (per la stratificazione)
    mini_train = [] # esempi che devono restare nel training

    for idx, example in enumerate(train_raw):
        if count_y[example['intent']] > 1:
            inputs.append(example)
            labels.append(example['intent'])
        else:
            mini_train.append(example)

    # Split stratificata: sklearn garantisce la distribuzione uniforme degli intent
    X_train, X_dev, _, _ = train_test_split(
        inputs, labels,
        test_size=dev_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels  # chiave: mantieni le proporzioni degli intent
    )

    # Riaggiungiamo gli esempi con intent rari al training
    X_train.extend(mini_train)

    return X_train, X_dev


# ===========================================================================
# 3. CLASSE LANG — VOCABOLARIO
# ===========================================================================

class Lang:
    """
    Gestisce i vocabolari per parole, slot e intent.

    Ogni vocabolario è un dizionario che mappa stringhe a interi:
      - word2id  : parola → indice intero
      - slot2id  : etichetta slot → indice intero
      - intent2id: etichetta intent → indice intero

    Costruiamo anche i dizionari inversi (id → stringa) per la valutazione:
      - id2word, id2slot, id2intent

    Token speciali:
      - 'pad' (id=0): usato per il padding delle sequenze
      - 'unk' (id=1): parole sconosciute a test time
      - 'cls' (id=2): token CLS aggiunto alla fine di ogni frase

    NOTA IMPORTANTE sul CLS nei slot:
    Quando processiamo una frase, aggiungiamo CLS alla fine. Il modello
    produrrà una previsione slot anche per il CLS, ma NON vogliamo includerla
    nella loss. Per questo, nel slot2id assegniamo al 'cls' lo stesso id del
    'pad' (id=0). La CrossEntropyLoss usa ignore_index=PAD_TOKEN=0, quindi
    ignorerà automaticamente le previsioni slot del CLS token.
    """

    def __init__(self, words, intents, slots, cutoff=0):
        """
        Args:
            words   : lista di tutte le parole del training set (con duplicati)
            intents : insieme di tutti gli intent del corpus
            slots   : insieme di tutte le etichette slot del corpus
            cutoff  : frequenza minima per includere una parola nel vocabolario
                      (0 = includi tutte le parole)
        """
        self.word2id   = self._build_word_vocab(words, cutoff=cutoff)
        self.slot2id   = self._build_label_vocab(slots, pad=True,  cls=True)
        self.intent2id = self._build_label_vocab(intents, pad=False, cls=False)

        # Dizionari inversi per la decodifica a tempo di valutazione
        self.id2word   = {v: k for k, v in self.word2id.items()}
        # Per id2slot escludiamo il 'cls' (che ha id=0=pad) per evitare ambiguità
        self.id2slot   = {v: k for k, v in self.slot2id.items()
                          if k != 'cls'}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def _build_word_vocab(self, words, cutoff=0):
        """
        Costruisce il vocabolario delle parole con token speciali.

        Usa Counter per contare le frequenze e filtra le parole con
        frequenza <= cutoff (gestione di parole rare).

        Struttura del vocabolario:
          0: 'pad'  — padding
          1: 'unk'  — unknown word
          2: 'cls'  — CLS token
          3+: parole del training set (ordinate per frequenza decrescente)
        """
        vocab = {
            'pad': PAD_TOKEN,   # 0
            'unk': 1,           # 1
            'cls': 2,           # 2
        }
        count = Counter(words)
        for word, freq in count.items():
            if freq > cutoff and word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    def _build_label_vocab(self, labels, pad=True, cls=True):
        """
        Costruisce il vocabolario per le etichette (slot o intent).

        Args:
            labels : insieme di etichette
            pad    : se True, aggiunge 'pad' con id=0
            cls    : se True, aggiunge 'cls' con id=PAD_TOKEN=0
                     (così le previsioni sul CLS vengono ignorate dalla loss)
        """
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN  # 0

        for label in sorted(labels):  # sorted per riproducibilità
            if label not in vocab:
                vocab[label] = len(vocab)

        if cls:
            # Il CLS token riceve id=0 (stesso del pad) → viene ignorato dalla loss
            # Questo è il meccanismo che permette di usare il CLS solo per l'intent
            # e non per lo slot filling
            vocab['cls'] = PAD_TOKEN

        return vocab


def build_lang(train_raw, dev_raw, test_raw, cutoff=0):
    """
    Costruisce l'oggetto Lang a partire dai tre split del dataset.

    REGOLA IMPORTANTE:
    - Le PAROLE vengono prese SOLO dal training set. Parole mai viste durante
      il training vengono mappate a 'unk' a test time. Questo simula le
      condizioni reali di deployment.
    - Le ETICHETTE (slot e intent) vengono raccolte da TUTTI gli split
      (train + dev + test). Questo perché non vogliamo etichette sconosciute
      a test time — il modello deve essere in grado di predire tutte le
      etichette possibili.

    Args:
        train_raw : lista di esempi del training set
        dev_raw   : lista di esempi del dev set
        test_raw  : lista di esempi del test set
        cutoff    : soglia di frequenza minima per le parole

    Returns:
        lang : istanza di Lang con tutti i vocabolari costruiti
    """
    # Parole: solo dal training set (lista con duplicati per contare frequenze)
    words = sum([x['utterance'].split() for x in train_raw], [])

    # Etichette: da tutti gli split (usiamo set per evitare duplicati)
    corpus = train_raw + dev_raw + test_raw
    slots   = set(sum([x['slots'].split()   for x in corpus], []))
    intents = set(x['intent'] for x in corpus)

    return Lang(words, intents, slots, cutoff=cutoff)


# ===========================================================================
# 4. DATASET CLASS
# ===========================================================================

class IntentsAndSlots(data.Dataset):
    """
    Dataset class per ATIS: gestisce utterances, slot labels e intent labels.

    Ogni esempio viene convertito da stringhe a sequenze di interi usando
    i vocabolari della classe Lang. Il token CLS viene aggiunto alla FINE
    di ogni utterance (e della sequenza di slot labels) per permettere
    al modello decoder-only GPT-2 di fare intent classification.

    La struttura di un esempio dopo __getitem__:

      utterance: [w1, w2, ..., wN, cls_id]   — N parole + CLS
      slots:     [s1, s2, ..., sN, pad_id]   — N slot labels + PAD (per CLS)
      intent:    int                          — un singolo intero

    Perché pad_id per il CLS nei slot?
    Il modello produrrà una previsione slot per il CLS, ma non ha senso
    (il CLS non è una parola con uno slot). Assegnando pad_id al CLS nei
    ground truth slots, la CrossEntropyLoss lo ignorerà automaticamente
    (grazie a ignore_index=PAD_TOKEN).
    """

    def __init__(self, dataset, lang, unk='unk', cls='cls', add_cls=True):
        """
        Args:
            dataset : lista di dizionari (output di load_data)
            lang    : istanza di Lang con i vocabolari
            unk     : nome del token unknown nel vocabolario
            cls     : nome del token CLS nel vocabolario
            add_cls : se True, aggiunge il token CLS alla fine di ogni frase
        """
        self.utterances = []
        self.intents    = []
        self.slots      = []
        self.unk        = unk
        self.cls        = cls
        self.add_cls    = add_cls

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        # Convertiamo le stringhe in sequenze di interi
        self.utt_ids    = self._mapping_seq(self.utterances, lang.word2id)
        self.slot_ids   = self._mapping_seq(self.slots,      lang.slot2id)
        self.intent_ids = self._mapping_lab(self.intents,    lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        """
        Restituisce un esempio come dizionario di tensori.

        Returns:
            sample : dict con chiavi 'utterance', 'slots', 'intent'
        """
        utt    = torch.Tensor(self.utt_ids[idx])
        slots  = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        return {'utterance': utt, 'slots': slots, 'intent': intent}

    def _mapping_lab(self, data, mapper):
        """Mappa una lista di stringhe (label singole) a una lista di interi."""
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def _mapping_seq(self, data, mapper):
        """
        Mappa una lista di sequenze di stringhe a una lista di sequenze di interi.
        Se add_cls=True, aggiunge il token CLS alla fine di ogni sequenza.
        Le parole non nel vocabolario vengono mappate a 'unk' (solo per utterances).
        """
        result = []
        for seq in data:
            tmp = []
            for token in seq.split():
                if token in mapper:
                    tmp.append(mapper[token])
                else:
                    tmp.append(mapper[self.unk])
            if self.add_cls:
                tmp.append(mapper[self.cls])
            result.append(tmp)
        return result


# ===========================================================================
# 5. COLLATE FUNCTION E DATALOADER
# ===========================================================================

def collate_fn(data):
    """
    Funzione di collating: assembla un batch di esempi in tensori paddizzati.

    Il problema principale è che le frasi hanno lunghezze diverse. Questa
    funzione aggiunge padding (token 0) a destra fino alla lunghezza massima
    nel batch, creando tensori rettangolari (batch_size × max_len).

    Schema di un batch di 3 frasi con max_len=8:
      [w1, w2, w3, w4, cls, 0,  0,  0 ]
      [w1, w2, w3, w4, w5,  w6, cls, 0 ]
      [w1, w2, cls, 0,  0,  0,  0,  0 ]

    Il valore di 'slots_len' per ogni frase è la lunghezza REALE (incluso CLS),
    usata nel forward del modello per estrarre il vettore del CLS token.

    Returns:
        batch: dizionario con:
          - 'utterances': (B, max_len) LongTensor
          - 'y_slots':    (B, max_len) LongTensor
          - 'intents':    (B,)         LongTensor
          - 'slots_len':  (B,)         LongTensor — lunghezze reali (senza padding)
    """

    def merge(sequences):
        """Pad una lista di sequenze alla lunghezza massima."""
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if max(lengths) > 0 else 1
        # Creiamo una matrice piena di PAD_TOKEN
        padded = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded, lengths

    # Raggruppa i campi: da lista di dict a dict di liste
    data_by_key = {key: [d[key] for d in data] for key in data[0].keys()}

    src_utt, _        = merge(data_by_key['utterance'])
    y_slots, y_lengths = merge(data_by_key['slots'])
    intent             = torch.LongTensor(data_by_key['intent'])

    return {
        'utterances': src_utt,
        'intents':    intent,
        'y_slots':    y_slots,
        'slots_len':  torch.LongTensor(y_lengths)
    }


def get_dataloaders(train_dataset, dev_dataset, test_dataset,
                    batch_size_train=128, batch_size_eval=64, device='cpu'):
    """
    Crea i tre DataLoader per train, dev e test.

    Batch size consigliati dal notebook:
      - train: 128 (il dataset è piccolo, batch grandi sono ok)
      - dev/test: 64

    Args:
        train_dataset    : istanza di IntentsAndSlots per il training
        dev_dataset      : istanza di IntentsAndSlots per la validation
        test_dataset     : istanza di IntentsAndSlots per il test
        batch_size_train : batch size per il training
        batch_size_eval  : batch size per dev/test
        device           : device su cui spostare i tensori ('cpu', 'cuda', 'mps')

    Returns:
        train_loader, dev_loader, test_loader
    """
    def collate_fn_device(data):
        batch = collate_fn(data)
        return {k: v.to(device) for k, v in batch.items()}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        collate_fn=collate_fn_device,
        shuffle=True   # shuffle solo per il training
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size_eval,
        collate_fn=collate_fn_device
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_eval,
        collate_fn=collate_fn_device
    )
    return train_loader, dev_loader, test_loader
