"""
model.py — Part 2.A: GPT-2 from scratch per NLU (Intent Classification + Slot Filling)
========================================================================================

QUADRO GENERALE
---------------
Questo modello è la versione NLU del GPT-2 implementato nella Parte 1.A.
L'architettura è identica (MultiHeadAttention, FeedForward, TransformerBlock,
embedding token+posizionale, causal mask), ma la TESTA DI OUTPUT cambia:

  Parte 1.A (Language Modeling):
    output = lm_head(x)        → (B, L, vocab_size)  — predice il token successivo

  Parte 2.A (NLU):
    slot_out   = slot_head(x)  → (B, L, slots_size)  — predice lo slot per ogni token
    intent_out = intent_head(x[CLS]) → (B, n_intents) — predice l'intent dall'ultimo token

TASK JOINT: SLOT FILLING + INTENT CLASSIFICATION
-------------------------------------------------
I due task vengono addestrati INSIEME (joint learning): il modello condivide
l'encoder (tutti i TransformerBlock) e ha due teste separate.

  1. SLOT FILLING — Sequence Labeling:
     Per ogni token della frase viene predetta un'etichetta BIO (es. "B-fromloc.city_name").
     L'output `slots` ha shape (B, L, slots_size): per ogni batch, per ogni posizione,
     uno score per ogni possibile etichetta slot.
     Il token CLS alla fine NON contribuisce: gli assegniamo slot=PAD_TOKEN nel ground truth
     così CrossEntropyLoss(ignore_index=PAD_TOKEN) lo ignora automaticamente.

  2. INTENT CLASSIFICATION — Sequence Classification:
     L'intera frase ha UN SOLO intent (es. "flight"). Per classificarla usiamo il
     vettore hidden state del token CLS (l'ultimo token della sequenza reale).
     Poiché GPT-2 è causal (ogni token vede solo i precedenti), il CLS essendo l'ULTIMO
     ha "visto" tutti i token della frase → contiene un riassunto dell'intera sequenza.
     L'output `intent` ha shape (B, n_intents): per ogni esempio del batch, uno score
     per ogni possibile intent.

PERCHÉ IL CLS È IN FONDO (e non in testa come in BERT)?
--------------------------------------------------------
BERT è encoder-only e BIDIREZIONALE: il [CLS] all'inizio del testo "vede" tutto
grazie all'attenzione bidirezionale.
GPT-2 è decoder-only e CAUSALE: il token alla posizione t vede solo i token 0..t-1.
Mettere CLS in fondo significa che vedrà TUTTE le parole della frase → è l'unico
posto che da accesso all'intera sequenza in un modello causal.

INIZIALIZZAZIONE RANDOM DEI PESI
---------------------------------
Il notebook usa una funzione `init_weights` che inizializza:
  - Ogni nn.Linear con uniform(-0.01, 0.01) per i pesi
  - Ogni nn.Linear con 0.01 per il bias
Questo "reset" dei pesi predefiniti di PyTorch migliora la convergenza iniziale.

STRUTTURA DELLE CLASSI
-----------------------
  MultiHeadAttention — identica a LM/A (causal self-attention)
  FeedForward        — identica a LM/A (Linear → GELU → Linear → Dropout)
  TransformerBlock   — identica a LM/A (Pre-LayerNorm + residual connections)
  GPT2               — come LM/A ma con slot_out e intent_out invece di lm_head
                       e forward(idx, seq_lens) che estrae il vettore CLS
  init_weights       — funzione di inizializzazione random dei parametri
  count_parameters   — utility per contare i parametri addestrabili
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. MULTI-HEAD ATTENTION (Causal Self-Attention)
# ===========================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention con causal mask — identica alla Parte 1.A.

    L'attenzione CAUSALE garantisce che ogni token possa "vedere" solo i token
    precedenti (e se stesso), non quelli futuri. Questo rispetta il paradigma
    auto-regressivo di GPT-2.

    STRUTTURA PER OGNI TESTA (head):
      Q = w_q(x)       → query: "cosa sto cercando?"
      K = w_k(x)       → key:   "cosa offro come informazione?"
      V = w_v(x)       → value: "quale informazione passo se vieni selezionato?"
      score = QK^T / sqrt(d_k)  → similarità scalata
      score[j > i] = -inf       → causal mask: azzera i futuri
      attn = softmax(score)     → distribuzione di probabilità
      output = attn @ V         → aggregazione pesata dei valori

    Usando più teste (n_heads), il modello cattura diversi tipi di relazioni
    linguistiche contemporaneamente (es. una testa per sintassi, una per semantica).
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model : dimensione del modello (es. 128, 256, 512)
            n_heads : numero di teste di attenzione (es. 4, 8)
            dropout : probabilità di dropout (sugli attention weights e sull'output)
        """
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) deve essere divisibile per n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimensione per testa

        # Proiezioni lineari per Q, K, V (ciascuna: d_model → d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Proiezione di output: riproietta le n_heads concatenate
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout sugli attention weights (after softmax) e sull'output
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        Trasforma (B, L, d_model) → (B, n_heads, L, d_k).
        Permette di elaborare tutte le teste in parallelo con operazioni matriciali.
        """
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x):
        """
        Operazione inversa: (B, n_heads, L, d_k) → (B, L, d_model).
        .contiguous() è necessario perché .view() richiede un tensore contiguo in memoria.
        """
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x    : (B, L, d_model) — input (hidden states)
            mask : (1, 1, L, L)   — causal mask: 0.0 dove si può attendere, -inf dove no

        Returns:
            out  : (B, L, d_model) — output dell'attenzione multi-testa
        """
        # Proietta in Q, K, V e divide in n_heads
        Q = self.split_heads(self.w_q(x))  # (B, n_heads, L, d_k)
        K = self.split_heads(self.w_k(x))
        V = self.split_heads(self.w_v(x))

        # Calcola gli attention scores: QK^T / sqrt(d_k)
        # La divisione per sqrt(d_k) previene che i dot product crescano troppo
        # con d_k grande → zone a gradiente quasi zero nel softmax
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, n_heads, L, L)

        # Applica la causal mask: dove mask=-inf, il softmax assegna peso ~0
        if mask is not None:
            scores = scores + mask

        # Softmax → distribuzione di probabilità per ogni query
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Aggregazione pesata dei valori
        context = torch.matmul(attn_weights, V)  # (B, n_heads, L, d_k)

        # Unisce le teste e proietta l'output
        out = self.out_proj(self.merge_heads(context))
        return self.proj_dropout(out)


# ===========================================================================
# 2. FEED-FORWARD NETWORK
# ===========================================================================

class FeedForward(nn.Module):
    """
    Feed-Forward Network position-wise — identica alla Parte 1.A.

    Applicata a ogni token INDIPENDENTEMENTE (è una rete condivisa su tutte le posizioni).
    Struttura: Linear(d_model → ff_dim) → GELU → Linear(ff_dim → d_model) → Dropout

    Il rapporto ff_dim / d_model è tipicamente 4 (es. 256/64, 512/128, 3072/768).
    L'espansione e la contrazione creano uno "spazio latente più ricco" dove il modello
    può fare trasformazioni non lineari complesse prima di tornare a d_model dimensioni.

    GELU: Gaussian Error Linear Unit. Più smooth di ReLU → gradienti più stabili.
    """

    def __init__(self, d_model, hidden_dim, dropout=0.1):
        """
        Args:
            d_model    : dimensione input/output
            hidden_dim : dimensione dello strato nascosto (tipicamente 4 * d_model)
            dropout    : probabilità di dropout
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ===========================================================================
# 3. TRANSFORMER BLOCK (Pre-LayerNorm)
# ===========================================================================

class TransformerBlock(nn.Module):
    """
    Singolo blocco Transformer con Pre-LayerNorm — identico alla Parte 1.A.

    Schema (Pre-LN, usato da GPT-2):
        x → LN1 → MHA → residual (+x) → LN2 → FFN → residual (+x)

    Le RESIDUAL CONNECTIONS permettono al gradiente di fluire anche attraverso
    molti strati (alleviando il vanishing gradient) e al modello di imparare
    "modifiche incrementali" piuttosto che trasformazioni complete.

    PRE-LN (normalizzazione PRIMA del sublayer) è più stabile di POST-LN
    e permette il training di modelli più profondi senza warmup aggressivo.
    """

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)  # Attention + residual
        x = x + self.ff(self.ln2(x))           # FFN + residual
        return x


# ===========================================================================
# 4. GPT-2 PER NLU — MODELLO COMPLETO
# ===========================================================================

class GPT2(nn.Module):
    """
    GPT-2 from scratch adattato per Intent Classification e Slot Filling.

    DIFFERENZE RISPETTO ALLA PARTE 1.A (Language Modeling)
    -------------------------------------------------------
    Parte 1.A ha come output:
        lm_head: Linear(d_model, vocab_size)
        Predice il token successivo per OGNI posizione della sequenza.

    Questa versione ha due teste di output:
        slot_out:   Linear(d_model, slots_size)  — una predizione per ogni token
        intent_out: Linear(d_model, n_intents)   — una predizione per l'intera frase

    Inoltre il forward prende `seq_lens` (le lunghezze reali delle sequenze incluso CLS)
    per estrarre il vettore del token CLS dalla posizione corretta.

    FLUSSO DEL FORWARD PASS:
    ─────────────────────────────────────────────────────────────────────────
    input_ids (B, L)
        ↓
    token_embed(input_ids) + pos_embed(arange(L))   → x: (B, L, d_model)
        ↓
    N × TransformerBlock(x, causal_mask)             → x: (B, L, d_model)
        ↓
    ln_f(x)                                          → x: (B, L, d_model)
        ↓
    [Per ogni token]   slot_out(x)                   → slots: (B, L, slots_size)
    [Solo CLS token]   intent_out(x[:, seq_len-1, :]) → intent: (B, n_intents)
    ─────────────────────────────────────────────────────────────────────────

    NOTA: Non usiamo weight tying perché non abbiamo una lm_head. I due
    compiti (slot e intent) sono strutturalmente diversi da LM.
    """

    def __init__(
        self,
        vocab_size,
        slots_size,
        n_intents,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.1,
    ):
        """
        Args:
            vocab_size   : dimensione del vocabolario word-level (da Lang.word2id)
            slots_size   : numero di etichette slot possibili (da Lang.id2slot)
            n_intents    : numero di intent possibili (da Lang.intent2id)
            pos_emb_size : lunghezza massima della sequenza (context window)
            d_model      : dimensione degli embedding e hidden states
            n_heads      : numero di teste di attenzione per blocco
            num_layers   : numero di TransformerBlock impilati
            ff_dim       : dimensione nascosta della FFN (tipicamente 4*d_model)
            dropout      : probabilità di dropout
        """
        super().__init__()

        self.pos_emb_size = pos_emb_size

        # -----------------------------------------------------------------------
        # EMBEDDINGS
        # -----------------------------------------------------------------------

        # Token embedding: mappa ogni ID parola in un vettore d_model-dimensionale.
        # Questo include anche i token speciali (pad=0, unk=1, cls=2).
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Positional embedding: mappa ogni posizione (0..pos_emb_size-1) in un vettore.
        # A differenza dei Transformer originali (encoding sinusoidale), GPT-2 usa
        # embedding posizionali APPRESI durante il training.
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        # Dropout sull'embedding combinato (token + posizionale)
        self.emb_dropout = nn.Dropout(dropout)

        # -----------------------------------------------------------------------
        # STACK DI TRANSFORMER BLOCKS
        # -----------------------------------------------------------------------

        # Il cuore del modello: N blocchi Transformer impilati.
        # Ogni blocco raffina la rappresentazione di ogni token guardando
        # i token precedenti (causal attention) e applicando trasformazioni FFN.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # -----------------------------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------------------------

        # Final LayerNorm: normalizzazione dopo l'ultimo TransformerBlock
        self.ln_f = nn.LayerNorm(d_model)

        # TESTA PER SLOT FILLING: un'etichetta per ogni token della sequenza.
        # Input:  (B, L, d_model) — hidden states di ogni token
        # Output: (B, L, slots_size) — score per ogni etichetta slot, per ogni token
        # Nota: includiamo anche una predizione per il CLS, ma nel ground truth
        #       il CLS ha etichetta PAD_TOKEN (0), che viene ignorata dalla loss.
        self.slot_out = nn.Linear(d_model, slots_size)

        # TESTA PER INTENT CLASSIFICATION: un intent per l'intera sequenza.
        # Input:  (B, d_model) — hidden state del CLS token (estratto nel forward)
        # Output: (B, n_intents) — score per ogni intent possibile
        # Il CLS token, essendo l'ultimo, ha "visto" tutta la frase → riassume il significato.
        self.intent_out = nn.Linear(d_model, n_intents)

        # -----------------------------------------------------------------------
        # CAUSAL MASK (buffer, non è un parametro addestrabile)
        # -----------------------------------------------------------------------
        # Matrice (pos_emb_size, pos_emb_size) con 0 dove l'attenzione è permessa
        # e -inf dove non lo è. Il token alla posizione i può vedere solo j <= i.
        #
        # Esempio per pos_emb_size=4:
        #   [ 0,   -inf, -inf, -inf ]   ← token 0 vede solo se stesso
        #   [ 0,    0,   -inf, -inf ]   ← token 1 vede 0 e 1
        #   [ 0,    0,    0,   -inf ]   ← token 2 vede 0, 1, 2
        #   [ 0,    0,    0,    0   ]   ← token 3 (CLS) vede tutta la frase
        #
        # Per il CLS token (in fondo), questo significa che vedrà TUTTI i token
        # della frase → perfetto per aggregare informazioni per l'intent.
        mask = torch.triu(
            torch.full((pos_emb_size, pos_emb_size), float('-inf')),
            diagonal=1  # 0 sulla diagonale (auto-attenzione permessa)
        )
        self.register_buffer("mask", mask)

    def forward(self, idx, seq_lens):
        """
        Forward pass del modello NLU.

        Args:
            idx      : (B, L) LongTensor — sequenze di token ID (con padding e CLS)
            seq_lens : (B,)   LongTensor — lunghezze REALI di ogni sequenza (incluso CLS,
                               escluso il padding). Usate per estrarre il vettore CLS.

        Returns:
            slots  : (B, L, slots_size) — logit per le etichette slot (tutti i token)
            intent : (B, n_intents)     — logit per gli intent (solo CLS)
        """
        B, L = idx.shape
        assert L <= self.pos_emb_size, \
            f"Sequenza troppo lunga: {L} > {self.pos_emb_size}"

        # -----------------------------------------------------------------------
        # 1. EMBEDDINGS
        # -----------------------------------------------------------------------

        # Embedding dei token: ogni parola diventa un vettore d_model-dimensionale
        tok_emb = self.token_embed(idx)  # (B, L, d_model)

        # Embedding posizionale: il vettore di posizione viene sommato al token embedding.
        # Questo è il modo in cui il modello "sa" la posizione di ogni token.
        # positions = [0, 1, 2, ..., L-1], broadcastato su tutto il batch
        positions = torch.arange(L, device=idx.device)
        pos_emb = self.pos_embed(positions)  # (L, d_model) → broadcastato a (B, L, d_model)

        # Somma token + posizionale, poi dropout per regolarizzazione
        x = self.emb_dropout(tok_emb + pos_emb)  # (B, L, d_model)

        # -----------------------------------------------------------------------
        # 2. CAUSAL MASK (slicata alla lunghezza corrente del batch)
        # -----------------------------------------------------------------------
        # Prendiamo il sotto-blocco (L, L) della mask pre-calcolata e aggiungiamo
        # le dimensioni batch e heads per il broadcasting nella MultiHeadAttention:
        # (L, L) → (1, 1, L, L) — broadcastato su (B, n_heads, L, L)
        causal_mask = self.mask[:L, :L].unsqueeze(0).unsqueeze(0)

        # -----------------------------------------------------------------------
        # 3. TRANSFORMER BLOCKS
        # -----------------------------------------------------------------------
        for block in self.blocks:
            x = block(x, causal_mask)  # (B, L, d_model) dopo ogni blocco

        # Final LayerNorm
        x = self.ln_f(x)  # (B, L, d_model)

        # -----------------------------------------------------------------------
        # 4. SLOT FILLING HEAD
        # -----------------------------------------------------------------------
        # Applichiamo la testa slot a TUTTI i token (incluso CLS).
        # Il CLS riceve un'etichetta PAD_TOKEN nel ground truth → ignorata dalla loss.
        slots = self.slot_out(x)  # (B, L, slots_size)

        # -----------------------------------------------------------------------
        # 5. INTENT CLASSIFICATION HEAD — estrazione del vettore CLS
        # -----------------------------------------------------------------------
        # Per ogni esempio del batch, estraiamo il vettore dell'ultimo token reale
        # (= il token CLS, che è all'indice seq_lens[i] - 1).
        #
        # Non possiamo semplicemente fare x[:, -1, :] perché le sequenze hanno
        # lunghezze diverse e sono paddizzate → il token CLS non è sempre all'ultima
        # posizione del tensore, ma all'ultima posizione REALE (prima del padding).
        #
        # Esempio (max_len=8):
        #   frase 1: [w1, w2, w3, cls, pad, pad, pad, pad]  → seq_len=4 → indice 3
        #   frase 2: [w1, w2, w3, w4,  w5,  cls, pad, pad]  → seq_len=6 → indice 5
        cls_vectors = []
        for i in range(B):
            cls_vectors.append(x[i, seq_lens[i] - 1])  # -1: indici 0-based
        cls_tokens = torch.stack(cls_vectors)  # (B, d_model)

        # Proieziamo il vettore CLS sullo spazio degli intent
        intent = self.intent_out(cls_tokens)  # (B, n_intents)

        return slots, intent


# ===========================================================================
# 5. INIZIALIZZAZIONE DEI PESI
# ===========================================================================

def init_weights(mat):
    """
    Inizializza i pesi di tutti gli strati Linear con una distribuzione uniforme
    in [-0.01, 0.01] e il bias a 0.01.

    Da notebook cell f47fe3fe — IDENTICA.

    Perché serve questa funzione?
    PyTorch di default inizializza i Linear con una distribuzione di Kaiming,
    che è progettata per ReLU. Per reti Transformer con GELU, un'inizializzazione
    più piccola (uniforme in [-0.01, 0.01]) aiuta la convergenza nelle prime epoche.
    Il bias a 0.01 (anziché 0) introduce una piccola attivazione iniziale che
    evita il problema dei "dead neurons" nelle prime iterazioni.

    Args:
        mat : il modello (nn.Module) su cui applicare l'inizializzazione
              Si usa come: model.apply(init_weights)
    """
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


# ===========================================================================
# 6. UTILITY — CONTEGGIO PARAMETRI
# ===========================================================================

def count_parameters(model):
    """
    Conta e stampa i parametri addestrabili del modello.

    Args:
        model : istanza di GPT2

    Returns:
        n_params : numero di parametri addestrabili (int)
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri addestrabili: {n_params:,}")
    return n_params
