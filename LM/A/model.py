"""
model.py — Part 1.A: GPT-2 from scratch
=========================================

QUADRO GENERALE — L'ARCHITETTURA GPT-2
---------------------------------------
GPT-2 è un Transformer *decoder-only*, cioè usa solo il blocco "decoder" del
Transformer originale (Vaswani et al., 2017), ma senza cross-attention (non c'è
un encoder a cui prestare attenzione). È progettato per il Language Modeling
auto-regressivo: ogni token può "vedere" solo i token precedenti, mai quelli futuri.

Questo vincolo è imposto dalla CAUSAL MASK (o "look-ahead mask"):
una matrice triangolare inferiore di 1 e -inf che, quando sommata agli attention
scores, azzera l'attenzione verso i token futuri.

PIPELINE FORWARD PASS
----------------------
  token_ids (batch, seq_len)
       ↓
  Token Embedding (vocab_size → d_model)   +   Positional Embedding (seq_len → d_model)
       ↓  [Embedding Dropout]
  N × TransformerBlock (Pre-LayerNorm + MultiHeadAttention + FeedForward)
       ↓
  Final LayerNorm
       ↓
  LM Head (d_model → vocab_size) — proietta su tutti i possibili token
       ↓
  Logits (batch, seq_len, vocab_size)

DIFFERENZE DA GPT-1 / Transformer originale
--------------------------------------------
1. PRE-LayerNorm: la normalizzazione avviene *prima* dell'attenzione e della
   FeedForward (non dopo come nel Transformer originale). Questo stabilizza
   il training con modelli profondi.
2. Nessuna positional encoding sinusoidale: usa positional embeddings *appresi*.
3. GELU invece di ReLU nella FeedForward.
4. Weight Tying: la matrice del token embedding viene riutilizzata come testa LM.

WEIGHT TYING
------------
L'idea è che la matrice W (vocab_size × d_model) usata per proiettare il vettore
nascosto sui logit (LM Head) sia la *stessa* della matrice usata nell'embedding
(token_embed.weight). Questo:
  - Riduce il numero di parametri (risparmio di ~38M param per vocab=50k, d=768)
  - Funziona perché l'embedding e la testa LM fanno operazioni "duali":
    l'embedding mappa token → spazio latente, la testa fa spazio latente → token
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
    Multi-Head Self-Attention con causal mask.

    INTUIZIONE
    ----------
    L'attenzione permette al modello di "guardare" altri token nella sequenza
    per costruire la rappresentazione di ogni token. Con più "teste" (heads),
    il modello può catturare diversi tipi di relazioni in parallelo.

    PER OGNI HEAD:
      1. Proietta gli input in spazi Q (query), K (key), V (value)
      2. Calcola gli scores: score(Q, K) = QK^T / sqrt(d_k)
      3. Applica la causal mask (azzera i futuri)
      4. Softmax → attention weights
      5. Prodotto pesato con V → output per questa testa

    Poi tutte le teste vengono concatenate e riproiettate con out_proj.

    In pratica, invece di avere n_heads moduli separati, facciamo tutto in
    un'unica grande matrice e poi riorganizziamo le dimensioni (split heads).
    Questo è più efficiente perché sfrutta meglio il parallelismo GPU.

    DROPOUT NELL'ATTENZIONE
    -----------------------
    Applichiamo dropout sugli attention weights (dopo softmax, prima del prodotto
    con V). Questo "disattiva" alcune connessioni casualmente durante il training,
    forzando il modello a distribuire l'attenzione su più token.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model  : dimensione del modello (es. 768 per GPT-2 base)
            n_heads  : numero di teste di attenzione (es. 12)
            dropout  : probabilità di dropout sugli attention weights
        """
        super().__init__()

        # Verifica che d_model sia divisibile per n_heads
        # Ogni testa lavora su d_model // n_heads dimensioni (= d_k)
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) deve essere divisibile per n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimensioni per testa

        # Proiezioni lineari per Query, Key, Value (bias=True di default)
        # Ciascuna proietta da d_model → d_model (poi splittata in n_heads)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Proiezione di output: concatena i risultati delle n_heads e riproietta
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout sugli attention weights (after softmax)
        self.attn_dropout = nn.Dropout(dropout)
        # Dropout sull'output dopo out_proj
        self.proj_dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        Riorganizza il tensore da (batch, seq, d_model) a (batch, n_heads, seq, d_k).
        Questo permette di elaborare tutte le teste in parallelo.
        """
        batch, seq, d_model = x.shape
        # Reshape: (batch, seq, n_heads, d_k) → poi trasponi heads e seq
        x = x.view(batch, seq, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq, d_k)

    def merge_heads(self, x):
        """
        Operazione inversa: (batch, n_heads, seq, d_k) → (batch, seq, d_model).
        """
        batch, n_heads, seq, d_k = x.shape
        # Trasponi e rendi contiguo in memoria, poi fai reshape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)  # (batch, seq, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x    : (batch, seq_len, d_model) — input
            mask : (1, 1, seq_len, seq_len) — causal mask (già creata nel GPT2)

        Returns:
            out  : (batch, seq_len, d_model) — output dopo self-attention
        """
        # 1. Proietta in Q, K, V e splitta in n_heads
        Q = self.split_heads(self.w_q(x))  # (batch, n_heads, seq, d_k)
        K = self.split_heads(self.w_k(x))
        V = self.split_heads(self.w_v(x))

        # 2. Attention scores: QK^T / sqrt(d_k)
        #    Scaling per d_k evita che i dot product crescano troppo con d_k grande,
        #    che spingerebbe il softmax in zone a gradiente quasi zero.
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (batch, n_heads, seq, seq)

        # 3. Applica la causal mask
        #    La mask contiene 0.0 dove l'attenzione è permessa e -inf dove non lo è.
        #    Sommando -inf agli scores, il softmax assegnerà peso ~0 a quei token.
        if mask is not None:
            scores = scores + mask

        # 4. Softmax → attention weights (distribuzione di probabilità per ogni query)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq, seq)

        # 5. Dropout sugli attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # 6. Prodotto pesato con V
        context = torch.matmul(attn_weights, V)  # (batch, n_heads, seq, d_k)

        # 7. Merge heads e proiezione di output
        context = self.merge_heads(context)  # (batch, seq, d_model)
        out = self.out_proj(context)          # (batch, seq, d_model)
        out = self.proj_dropout(out)

        return out


# ===========================================================================
# 2. FEED-FORWARD NETWORK
# ===========================================================================

class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) applicata position-wise.

    Struttura: Linear → GELU → Linear → Dropout

    Ogni token viene elaborato indipendentemente dalla stessa FFN.
    La dimensione interna (hidden_dim) è tipicamente 4x la dimensione del modello:
    per GPT-2 base: d_model=768, ff_dim=3072.

    GELU vs ReLU:
    GELU (Gaussian Error Linear Unit) è una variante smooth della ReLU che
    dà risultati migliori nei Transformer. La differenza principale è che
    GELU non azzera bruscamente i valori negativi ma li "attenua" dolcemente.
    """

    def __init__(self, d_model, hidden_dim, dropout=0.1):
        """
        Args:
            d_model    : dimensione input/output
            hidden_dim : dimensione dello strato nascosto (tipicamente 4 * d_model)
            dropout    : probabilità di dropout dopo l'ultimo Linear
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),  # espansione
            nn.GELU(),                        # attivazione non lineare
            nn.Linear(hidden_dim, d_model),  # proiezione di ritorno
            nn.Dropout(dropout)              # regolarizzazione
        )

    def forward(self, x):
        """
        Args:
            x : (batch, seq_len, d_model)
        Returns:
            : (batch, seq_len, d_model)
        """
        return self.net(x)


# ===========================================================================
# 3. TRANSFORMER BLOCK (Pre-LayerNorm)
# ===========================================================================

class TransformerBlock(nn.Module):
    """
    Un singolo blocco Transformer con Pre-LayerNorm.

    SCHEMA (Pre-LN, usato da GPT-2):
        x → LN → MultiHeadAttention → + x  (residual connection)
          → LN → FeedForward         → + x  (residual connection)

    Le RESIDUAL CONNECTIONS (x + sublayer(x)) sono fondamentali:
    - Permettono il flusso del gradiente anche attraverso molti strati
    - Permettono al modello di imparare "modifiche" piuttosto che trasformazioni complete
    - Rendono possibile addestrare reti molto profonde (es. GPT-3 con 96 layer)

    PRE-LN vs POST-LN:
    - POST-LN (Transformer originale): LN applicata *dopo* il residual
    - PRE-LN (GPT-2): LN applicata *prima* del sublayer → più stabile,
      meno problemi di vanishing gradient, spesso converge più velocemente
    """

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        """
        Args:
            d_model  : dimensione del modello
            n_heads  : numero di teste di attenzione
            ff_dim   : dimensione nascosta della FFN
            dropout  : probabilità di dropout
        """
        super().__init__()

        # Layer normalization prima dell'attenzione
        self.ln1 = nn.LayerNorm(d_model)
        # Multi-Head Self-Attention
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

        # Layer normalization prima della FFN
        self.ln2 = nn.LayerNorm(d_model)
        # Feed-Forward Network
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x    : (batch, seq_len, d_model)
            mask : causal mask per l'attenzione
        Returns:
            x    : (batch, seq_len, d_model) — trasformato
        """
        # Pre-LN + Attention + Residual
        x = x + self.attn(self.ln1(x), mask)

        # Pre-LN + FFN + Residual
        x = x + self.ff(self.ln2(x))

        return x


# ===========================================================================
# 4. GPT-2 — MODELLO COMPLETO
# ===========================================================================

class GPT2(nn.Module):
    """
    Modello GPT-2 completo implementato da zero.

    COMPONENTI:
      - Token Embedding: mappa ogni token ID in un vettore d_model-dimensionale
      - Positional Embedding: mappa ogni posizione (0, 1, ..., seq_len-1) in
        un vettore d_model-dimensionale; appreso durante il training
      - Embedding Dropout: applicato dopo la somma token+pos embedding
      - N × TransformerBlock: stack di blocchi Transformer
      - Final LayerNorm: normalizzazione finale prima della proiezione
      - LM Head: proietta da d_model → vocab_size per ottenere i logit

    DIMENSIONI DEFAULT (GPT-2 base):
      vocab_size = 50.257 (tokenizer GPT-2)
      d_model    = 768
      n_heads    = 12
      num_layers = 12
      ff_dim     = 3072  (= 4 * d_model)
      pos_emb_size = 1024 (contesto massimo)

    Per gli esperimenti iniziali useremo dimensioni molto più piccole
    per permettere training rapido sulla CPU/MPS.
    """

    def __init__(self, vocab_size, pos_emb_size=1024, d_model=768,
                 n_heads=12, num_layers=12, ff_dim=3072,
                 dropout=0.1, weight_tying=False):
        """
        Args:
            vocab_size   : dimensione del vocabolario (50.257 per GPT-2)
            pos_emb_size : lunghezza massima della sequenza (context window)
            d_model      : dimensione degli embedding e degli hidden states
            n_heads      : numero di teste di attenzione per blocco
            num_layers   : numero di TransformerBlock impilati
            ff_dim       : dimensione nascosta della FFN (tipicamente 4*d_model)
            dropout      : probabilità di dropout (0.0 = nessun dropout)
            weight_tying : se True, condivide i pesi tra token_embed e lm_head
        """
        super().__init__()

        # ---------------------------------------------------------------
        # EMBEDDINGS
        # ---------------------------------------------------------------

        # Token embedding: (vocab_size, d_model)
        # Mappa ogni token ID in un vettore continuo d-dimensionale
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Positional embedding: (pos_emb_size, d_model)
        # Ogni posizione nella sequenza ha un suo vettore apprso (non sinusoidale)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        # Dropout applicato dopo la somma token + positional embedding
        self.emb_dropout = nn.Dropout(dropout)

        # ---------------------------------------------------------------
        # TRANSFORMER BLOCKS
        # ---------------------------------------------------------------

        # Stack di num_layers blocchi Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # ---------------------------------------------------------------
        # OUTPUT
        # ---------------------------------------------------------------

        # Layer Norm finale (dopo tutti i blocchi, prima della proiezione)
        self.ln_f = nn.LayerNorm(d_model)

        # LM Head: proietta da d_model → vocab_size
        # bias=False è standard per la LM head con weight tying
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ---------------------------------------------------------------
        # WEIGHT TYING
        # ---------------------------------------------------------------
        if weight_tying:
            # Condivide i pesi: lm_head usa la stessa matrice di token_embed
            # Intuizione: la matrice di embedding codifica "dove si trova un token
            # nello spazio semantico"; la LM head fa la stessa cosa al contrario.
            # Condividerle forza consistenza e riduce i parametri.
            self.lm_head.weight = self.token_embed.weight

        # ---------------------------------------------------------------
        # CAUSAL MASK (registrata come buffer, non è un parametro)
        # ---------------------------------------------------------------
        # Crea una matrice triangolare superiore di -inf (eccetto la diagonale)
        # Shape: (pos_emb_size, pos_emb_size) — poi slicata alla seq_len reale
        #
        # Esempio per seq_len=4:
        #   [0,   -inf, -inf, -inf]
        #   [0,    0,   -inf, -inf]
        #   [0,    0,    0,   -inf]
        #   [0,    0,    0,    0  ]
        #
        # La riga i può "guardare" solo le colonne j <= i (token precedenti)
        mask = torch.triu(
            torch.full((pos_emb_size, pos_emb_size), float('-inf')),
            diagonal=1  # 0 sulla diagonale principale (token può vedere se stesso)
        )
        # register_buffer: il tensore viene spostato su GPU con .to(device),
        # ma NON è considerato un parametro addestrabile
        self.register_buffer("mask", mask)

    def forward(self, input_ids):
        """
        Args:
            input_ids : (batch_size, seq_len) — sequenza di token ID

        Returns:
            logits : (batch_size, seq_len, vocab_size) — punteggi non normalizzati
                     Per ogni posizione nella sequenza, un punteggio per ogni token
                     del vocabolario. Si usa poi CrossEntropyLoss per calcolare la loss.
        """
        batch_size, seq_len = input_ids.shape

        # ---------------------------------------------------------------
        # 1. EMBEDDINGS
        # ---------------------------------------------------------------

        # Token embedding: converte ogni ID in un vettore
        tok_emb = self.token_embed(input_ids)  # (batch, seq_len, d_model)

        # Positional embedding: crea vettori di posizione [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.pos_embed(positions)    # (seq_len, d_model) — broadcastato

        # Somma token + positional embedding, poi dropout
        x = self.emb_dropout(tok_emb + pos_emb)  # (batch, seq_len, d_model)

        # ---------------------------------------------------------------
        # 2. CAUSAL MASK (sliced alla lunghezza corrente)
        # ---------------------------------------------------------------
        # Prendiamo solo il sotto-blocco (seq_len, seq_len) della mask completa
        # e aggiungiamo le dimensioni batch e heads per il broadcasting
        causal_mask = self.mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        # Shape: (1, 1, seq_len, seq_len) — viene broadcastato su (batch, n_heads, ...)

        # ---------------------------------------------------------------
        # 3. TRANSFORMER BLOCKS
        # ---------------------------------------------------------------
        for block in self.blocks:
            x = block(x, causal_mask)

        # ---------------------------------------------------------------
        # 4. OUTPUT
        # ---------------------------------------------------------------
        x = self.ln_f(x)           # Final LayerNorm: (batch, seq_len, d_model)
        logits = self.lm_head(x)   # LM Head: (batch, seq_len, vocab_size)

        return logits

    def count_parameters(self):
        """
        Conta il numero di parametri addestrabili del modello.
        Utile per monitorare la dimensione del modello durante gli esperimenti.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
