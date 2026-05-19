"""
model.py — Part 1.B: GPT-2 con adattatori LoRA (Low-Rank Adaptation)
======================================================================

IDEA DI LORA (paper: Hu et al., 2022 — https://arxiv.org/pdf/2106.09685)
--------------------------------------------------------------------------
Fine-tuning di un modello grande richiede aggiornare tutti i suoi parametri,
il che è costoso in termini di memoria e calcolo. LoRA propone un'alternativa:
mantenere i pesi pre-addestrati CONGELATI e aggiungere matrici addestrabili
di rango basso che "adattano" la computazione:

    W_new(x) = W_frozen(x) + ΔW(x)
    ΔW = B @ A   dove  A ∈ R^{r × d_in},  B ∈ R^{d_out × r}

Il rango r è molto più piccolo della dimensione del modello (r << d_model).
Per GPT-2 base (d_model=768), r=8 → ΔW ha solo 2 × 768 × 8 = 12.288 parametri
per ogni matrice Q/K/V, contro 768 × 768 = 589.824 della matrice originale.

Il fattore di scaling alpha/r permette di regolare l'intensità dell'adattamento
senza dover riscalare il learning rate.

INIZIALIZZAZIONE CRUCIALE:
  A ~ N(0, 1/r)  →  introduce variazione casuale nella proiezione verso il basso
  B = 0          →  garantisce che ΔW = B @ A = 0 all'inizio del training

Con B=0, il modello parte IDENTICO al pre-addestrato (zero delta), e durante
il training impara gradualmente ad adattarsi al nuovo dominio (PTB).

STRUTTURA DELLE CLASSI
-----------------------
  CustomGPT2Attention(GPT2Attention)
      Estende l'attention di HuggingFace aggiungendo matrici LoRA per Q, K, V.
      Il forward è identico all'originale, con l'aggiunta dei delta LoRA
      sulle proiezioni Q, K, V nel ramo di self-attention.

  GPT2_LoRA(GPT2LMHeadModel)
      Modello GPT-2 completo che:
        1. Carica i pesi pre-addestrati via from_pretrained()
        2. Sostituisce ogni GPT2Attention con CustomGPT2Attention
        3. Espone lo stesso forward di GPT2LMHeadModel (nessuna modifica)
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


# ===========================================================================
# 1. CUSTOM ATTENTION CON LORA
# ===========================================================================

class CustomGPT2Attention(GPT2Attention):
    """
    GPT2Attention esteso con adattatori LoRA su Query, Key, Value.

    Eredita TUTTO da GPT2Attention (pesi, metodi, struttura) e aggiunge:
      - Sei nn.Linear senza bias: lora_A_q/k/v (d → r) e lora_B_q/k/v (r → d)
      - Un fattore di scaling: alpha / rank
      - Il forward modificato per applicare ΔQ, ΔK, ΔV

    I pesi pre-addestrati (c_attn, c_proj, ecc.) restano frozen.
    Solo le sei matrici LoRA sono addestrabili.
    """

    def __init__(self, config, rank, alpha):
        """
        Args:
            config : GPT2Config con hidden_size, num_attention_heads, ecc.
            rank   : rango r delle matrici LoRA (tipicamente 4, 8, 16)
            alpha  : fattore di scaling per il delta LoRA (scaling = alpha / rank)
        """
        # Inizializza tutte le componenti originali di GPT2Attention
        # (c_attn, c_proj, attn_dropout, resid_dropout, mask, ecc.)
        super().__init__(config)

        embed_dim = config.hidden_size  # 768 per GPT-2 base

        self.rank = rank
        # scaling = alpha / rank è il fattore moltiplicativo del delta LoRA.
        # Permette di controllare l'intensità dell'adattamento separatamente dal lr.
        # (Con alpha=rank si ottiene scaling=1; con alpha=2*rank si raddoppia.)
        self.scaling = alpha / rank

        # -----------------------------------------------------------------------
        # MATRICI LORA PER QUERY
        # -----------------------------------------------------------------------
        # lora_A_q: proietta da embed_dim → rank (down-projection)
        # lora_B_q: proietta da rank → embed_dim (up-projection)
        # Il prodotto lora_B_q(lora_A_q(x)) ha la stessa forma di c_attn(x)/3
        self.lora_A_q = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B_q = nn.Linear(rank, embed_dim, bias=False)

        # -----------------------------------------------------------------------
        # MATRICI LORA PER KEY
        # -----------------------------------------------------------------------
        self.lora_A_k = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B_k = nn.Linear(rank, embed_dim, bias=False)

        # -----------------------------------------------------------------------
        # MATRICI LORA PER VALUE
        # -----------------------------------------------------------------------
        self.lora_A_v = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B_v = nn.Linear(rank, embed_dim, bias=False)

        # -----------------------------------------------------------------------
        # INIZIALIZZAZIONE LORA (dal paper: sezione 4.1)
        # -----------------------------------------------------------------------
        # A ~ N(0, 1): down-projection con valori casuali
        #   → garantisce che il gradiente scorra sin dall'inizio del training
        # B = 0:      up-projection azzerata
        #   → ΔW = B @ A = 0 → il modello parte IDENTICO al pre-addestrato
        #   → fondamentale per preservare le capacità acquisite durante pre-training
        for lora_A in [self.lora_A_q, self.lora_A_k, self.lora_A_v]:
            nn.init.normal_(lora_A.weight)
        for lora_B in [self.lora_B_q, self.lora_B_k, self.lora_B_v]:
            nn.init.zeros_(lora_B.weight)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """
        Forward preso dal notebook (cell 37) — identico all'implementazione
        originale di GPT2Attention in transformers==4.38.0, con la sola
        aggiunta dei delta LoRA su query, key, value.

        MODIFICHE RISPETTO AL NOTEBOOK (cell 37):
          Nel ramo else (self-attention), dopo lo split di c_attn aggiungiamo:
            query += lora_B_q(lora_A_q(hidden_states)) * scaling
            key   += lora_B_k(lora_A_k(hidden_states)) * scaling
            value += lora_B_v(lora_A_v(hidden_states)) * scaling
          Il resto è IDENTICO all'originale.

        Args:
            hidden_states           : (B, L, embed_dim) — input del blocco
            layer_past              : tuple (key, value) per KV-cache (inference)
            attention_mask          : maschera di attenzione HuggingFace
            head_mask               : maschera per singole teste
            encoder_hidden_states   : per cross-attention (non usata in decoder-only)
            encoder_attention_mask  : maschera encoder (non usata)
            use_cache               : True → restituisce (key, value) per caching
            output_attentions       : True → include attention weights nell'output

        Returns:
            outputs : (attn_output, present, [attn_weights])
        """
        if encoder_hidden_states is not None:
            # --- CROSS-ATTENTION ---
            # Questo ramo NON viene usato nel Language Modeling decoder-only.
            # Lo manteniamo invariato (identico all'originale) per completezza.
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # --- SELF-ATTENTION (ramo normale per il LM) ---
            # c_attn è una Conv1D frozen (pesi pre-addestrati) che proietta
            # hidden_states → 3 * embed_dim, poi splittiamo in Q, K, V.
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

            # --- LORA DELTA ---
            # Calcoliamo il delta per Q, K, V usando le matrici addestrabili:
            #   delta_q = lora_B_q(lora_A_q(hidden_states)) * (alpha / rank)
            # Shape: (B, L, embed_dim) → identica a query/key/value
            # Questo somma un "aggiustamento" apprendo alle proiezioni frozen.
            query = query + self.lora_B_q(self.lora_A_q(hidden_states)) * self.scaling
            key   = key   + self.lora_B_k(self.lora_A_k(hidden_states)) * self.scaling
            value = value + self.lora_B_v(self.lora_A_v(hidden_states)) * self.scaling

        # _split_heads: (B, L, embed_dim) → (B, num_heads, L, head_dim)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key   = self._split_heads(key,   self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # KV-cache: concatena i token passati (usato durante generazione/inference)
        if layer_past is not None:
            past_key, past_value = layer_past
            key   = torch.cat((past_key,   key),   dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # Calcolo dell'attention (standard o con upcast per precisione)
        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )

        # _merge_heads: (B, num_heads, L, head_dim) → (B, L, embed_dim)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        # Proiezione di output (frozen) + dropout residuale
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # (attn_output, present, [attn_weights])


# ===========================================================================
# 2. GPT-2 CON LORA — MODELLO COMPLETO
# ===========================================================================

class GPT2_LoRA(GPT2LMHeadModel):
    """
    GPT-2 pre-addestrato con adattatori LoRA iniettati in tutti i blocchi.

    Eredita da GPT2LMHeadModel (HuggingFace) e sostituisce ogni GPT2Attention
    con CustomGPT2Attention durante __init__. Tutti gli altri componenti
    (embedding, LayerNorm, LM head, ecc.) rimangono invariati.

    FLUSSO DI from_pretrained:
      1. Viene chiamato __init__ (random weights)
      2. __init__ sostituisce i blocchi di attenzione con CustomGPT2Attention
      3. from_pretrained carica i pesi pre-addestrati sull'intero modello
         → i pesi di c_attn, c_proj, ecc. vengono sovrascritti con i valori GPT-2
         → i pesi LoRA (lora_A_q, ecc.) non sono nel checkpoint → mantengono
           l'inizializzazione (A~N(0,1), B=0) definita in CustomGPT2Attention
      4. Solo i pesi LoRA vengono resi addestrabili (vedi freeze in functions.py)
    """

    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        """
        Args:
            *model_args   : argomenti posizionali per GPT2LMHeadModel (tipicamente: config)
            rank          : rango r delle matrici LoRA
            alpha         : fattore di scaling (scaling = alpha / rank)
            **model_kwargs: altri kwargs per GPT2LMHeadModel (es. torch_dtype)
                            rank e alpha vengono estratti PRIMA di chiamare super(),
                            così non vengono passati a GPT2LMHeadModel (che li rifiuterebbe)
        """
        # Inizializza GPT2LMHeadModel completo (senza rank/alpha che sono solo per LoRA)
        super().__init__(*model_args, **model_kwargs)

        # Sostituiamo ogni blocco di attenzione con la versione LoRA.
        # self.transformer.h è la lista dei GPT2Block (12 per GPT-2 base).
        for block in self.transformer.h:
            old_attn = block.attn  # GPT2Attention originale (pesi random a questo punto)

            # Creiamo la nuova attention con LoRA usando la config del modello
            new_attn = CustomGPT2Attention(self.config, rank=rank, alpha=alpha)

            # Copiamo i pesi dal vecchio al nuovo modulo.
            # strict=False: ignora le chiavi mancanti (le matrici LoRA non sono
            # nel state_dict di old_attn, vengono mantenute con l'inizializzazione).
            # NOTA: a questo punto old_attn ha pesi random (from_pretrained non ha
            # ancora caricato i checkpoint). I pesi veri arriveranno nel passo 3
            # del flusso from_pretrained, che sovrascrive c_attn, c_proj, ecc.
            # tramite model.load_state_dict() sull'intero modello.
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)

            # Sostituiamo il modulo nel blocco corrente
            block.attn = new_attn

    def forward(self, *args, **kwargs):
        """
        Forward identico a GPT2LMHeadModel — passthrough al genitore.

        Il notebook (cell 38) lo definisce esplicitamente per chiarezza.
        GPT2LMHeadModel gestisce internamente:
          - Il calcolo dei logits
          - Lo shift dei labels (shift_logits = logits[..., :-1, :])
          - Il calcolo della loss CrossEntropy quando labels è fornito
        """
        return super().forward(*args, **kwargs)


# ===========================================================================
# 3. UTILITÀ
# ===========================================================================

def param_stats(model):
    """
    Stampa le statistiche dei parametri del modello.

    Da notebook cell 36 — IDENTICO.
    Utile per verificare che solo le matrici LoRA siano addestrabili dopo
    il freeze. Per GPT-2 base con rank=8:
      - Total: ~124M parametri
      - Trainable: ~295K parametri LoRA (0.24% del totale)
      - Frozen:   ~124M parametri pre-addestrati

    Args:
        model : istanza di GPT2_LoRA (o qualsiasi nn.Module)
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params:     {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"frozen params:    {total - trainable:,}")
    print(f"trainable ratio:  {100 * trainable / total:.4f}%")
