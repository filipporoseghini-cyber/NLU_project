"""
functions.py — Part 1.B: Training e valutazione con GPT2_LoRA (HuggingFace)
============================================================================

QUADRO GENERALE
---------------
Questo file implementa la logica di training e valutazione per il fine-tuning
di GPT-2 con LoRA. Rispetto alla Parte 1.A, ci sono differenze importanti
nel modo in cui viene calcolata la loss.

DIFFERENZA PRINCIPALE CON PARTE 1.A: LOSS INTERNA AL MODELLO
-------------------------------------------------------------
In Parte 1.A usavamo un criterion (CrossEntropyLoss) esterno:
    logits = model(input_ids)                  # (B, L, vocab)
    loss = criterion(logits.view(B*L, vocab), labels.view(B*L))

In Parte 1.B, GPT2LMHeadModel computa la loss internamente quando passiamo
'labels' al forward:
    output = model(input_ids, labels=labels)   # output.loss = scalar
    loss = output.loss

Il modello HuggingFace internamente:
  1. Calcola logits: (B, L, vocab)
  2. Shift: shift_logits = logits[..., :-1, :], shift_labels = labels[..., 1:]
  3. Loss = CrossEntropyLoss(shift_logits, shift_labels) con mean per token

GESTIONE DEL PADDING IN PARTE 1.B
-----------------------------------
In Parte 1.A usavamo criterion con ignore_index=tokenizer.pad_token_id.
In Parte 1.B non possiamo specificare ignore_index (la loss è interna), quindi
sostituiamo i token di padding con -100 nelle labels:
    labels[labels == tokenizer.pad_token_id] = -100
HuggingFace ignora automaticamente gli indici -100 nella loss.

PERCHÉ labels = input_ids.clone() E NON LE labels DAL DATALOADER?
------------------------------------------------------------------
La collate_fn restituisce (input_ids, labels, n_tokens) dove:
  - input_ids = token[0..T-2]  (senza l'ultimo token)
  - labels    = token[1..T-1]  (shifted di 1, usate in Parte 1.A)

In Parte 1.B, il training loop IGNORA le labels dal dataloader e ricrea:
  - labels = input_ids.clone()  (stessa sequenza di input)
  - Il modello fa internamente lo shift dei labels

Questo equivale a fare training su tutti i token di input (tranne il pad):
ogni token predice quello successivo nella stessa sequenza.

ACCUMULAZIONE DELLA LOSS
------------------------
output.loss è la CE loss MEDIA per token non-ignorato nel batch.
Per ottenere la loss totale del batch la moltiplichiamo per n_tokens,
poi dividiamo per la somma dei token sull'intera epoca (weighted average).
Questo è esattamente l'approccio del notebook (cell 41).

NOTA su n_tokens: c'è una leggera sovrastima perché n_tokens conta i token
non-pad in input_ids (che include tutti i token tranne il padding), mentre
il modello usa solo L-1 posizioni per sequenza dopo lo shift interno.
Questo disallineamento minore è presente anche nel notebook originale.

FREEZE E TRAINING SELETTIVO
-----------------------------
Solo le matrici LoRA sono addestrabili. La funzione freeze_pretrained_and_enable_lora()
(in questo file) congela tutti i parametri e poi riabilita solo i parametri
LoRA (lora_A_q/k/v e lora_B_q/k/v in ogni CustomGPT2Attention).
"""

import os
import math
import copy

import torch
from tqdm import tqdm

from model import CustomGPT2Attention


# ===========================================================================
# 1. FREEZE / UNFREEZE PARAMETRI
# ===========================================================================

def freeze_pretrained_and_enable_lora(model):
    """
    Congela tutti i parametri del modello, poi riabilita solo le matrici LoRA.

    Questo è il passo fondamentale per il PEFT (Parameter-Efficient Fine-Tuning):
    i 124M parametri di GPT-2 restano frozen, mentre solo i ~295K parametri
    LoRA (per rank=8) vengono addestrati.

    Da notebook cell 42 — adattato per usare isinstance(module, CustomGPT2Attention)
    invece della stringa del nome dell'attributo, per maggiore robustezza.

    Args:
        model : istanza di GPT2_LoRA con CustomGPT2Attention già iniettati
    """
    # Step 1: congela TUTTO — nessun parametro è addestrabile
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: riabilita SOLO le matrici LoRA in ogni CustomGPT2Attention
    # lora_A_q/k/v: down-projection (embed_dim → rank)
    # lora_B_q/k/v: up-projection (rank → embed_dim)
    lora_attrs = ['lora_A_q', 'lora_A_k', 'lora_A_v',
                  'lora_B_q', 'lora_B_k', 'lora_B_v']

    for module in model.modules():
        if isinstance(module, CustomGPT2Attention):
            for attr_name in lora_attrs:
                lora_module = getattr(module, attr_name)
                for param in lora_module.parameters():
                    param.requires_grad = True


# ===========================================================================
# 2. TRAINING LOOP
# ===========================================================================

def train_loop(data, optimizer, model, tokenizer, clip=5.0):
    """
    Esegue un'epoca di training sul modello GPT2_LoRA (HuggingFace).

    Da notebook cell 41 — modifiche rispetto all'originale:
      1. Aggiunto gradient clipping con torch.nn.utils.clip_grad_norm_()
         per maggiore stabilità del training (non presente nel notebook).
         Previene l'esplosione dei gradienti, importante anche con LoRA
         dato che i gradienti devono "propagarsi" attraverso il modello frozen.
      2. Usato n_tokens.item() per convertire il tensore in float Python
         (evita potenziali problemi con math.exp() e accumulo di grafi).
      3. Usata tqdm standard invece di tqdm.notebook (per esecuzione da script).

    ALGORITMO PER OGNI BATCH:
      1. Crea labels = input_ids.clone() con pad → -100
      2. Forward: output = model(input_ids, labels=labels)
      3. output.loss = CE loss media per token (interna al modello HuggingFace)
      4. Accumula loss * n_tokens (per calcolare la media pesata finale)
      5. Backward + gradient clipping + optimizer step

    Args:
        data      : DataLoader per il training
        optimizer : ottimizzatore (AdamW, solo sui parametri LoRA)
        model     : istanza di GPT2_LoRA
        tokenizer : tokenizer per identificare il pad token id
        clip      : soglia per gradient clipping (default 5.0)

    Returns:
        loss_per_token : float, perdita media per token sull'intera epoca
    """
    model.train()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="  train", leave=False)

    for i, (input_ids, _, n_tokens) in enumerate(pbar):
        # '_' ignora le labels dello shift dal dataloader: in Parte 1.B
        # le labels vengono ricreate internamente (vedi note nel docstring del modulo)

        optimizer.zero_grad()

        # Creiamo le labels come copia di input_ids.
        # Il modello HuggingFace farà internamente lo shift:
        #   shift_labels = labels[..., 1:]   → predice il token successivo
        labels = input_ids.clone().detach()

        # Sostituiamo i pad token con -100: HuggingFace ignora automaticamente
        # le posizioni con target=-100 nel calcolo della CrossEntropyLoss.
        # Questo equivale a usare ignore_index nella Parte 1.A.
        labels[labels == tokenizer.pad_token_id] = -100

        # Forward: il modello calcola logits E loss internamente
        output = model(input_ids, labels=labels)

        # output.loss è la CE loss MEDIA per token non-ignorato.
        # Moltiplichiamo per n_tokens per ottenere la loss TOTALE del batch.
        # Usiamo .item() per convertire sia output.loss che n_tokens in float Python
        # (evita accumulo di grafi computazionali nella lista loss_array).
        loss_array.append(output.loss.item() * n_tokens.item())
        number_of_tokens.append(n_tokens.item())

        # Backward: calcola i gradienti (solo per i parametri LoRA, che sono requires_grad=True)
        output.loss.backward()

        # Gradient clipping: previene l'esplosione dei gradienti.
        # AGGIUNTO rispetto al notebook per maggiore stabilità.
        # Anche se i pesi frozen non vengono aggiornati, i gradienti fluiscono
        # attraverso di essi durante il backward verso le matrici LoRA.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Aggiorna solo i parametri con requires_grad=True (le matrici LoRA)
        optimizer.step()

        # Aggiorna la progress bar ogni 100 batch con la loss corrente
        if i % 100 == 0:
            current_loss = sum(loss_array) / sum(number_of_tokens)
            pbar.set_postfix(loss=f"{current_loss:.4f}")

    # Loss media per token sull'intera epoca (weighted average per batch)
    return sum(loss_array) / sum(number_of_tokens)


# ===========================================================================
# 3. EVALUATION LOOP
# ===========================================================================

def eval_loop(data, model, tokenizer):
    """
    Valuta il modello su un dataset (dev o test) senza aggiornare i pesi.

    Da notebook cell 41 — modifiche rispetto all'originale:
      - Usato n_tokens.item() per robustezza (vedi train_loop)
      - Aggiunto cap a loss_avg per evitare overflow di math.exp() (come Parte 1.A)
      - Usata tqdm standard invece di tqdm.notebook

    Args:
        data      : DataLoader per dev o test
        model     : istanza di GPT2_LoRA
        tokenizer : tokenizer per il pad token id

    Returns:
        ppl      : float, perplexity sul dataset
        loss_avg : float, cross-entropy media per token
    """
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():  # disabilita il calcolo dei gradienti → risparmio memoria
        for input_ids, _, n_tokens in tqdm(data, desc="  eval", leave=False):
            labels = input_ids.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100

            output = model(input_ids, labels=labels)

            loss_array.append(output.loss.item() * n_tokens.item())
            number_of_tokens.append(n_tokens.item())

    # Cross-entropy media per token su tutto il dataset
    loss_avg = sum(loss_array) / sum(number_of_tokens)

    # Perplexity: PPL = exp(cross_entropy_per_token)
    # min con 100 per prevenire overflow aritmetico (e^100 ≈ 2.7×10^43 è già enorme)
    ppl = math.exp(min(loss_avg, 100))

    return ppl, loss_avg


# ===========================================================================
# 4. TRAINING CON EARLY STOPPING
# ===========================================================================

def train_model(model, train_loader, dev_loader, tokenizer, optimizer,
                n_epochs=100, patience=3, experiment_name="exp"):
    """
    Ciclo di training completo con early stopping.

    Struttura identica alla Parte 1.A. Monitoriamo la PPL sul dev set:
    se non migliora per 'patience' epoche consecutive, fermiamo il training
    e restituiamo il modello con la PPL migliore.

    NOTA: per modelli pre-addestrati come GPT-2, la convergenza è MOLTO
    più rapida rispetto al training from scratch. In genere 3-10 epoche
    sono sufficienti per un buon adattamento al dominio PTB.

    Args:
        model           : istanza di GPT2_LoRA con LoRA abilitata
        train_loader    : DataLoader per il training
        dev_loader      : DataLoader per la validation
        tokenizer       : tokenizer GPT-2
        optimizer       : AdamW configurato solo sui parametri LoRA
        n_epochs        : numero massimo di epoche
        patience        : epoche senza miglioramento prima di fermarsi
        experiment_name : nome dell'esperimento (per i log e la progress bar)

    Returns:
        best_model : modello con la migliore PPL di validazione (su CPU)
        best_ppl   : migliore PPL di validazione raggiunta
        history    : dizionario {'train_loss', 'dev_ppl', 'dev_loss'} per epoch
    """
    best_ppl = float('inf')
    best_model = None
    patience_counter = 0

    # Storico delle metriche per ogni epoca (utile per analisi e grafici)
    history = {
        'train_loss': [],
        'dev_ppl': [],
        'dev_loss': []
    }

    epoch_bar = tqdm(range(1, n_epochs + 1), desc=f"[{experiment_name}]", unit="ep")
    for epoch in epoch_bar:
        # -----------------------------------------------------------------------
        # TRAINING
        # -----------------------------------------------------------------------
        train_loss = train_loop(train_loader, optimizer, model, tokenizer)

        # -----------------------------------------------------------------------
        # EVALUATION SUL DEV SET
        # -----------------------------------------------------------------------
        dev_ppl, dev_loss = eval_loop(dev_loader, model, tokenizer)

        # Salva le metriche nello storico
        history['train_loss'].append(train_loss)
        history['dev_ppl'].append(dev_ppl)
        history['dev_loss'].append(dev_loss)

        epoch_bar.set_postfix(loss=f"{train_loss:.4f}", dev_ppl=f"{dev_ppl:.2f}")

        # -----------------------------------------------------------------------
        # EARLY STOPPING
        # -----------------------------------------------------------------------
        if dev_ppl < best_ppl:
            # Miglioramento trovato: aggiorna il best e salva copia del modello
            best_ppl = dev_ppl
            # deepcopy su CPU: evita di tenere due copie del modello in GPU
            best_model = copy.deepcopy(model).cpu()
            patience_counter = 0
            print(f"  ✓ Nuovo best dev PPL: {best_ppl:.2f} (epoch {epoch})")
        else:
            patience_counter += 1
            print(f"  · No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"  ✗ Early stopping dopo {epoch} epoche.")
                break

    return best_model, best_ppl, history


# ===========================================================================
# 5. SALVATAGGIO DEI RISULTATI
# ===========================================================================

def save_results_to_csv(results, csv_path="results_1B.csv"):
    """
    Salva i risultati degli esperimenti in un file CSV.

    Identico alla Parte 1.A. Crea il file se non esiste, aggiunge in fondo
    se esiste già (append mode) — utile per esperimenti eseguiti in sequenza.

    Args:
        results  : dizionario con i risultati dell'esperimento
        csv_path : percorso del file CSV
    """
    import csv
    fieldnames = list(results.keys())
    write_header = not os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(results)

    print(f"  Risultati salvati in {csv_path}")
