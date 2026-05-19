"""
functions.py — Part 1.A: Training, Evaluation, Early Stopping
==============================================================

QUADRO GENERALE
---------------
Questo file contiene la logica di addestramento e valutazione del modello GPT-2.

LOSS FUNCTION: CrossEntropyLoss
--------------------------------
La CrossEntropyLoss calcola:
    loss = -log P(label | context)
Per un batch con N token non-padding, vogliamo la perdita MEDIA per token.

Usiamo reduction='sum' nella loss e dividiamo manualmente per il numero di
token non-padding, invece di usare reduction='mean'. Questo perché con il
padding, reduction='mean' dividerebbe per tutti i token (inclusi i pad),
sottostimando la perdita reale.

PERPLEXITY
----------
    PPL = exp(cross_entropy_per_token)
Se la loss media per token è H, allora PPL = e^H.
Target: PPL < 250 sul test set (per questa parte).

GRADIENT CLIPPING
-----------------
Durante il backprop, i gradienti possono esplodere (diventare enormi),
specialmente con reti profonde. clip_grad_norm_ tronca il vettore dei
gradienti se la sua norma L2 supera una soglia (clip=5.0).
Questo stabilizza il training senza eliminare completamente il segnale.

EARLY STOPPING
--------------
Fermiamo il training se la PPL di validazione non migliora per `patience`
epoche consecutive. Questo evita overfitting e spreco di risorse GPU.
Salviamo sempre il modello con la migliore PPL di validazione.
"""

import os
import math
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===========================================================================
# 1. INIZIALIZZAZIONE PESI
# ===========================================================================

def init_weights(model):
    """
    Inizializza i pesi di tutti i layer Linear con una distribuzione uniforme
    piccola: U(-0.01, 0.01).

    Perché è importante? I pesi iniziali influenzano la velocità di convergenza.
    Con pesi troppo grandi, i gradienti esplodono; con pesi troppo piccoli,
    scompaiono. Una distribuzione uniforme piccola è una scelta sicura per
    modelli Transformer piccoli.

    I bias vengono inizializzati a zero (default di PyTorch, ma lo facciamo
    esplicito per chiarezza).

    Args:
        model : istanza di nn.Module da inizializzare in-place
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


# ===========================================================================
# 2. TRAINING LOOP
# ===========================================================================

def train_loop(data, optimizer, criterion, model, clip=5.0):
    """
    Esegue un'epoca di training su tutto il dataset.

    ALGORITMO PER OGNI BATCH:
      1. Forward pass → logits (batch, seq_len, vocab_size)
      2. Reshape per CrossEntropyLoss: (batch*seq_len, vocab_size) e (batch*seq_len,)
      3. Calcola loss SUM (non mean, per gestire correttamente il padding)
      4. Backward pass → calcola i gradienti
      5. Gradient clipping → tronca la norma del gradiente se > clip
      6. Optimizer step → aggiorna i pesi
      7. Zero grad → azzera i gradienti per il batch successivo

    Args:
        data      : DataLoader per il training
        optimizer : ottimizzatore (AdamW)
        criterion : CrossEntropyLoss con reduction='sum'
        model     : istanza di GPT2
        clip      : soglia per gradient clipping (default 5.0)

    Returns:
        loss_per_token : float, perdita media per token sull'intera epoca
    """
    model.train()  # mette il modello in modalità training (abilita dropout)

    loss_array = []          # perdita per ogni batch
    number_of_tokens = []   # token non-padding per ogni batch

    for input_ids, labels, n_tokens in tqdm(data, desc="  train", leave=False):
        # -----------------------------------------------
        # FORWARD PASS
        # -----------------------------------------------
        optimizer.zero_grad()  # azzera i gradienti accumulati dal batch precedente

        # Forward pass del modello: otteniamo i logit
        logits = model(input_ids)  # (batch, seq_len, vocab_size)

        # CrossEntropyLoss si aspetta:
        #   - input: (N, C) dove N = campioni, C = classi (vocab_size)
        #   - target: (N,) con i class indici
        # Quindi facciamo reshape: (batch * seq_len, vocab_size) e (batch * seq_len,)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat  = logits.view(batch_size * seq_len, vocab_size)
        labels_flat  = labels.view(batch_size * seq_len)

        # Calcola la loss con reduction='sum' (somma su tutti i token del batch)
        loss = criterion(logits_flat, labels_flat)

        # -----------------------------------------------
        # BACKWARD PASS
        # -----------------------------------------------
        loss.backward()  # calcola i gradienti con la backpropagation

        # Gradient clipping: normalizza il gradiente se la norma > clip
        # Questo previene l'esplosione dei gradienti con reti profonde
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Aggiorna i parametri del modello
        optimizer.step()

        # -----------------------------------------------
        # TRACCIAMENTO
        # -----------------------------------------------
        # Salviamo la loss e il numero di token per calcolare la media alla fine
        loss_array.append(loss.item())
        number_of_tokens.append(n_tokens.item())

    # Loss media per token: somma di tutte le loss / somma di tutti i token
    # (questo è corretto anche con batch di lunghezze diverse)
    return sum(loss_array) / sum(number_of_tokens)


# ===========================================================================
# 3. EVALUATION LOOP
# ===========================================================================

def eval_loop(data, criterion, model):
    """
    Valuta il modello su un dataset (dev o test) senza aggiornare i pesi.

    Differenze rispetto a train_loop:
      - model.eval(): disabilita dropout (ogni neurone è sempre attivo)
      - torch.no_grad(): non calcola i gradienti → risparmio di memoria e velocità

    Args:
        data      : DataLoader per dev o test
        criterion : CrossEntropyLoss con reduction='sum'
        model     : istanza di GPT2

    Returns:
        ppl      : float, perplexity media sul dataset
        loss_avg : float, cross-entropy media per token
    """
    model.eval()  # disabilita dropout e batchnorm

    loss_array = []
    number_of_tokens = []

    with torch.no_grad():  # nessun calcolo del gradiente
        for input_ids, labels, n_tokens in data:
            logits = model(input_ids)

            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            labels_flat = labels.view(batch_size * seq_len)

            loss = criterion(logits_flat, labels_flat)

            loss_array.append(loss.item())
            number_of_tokens.append(n_tokens.item())

    # Cross-entropy media per token
    loss_avg = sum(loss_array) / sum(number_of_tokens)

    # Perplexity: PPL = exp(cross_entropy_per_token)
    # Se loss_avg è molto grande, exp può andare in overflow → usiamo min con 1e9
    ppl = math.exp(min(loss_avg, 100))  # cap a e^100 per evitare overflow

    return ppl, loss_avg


# ===========================================================================
# 4. TRAINING CON EARLY STOPPING
# ===========================================================================

def train_model(model, train_loader, dev_loader, optimizer,
                criterion_train, criterion_eval,
                n_epochs=100, patience=3, experiment_name="exp"):
    """
    Ciclo di training completo con early stopping.

    EARLY STOPPING
    --------------
    Monitoriamo la PPL sul dev set. Se non migliora per `patience` epoche
    consecutive, fermiamo il training e restituiamo il modello con la PPL
    migliore. Questo evita overfitting e non spreca GPU time.

    Algoritmo:
      - best_ppl = inf (inizialmente)
      - patience_counter = 0
      - Se dev_ppl < best_ppl: aggiorna best_ppl, salva modello, reset counter
      - Altrimenti: incrementa counter
      - Se counter >= patience: stop

    Il modello migliore viene salvato su CPU per risparmiare memoria GPU
    durante il training (deepcopy + .cpu()).

    Args:
        model           : istanza di GPT2
        train_loader    : DataLoader per training
        dev_loader      : DataLoader per validazione
        optimizer       : ottimizzatore AdamW
        criterion_train : loss per training (reduction='sum', ignora padding)
        criterion_eval  : loss per evaluation (reduction='sum', ignora padding)
        n_epochs        : numero massimo di epoche
        patience        : numero di epoche senza miglioramento prima di fermarsi
        experiment_name : nome dell'esperimento (per i log)

    Returns:
        best_model : il modello con la migliore PPL di validazione (su CPU)
        best_ppl   : la migliore PPL di validazione raggiunta
        history    : dizionario con le metriche per ogni epoca
    """
    # Stato per early stopping
    best_ppl = float('inf')
    best_model = None
    patience_counter = 0

    # Storico delle metriche per ogni epoca (utile per i grafici)
    history = {
        'train_loss': [],
        'dev_ppl': [],
        'dev_loss': []
    }

    epoch_bar = tqdm(range(1, n_epochs + 1), desc=f"[{experiment_name}]", unit="ep")
    for epoch in epoch_bar:
        # -----------------------------------------------
        # TRAINING
        # -----------------------------------------------
        train_loss = train_loop(train_loader, optimizer, criterion_train, model)

        # -----------------------------------------------
        # EVALUATION SUL DEV SET
        # -----------------------------------------------
        dev_ppl, dev_loss = eval_loop(dev_loader, criterion_eval, model)

        # Salva le metriche nello storico
        history['train_loss'].append(train_loss)
        history['dev_ppl'].append(dev_ppl)
        history['dev_loss'].append(dev_loss)

        epoch_bar.set_postfix(loss=f"{train_loss:.4f}", dev_ppl=f"{dev_ppl:.2f}")

        # -----------------------------------------------
        # EARLY STOPPING
        # -----------------------------------------------
        if dev_ppl < best_ppl:
            # Miglioramento! Aggiorna il migliore e salva il modello
            best_ppl = dev_ppl
            # Salva su CPU: evita di tenere due copie del modello sulla GPU
            best_model = copy.deepcopy(model).cpu()
            patience_counter = 0
            print(f"  ✓ Nuovo best dev PPL: {best_ppl:.2f}")
        else:
            # Nessun miglioramento
            patience_counter += 1
            print(f"  · No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"  ✗ Early stopping dopo {epoch} epoche.")
                break

    return best_model, best_ppl, history


# ===========================================================================
# 5. SALVATAGGIO DEI RISULTATI
# ===========================================================================

def save_results_to_csv(results, csv_path="results_1A.csv"):
    """
    Salva i risultati degli esperimenti in un file CSV.
    Crea il file se non esiste, altrimenti aggiunge in fondo (append).

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

    print(f"Risultati salvati in {csv_path}")


# ===========================================================================
# 6. PLOTTING
# ===========================================================================

def plot_history(history, experiment_name="exp", save_path=None):
    """
    Visualizza le curve di training loss e dev PPL nel tempo.

    Args:
        history         : dizionario con 'train_loss', 'dev_ppl', 'dev_loss'
        experiment_name : titolo del grafico
        save_path       : se specificato, salva il grafico in questo percorso
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title(f'{experiment_name} — Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['dev_ppl'], 'r-o', markersize=3, label='Dev PPL')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title(f'{experiment_name} — Dev Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
