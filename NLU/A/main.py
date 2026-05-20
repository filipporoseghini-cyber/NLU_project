"""
main.py — Part 2.A: Esperimenti NLU from scratch (Intent + Slot Filling)
=========================================================================

STRUTTURA DEGLI ESPERIMENTI
----------------------------
Seguendo le indicazioni del notebook (cell 8e2fad3c), gli esperimenti sono
organizzati in tre step incrementali:

  STEP 0 — Baseline: trova il learning rate ottimale
    Iperparametri fissi: d_model=64, n_heads=2, num_layers=2, ff_dim=256, dropout=0.0
    Variabile: lr ∈ {0.1, 0.01, 0.001, 0.0001}

    Obiettivo: capire l'ordine di grandezza del lr giusto prima di variare
    la dimensione del modello. Con AdamW, lr=0.001 è spesso un buon punto di partenza
    per i Transformer NLU.

  STEP 1 — Hyperparameter optimization (incrementale)
    Partiamo dal miglior lr trovato nello Step 0 e modifichiamo UN iperparametro
    alla volta (d_model, n_heads, num_layers, ff_dim). Questo approccio "one change
    at a time" permette di capire l'effetto di ogni iperparametro isolatamente.

    Valori da provare:
      d_model    : 64 (baseline) → 128 → 256
      n_heads    : 2  (baseline) → 4   → 8
      num_layers : 2  (baseline) → 4   → 6
      ff_dim     : 256 (baseline) → 512 → 1024

    NOTA: con modelli più grandi potrebbe essere necessario riadeguare il lr.

  STEP 2 — Aggiunta del dropout
    Con il miglior modello trovato nello Step 1, proviamo ad aggiungere dropout
    prima delle teste di output (slot_out e intent_out):
      dropout : 0.0 (baseline) → 0.1 → 0.2 → 0.3

METRICHE E RISULTATI
--------------------
Per ogni esperimento:
  - 5 run indipendenti (init random diversa)
  - Slot F1: media ± std    (metrica principale)
  - Intent Acc: media ± std (metrica secondaria)
  - Risultati salvati in results_2A.csv

TARGET INDICATIVO (da letteratura NLU su ATIS):
  Slot F1:    > 0.90
  Intent Acc: > 0.95

DEVICE
------
Il codice rileva automaticamente CUDA > MPS > CPU.
Su GPU Tesla V100 (VM del corso), ogni run completa in ~2-5 minuti.
Con 5 run × 4 esperimenti Step 0 = ~40-100 minuti totali.
"""

import os
import torch

from utils import (
    load_data, create_dev_split, build_lang,
    IntentsAndSlots, get_dataloaders, PAD_TOKEN
)
from functions import run_experiments, save_results_to_csv


# ===========================================================================
# SETUP DEVICE
# ===========================================================================

def get_device():
    """Rileva il miglior dispositivo disponibile."""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")
    return device


# ===========================================================================
# CARICAMENTO DATI
# ===========================================================================

def setup_data(device):
    """
    Carica ATIS, crea il dev set, costruisce i vocabolari e i DataLoader.

    Restituisce tutto ciò che serve per gli esperimenti.
    """
    # Percorsi al dataset ATIS
    data_dir  = os.path.join('dataset', 'ATIS')
    train_path = os.path.join(data_dir, 'train.json')
    test_path  = os.path.join(data_dir, 'test.json')

    # Carica i dati raw
    tmp_train_raw = load_data(train_path)
    test_raw      = load_data(test_path)

    print(f"Training set originale: {len(tmp_train_raw)} esempi")
    print(f"Test set:               {len(test_raw)} esempi")

    # Crea il dev set con split stratificata (10% del training)
    train_raw, dev_raw = create_dev_split(tmp_train_raw, dev_size=0.10)
    print(f"Training (dopo split): {len(train_raw)} esempi")
    print(f"Dev set:               {len(dev_raw)} esempi")

    # Costruisce i vocabolari (parole da train, label da tutto il corpus)
    lang = build_lang(train_raw, dev_raw, test_raw, cutoff=0)
    print(f"Vocabolario: {len(lang.word2id)} parole "
          f"(include pad, unk, cls)")
    print(f"Slot labels: {len(lang.id2slot)} etichette")
    print(f"Intent:      {len(lang.intent2id)} classi")

    # Crea i Dataset e i DataLoader
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset   = IntentsAndSlots(dev_raw,   lang)
    test_dataset  = IntentsAndSlots(test_raw,  lang)

    train_loader, dev_loader, test_loader = get_dataloaders(
        train_dataset, dev_dataset, test_dataset,
        batch_size_train=128,
        batch_size_eval=64
    )

    # Dimensioni per la costruzione del modello
    vocab_len  = len(lang.word2id)
    slots_len  = len(lang.id2slot)   # pad e cls hanno stesso id → len conta i veri slot
    n_intents  = len(lang.intent2id)

    print(f"\nvocab_len={vocab_len}, slots_len={slots_len}, n_intents={n_intents}")

    return (train_loader, dev_loader, test_loader,
            lang, vocab_len, slots_len, n_intents)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':

    device = get_device()
    (train_loader, dev_loader, test_loader,
     lang, vocab_len, slots_len, n_intents) = setup_data(device)

    # Raccoglie tutti i risultati per la stampa finale
    all_results = []

    # =========================================================================
    # STEP 0 — BASELINE: cerca il learning rate ottimale
    # =========================================================================
    # Iperparametri fissi (modello piccolo per velocità):
    # d_model=64, n_heads=2, num_layers=2, ff_dim=256, dropout=0.0
    #
    # Perché questi valori?
    #   - d_model=64 è abbastanza piccolo da girare velocemente su CPU/MPS/GPU
    #   - n_heads=2 con d_model=64 → d_k=32 per testa (ragionevole)
    #   - num_layers=2 è il minimo utile per catturare dipendenze a corto raggio
    #   - ff_dim=256 = 4 * d_model (rapporto standard)
    #   - dropout=0.0 per ora: vogliamo vedere l'effetto del lr "puro"
    #
    # Learning rate da testare:
    #   - 0.1:    probabilmente troppo alto → instabilità
    #   - 0.01:   alto → convergenza rapida ma potenziale oscillazione
    #   - 0.001:  standard per AdamW sui Transformer
    #   - 0.0001: conservativo → convergenza lenta ma stabile

    step0_configs = [
        {"lr": 0.1,    "name": "step0_lr0.1"},
        {"lr": 0.01,   "name": "step0_lr0.01"},
        {"lr": 0.001,  "name": "step0_lr0.001"},
        {"lr": 0.0001, "name": "step0_lr0.0001"},
    ]

    # Iperparametri fissi per Step 0
    base_d_model    = 64
    base_n_heads    = 2
    base_num_layers = 2
    base_ff_dim     = 256
    base_dropout    = 0.0

    print("\n" + "="*70)
    print("STEP 0 — Ricerca del Learning Rate")
    print("="*70)

    for cfg in step0_configs:
        res = run_experiments(
            train_loader, dev_loader, test_loader,
            lang, vocab_len, slots_len, n_intents,
            lr=cfg["lr"],
            d_model=base_d_model,
            n_heads=base_n_heads,
            num_layers=base_num_layers,
            ff_dim=base_ff_dim,
            dropout=base_dropout,
            n_runs=5,
            n_epochs=200,
            patience=3,
            experiment_name=cfg["name"],
            device=device
        )
        save_results_to_csv(res)
        all_results.append(res)

    # Identifica il miglior lr dallo step 0
    best_step0 = max(all_results, key=lambda x: x['slot_f1_mean'])
    best_lr = best_step0['lr']
    print(f"\nMiglior lr da Step 0: {best_lr} "
          f"(Slot F1={best_step0['slot_f1_mean']:.3f})")

    # =========================================================================
    # STEP 1 — HYPERPARAMETER OPTIMIZATION (incrementale)
    # =========================================================================
    # Partiamo dalla configurazione baseline (d_model=64, n_heads=2, ...)
    # e modifichiamo UN parametro alla volta.
    #
    # ORDINE: prima aumentiamo d_model (il più impattante), poi n_heads,
    # poi num_layers, infine ff_dim.
    #
    # Con modelli più grandi il lr potrebbe dover essere ridotto:
    # se un esperimento non converge, proviamo lr/10.

    print("\n" + "="*70)
    print("STEP 1 — Hyperparameter Optimization (incrementale)")
    print("="*70)

    step1_results = []

    # --- 1a. d_model ---
    for d_model in [128, 256]:
        # Con d_model=128, n_heads=2 → d_k=64 (ragionevole)
        # Con d_model=256, n_heads=2 → d_k=128 (un po' grande per teste solo 2)
        # Potremmo dover aumentare n_heads per d_model=256
        res = run_experiments(
            train_loader, dev_loader, test_loader,
            lang, vocab_len, slots_len, n_intents,
            lr=best_lr,
            d_model=d_model,
            n_heads=base_n_heads,
            num_layers=base_num_layers,
            ff_dim=base_ff_dim,
            dropout=base_dropout,
            n_runs=5, n_epochs=200, patience=3,
            experiment_name=f"step1_dmodel{d_model}",
            device=device
        )
        save_results_to_csv(res)
        step1_results.append(res)
        all_results.append(res)

    # Aggiorna il miglior d_model
    all_so_far = [best_step0] + step1_results
    best_d_model_cfg = max(all_so_far, key=lambda x: x['slot_f1_mean'])
    best_d_model = best_d_model_cfg['d_model']
    print(f"\nMiglior d_model: {best_d_model} "
          f"(Slot F1={best_d_model_cfg['slot_f1_mean']:.3f})")

    # --- 1b. n_heads ---
    for n_heads in [4, 8]:
        # n_heads deve dividere d_model esattamente
        # Con d_model=64: n_heads=4 → d_k=16; n_heads=8 → d_k=8 (piccolo)
        # Con d_model=128: n_heads=4 → d_k=32; n_heads=8 → d_k=16
        # Con d_model=256: n_heads=4 → d_k=64; n_heads=8 → d_k=32
        if best_d_model % n_heads != 0:
            print(f"  Skipping n_heads={n_heads}: non divisibile per d_model={best_d_model}")
            continue
        res = run_experiments(
            train_loader, dev_loader, test_loader,
            lang, vocab_len, slots_len, n_intents,
            lr=best_lr,
            d_model=best_d_model,
            n_heads=n_heads,
            num_layers=base_num_layers,
            ff_dim=base_ff_dim,
            dropout=base_dropout,
            n_runs=5, n_epochs=200, patience=3,
            experiment_name=f"step1_nheads{n_heads}",
            device=device
        )
        save_results_to_csv(res)
        step1_results.append(res)
        all_results.append(res)

    all_so_far = [best_step0] + step1_results
    best_nheads_cfg = max(all_so_far, key=lambda x: x['slot_f1_mean'])
    best_n_heads = best_nheads_cfg['n_heads']
    print(f"\nMiglior n_heads: {best_n_heads} "
          f"(Slot F1={best_nheads_cfg['slot_f1_mean']:.3f})")

    # --- 1c. num_layers ---
    for num_layers in [4, 6]:
        # Più layer = più capacità espressiva ma più rischio di overfitting
        # su ATIS (corpus piccolo). num_layers=4 è spesso il punto ottimale.
        res = run_experiments(
            train_loader, dev_loader, test_loader,
            lang, vocab_len, slots_len, n_intents,
            lr=best_lr,
            d_model=best_d_model,
            n_heads=best_n_heads,
            num_layers=num_layers,
            ff_dim=base_ff_dim,
            dropout=base_dropout,
            n_runs=5, n_epochs=200, patience=3,
            experiment_name=f"step1_layers{num_layers}",
            device=device
        )
        save_results_to_csv(res)
        step1_results.append(res)
        all_results.append(res)

    all_so_far = [best_step0] + step1_results
    best_layers_cfg = max(all_so_far, key=lambda x: x['slot_f1_mean'])
    best_num_layers = best_layers_cfg['num_layers']
    print(f"\nMiglior num_layers: {best_num_layers} "
          f"(Slot F1={best_layers_cfg['slot_f1_mean']:.3f})")

    # --- 1d. ff_dim ---
    for ff_dim in [512, 1024]:
        # ff_dim = 4 * d_model è il rapporto standard.
        # Con d_model=64: ff_dim=256 (baseline), 512, 1024
        # Con d_model=128: ff_dim=512 è già il rapporto 4x standard
        res = run_experiments(
            train_loader, dev_loader, test_loader,
            lang, vocab_len, slots_len, n_intents,
            lr=best_lr,
            d_model=best_d_model,
            n_heads=best_n_heads,
            num_layers=best_num_layers,
            ff_dim=ff_dim,
            dropout=base_dropout,
            n_runs=5, n_epochs=200, patience=3,
            experiment_name=f"step1_ffdim{ff_dim}",
            device=device
        )
        save_results_to_csv(res)
        step1_results.append(res)
        all_results.append(res)

    all_so_far = [best_step0] + step1_results
    best_ffdim_cfg = max(all_so_far, key=lambda x: x['slot_f1_mean'])
    best_ff_dim = best_ffdim_cfg['ff_dim']
    print(f"\nMiglior ff_dim: {best_ff_dim} "
          f"(Slot F1={best_ffdim_cfg['slot_f1_mean']:.3f})")

    # Configurazione migliore complessiva da Step 1
    best_step1 = max(all_so_far, key=lambda x: x['slot_f1_mean'])
    print(f"\nMiglior configurazione Step 1:")
    print(f"  lr={best_step1['lr']}, d_model={best_step1['d_model']}, "
          f"n_heads={best_step1['n_heads']}, num_layers={best_step1['num_layers']}, "
          f"ff_dim={best_step1['ff_dim']}")
    print(f"  Slot F1={best_step1['slot_f1_mean']:.3f} +- {best_step1['slot_f1_std']:.3f}")

    # =========================================================================
    # STEP 2 — DROPOUT
    # =========================================================================
    # Aggiungiamo dropout con il modello ottimale trovato nello Step 1.
    # Il dropout agisce sugli attention weights, sull'output della FFN,
    # e sull'embedding (vedi MultiHeadAttention e FeedForward).
    #
    # Su corpus piccoli come ATIS:
    #   - dropout=0.1 è spesso il valore ottimale
    #   - dropout=0.3 può aiutare con modelli grandi (riduce overfitting)
    #   - dropout>0.5 di solito peggiora (troppa regolarizzazione)

    print("\n" + "="*70)
    print("STEP 2 — Aggiunta del Dropout")
    print("="*70)

    step2_configs = [
        {"dropout": 0.1, "name": "step2_dropout0.1"},
        {"dropout": 0.2, "name": "step2_dropout0.2"},
        {"dropout": 0.3, "name": "step2_dropout0.3"},
    ]

    for cfg in step2_configs:
        res = run_experiments(
            train_loader, dev_loader, test_loader,
            lang, vocab_len, slots_len, n_intents,
            lr=best_step1['lr'],
            d_model=best_step1['d_model'],
            n_heads=best_step1['n_heads'],
            num_layers=best_step1['num_layers'],
            ff_dim=best_step1['ff_dim'],
            dropout=cfg["dropout"],
            n_runs=5, n_epochs=200, patience=3,
            experiment_name=cfg["name"],
            device=device
        )
        save_results_to_csv(res)
        all_results.append(res)

    # =========================================================================
    # RIEPILOGO FINALE
    # =========================================================================

    print("\n" + "="*70)
    print("RIEPILOGO FINALE — Tutti gli esperimenti")
    print("="*70)
    print(f"{'Esperimento':<30} {'Slot F1':>10} {'±':>4} {'Intent Acc':>12} {'±':>4}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['experiment']:<30} "
              f"{r['slot_f1_mean']:>10.3f} "
              f"{r['slot_f1_std']:>4.3f} "
              f"{r['intent_acc_mean']:>12.3f} "
              f"{r['intent_acc_std']:>4.3f}")

    best_overall = max(all_results, key=lambda x: x['slot_f1_mean'])
    print(f"\nMigliore: {best_overall['experiment']}")
    print(f"  Slot F1    = {best_overall['slot_f1_mean']:.3f} +- {best_overall['slot_f1_std']:.3f}")
    print(f"  Intent Acc = {best_overall['intent_acc_mean']:.3f} +- {best_overall['intent_acc_std']:.3f}")
    print(f"  Config: lr={best_overall['lr']}, d_model={best_overall['d_model']}, "
          f"n_heads={best_overall['n_heads']}, num_layers={best_overall['num_layers']}, "
          f"ff_dim={best_overall['ff_dim']}, dropout={best_overall['dropout']}")
    print(f"\nRisultati salvati in results_2A.csv")
