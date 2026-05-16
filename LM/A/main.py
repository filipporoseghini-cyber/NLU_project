"""
main.py — Part 1.A: Language Modeling con GPT-2 from scratch
=============================================================

SCOPO DI QUESTO SCRIPT
-----------------------
Questo script esegue una serie sistematica di esperimenti per trovare
la configurazione ottimale del modello GPT-2 sul dataset Penn TreeBank.

STRATEGIA DI RICERCA DEGLI IPERPARAMETRI
-----------------------------------------
L'ottimizzazione degli iperparametri segue un approccio a STEP PROGRESSIVI
(greedy / one-at-a-time), molto comune in NLP:

  STEP 0 — Learning Rate Search:
    Teniamo il modello piccolo (d_model=20, n_heads=1, layers=1) e proviamo
    diversi learning rate per trovare quello che converge meglio.
    → scegliamo il lr con la PPL di validazione più bassa

  STEP 1 — Architettura:
    Con il lr migliore, proviamo diverse combinazioni di:
    1.1 d_model: 64, 128, 256   (dimensione degli embedding)
    1.2 n_heads: 4              (con il miglior d_model)
    1.3 num_layers: 2, 4        (profondità del modello)
    1.4 ff_dim: 512, 1024       (dimensione della FFN)

  STEP 2 — Dropout:
    Con la migliore architettura, proviamo dropout 0.1 e 0.2

  STEP 3 — Weight Tying:
    Con la migliore configurazione, proviamo weight_tying=True

NOTA: ogni step parte dal miglior risultato dello step precedente (greedy search).
Questo non garantisce l'ottimo globale ma è pratico con risorse limitate.

TARGET: PPL < 250 sul test set.

COME ESEGUIRE
-------------
    python main.py

Oppure su VM con GPU in background (usando screen):
    screen -S lm_experiment
    conda activate nlu26
    python main.py
    # Ctrl+A, D per detacchare lo screen
"""

import os
import gc
import torch
import torch.nn as nn

from utils import read_file, get_dataloaders, get_tokenizer
from model import GPT2
from functions import (init_weights, train_model, eval_loop,
                       save_results_to_csv, plot_history)


# ===========================================================================
# 0. SETUP: DEVICE, PATHS, TOKENIZER, DATASET
# ===========================================================================

# -----------------------------------------------------------------------
# DEVICE DETECTION
# Priorità: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
# -----------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"[Device] CUDA disponibile: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
    print("[Device] Apple MPS disponibile (M1/M2 GPU)")
else:
    DEVICE = 'cpu'
    print("[Device] Nessuna GPU trovata, uso CPU")

print(f"[Device] Usando: {DEVICE}\n")

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
DATASET_DIR = os.path.join("dataset", "PennTreeBank")
BIN_DIR     = "bin"
os.makedirs(BIN_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATASET_DIR, "ptb.train.txt")
DEV_PATH   = os.path.join(DATASET_DIR, "ptb.valid.txt")
TEST_PATH  = os.path.join(DATASET_DIR, "ptb.test.txt")

CSV_PATH = "results_1A.csv"

# -----------------------------------------------------------------------
# TOKENIZER
# Usiamo il tokenizer BPE di GPT-2 (50.257 token nel vocabolario)
# -----------------------------------------------------------------------
print("[Setup] Caricamento tokenizer GPT-2...")
tokenizer = get_tokenizer()
VOCAB_SIZE = len(tokenizer)  # 50257
print(f"[Setup] Vocab size: {VOCAB_SIZE}\n")

# -----------------------------------------------------------------------
# DATASET
# Leggiamo i file raw e aggiungiamo <eos> a fine di ogni frase
# -----------------------------------------------------------------------
print("[Setup] Caricamento dataset Penn TreeBank...")
train_raw = read_file(TRAIN_PATH)
dev_raw   = read_file(DEV_PATH)
test_raw  = read_file(TEST_PATH)
print(f"[Setup] Train: {len(train_raw)} frasi | "
      f"Dev: {len(dev_raw)} frasi | "
      f"Test: {len(test_raw)} frasi\n")


# ===========================================================================
# 1. DEFINIZIONE DEGLI ESPERIMENTI
# ===========================================================================
#
# Ogni esperimento è un dizionario con:
#   - name          : identificativo univoco (usato per salvare il .pt e nel CSV)
#   - step          : a quale step di ottimizzazione appartiene
#   - lr            : learning rate per AdamW
#   - d_model       : dimensione degli embedding
#   - n_heads       : numero di teste di attenzione
#   - num_layers    : numero di TransformerBlock
#   - ff_dim        : dimensione nascosta FFN (tipicamente 4 * d_model)
#   - dropout       : probabilità di dropout
#   - weight_tying  : condivisione pesi embedding / lm_head
#   - batch_size    : numero di frasi per batch (più piccolo = meno memoria GPU)
#   - n_epochs      : epoche massime (l'early stopping ferma prima se necessario)
#   - patience      : epoche senza miglioramento prima di fermarsi
#
# ===========================================================================

experiments = [

    # -----------------------------------------------------------------------
    # STEP 0: RICERCA DEL LEARNING RATE
    # Modello molto piccolo per velocità: d_model=20, n_heads=1, layers=1
    # Proviamo lr: 0.1, 0.01, 0.001, 0.0001
    # -----------------------------------------------------------------------
    # Con AdamW, lr=0.1 è quasi sempre troppo grande per i Transformer
    # → ci aspettiamo che diverga o oscilli. lr=1e-3 o 1e-4 sono tipicamente i migliori.
    {
        "name": "step0_lr0.1",
        "step": 0,
        "lr": 0.1,
        "d_model": 20, "n_heads": 1, "num_layers": 1, "ff_dim": 20,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 50, "patience": 3
    },
    {
        "name": "step0_lr0.01",
        "step": 0,
        "lr": 0.01,
        "d_model": 20, "n_heads": 1, "num_layers": 1, "ff_dim": 20,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 50, "patience": 3
    },
    {
        "name": "step0_lr0.001",
        "step": 0,
        "lr": 0.001,
        "d_model": 20, "n_heads": 1, "num_layers": 1, "ff_dim": 20,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 50, "patience": 3
    },
    {
        "name": "step0_lr0.0001",
        "step": 0,
        "lr": 0.0001,
        "d_model": 20, "n_heads": 1, "num_layers": 1, "ff_dim": 20,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 50, "patience": 3
    },

    # -----------------------------------------------------------------------
    # STEP 1.1: d_model (con il lr migliore dallo step 0)
    # MODIFICA: inserire qui il lr migliore trovato nello step 0
    # -----------------------------------------------------------------------
    {
        "name": "step1_dmodel64",
        "step": "1.1",
        "lr": 0.001,      # <-- aggiornare con il best lr dallo step 0
        "d_model": 64, "n_heads": 1, "num_layers": 1, "ff_dim": 64,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },
    {
        "name": "step1_dmodel128",
        "step": "1.1",
        "lr": 0.001,
        "d_model": 128, "n_heads": 1, "num_layers": 1, "ff_dim": 128,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },
    {
        "name": "step1_dmodel256",
        "step": "1.1",
        "lr": 0.001,
        "d_model": 256, "n_heads": 1, "num_layers": 1, "ff_dim": 256,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },

    # -----------------------------------------------------------------------
    # STEP 1.2: n_heads (con il miglior d_model; deve essere divisore di d_model)
    # MODIFICA: aggiornare d_model con il best dallo step 1.1
    # -----------------------------------------------------------------------
    {
        "name": "step1_nheads4",
        "step": "1.2",
        "lr": 0.001,
        "d_model": 256,  # <-- aggiornare con il best d_model
        "n_heads": 4,
        "num_layers": 1, "ff_dim": 256,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },

    # -----------------------------------------------------------------------
    # STEP 1.3: num_layers (con il miglior d_model e n_heads)
    # -----------------------------------------------------------------------
    {
        "name": "step1_layers2",
        "step": "1.3",
        "lr": 0.001,
        "d_model": 256, "n_heads": 4, "num_layers": 2, "ff_dim": 256,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },
    {
        "name": "step1_layers4",
        "step": "1.3",
        "lr": 0.001,
        "d_model": 256, "n_heads": 4, "num_layers": 4, "ff_dim": 256,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },

    # -----------------------------------------------------------------------
    # STEP 1.4: ff_dim (con la migliore architettura finora)
    # -----------------------------------------------------------------------
    {
        "name": "step1_ffdim512",
        "step": "1.4",
        "lr": 0.001,
        "d_model": 256, "n_heads": 4, "num_layers": 4, "ff_dim": 512,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },
    {
        "name": "step1_ffdim1024",
        "step": "1.4",
        "lr": 0.001,
        "d_model": 256, "n_heads": 4, "num_layers": 4, "ff_dim": 1024,
        "dropout": 0.0, "weight_tying": False,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },

    # -----------------------------------------------------------------------
    # STEP 2: DROPOUT (con la migliore architettura dallo step 1)
    # MODIFICA: aggiornare i valori con i best da step 1
    # -----------------------------------------------------------------------
    {
        "name": "step2_dropout0.1",
        "step": 2,
        "lr": 0.001,
        "d_model": 256, "n_heads": 4, "num_layers": 4, "ff_dim": 1024,
        "dropout": 0.1, "weight_tying": False,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },
    {
        "name": "step2_dropout0.2",
        "step": 2,
        "lr": 0.001,
        "d_model": 256, "n_heads": 4, "num_layers": 4, "ff_dim": 1024,
        "dropout": 0.2, "weight_tying": False,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },

    # -----------------------------------------------------------------------
    # STEP 3: WEIGHT TYING (con la migliore configurazione dallo step 2)
    # MODIFICA: aggiornare dropout con il best da step 2
    # -----------------------------------------------------------------------
    {
        "name": "step3_weight_tying",
        "step": 3,
        "lr": 0.001,
        "d_model": 256, "n_heads": 4, "num_layers": 4, "ff_dim": 1024,
        "dropout": 0.1,  # <-- best da step 2
        "weight_tying": True,
        "batch_size": 32, "n_epochs": 100, "patience": 5
    },
]


# ===========================================================================
# 2. ESECUZIONE DEGLI ESPERIMENTI
# ===========================================================================

print("=" * 70)
print(f"Avvio di {len(experiments)} esperimenti su device: {DEVICE}")
print("=" * 70)

for i, exp in enumerate(experiments):
    name = exp["name"]
    print(f"\n{'='*70}")
    print(f"[{i+1}/{len(experiments)}] Esperimento: {name}")
    print(f"{'='*70}")
    print(f"  Configurazione: {exp}")

    # -----------------------------------------------------------------------
    # 2.1 DATALOADER
    # (li ricreaimo per ogni esperimento in caso cambi il batch_size)
    # -----------------------------------------------------------------------
    train_loader, dev_loader, test_loader = get_dataloaders(
        train_raw, dev_raw, test_raw, tokenizer, DEVICE,
        batch_size_train=exp["batch_size"],
        batch_size_eval=exp["batch_size"] * 2  # eval: nessun gradiente → più grande
    )

    # -----------------------------------------------------------------------
    # 2.2 MODELLO
    # -----------------------------------------------------------------------
    model = GPT2(
        vocab_size   = VOCAB_SIZE,
        pos_emb_size = 1024,              # contesto massimo
        d_model      = exp["d_model"],
        n_heads      = exp["n_heads"],
        num_layers   = exp["num_layers"],
        ff_dim       = exp["ff_dim"],
        dropout      = exp["dropout"],
        weight_tying = exp["weight_tying"]
    ).to(DEVICE)

    # Inizializzazione dei pesi con distribuzione uniforme piccola
    init_weights(model)

    n_params = model.count_parameters()
    print(f"  Parametri addestrabili: {n_params:,}")

    # -----------------------------------------------------------------------
    # 2.3 OTTIMIZZATORE E LOSS
    # -----------------------------------------------------------------------

    # AdamW: Adam con weight decay corretto (non applica decay ai bias e LayerNorm)
    # weight_decay=0.01 è una regolarizzazione aggiuntiva
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp["lr"],
        weight_decay=0.01
    )

    # CrossEntropyLoss con reduction='sum' per gestire correttamente il padding
    # ignore_index: ignora i token di padding nel calcolo della loss
    # (il pad token ha lo stesso ID dell'eos token nel tokenizer GPT-2)
    criterion_train = nn.CrossEntropyLoss(
        reduction='sum',
        ignore_index=tokenizer.pad_token_id
    )
    criterion_eval = nn.CrossEntropyLoss(
        reduction='sum',
        ignore_index=tokenizer.pad_token_id
    )

    # -----------------------------------------------------------------------
    # 2.4 TRAINING
    # -----------------------------------------------------------------------
    best_model, best_dev_ppl, history = train_model(
        model       = model,
        train_loader = train_loader,
        dev_loader   = dev_loader,
        optimizer    = optimizer,
        criterion_train = criterion_train,
        criterion_eval  = criterion_eval,
        n_epochs     = exp["n_epochs"],
        patience     = exp["patience"],
        experiment_name = name
    )

    # -----------------------------------------------------------------------
    # 2.5 VALUTAZIONE SUL TEST SET (con il miglior modello)
    # -----------------------------------------------------------------------
    # Spostiamo il miglior modello sul device corretto per la valutazione
    best_model = best_model.to(DEVICE)
    test_ppl, test_loss = eval_loop(test_loader, criterion_eval, best_model)

    print(f"\n  [RISULTATI FINALI] {name}")
    print(f"  Best Dev PPL : {best_dev_ppl:.2f}")
    print(f"  Test PPL     : {test_ppl:.2f}")
    print(f"  Test Loss    : {test_loss:.4f}")

    # -----------------------------------------------------------------------
    # 2.6 SALVATAGGIO
    # -----------------------------------------------------------------------

    # Salva il modello in bin/{name}.pt (su CPU per portabilità)
    model_path = os.path.join(BIN_DIR, f"{name}.pt")
    torch.save(best_model.cpu().state_dict(), model_path)
    print(f"  Modello salvato in: {model_path}")

    # Salva i risultati nel CSV
    results = {
        "experiment"  : name,
        "step"        : exp["step"],
        "lr"          : exp["lr"],
        "d_model"     : exp["d_model"],
        "n_heads"     : exp["n_heads"],
        "num_layers"  : exp["num_layers"],
        "ff_dim"      : exp["ff_dim"],
        "dropout"     : exp["dropout"],
        "weight_tying": exp["weight_tying"],
        "batch_size"  : exp["batch_size"],
        "n_params"    : n_params,
        "best_dev_ppl": round(best_dev_ppl, 4),
        "test_ppl"    : round(test_ppl, 4),
        "test_loss"   : round(test_loss, 4),
    }
    save_results_to_csv(results, CSV_PATH)

    # -----------------------------------------------------------------------
    # 2.7 PULIZIA MEMORIA GPU
    # Importante con GPU a memoria limitata: eliminare modello e loader
    # tra un esperimento e l'altro per evitare OOM (Out Of Memory)
    # -----------------------------------------------------------------------
    del model, best_model, optimizer, criterion_train, criterion_eval
    del train_loader, dev_loader, test_loader
    gc.collect()  # garbage collector Python

    if DEVICE == 'cuda':
        torch.cuda.empty_cache()  # svuota la cache della GPU NVIDIA
    elif DEVICE == 'mps':
        torch.mps.empty_cache()   # svuota la cache della GPU Apple Silicon


# ===========================================================================
# 3. RIEPILOGO FINALE
# ===========================================================================

print("\n" + "=" * 70)
print("TUTTI GLI ESPERIMENTI COMPLETATI")
print(f"Risultati salvati in: {CSV_PATH}")
print("=" * 70)

# Leggi e stampa la tabella dei risultati
import csv
print("\nRIEPILOGO:")
print(f"{'Experiment':<25} {'Step':<6} {'Dev PPL':>10} {'Test PPL':>10}")
print("-" * 55)
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    for row in rows:
        print(f"{row['experiment']:<25} {row['step']:<6} "
              f"{float(row['best_dev_ppl']):>10.2f} {float(row['test_ppl']):>10.2f}")

# Trova il miglior esperimento per test PPL
best_row = min(rows, key=lambda r: float(r['test_ppl']))
print(f"\nMIGLIOR ESPERIMENTO: {best_row['experiment']}")
print(f"  Test PPL: {best_row['test_ppl']}")
print(f"  Configurazione: d_model={best_row['d_model']}, "
      f"n_heads={best_row['n_heads']}, "
      f"layers={best_row['num_layers']}, "
      f"ff_dim={best_row['ff_dim']}, "
      f"dropout={best_row['dropout']}, "
      f"weight_tying={best_row['weight_tying']}")
