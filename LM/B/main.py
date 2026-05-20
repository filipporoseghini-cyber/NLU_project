"""
main.py — Part 1.B: Fine-tuning GPT-2 con LoRA sul Penn TreeBank
=================================================================

SCOPO DI QUESTO SCRIPT
-----------------------
Fine-tuning del modello GPT-2 pre-addestrato (da HuggingFace) sul corpus
Penn TreeBank usando LoRA. Solo le matrici LoRA vengono addestrate (~0.2%
dei parametri totali), mentre i 124M pesi originali restano frozen.

STRATEGIA DEGLI ESPERIMENTI
-----------------------------
Gli esperimenti sono organizzati in step progressivi (greedy search),
seguendo lo stesso schema della Parte 1.A:

  STEP 0 — Learning Rate Search (rank=4, alpha=8):
    Con LoRA e modelli pre-addestrati, il lr ottimale è molto più basso
    rispetto al training from scratch. Valori tipici: 1e-4 ÷ 1e-3.
    Rank piccolo (r=4) per velocizzare la ricerca del lr.
    → Troviamo il lr con la PPL di validazione più bassa.

  STEP 1 — Rank Search (miglior lr da step 0):
    Il rank r controlla la capacità di adattamento di LoRA:
    - r piccolo (es. 4): pochi parametri, meno overfitting, meno espressività
    - r grande (es. 16): più parametri, più flessibilità, rischio overfitting
    Testiamo: r = 4, 8, 16
    → Troviamo il rank con la PPL migliore.

  STEP 2 — Alpha Search (miglior rank e lr da step 1):
    alpha controlla lo scaling: scaling = alpha / rank.
    - alpha = rank     → scaling = 1.0 (effetto unitario)
    - alpha = 2*rank   → scaling = 2.0 (effetto amplificato)
    - alpha = rank//2  → scaling = 0.5 (effetto attenuato)
    → Troviamo l'alpha ottimale.

TARGET: PPL < 250 (obbligatorio) e migliore della Parte 1.A.

COME ESEGUIRE
-------------
    conda activate nlu26
    cd LM/B
    python main.py

Per eseguire in background su server con screen:
    screen -S lm_1B
    conda activate nlu26
    python main.py
    # Ctrl+A, D per detacchare

SELEZIONARE GLI ESPERIMENTI
----------------------------
Decommentare UN SOLO step alla volta. Dopo aver visto i risultati,
scegliere i migliori iperparametri e passare allo step successivo.
"""

import os
import gc
import csv

import torch

from utils import read_file, get_dataloaders, get_tokenizer
from model import GPT2_LoRA, CustomGPT2Attention, param_stats
from functions import (freeze_pretrained_and_enable_lora,
                       train_model, eval_loop, save_results_to_csv)


# ===========================================================================
# 0. SETUP: DEVICE, PATHS, TOKENIZER, DATASET
# ===========================================================================

# -----------------------------------------------------------------------
# DEVICE DETECTION
# Priorità: CUDA (GPU NVIDIA) > MPS (Apple Silicon) > CPU
# -----------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"[Device] CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
    print("[Device] Apple MPS (M1/M2)")
else:
    DEVICE = 'cpu'
    print("[Device] CPU (nessuna GPU trovata)")

print(f"[Device] Usando: {DEVICE}\n")

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
DATASET_DIR = os.path.join("dataset", "PennTreeBank")
BIN_DIR = "bin"
os.makedirs(BIN_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATASET_DIR, "ptb.train.txt")
DEV_PATH   = os.path.join(DATASET_DIR, "ptb.valid.txt")
TEST_PATH  = os.path.join(DATASET_DIR, "ptb.test.txt")
CSV_PATH   = "results_1B.csv"

# -----------------------------------------------------------------------
# TOKENIZER
# Usiamo AutoTokenizer con il modello ufficiale su HuggingFace Hub.
# Il tokenizer è identico a quello di GPT-2 (BPE, 50.257 token nel vocabolario).
# -----------------------------------------------------------------------
print("[Setup] Caricamento tokenizer GPT-2...")
tokenizer = get_tokenizer()
print(f"[Setup] Vocab size: {len(tokenizer)}\n")

# -----------------------------------------------------------------------
# DATASET
# Leggiamo i file raw: ogni riga è una frase, aggiungiamo <eos> alla fine.
# -----------------------------------------------------------------------
print("[Setup] Caricamento Penn TreeBank...")
train_raw = read_file(TRAIN_PATH)
dev_raw   = read_file(DEV_PATH)
test_raw  = read_file(TEST_PATH)
print(f"[Setup] Train: {len(train_raw)} | Dev: {len(dev_raw)} | Test: {len(test_raw)}\n")


# ===========================================================================
# 1. DEFINIZIONE DEGLI ESPERIMENTI
# ===========================================================================
#
# Ogni esperimento è un dizionario con:
#   - name       : identificativo univoco
#   - step       : a quale step appartiene (per documentazione)
#   - rank       : rango r delle matrici LoRA
#   - alpha      : fattore di scaling (scaling = alpha / rank)
#   - lr         : learning rate per AdamW
#   - batch_size : frasi per batch (più piccolo → meno memoria GPU)
#   - n_epochs   : epoche massime (l'early stopping può fermare prima)
#   - patience   : epoche senza miglioramento prima di fermarsi
#
# ===========================================================================

experiments = [

    # -----------------------------------------------------------------------
    # STEP 0: LEARNING RATE SEARCH (rank=4, alpha=8 fissi)
    # Modello: GPT-2 base (124M param) + LoRA rank=4 (~295K param trainabili)
    # Selezioniamo il lr con la PPL di validazione più bassa.
    #
    # Valori tipici per LoRA con AdamW su LM:
    #   lr = 1e-3  → tendenzialmente troppo alto (instabilità)
    #   lr = 5e-4  → possibile
    #   lr = 1e-4  → valore di riferimento nella letteratura LoRA
    #
    # RISULTATI:
    #   lr=1e-3  → vedere results_1B.csv
    #   lr=5e-4  → vedere results_1B.csv
    #   lr=1e-4  → vedere results_1B.csv
    # → Best LR: 5e-4
    # -----------------------------------------------------------------------
    # {
    #     "name": "step0_lr1e-3",
    #     "step": 0,
    #     "rank": 4, "alpha": 8,
    #     "lr": 1e-3,
    #     "batch_size": 8, "n_epochs": 20, "patience": 3
    # },
    # {
    #     "name": "step0_lr5e-4",
    #     "step": 0,
    #     "rank": 4, "alpha": 8,
    #     "lr": 5e-4,
    #     "batch_size": 8, "n_epochs": 20, "patience": 3
    # },
    # {
    #     "name": "step0_lr1e-4",
    #     "step": 0,
    #     "rank": 4, "alpha": 8,
    #     "lr": 1e-4,
    #     "batch_size": 8, "n_epochs": 20, "patience": 3
    # },

    # -----------------------------------------------------------------------
    # STEP 1: RANK SEARCH (lr=5e-4 da step 0)
    # alpha = 2 * rank (scaling = 2.0) come valore di default iniziale.
    #
    # r=4  →  2 × 12 × 2 × (768×4)  ≈  295K trainable params
    # r=8  →  2 × 12 × 2 × (768×8)  ≈  590K trainable params
    # r=16 →  2 × 12 × 2 × (768×16) ≈ 1.18M trainable params
    #
    # RISULTATI:
    #   rank=4  → Dev PPL: 23.33  Test PPL: 21.05
    #   rank=8  → Dev PPL: 22.00  Test PPL: 19.92
    #   rank=16 → Dev PPL: 21.18  Test PPL: 19.30
    # → Best rank: 16 (trend monotono decrescente, guadagno ancora significativo)
    # -----------------------------------------------------------------------
    # {
    #     "name": "step1_rank4",
    #     "step": 1,
    #     "rank": 4, "alpha": 8,
    #     "lr": 5e-4,
    #     "batch_size": 8, "n_epochs": 20, "patience": 3
    # },
    # {
    #     "name": "step1_rank8",
    #     "step": 1,
    #     "rank": 8, "alpha": 16,
    #     "lr": 5e-4,
    #     "batch_size": 8, "n_epochs": 20, "patience": 3
    # },
    # {
    #     "name": "step1_rank16",
    #     "step": 1,
    #     "rank": 16, "alpha": 32,
    #     "lr": 5e-4,
    #     "batch_size": 8, "n_epochs": 20, "patience": 3
    # },

    # -----------------------------------------------------------------------
    # STEP 2: ALPHA SEARCH (rank=16, lr=5e-4 da step 1)
    # Testiamo tre valori di alpha per rank=16.
    #
    # scaling = alpha / rank:
    #   alpha = 8   → scaling = 0.5  (effetto attenuato)
    #   alpha = 16  → scaling = 1.0  (effetto unitario)
    #   alpha = 32  → scaling = 2.0  (già testato in step1_rank16: Test PPL 19.30)
    #
    # RISULTATI (da compilare dopo l'esecuzione):
    #   alpha=8  (scaling=0.5) → Dev PPL: ___  Test PPL: ___
    #   alpha=16 (scaling=1.0) → Dev PPL: ___  Test PPL: ___
    #   alpha=32 (scaling=2.0) → Dev PPL: 21.18  Test PPL: 19.30  (già noto)
    # → Best alpha: ___
    # -----------------------------------------------------------------------
    {
        "name": "step2_alpha_half",
        "step": 2,
        "rank": 16, "alpha": 8,     # alpha = rank//2, scaling = 0.5
        "lr": 5e-4,
        "batch_size": 8, "n_epochs": 20, "patience": 3
    },
    {
        "name": "step2_alpha_eq",
        "step": 2,
        "rank": 16, "alpha": 16,    # alpha = rank, scaling = 1.0
        "lr": 5e-4,
        "batch_size": 8, "n_epochs": 20, "patience": 3
    },

]


# ===========================================================================
# 2. ESECUZIONE DEGLI ESPERIMENTI
# ===========================================================================

print("=" * 70)
print(f"Avvio di {len(experiments)} esperimenti su device: {DEVICE}")
print("Target: PPL < 250 (obbligatorio) e < best Parte 1.A")
print("=" * 70)

for i, exp in enumerate(experiments):
    name = exp["name"]
    print(f"\n{'='*70}")
    print(f"[{i+1}/{len(experiments)}] Esperimento: {name}")
    print(f"{'='*70}")
    print(f"  rank={exp['rank']}, alpha={exp['alpha']}, "
          f"scaling={exp['alpha']/exp['rank']:.2f}, lr={exp['lr']}")

    # -----------------------------------------------------------------------
    # 2.1 DATALOADER
    # Ricreati a ogni esperimento per gestire variazioni del batch_size
    # -----------------------------------------------------------------------
    train_loader, dev_loader, test_loader = get_dataloaders(
        train_raw, dev_raw, test_raw, tokenizer, DEVICE,
        batch_size_train=exp["batch_size"],
        batch_size_eval=exp["batch_size"] * 2  # eval: no gradients → batch più grande
    )

    # -----------------------------------------------------------------------
    # 2.2 MODELLO PRE-ADDESTRATO CON LORA
    # from_pretrained:
    #   1. Chiama GPT2_LoRA.__init__() → sostituisce i blocchi di attenzione
    #   2. Scarica e carica i pesi di GPT-2 base (~467MB la prima volta)
    #   3. I pesi LoRA mantengono l'inizializzazione (A~N(0,1), B=0)
    # -----------------------------------------------------------------------
    print(f"  Caricamento GPT-2 pre-addestrato con LoRA (rank={exp['rank']})...")
    model = GPT2_LoRA.from_pretrained(
        "openai-community/gpt2",
        rank=exp["rank"],
        alpha=exp["alpha"]
    )
    model.to(DEVICE)

    # -----------------------------------------------------------------------
    # 2.3 FREEZE: congela tutti i pesi, poi riabilita solo le matrici LoRA
    # -----------------------------------------------------------------------
    freeze_pretrained_and_enable_lora(model)
    print(f"  Parametri del modello dopo freeze:")
    param_stats(model)

    # Stampa i nomi dei parametri addestrabili per verifica
    trainable_params = [(n, p.shape) for n, p in model.named_parameters()
                        if p.requires_grad]
    print(f"  Parametri addestrabili ({len(trainable_params)} tensori):")
    for param_name, param_shape in trainable_params[:6]:  # mostra i primi 6
        print(f"    {param_name}: {list(param_shape)}")
    if len(trainable_params) > 6:
        print(f"    ... e altri {len(trainable_params) - 6} tensori analoghi")

    # -----------------------------------------------------------------------
    # 2.4 OTTIMIZZATORE
    # AdamW: ottimizzatore standard per fine-tuning con LoRA.
    # Passiamo SOLO i parametri con requires_grad=True (le matrici LoRA).
    # weight_decay=0.01: piccola regolarizzazione L2.
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=exp["lr"],
        weight_decay=0.01
    )

    # -----------------------------------------------------------------------
    # 2.5 TRAINING CON EARLY STOPPING
    # -----------------------------------------------------------------------
    best_model, best_dev_ppl, history = train_model(
        model           = model,
        train_loader    = train_loader,
        dev_loader      = dev_loader,
        tokenizer       = tokenizer,
        optimizer       = optimizer,
        n_epochs        = exp["n_epochs"],
        patience        = exp["patience"],
        experiment_name = name
    )

    # -----------------------------------------------------------------------
    # 2.6 VALUTAZIONE SUL TEST SET (con il modello migliore)
    # -----------------------------------------------------------------------
    best_model = best_model.to(DEVICE)
    test_ppl, test_loss = eval_loop(test_loader, best_model, tokenizer)

    print(f"\n  [RISULTATI FINALI] {name}")
    print(f"  Best Dev PPL : {best_dev_ppl:.2f}")
    print(f"  Test PPL     : {test_ppl:.2f}")
    print(f"  Test Loss    : {test_loss:.4f}")
    print(f"  Target PPL < 250: {'✓ OK' if test_ppl < 250 else '✗ NON RAGGIUNTO'}")

    # -----------------------------------------------------------------------
    # 2.7 SALVATAGGIO
    # Salviamo solo i pesi LoRA (molto più leggeri del modello completo):
    #   ~295KB per rank=4 vs ~467MB per il modello GPT-2 completo.
    # Per ricaricare: creare GPT2_LoRA, fare freeze, poi load_state_dict con strict=False.
    # -----------------------------------------------------------------------
    lora_state_dict = {
        k: v for k, v in best_model.state_dict().items()
        if 'lora_' in k
    }
    model_path = os.path.join(BIN_DIR, f"{name}_lora.pt")
    torch.save(lora_state_dict, model_path)
    print(f"  Pesi LoRA salvati in: {model_path} "
          f"({len(lora_state_dict)} tensori)")

    # Salviamo i risultati nel CSV
    results = {
        "experiment"  : name,
        "step"        : exp["step"],
        "rank"        : exp["rank"],
        "alpha"       : exp["alpha"],
        "scaling"     : exp["alpha"] / exp["rank"],
        "lr"          : exp["lr"],
        "batch_size"  : exp["batch_size"],
        "best_dev_ppl": round(best_dev_ppl, 4),
        "test_ppl"    : round(test_ppl, 4),
        "test_loss"   : round(test_loss, 4),
    }
    save_results_to_csv(results, CSV_PATH)

    # -----------------------------------------------------------------------
    # 2.8 PULIZIA MEMORIA GPU
    # Fondamentale con GPU a memoria limitata: eliminare modello e loader
    # prima del prossimo esperimento per evitare OOM (Out Of Memory).
    # -----------------------------------------------------------------------
    del model, best_model, optimizer
    del train_loader, dev_loader, test_loader
    gc.collect()

    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE == 'mps':
        torch.mps.empty_cache()


# ===========================================================================
# 3. RIEPILOGO FINALE
# ===========================================================================

print("\n" + "=" * 70)
print("TUTTI GLI ESPERIMENTI COMPLETATI")
print(f"Risultati salvati in: {CSV_PATH}")
print("=" * 70)

# Stampa la tabella dei risultati
if os.path.exists(CSV_PATH):
    print("\nRIEPILOGO:")
    print(f"{'Experiment':<25} {'Step':>5} {'Rank':>5} {'Alpha':>6} "
          f"{'LR':>8} {'Dev PPL':>10} {'Test PPL':>10}")
    print("-" * 75)
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in rows:
            print(f"{row['experiment']:<25} {row['step']:>5} "
                  f"{row['rank']:>5} {row['alpha']:>6} "
                  f"{float(row['lr']):>8.0e} "
                  f"{float(row['best_dev_ppl']):>10.2f} "
                  f"{float(row['test_ppl']):>10.2f}")

    # Trova il miglior esperimento per test PPL
    if rows:
        best_row = min(rows, key=lambda r: float(r['test_ppl']))
        print(f"\nMIGLIOR ESPERIMENTO: {best_row['experiment']}")
        print(f"  rank={best_row['rank']}, alpha={best_row['alpha']}, "
              f"lr={best_row['lr']}")
        print(f"  Dev PPL: {best_row['best_dev_ppl']}")
        print(f"  Test PPL: {best_row['test_ppl']}")
        print(f"  Target PPL < 250: "
              f"{'✓ OK' if float(best_row['test_ppl']) < 250 else '✗ NON RAGGIUNTO'}")
