"""
main.py — Part 2.B: Fine-tuning BERT e GPT-2 su ATIS
======================================================

STRUTTURA DEGLI ESPERIMENTI
-----------------------------
La Part 2.B richiede di fare fine-tuning di modelli pre-addestrati
(encoder BERT + decoder GPT-2) per il task NLU congiunto su ATIS.

SEZIONE A — BERT (encoder bidirezionale, bert-base-uncased)
  Step 0: ricerca learning rate ∈ {1e-5, 2e-5, 5e-5}
           (valori tipici per il fine-tuning di Transformer pre-addestrati)
  Step 1: prova bert-large-uncased con il miglior lr trovato

SEZIONE B — GPT-2 (decoder causale, openai-community/gpt2)
  Step 0: ricerca learning rate ∈ {1e-5, 2e-5, 5e-5}
  Step 1: prova openai-community/gpt2-medium con il miglior lr trovato

PERCHÉ LR COSÌ PICCOLI RISPETTO A 2.A?
-----------------------------------------
In 2.A addestravo da zero con AdamW e lr=0.01 (ottimale per pesi random).
In 2.B i pesi del backbone sono già in un ottimo locale pre-addestrato.
Un lr grande distruggerebbe queste rappresentazioni già utili ("catastrophic
forgetting"). I valori {1e-5, 2e-5, 5e-5} sono lo standard per il fine-tuning
di BERT/GPT-2 (Devlin et al., 2019; Radford et al., 2019).

METRICHE
--------
  Slot F1:    media ± std su n_runs run indipendenti  (metrica principale)
  Intent Acc: media ± std  (metrica secondaria)

NUMERO DI RUN
-------------
  3 run (vs 5 in 2.A): il fine-tuning parte sempre dai pesi pre-addestrati,
  quindi la varianza tra run è più bassa. 3 run sono sufficienti per ATIS.

TARGET INDICATIVO (letteratura NLU su ATIS con BERT-base):
  Slot F1:    > 0.95
  Intent Acc: > 0.97
"""

import os
import torch

from utils import (
    load_data, create_dev_split, build_lang,
    BERTIntentsAndSlots, GPT2IntentsAndSlots,
    collate_fn_bert, collate_fn_gpt2,
    get_dataloaders,
    get_bert_tokenizer, get_gpt2_tokenizer,
)
from functions import run_experiments, save_results_to_csv


# ===========================================================================
# SETUP DEVICE
# ===========================================================================

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")
    return device


# ===========================================================================
# CARICAMENTO DATI E VOCABOLARIO (condiviso tra BERT e GPT-2)
# ===========================================================================

def setup_raw_data():
    """Carica i dati ATIS e crea il dev set stratificato."""
    data_dir   = os.path.join('dataset', 'ATIS')
    train_path = os.path.join(data_dir, 'train.json')
    test_path  = os.path.join(data_dir, 'test.json')

    tmp_train_raw = load_data(train_path)
    test_raw      = load_data(test_path)

    print(f"Training set originale: {len(tmp_train_raw)} esempi")
    print(f"Test set:               {len(test_raw)} esempi")

    train_raw, dev_raw = create_dev_split(tmp_train_raw, dev_size=0.10)
    print(f"Training (dopo split): {len(train_raw)} esempi")
    print(f"Dev set:               {len(dev_raw)} esempi")

    return train_raw, dev_raw, test_raw


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':

    device = get_device()

    # Carica i dati raw e costruisce il vocabolario
    # (lang è condiviso tra BERT e GPT-2: stessi slot e intent)
    train_raw, dev_raw, test_raw = setup_raw_data()
    lang = build_lang(train_raw, dev_raw, test_raw)

    slots_size = len(lang.slot2id)
    n_intents  = len(lang.intent2id)
    print(f"\nSlot labels: {slots_size}  —  Intent: {n_intents}")

    # Raccoglie tutti i risultati per il riepilogo finale
    all_results = []

    # =========================================================================
    # SEZIONE A — BERT (encoder bidirezionale)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SEZIONE A — BERT fine-tuning")
    print("=" * 70)

    bert_tokenizer = get_bert_tokenizer("bert-base-uncased")

    # Crea i dataset con tokenizzazione WordPiece e allineamento slot labels
    bert_train_ds = BERTIntentsAndSlots(train_raw, lang, bert_tokenizer, max_len=128)
    bert_dev_ds   = BERTIntentsAndSlots(dev_raw,   lang, bert_tokenizer, max_len=128)
    bert_test_ds  = BERTIntentsAndSlots(test_raw,  lang, bert_tokenizer, max_len=128)

    bert_train_loader, bert_dev_loader, bert_test_loader = get_dataloaders(
        bert_train_ds, bert_dev_ds, bert_test_ds,
        collate_fn=collate_fn_bert,
        batch_size_train=32,
        batch_size_eval=64,
    )

    # -------------------------------------------------------------------
    # BERT Step 0 — ricerca learning rate
    # -------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("BERT — Step 0: ricerca learning rate")
    print("-" * 50)

    bert_lr_configs = [
        {"lr": 1e-5, "name": "bert_step0_lr1e-5"},
        {"lr": 2e-5, "name": "bert_step0_lr2e-5"},
        {"lr": 5e-5, "name": "bert_step0_lr5e-5"},
    ]

    bert_step0_results = []
    for cfg in bert_lr_configs:
        res = run_experiments(
            bert_train_loader, bert_dev_loader, bert_test_loader,
            lang,
            slots_size=slots_size,
            n_intents=n_intents,
            lr=cfg["lr"],
            model_name="bert-base-uncased",
            model_type="bert",
            dropout=0.1,
            n_runs=3,
            n_epochs=30,
            patience=3,
            experiment_name=cfg["name"],
            device=device,
        )
        save_results_to_csv(res)
        bert_step0_results.append(res)
        all_results.append(res)

    best_bert_base = max(bert_step0_results, key=lambda x: x['slot_f1_mean'])
    best_bert_lr   = best_bert_base['lr']
    print(f"\nMiglior lr per BERT-base: {best_bert_lr} "
          f"(Slot F1={best_bert_base['slot_f1_mean']:.3f})")

    # -------------------------------------------------------------------
    # BERT Step 1 — bert-large-uncased
    # -------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("BERT — Step 1: bert-large-uncased")
    print("-" * 50)

    # NOTA: bert-large ha hidden_size=1024 → più lento ma potenzialmente migliore
    bert_large_tokenizer = get_bert_tokenizer("bert-large-uncased")

    bert_large_train_ds = BERTIntentsAndSlots(train_raw, lang, bert_large_tokenizer, max_len=128)
    bert_large_dev_ds   = BERTIntentsAndSlots(dev_raw,   lang, bert_large_tokenizer, max_len=128)
    bert_large_test_ds  = BERTIntentsAndSlots(test_raw,  lang, bert_large_tokenizer, max_len=128)

    bert_large_train_loader, bert_large_dev_loader, bert_large_test_loader = get_dataloaders(
        bert_large_train_ds, bert_large_dev_ds, bert_large_test_ds,
        collate_fn=collate_fn_bert,
        batch_size_train=16,  # batch size ridotto per bert-large (più memoria)
        batch_size_eval=32,
    )

    res = run_experiments(
        bert_large_train_loader, bert_large_dev_loader, bert_large_test_loader,
        lang,
        slots_size=slots_size,
        n_intents=n_intents,
        lr=best_bert_lr,
        model_name="bert-large-uncased",
        model_type="bert",
        dropout=0.1,
        n_runs=3,
        n_epochs=30,
        patience=3,
        experiment_name="bert_step1_large",
        device=device,
    )
    save_results_to_csv(res)
    all_results.append(res)

    best_bert = max([best_bert_base, res], key=lambda x: x['slot_f1_mean'])
    print(f"\nMiglior modello BERT: {best_bert['experiment']} "
          f"(Slot F1={best_bert['slot_f1_mean']:.3f})")

    # =========================================================================
    # SEZIONE B — GPT-2 (decoder causale)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SEZIONE B — GPT-2 fine-tuning")
    print("=" * 70)

    gpt2_tokenizer = get_gpt2_tokenizer("openai-community/gpt2")

    # Crea i dataset con tokenizzazione BPE e EOS come token CLS alla fine
    gpt2_train_ds = GPT2IntentsAndSlots(train_raw, lang, gpt2_tokenizer, max_len=128)
    gpt2_dev_ds   = GPT2IntentsAndSlots(dev_raw,   lang, gpt2_tokenizer, max_len=128)
    gpt2_test_ds  = GPT2IntentsAndSlots(test_raw,  lang, gpt2_tokenizer, max_len=128)

    gpt2_train_loader, gpt2_dev_loader, gpt2_test_loader = get_dataloaders(
        gpt2_train_ds, gpt2_dev_ds, gpt2_test_ds,
        collate_fn=collate_fn_gpt2,
        batch_size_train=32,
        batch_size_eval=64,
    )

    # -------------------------------------------------------------------
    # GPT-2 Step 0 — ricerca learning rate
    # -------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("GPT-2 — Step 0: ricerca learning rate")
    print("-" * 50)

    gpt2_lr_configs = [
        {"lr": 1e-5, "name": "gpt2_step0_lr1e-5"},
        {"lr": 2e-5, "name": "gpt2_step0_lr2e-5"},
        {"lr": 5e-5, "name": "gpt2_step0_lr5e-5"},
    ]

    gpt2_step0_results = []
    for cfg in gpt2_lr_configs:
        res = run_experiments(
            gpt2_train_loader, gpt2_dev_loader, gpt2_test_loader,
            lang,
            slots_size=slots_size,
            n_intents=n_intents,
            lr=cfg["lr"],
            model_name="openai-community/gpt2",
            model_type="gpt2",
            dropout=0.1,
            n_runs=3,
            n_epochs=30,
            patience=3,
            experiment_name=cfg["name"],
            device=device,
        )
        save_results_to_csv(res)
        gpt2_step0_results.append(res)
        all_results.append(res)

    best_gpt2_base = max(gpt2_step0_results, key=lambda x: x['slot_f1_mean'])
    best_gpt2_lr   = best_gpt2_base['lr']
    print(f"\nMiglior lr per GPT-2 base: {best_gpt2_lr} "
          f"(Slot F1={best_gpt2_base['slot_f1_mean']:.3f})")

    # -------------------------------------------------------------------
    # GPT-2 Step 1 — gpt2-medium
    # -------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("GPT-2 — Step 1: gpt2-medium")
    print("-" * 50)

    gpt2_medium_tokenizer = get_gpt2_tokenizer("openai-community/gpt2-medium")

    gpt2_med_train_ds = GPT2IntentsAndSlots(train_raw, lang, gpt2_medium_tokenizer, max_len=128)
    gpt2_med_dev_ds   = GPT2IntentsAndSlots(dev_raw,   lang, gpt2_medium_tokenizer, max_len=128)
    gpt2_med_test_ds  = GPT2IntentsAndSlots(test_raw,  lang, gpt2_medium_tokenizer, max_len=128)

    gpt2_med_train_loader, gpt2_med_dev_loader, gpt2_med_test_loader = get_dataloaders(
        gpt2_med_train_ds, gpt2_med_dev_ds, gpt2_med_test_ds,
        collate_fn=collate_fn_gpt2,
        batch_size_train=16,  # ridotto per gpt2-medium
        batch_size_eval=32,
    )

    res = run_experiments(
        gpt2_med_train_loader, gpt2_med_dev_loader, gpt2_med_test_loader,
        lang,
        slots_size=slots_size,
        n_intents=n_intents,
        lr=best_gpt2_lr,
        model_name="openai-community/gpt2-medium",
        model_type="gpt2",
        dropout=0.1,
        n_runs=3,
        n_epochs=30,
        patience=3,
        experiment_name="gpt2_step1_medium",
        device=device,
    )
    save_results_to_csv(res)
    all_results.append(res)

    best_gpt2 = max([best_gpt2_base, res], key=lambda x: x['slot_f1_mean'])
    print(f"\nMiglior modello GPT-2: {best_gpt2['experiment']} "
          f"(Slot F1={best_gpt2['slot_f1_mean']:.3f})")

    # =========================================================================
    # RIEPILOGO FINALE
    # =========================================================================
    print("\n" + "=" * 70)
    print("RIEPILOGO FINALE — Tutti gli esperimenti")
    print("=" * 70)
    print(f"{'Esperimento':<35} {'Slot F1':>10} {'±':>6} {'Intent Acc':>12} {'±':>6}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['experiment']:<35} "
              f"{r['slot_f1_mean']:>10.3f} "
              f"{r['slot_f1_std']:>6.3f} "
              f"{r['intent_acc_mean']:>12.3f} "
              f"{r['intent_acc_std']:>6.3f}")

    best_overall = max(all_results, key=lambda x: x['slot_f1_mean'])
    print(f"\nMigliore: {best_overall['experiment']}")
    print(f"  Slot F1    = {best_overall['slot_f1_mean']:.3f} +- {best_overall['slot_f1_std']:.3f}")
    print(f"  Intent Acc = {best_overall['intent_acc_mean']:.3f} +- {best_overall['intent_acc_std']:.3f}")
    print(f"  Config: model={best_overall['model_name']}, lr={best_overall['lr']}")
    print(f"\nRisultati salvati in results_2B.csv")
