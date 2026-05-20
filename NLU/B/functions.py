"""
functions.py — Part 2.B: Training e Valutazione per BERT e GPT-2 fine-tuning
=============================================================================

DIFFERENZE RISPETTO ALLA PARTE 2.A
------------------------------------
  1. IGNORE_INDEX = -100 al posto di PAD_TOKEN = 0
     CrossEntropyLoss(ignore_index=-100) ignora le posizioni con:
       - token speciali [CLS], [SEP], EOS
       - sub-token dopo il primo (solo il primo subtoken di ogni parola ha
         un'etichetta reale; gli altri hanno -100)
       - padding

  2. DECODIFICA SLOT tramite maschera invece di seq_len
     In 2.A usavamo batch['slots_len'] - 1 per trovare la lunghezza reale.
     In 2.B usiamo: mask = (y_slots != IGNORE_INDEX)
     Le posizioni True corrispondono ai primi subtoken delle parole reali.
     Il numero di posizioni True = len(words) per quella sequenza.

  3. 'words' invece di id2word
     In 2.A ricostruivamo le parole da batch['utterances'] tramite lang.id2word.
     In 2.B le parole originali sono già in batch['words'] (lista di stringhe),
     perché non abbiamo word2id (il tokenizer pretrained gestisce i token).

  4. Gestione del batch device-aware
     batch contiene sia Tensor sia liste Python (words = lista di liste).
     Non si può chiamare .to(device) su tutto il batch: usiamo isinstance().

  5. n_epochs e patience diversi
     Il fine-tuning converge molto più velocemente del training from-scratch.
     Usiamo n_epochs=30, patience=3 come default (vs 200/3 per 2.A).

ARCHITETTURA BERT vs GPT-2 nel LOOP
-------------------------------------
Le due architetture hanno forward signatures diverse:
  - BERT:  model(input_ids, attention_mask, token_type_ids)
  - GPT-2: model(input_ids, attention_mask, seq_lens)

Gestiamo questa differenza tramite il parametro model_type ('bert' o 'gpt2').
"""

import os
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from conll import evaluate
from sklearn.metrics import classification_report

from model import BERTforNLU, GPT2forNLU, count_parameters

IGNORE_INDEX = -100  # standard HuggingFace, ignorato da CrossEntropyLoss


# ===========================================================================
# 1. TRAINING LOOP (singola epoca)
# ===========================================================================

def train_loop(data, optimizer, criterion_slots, criterion_intents, model,
               model_type='bert'):
    """
    Esegue un'epoca di training sul modello BERT o GPT-2 NLU.

    ALGORITMO PER OGNI BATCH:
      1. Zero gradienti
      2. Forward: model(input_ids, attention_mask, ...) → (slots, intent)
      3. Permutazione slots: (B, L, slots_size) → (B, slots_size, L) per CrossEntropyLoss
      4. loss = criterion_intents(intent, intents) + criterion_slots(slots, y_slots)
      5. Backward + optimizer step

    Args:
        data              : DataLoader per il training
        optimizer         : AdamW con lr da fine-tuning (tipicamente 1e-5 – 5e-5)
        criterion_slots   : CrossEntropyLoss(ignore_index=-100)
        criterion_intents : CrossEntropyLoss()
        model             : BERTforNLU o GPT2forNLU
        model_type        : 'bert' o 'gpt2' (seleziona quali chiavi del batch usare)

    Returns:
        loss_array : lista di loss per batch (per monitoraggio)
    """
    model.train()
    loss_array = []
    device = next(model.parameters()).device

    pbar = tqdm(data, desc="  train", leave=False)

    for batch in pbar:
        optimizer.zero_grad()

        # Sposta solo i Tensor sul device; 'words' è una lista Python → lasciala
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward: firma diversa per BERT e GPT-2
        if model_type == 'bert':
            slots, intent = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['token_type_ids'],
            )
        else:  # gpt2
            slots, intent = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['seq_lens'],
            )

        # CrossEntropyLoss per sequenze si aspetta (B, n_classes, L)
        slots = slots.permute(0, 2, 1)  # (B, L, slots_size) → (B, slots_size, L)

        loss_intent = criterion_intents(intent, batch['intents'])
        loss_slot   = criterion_slots(slots, batch['y_slots'])
        loss        = loss_intent + loss_slot

        loss_array.append(loss.item())
        loss.backward()
        optimizer.step()

    return loss_array


# ===========================================================================
# 2. EVALUATION LOOP (dev o test)
# ===========================================================================

def eval_loop(data, criterion_slots, criterion_intents, model, lang,
              model_type='bert'):
    """
    Valuta il modello su un dataset senza aggiornare i pesi.

    DECODIFICA SLOT — differenza rispetto a 2.A:
    In 2.A usavamo batch['slots_len'] - 1 come lunghezza reale della frase.
    In 2.B usiamo la maschera: mask = (y_slots != IGNORE_INDEX)
      - True  → posizione con etichetta reale (primo subtoken di una parola)
      - False → posizione da ignorare ([CLS], [SEP], EOS, sub-token dopo il primo, padding)
    Il numero di True = numero di parole originali (o parole rimaste dopo truncation).

    PAROLE ORIGINALI:
    In 2.A usavamo lang.id2word per ricostruire le parole dai loro id numerici.
    In 2.B non esiste word2id: le parole originali sono in batch['words'].

    Args:
        data              : DataLoader per dev o test
        criterion_slots   : CrossEntropyLoss(ignore_index=-100)
        criterion_intents : CrossEntropyLoss()
        model             : BERTforNLU o GPT2forNLU
        lang              : istanza di Lang (id2slot, id2intent)
        model_type        : 'bert' o 'gpt2'

    Returns:
        results       : dizionario conll (results['total']['f'] = Slot F1)
        report_intent : dizionario classification_report (report_intent['accuracy'])
        loss_array    : lista di loss per batch
    """
    model.eval()
    loss_array = []
    device = next(model.parameters()).device

    ref_intents = []
    hyp_intents = []
    ref_slots   = []
    hyp_slots   = []

    with torch.no_grad():
        for batch in tqdm(data, desc="  eval", leave=False):

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if model_type == 'bert':
                slots, intent = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['token_type_ids'],
                )
            else:
                slots, intent = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['seq_lens'],
                )

            slots = slots.permute(0, 2, 1)  # (B, slots_size, L)

            loss_intent = criterion_intents(intent, batch['intents'])
            loss_slot   = criterion_slots(slots, batch['y_slots'])
            loss_array.append((loss_intent + loss_slot).item())

            # -------------------------------------------------------------------
            # DECODIFICA INTENT
            # -------------------------------------------------------------------
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intent, dim=1).tolist()]
            gt_intents  = [lang.id2intent[x]
                           for x in batch['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # -------------------------------------------------------------------
            # DECODIFICA SLOT
            # -------------------------------------------------------------------
            output_slots = torch.argmax(slots, dim=1)  # (B, L)

            for id_seq in range(output_slots.size(0)):
                words  = batch['words'][id_seq]          # lista di parole originali
                y      = batch['y_slots'][id_seq]        # (L,) con -100 nelle posizioni ignorate
                preds  = output_slots[id_seq]            # (L,) predizioni argmax

                # Posizioni con etichetta reale = primo subtoken di ogni parola
                mask = (y != IGNORE_INDEX)   # (L,) bool

                real_gt_ids   = y[mask].tolist()      # ID slot ground truth per parola reale
                real_pred_ids = preds[mask].tolist()  # ID slot predetti per parola reale

                # Numero di parole reali (può essere < len(words) se sequenza truncata)
                n_real = len(real_gt_ids)
                words_aligned = words[:n_real]

                ref_seq = [(words_aligned[j], lang.id2slot[real_gt_ids[j]])
                           for j in range(n_real)]
                hyp_seq = [(words_aligned[j], lang.id2slot[real_pred_ids[j]])
                           for j in range(n_real)]

                ref_slots.append(ref_seq)
                hyp_slots.append(hyp_seq)

    # -----------------------------------------------------------------------
    # METRICHE
    # -----------------------------------------------------------------------
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print(f"  Attenzione conll.evaluate: {ex}")
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents,
        zero_division=False,
        output_dict=True,
    )

    return results, report_intent, loss_array


# ===========================================================================
# 3. TRAINING COMPLETO CON EARLY STOPPING (singola run)
# ===========================================================================

def train_model(model, train_loader, dev_loader, lang, optimizer,
                criterion_slots, criterion_intents,
                model_type='bert', n_epochs=30, patience=3,
                experiment_name="exp"):
    """
    Ciclo di training completo con early stopping basato sulla Slot F1 sul dev.

    Identico nella struttura a 2.A ma con:
      - n_epochs=30 di default (il fine-tuning converge in poche epoche)
      - model_type passato a train_loop e eval_loop

    Args:
        model              : BERTforNLU o GPT2forNLU (su device)
        train_loader       : DataLoader train
        dev_loader         : DataLoader dev
        lang               : istanza di Lang
        optimizer          : AdamW
        criterion_slots    : CrossEntropyLoss(ignore_index=-100)
        criterion_intents  : CrossEntropyLoss()
        model_type         : 'bert' o 'gpt2'
        n_epochs           : epoche massime
        patience           : epoche senza miglioramento prima di fermarsi
        experiment_name    : nome per la progress bar

    Returns:
        best_model      : modello con la migliore Slot F1 dev (su CPU)
        best_f1         : migliore Slot F1 sul dev
        best_intent_acc : accuracy intent corrispondente
    """
    best_f1         = 0.0
    best_intent_acc = 0.0
    best_model      = None
    patience_count  = 0

    epoch_bar = tqdm(range(1, n_epochs + 1), desc=f"[{experiment_name}]", unit="ep")

    for epoch in epoch_bar:
        train_losses = train_loop(
            train_loader, optimizer,
            criterion_slots, criterion_intents, model, model_type,
        )
        dev_results, dev_intent, dev_losses = eval_loop(
            dev_loader, criterion_slots, criterion_intents, model, lang, model_type,
        )

        dev_f1     = dev_results['total']['f']
        dev_acc    = dev_intent['accuracy']
        train_loss = np.mean(train_losses)
        dev_loss   = np.mean(dev_losses)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.3f}",
            dev_f1=f"{dev_f1:.3f}",
            intent_acc=f"{dev_acc:.3f}",
        )

        if dev_f1 > best_f1:
            best_f1         = dev_f1
            best_intent_acc = dev_acc
            best_model      = copy.deepcopy(model).cpu()
            patience_count  = 0
            print(f"  [ep {epoch:3d}] loss={train_loss:.4f} | "
                  f"dev F1={dev_f1:.4f}  acc={dev_acc:.4f}  ✓ nuovo best")
        else:
            patience_count += 1
            print(f"  [ep {epoch:3d}] loss={train_loss:.4f} | "
                  f"dev F1={dev_f1:.4f}  acc={dev_acc:.4f}  "
                  f"· patience {patience_count}/{patience}")
            if patience_count >= patience:
                print(f"  Early stopping dopo {epoch} epoche. Best dev F1: {best_f1:.4f}")
                break

    return best_model, best_f1, best_intent_acc


# ===========================================================================
# 4. MULTIPLE RUNS
# ===========================================================================

def run_experiments(train_loader, dev_loader, test_loader, lang,
                    slots_size, n_intents,
                    lr, model_name, model_type='bert', dropout=0.1,
                    n_runs=3, n_epochs=30, patience=3,
                    experiment_name="exp", device='cpu'):
    """
    Esegue n_runs addestramenti indipendenti e riporta media ± std.

    A differenza di 2.A, qui non ci sono parametri architetturali da variare:
    il backbone è fisso (pretrained). I parametri principali sono lr e model_name.

    NOTA SULLA VARIANZA:
    Nel fine-tuning, tutti i run partono dagli stessi pesi pretrained. La varianza
    proviene solo dall'ordine dei mini-batch (shuffle nel DataLoader). Per modelli
    grandi (BERT-large, GPT-2-medium), 3 run sono sufficienti.

    Args:
        train_loader, dev_loader, test_loader : DataLoader per i tre split
        lang         : istanza di Lang
        slots_size   : len(lang.slot2id)
        n_intents    : len(lang.intent2id)
        lr           : learning rate per AdamW (tipicamente 1e-5, 2e-5, 5e-5)
        model_name   : nome HuggingFace del modello
        model_type   : 'bert' o 'gpt2'
        dropout      : dropout nelle teste di output (default 0.1)
        n_runs       : numero di run (default 3 per il fine-tuning)
        n_epochs     : epoche massime per run (default 30)
        patience     : patience per early stopping (default 3)
        experiment_name : nome per i log
        device       : 'cuda', 'mps', o 'cpu'

    Returns:
        results : dizionario con metriche aggregate e per-run
    """
    slot_f1s    = []
    intent_accs = []

    criterion_slots   = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_intents = nn.CrossEntropyLoss()

    print(f"\n{'='*65}")
    print(f"Esperimento: {experiment_name}")
    print(f"  model={model_name}, model_type={model_type}")
    print(f"  lr={lr}, dropout={dropout}")
    print(f"  {n_runs} run x max {n_epochs} epoche (patience={patience})")
    print(f"{'='*65}")

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")

        # Crea un nuovo modello per ogni run (parte sempre dai pesi pretrained)
        if model_type == 'bert':
            model = BERTforNLU(slots_size, n_intents, model_name, dropout).to(device)
        else:
            model = GPT2forNLU(slots_size, n_intents, model_name, dropout).to(device)

        if run_idx == 0:
            print(f"  Parametri addestrabili: {count_parameters(model):,}")

        optimizer = optim.AdamW(model.parameters(), lr=lr)

        best_model, best_dev_f1, best_dev_acc = train_model(
            model, train_loader, dev_loader, lang, optimizer,
            criterion_slots, criterion_intents,
            model_type=model_type,
            n_epochs=n_epochs, patience=patience,
            experiment_name=f"{experiment_name}_run{run_idx + 1}",
        )

        # Valutazione finale sul TEST set con il miglior modello
        best_model = best_model.to(device)
        test_results, test_intent, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, best_model, lang, model_type,
        )

        test_f1  = test_results['total']['f']
        test_acc = test_intent['accuracy']
        print(f"  Run {run_idx + 1}: Test Slot F1={test_f1:.4f}, Test Intent Acc={test_acc:.4f}")

        slot_f1s.append(test_f1)
        intent_accs.append(test_acc)

    slot_f1s    = np.array(slot_f1s)
    intent_accs = np.array(intent_accs)

    print(f"\n{'='*65}")
    print(f"Risultati {experiment_name} ({n_runs} run):")
    print(f"  Slot F1:    {slot_f1s.mean():.3f} +- {slot_f1s.std():.3f}")
    print(f"  Intent Acc: {intent_accs.mean():.3f} +- {intent_accs.std():.3f}")
    print(f"{'='*65}\n")

    return {
        'experiment':      experiment_name,
        'model_name':      model_name,
        'model_type':      model_type,
        'lr':              lr,
        'dropout':         dropout,
        'slot_f1_mean':    round(float(slot_f1s.mean()), 4),
        'slot_f1_std':     round(float(slot_f1s.std()),  4),
        'intent_acc_mean': round(float(intent_accs.mean()), 4),
        'intent_acc_std':  round(float(intent_accs.std()),  4),
        'slot_f1_runs':    slot_f1s.tolist(),
        'intent_acc_runs': intent_accs.tolist(),
    }


# ===========================================================================
# 5. SALVATAGGIO RISULTATI
# ===========================================================================

def save_results_to_csv(results, csv_path="results_2B.csv"):
    """
    Salva i risultati di un esperimento in un file CSV (append mode).

    Identico nella struttura a 2.A ma con le colonne per model_name e model_type.

    Args:
        results  : dizionario output di run_experiments
        csv_path : percorso del file CSV
    """
    import csv
    scalar_keys = [
        'experiment', 'model_name', 'model_type', 'lr', 'dropout',
        'slot_f1_mean', 'slot_f1_std', 'intent_acc_mean', 'intent_acc_std',
    ]
    row = {k: results[k] for k in scalar_keys if k in results}

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Risultati salvati in {csv_path}")
