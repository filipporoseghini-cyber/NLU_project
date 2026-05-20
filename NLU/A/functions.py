"""
functions.py — Part 2.A: Training e Valutazione per Intent Classification + Slot Filling
==========================================================================================

QUADRO GENERALE
---------------
Questo file implementa la logica di training e valutazione per il task NLU con
GPT-2 from scratch sul dataset ATIS.

I DUE TASK E LE LORO METRICHE
------------------------------
Il modello è addestrato CONGIUNTAMENTE su:
  1. SLOT FILLING → valutato con F1 score (conll script)
     La predizione è corretta solo se il BIO tag è corretto (es. B-fromloc.city_name),
     non basta avere il chunk giusto ma nel tipo sbagliato.
     Usiamo conll.evaluate() che valuta a livello di CHUNK (non di token):
     "New York" = [B-toloc.city_name, I-toloc.city_name] deve essere predetto
     correttamente COME COPPIA per essere considerato corretto.

  2. INTENT CLASSIFICATION → valutato con accuracy
     La predizione è corretta se l'intent scelto (argmax) coincide con il ground truth.
     Usiamo sklearn.classification_report() che restituisce molte metriche.
     Leggiamo report['accuracy'] per l'accuracy totale.

LOSS JOINT
----------
La loss totale è la SOMMA delle due loss:
    loss = loss_intent + loss_slot

Questa è la formulazione più semplice del multi-task learning (MTL) a pesi uguali.
Alternativa: loss = λ_intent * loss_intent + λ_slot * loss_slot con λ iperparametri.

Per lo slot filling usiamo CrossEntropyLoss(ignore_index=PAD_TOKEN):
  - ignora automaticamente le predizioni al CLS token (che ha etichetta = PAD_TOKEN)
  - ignora il padding (che ha etichetta = PAD_TOKEN)

Per l'intent classification usiamo CrossEntropyLoss() senza ignore_index:
  - ogni esempio ha esattamente UN intent → no padding, no CLS da ignorare

VALUTAZIONE SLOT: COME FUNZIONA CONLL.EVALUATE()
-------------------------------------------------
conll.evaluate() prende due liste di sequenze di tuple (word, label):
  ref_slots = [ [(w1, label1), (w2, label2), ...],   ← sequenza 1 ground truth
                [...],                                 ← sequenza 2 ground truth
                ... ]
  hyp_slots = [ [(w1, pred1), (w2, pred2), ...],      ← sequenza 1 predizione
                [...],
                ... ]

Restituisce un dizionario con la F1 score totale in results['total']['f'].

NOTA CRITICA: la lunghezza delle sequenze in ref/hyp deve essere seq_len - 1
(escludiamo il CLS token). Usiamo batch['slots_len'] - 1 per ottenere la
lunghezza reale della frase senza CLS.

MULTIPLE RUNS (5 RUN)
---------------------
Il dataset ATIS è piccolo (~4500 esempi). Risultati singoli possono essere rumorosi
a seconda dell'inizializzazione random. Per risultati affidabili:
  - Addestriamo il modello 5 volte con seed diversi (o semplicemente random)
  - Calcoliamo media e deviazione standard su F1 slot e accuracy intent

Questo è lo standard nel campo NLP per corpus di dimensioni ridotte.

EARLY STOPPING
--------------
Usiamo il Slot F1 sul dev set come metrica di early stopping (invece della loss),
perché è la metrica più direttamente correlata alla qualità del modello sul task
principale. Patience = 3: aspettiamo 3 epoche consecutive senza miglioramento prima di fermarci.
"""

import os
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from conll import evaluate
from sklearn.metrics import classification_report

from model import GPT2, init_weights


# ===========================================================================
# 1. TRAINING LOOP (singola epoca)
# ===========================================================================

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    """
    Esegue un'epoca di training sul modello GPT2 NLU.

    Da notebook cell 6bf6dfca — adattato:
      - Usata tqdm standard invece di tqdm.notebook
      - Aggiunta progress bar con loss corrente

    ALGORITMO PER OGNI BATCH:
      1. Zero gradienti
      2. Forward: model(utterances, slots_len) → (slots, intent)
      3. Permutazione slots: (B, L, slots_size) → (B, slots_size, L) per CrossEntropyLoss
      4. loss = criterion_intents(intent, y_intents) + criterion_slots(slots, y_slots)
      5. Backward + optimizer step

    PERCHÉ IL PERMUTE?
    CrossEntropyLoss di PyTorch per sequenze si aspetta (B, C, L) dove C è il numero
    di classi. Il modello restituisce (B, L, C) → dobbiamo permutare a (B, C, L).
    Equivalentemente: slots.permute(0, 2, 1)

    Args:
        data               : DataLoader per il training
        optimizer          : ottimizzatore (AdamW)
        criterion_slots    : CrossEntropyLoss(ignore_index=PAD_TOKEN) per gli slot
        criterion_intents  : CrossEntropyLoss() per gli intent
        model              : istanza di GPT2 NLU

    Returns:
        loss_array : lista di loss per ogni batch (utile per il plot)
    """
    model.train()
    loss_array = []

    # Inferisce il device dal modello così non serve passarlo come argomento
    device = next(model.parameters()).device

    pbar = tqdm(data, desc="  train", leave=False)

    for batch in pbar:
        optimizer.zero_grad()

        # Sposta il batch sul device del modello (CPU, MPS o CUDA)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass: restituisce logit per slot e intent
        # slots:  (B, L, slots_size)
        # intent: (B, n_intents)
        slots, intent = model(batch['utterances'], batch['slots_len'])

        # Permutazione necessaria per CrossEntropyLoss:
        # (B, L, slots_size) → (B, slots_size, L)
        # CrossEntropyLoss si aspetta (batch, n_classes, seq_len)
        slots = slots.permute(0, 2, 1)

        # Loss per intent classification
        # criterion_intents: CrossEntropyLoss() senza ignore_index
        # batch['intents']: (B,) — un solo intero per ogni esempio
        loss_intent = criterion_intents(intent, batch['intents'])

        # Loss per slot filling
        # criterion_slots: CrossEntropyLoss(ignore_index=PAD_TOKEN)
        # → ignora automaticamente le posizioni con etichetta 0 (pad e CLS)
        # batch['y_slots']: (B, L) — etichetta per ogni token (con padding e CLS→0)
        loss_slot = criterion_slots(slots, batch['y_slots'])

        # JOINT LOSS: somma delle due loss (pesi uguali = 1.0 per entrambi)
        # Questa scelta semplice funziona bene in pratica per ATIS.
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())

        loss.backward()
        optimizer.step()

    return loss_array


# ===========================================================================
# 2. EVALUATION LOOP (dev o test)
# ===========================================================================

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """
    Valuta il modello su un dataset (dev o test) senza aggiornare i pesi.

    Da notebook cell 6bf6dfca — adattato:
      - Usata tqdm standard invece di tqdm.notebook
      - Aggiunto commento esteso sulle fasi di decodifica

    FASI:
      1. Forward pass (senza gradiente)
      2. Calcolo loss (per il plot, non per early stopping)
      3. Decodifica intent: argmax(intent_logits)
      4. Decodifica slot: argmax(slot_logits) per ogni token (escluso CLS e padding)
      5. Calcolo F1 con conll.evaluate()
      6. Calcolo accuracy con classification_report()

    DECODIFICA DEGLI SLOT — dettaglio critico:
    Per ogni sequenza nel batch, prendiamo solo i token della frase REALE:
      length = batch['slots_len'][i] - 1   ← -1: escludiamo il token CLS

    Questo perché:
      - slots_len[i] include il CLS (es. frase di 5 parole → slots_len=6)
      - conll.evaluate() si aspetta sequenze senza il CLS
      - Le parole ground truth sono batch['utterances'][i][:length]
      - Le etichette ground truth sono batch['y_slots'][i][:length]

    Args:
        data              : DataLoader per dev o test
        criterion_slots   : CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents : CrossEntropyLoss()
        model             : istanza di GPT2 NLU
        lang              : istanza di Lang (per id2slot, id2intent, id2word)

    Returns:
        results       : dizionario con metriche conll (results['total']['f'] = F1)
        report_intent : dizionario classification_report (report_intent['accuracy'])
        loss_array    : lista di loss per batch (per il plot)
    """
    model.eval()
    loss_array = []

    device = next(model.parameters()).device

    ref_intents = []  # ground truth intent (stringhe)
    hyp_intents = []  # predizioni intent (stringhe)

    ref_slots = []  # ground truth slot sequences (liste di tuple (word, label))
    hyp_slots = []  # predizioni slot sequences

    with torch.no_grad():
        for batch in tqdm(data, desc="  eval", leave=False):

            # Sposta il batch sul device del modello
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            slots, intent = model(batch['utterances'], batch['slots_len'])
            slots = slots.permute(0, 2, 1)  # (B, slots_size, L)

            # Loss (solo per monitoraggio, non per early stopping)
            loss_intent = criterion_intents(intent, batch['intents'])
            loss_slot   = criterion_slots(slots, batch['y_slots'])
            loss_array.append((loss_intent + loss_slot).item())

            # ---------------------------------------------------------------
            # DECODIFICA INTENT
            # ---------------------------------------------------------------
            # argmax(dim=1): per ogni esempio, prende la classe con score più alto
            # .tolist(): converte il tensore in lista Python
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intent, dim=1).tolist()]
            gt_intents  = [lang.id2intent[x]
                           for x in batch['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # ---------------------------------------------------------------
            # DECODIFICA SLOT
            # ---------------------------------------------------------------
            # output_slots: (B, L) — per ogni token, l'indice dello slot predetto
            output_slots = torch.argmax(slots, dim=1)

            for id_seq, seq in enumerate(output_slots):
                # Lunghezza reale della frase SENZA il CLS token
                # slots_len include CLS → sottraiamo 1
                length = batch['slots_len'].tolist()[id_seq] - 1

                # Recuperiamo le parole della frase (senza CLS e senza padding)
                utt_ids   = batch['utterances'][id_seq][:length].tolist()
                utterance = [lang.id2word[elem] for elem in utt_ids]

                # Ground truth: etichette slot per i token reali (senza CLS)
                gt_ids   = batch['y_slots'][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]

                # Predizioni: etichette slot per i token reali (senza CLS)
                to_decode = seq[:length].tolist()
                hyp_seq = [(utterance[j], lang.id2slot[elem])
                           for j, elem in enumerate(to_decode)]
                ref_seq = [(utterance[j], gt_slots[j])
                           for j in range(length)]

                ref_slots.append(ref_seq)
                hyp_slots.append(hyp_seq)

    # -----------------------------------------------------------------------
    # CALCOLO METRICHE
    # -----------------------------------------------------------------------
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Capita quando il modello predice uno slot mai visto nel ground truth
        # (raro, ma può succedere nelle prime epoche con learning rate alto)
        print(f"  Attenzione conll.evaluate: {ex}")
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents,
        zero_division=False,
        output_dict=True
    )

    return results, report_intent, loss_array


# ===========================================================================
# 3. TRAINING COMPLETO CON EARLY STOPPING (singola run)
# ===========================================================================

def train_model(model, train_loader, dev_loader, lang,
                optimizer, criterion_slots, criterion_intents,
                n_epochs=200, patience=3, experiment_name="exp"):
    """
    Ciclo di training completo con early stopping basato sulla Slot F1 sul dev.

    Da notebook cell 2d07777d — adattato:
      - Incapsulato in funzione per permettere multiple runs
      - Early stopping sul Slot F1 invece che sulla loss
      - Salvataggio del miglior modello via deepcopy

    PERCHÉ EARLY STOPPING SUL SLOT F1?
    -----------------------------------
    La loss su dev potrebbe scendere anche quando il modello inizia a overfittare,
    specialmente su un corpus piccolo come ATIS. La Slot F1 è una metrica più diretta
    della qualità del modello sul task reale.
    Patience = 3: aspettiamo 3 epoche consecutive senza miglioramento prima di fermarci.

    NOTA: evaluiamo ogni epoca (non ogni 5 come nel notebook di esempio), perché
    il training è già veloce su ATIS e vogliamo early stopping preciso.

    Args:
        model              : istanza di GPT2 NLU (su GPU/MPS/CPU)
        train_loader       : DataLoader per il training
        dev_loader         : DataLoader per la validation
        lang               : istanza di Lang per la valutazione
        optimizer          : AdamW
        criterion_slots    : CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents  : CrossEntropyLoss()
        n_epochs           : numero massimo di epoche (default 200)
        patience           : epoche senza miglioramento prima di fermarsi (default 3)
        experiment_name    : nome per la progress bar

    Returns:
        best_model      : modello con la migliore Slot F1 sul dev (su CPU)
        best_f1         : migliore Slot F1 sul dev raggiunta
        best_intent_acc : intent accuracy corrispondente alla migliore Slot F1
    """
    best_f1         = 0.0
    best_intent_acc = 0.0
    best_model      = None
    patience_count  = 0

    epoch_bar = tqdm(range(1, n_epochs + 1), desc=f"[{experiment_name}]", unit="ep")

    for epoch in epoch_bar:
        # -----------------------------------------------------------------------
        # TRAINING
        # -----------------------------------------------------------------------
        train_losses = train_loop(
            train_loader, optimizer,
            criterion_slots, criterion_intents, model
        )

        # -----------------------------------------------------------------------
        # EVALUATION SUL DEV SET
        # -----------------------------------------------------------------------
        dev_results, dev_intent, dev_losses = eval_loop(
            dev_loader, criterion_slots, criterion_intents, model, lang
        )

        dev_f1     = dev_results['total']['f']
        dev_acc    = dev_intent['accuracy']
        train_loss = np.mean(train_losses)
        dev_loss   = np.mean(dev_losses)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.3f}",
            dev_f1=f"{dev_f1:.3f}",
            intent_acc=f"{dev_acc:.3f}"
        )

        # -----------------------------------------------------------------------
        # EARLY STOPPING
        # -----------------------------------------------------------------------
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

def run_experiments(
    train_loader, dev_loader, test_loader,
    lang,
    vocab_len, slots_len, n_intents,
    lr, d_model, n_heads, num_layers, ff_dim, dropout,
    n_runs=5, n_epochs=200, patience=3,
    experiment_name="exp",
    device='cpu'
):
    """
    Esegue n_runs addestramenti indipendenti e riporta media ± std.

    Da notebook cell f08d0445 — incapsulato in funzione.

    PERCHÉ MULTIPLE RUNS?
    ----------------------
    Il dataset ATIS è piccolo (~4500 esempi di training). Risultati da una singola
    run possono variare significativamente a seconda dell'inizializzazione random dei pesi.
    Con 5 run otteniamo una stima affidabile delle performance del modello, con media
    e deviazione standard come metrica di variabilità.

    STRUTTURA:
      Per ogni run:
        1. Crea un nuovo modello con pesi random (via init_weights)
        2. Addestra con early stopping
        3. Valuta il best model sul TEST set
        4. Salva F1 e accuracy

      Alla fine: stampa media ± std per F1 e accuracy.

    Args:
        train_loader, dev_loader, test_loader : DataLoader per i tre split
        lang         : istanza di Lang
        vocab_len    : len(lang.word2id)
        slots_len    : len(lang.id2slot)
        n_intents    : len(lang.intent2id)
        lr           : learning rate per AdamW
        d_model, n_heads, num_layers, ff_dim, dropout : iperparametri del modello
        n_runs       : numero di run indipendenti (default 5)
        n_epochs     : numero massimo di epoche per run (default 200)
        patience     : patience per early stopping (default 3)
        experiment_name : nome per i log
        device       : 'cuda', 'mps', o 'cpu'

    Returns:
        results : dizionario con le metriche per ogni run e le statistiche aggregate
    """
    import torch.optim as optim

    slot_f1s    = []
    intent_accs = []

    criterion_slots   = nn.CrossEntropyLoss(ignore_index=0)   # PAD_TOKEN = 0
    criterion_intents = nn.CrossEntropyLoss()

    print(f"\n{'='*60}")
    print(f"Esperimento: {experiment_name}")
    print(f"  lr={lr}, d_model={d_model}, n_heads={n_heads}, "
          f"num_layers={num_layers}, ff_dim={ff_dim}, dropout={dropout}")
    print(f"  {n_runs} run x max {n_epochs} epoche")
    print(f"{'='*60}")

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")

        # Crea un nuovo modello con pesi random per ogni run
        model = GPT2(
            vocab_size=vocab_len,
            slots_size=slots_len,
            n_intents=n_intents,
            pos_emb_size=1024,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        ).to(device)

        # Inizializzazione uniforme dei pesi (come da notebook)
        model.apply(init_weights)

        # AdamW: buone performance sui Transformer, con weight decay integrato
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        # Training con early stopping → restituisce il modello con miglior dev F1
        best_model, best_dev_f1, best_dev_acc = train_model(
            model, train_loader, dev_loader, lang,
            optimizer, criterion_slots, criterion_intents,
            n_epochs=n_epochs, patience=patience,
            experiment_name=f"{experiment_name}_run{run_idx+1}"
        )

        # Valutazione finale sul TEST set con il miglior modello
        best_model = best_model.to(device)
        test_results, test_intent, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, best_model, lang
        )

        test_f1  = test_results['total']['f']
        test_acc = test_intent['accuracy']

        print(f"  Run {run_idx+1}: Test Slot F1={test_f1:.4f}, Test Intent Acc={test_acc:.4f}")

        slot_f1s.append(test_f1)
        intent_accs.append(test_acc)

    # -----------------------------------------------------------------------
    # STATISTICHE AGGREGATE
    # -----------------------------------------------------------------------
    slot_f1s    = np.array(slot_f1s)
    intent_accs = np.array(intent_accs)

    print(f"\n{'='*60}")
    print(f"Risultati {experiment_name} ({n_runs} run):")
    print(f"  Slot F1:    {slot_f1s.mean():.3f} +- {slot_f1s.std():.3f}")
    print(f"  Intent Acc: {intent_accs.mean():.3f} +- {intent_accs.std():.3f}")
    print(f"{'='*60}\n")

    return {
        'experiment':      experiment_name,
        'lr':              lr,
        'd_model':         d_model,
        'n_heads':         n_heads,
        'num_layers':      num_layers,
        'ff_dim':          ff_dim,
        'dropout':         dropout,
        'slot_f1_mean':    round(float(slot_f1s.mean()), 4),
        'slot_f1_std':     round(float(slot_f1s.std()), 4),
        'intent_acc_mean': round(float(intent_accs.mean()), 4),
        'intent_acc_std':  round(float(intent_accs.std()), 4),
        'slot_f1_runs':    slot_f1s.tolist(),
        'intent_acc_runs': intent_accs.tolist(),
    }


# ===========================================================================
# 5. SALVATAGGIO RISULTATI
# ===========================================================================

def save_results_to_csv(results, csv_path="results_2A.csv"):
    """
    Salva i risultati di un esperimento in un file CSV.

    Identico alla struttura delle Parti 1.A e 1.B. Crea il file se non esiste,
    aggiunge in append se esiste già.

    Args:
        results  : dizionario con i risultati dell'esperimento (output di run_experiments)
        csv_path : percorso del file CSV
    """
    import csv
    scalar_keys = ['experiment', 'lr', 'd_model', 'n_heads', 'num_layers',
                   'ff_dim', 'dropout', 'slot_f1_mean', 'slot_f1_std',
                   'intent_acc_mean', 'intent_acc_std']
    row = {k: results[k] for k in scalar_keys if k in results}

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Risultati salvati in {csv_path}")
