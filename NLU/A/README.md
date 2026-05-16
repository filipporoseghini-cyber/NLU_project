# Part 2.A — NLU from scratch (Intent Classification + Slot Filling)

## Overview
Joint model for Intent Classification and Slot Filling on the ATIS dataset, built from scratch.

## Tasks
- **Intent Classification**: classify utterance into one of ~18 intents (Accuracy metric)
- **Slot Filling**: BIO sequence labeling for ~80 slot types (F1 metric)

## TODO
- Implement dataset loading and vocabulary (utils.py)
- Implement joint NLU model (model.py)
- Implement training with joint loss (functions.py, main.py)

## How to run
```bash
conda activate nlu26
cd NLU/A
python main.py
```
