# Plivo ML — 2,000-Step LLM Speedrun (2024CH10821)

Train a GPT-style LLM from scratch under hard caps and minimize **bits-per-byte
(bpb)** on a hidden English+Hindi text file. Lower is better.

## Hard caps
- ≤ 2,000 optimizer steps for the checkpoint-producing run
- ≤ 2,000,000 total parameters
- Train only on `data/train_corpus.txt` (tokenizer too)
- Pure PyTorch + numpy + stdlib, **CPU only**, no pretrained weights

## Data
| file | size | what |
|---|---|---|
| `data/train_corpus.txt` | 7.0 MB (5.70M chars / 7.32M bytes) | the ONLY training data — mixed English + Hindi |
| `data/dev_eval.txt` | 156 KB | held-out text for local scoring |

Corpus composition: **85.8% ASCII/English, 14.1% Devanagari/Hindi, 0.1% other; 657
unique characters.** Because Devanagari is 3 bytes/char in UTF-8, a byte-level
tokenizer spends 3 model steps per Hindi character — the main reason a trained
BPE tokenizer is the biggest lever here.

## Baseline (stock starter code)
- **Model:** 4-layer GPT, `n_embd=160`, 4 heads, block 128, learned positional
  embeddings, LayerNorm, standard GELU MLP, byte tokenizer (vocab 256),
  **1,339,840 params**.
- **Trainer:** constant Adam `lr=3e-4`, no warmup, no schedule, no weight decay,
  no gradient clipping, flat `N(0, 0.05)` init, batch 8.
- **Result:** dev **bpb = 2.3718** (2000 steps, ~86s CPU).

This baseline is mediocre on purpose; the work is improving it under the caps.
See `RUNLOG.md` for the experiment log.

## Layout
```
starter/
  model.py            GPT (modifiable)
  train.py            trainer (modifiable)
  evaluate.py         official scorer — interface must not change
  tokenizer.py        tokenizer (load/encode/decode/vocab_size)
  train_tokenizer.py  byte-level BPE trainer (trained on the corpus only)
data/
  train_corpus.txt, dev_eval.txt
```

## Run
```bash
python starter/train_tokenizer.py --data data/train_corpus.txt --vocab 4096
python starter/train.py --data data/train_corpus.txt --steps 2000 --out ckpt.pt
python starter/evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt
```
