# RUNLOG — 2,000-step LLM speedrun

Metric: **bits-per-byte (bpb)** on `data/dev_eval.txt` (lower is better).
Caps: ≤2000 steps, ≤2,000,000 **non-embedding** params (tokeniser/embedding
exempt — confirmed by staff), CPU only, train on `train_corpus.txt` only.

Corpus facts that drive the choices: 5.70M chars / 7.32M bytes; **85.8%
English, 14.1% Hindi (Devanagari = 3 bytes/char in UTF-8), 0.1% other**; 657
unique chars. A byte tokeniser therefore (a) spends 3 steps per Hindi char and
(b) at fixed steps×batch×block only sees ~1 epoch of the data.

| run | change | tokeniser | arch (non-embed) | peak LR | dev bpb | Δ |
|---|---|---|---|---|---|---|
| R0 baseline | stock starter | byte 256 | L4 E160 (1.24M) | 3e-4 const | **2.3718** | — |
| R1 | +BPE +modern recipe +modern arch (all at once) | BPE 4096 | L4 E160 (1.24M) | 3e-3 cos | **1.6752** | −0.697 |
| R2 | peak LR 3e-3 → 6e-3 | BPE 4096 | L4 E160 | 6e-3 cos | (worse) | + |
| R3 | +vocab 8192 +width E192 | BPE 8192 | L4 E192 (1.77M) | 3e-3 cos | _pending_ | |

## R0 — baseline (reference)
Hypothesis: none; establish the number to beat.
Stock model: L4 E160, byte tokeniser, learned pos, LayerNorm, GELU, untied
head, flat N(0,0.05) init; plain Adam, constant lr 3e-4, no warmup/decay/clip,
batch 8. Final train loss ~1.73 nats/byte. **dev bpb 2.3718.**

## R1 — BPE + modern recipe + modern arch (bundled)
Hypothesis: the tokeniser is the biggest lever (Devanagari + effective-epochs
argument), and the baseline trainer leaves easy gains (schedule, init, tying).
Changed at once: BPE-4096 tokeniser (lossless, byte-fallback; 3.49 bytes/token
on train → ~3.4× shorter sequences → ~3.9 effective epochs); RoPE, RMSNorm,
SwiGLU; weight tying; init 0.02 + residual scaled 1/√(2L); AdamW (0.9,0.95),
wd 0.1, grad-clip 1.0; warmup 100 → cosine to 3e-4, peak 3e-3; batch 32.
Result: **dev bpb 1.6752** (−29% vs baseline). Conclusion: huge combined win;
attribution still owed (see planned ablations).

## R2 — LR range test + confirm
Hypothesis: the LR range test (lr_find.py) showed smoothed loss still dropping
up to ~3.5e-2, suggesting peak 3e-3 was too low; try 6e-3.
Result: 6e-3 gave **higher** final train loss (3.53 vs R1's 3.24) → worse.
Conclusion: the range test finds the fastest *short-term* descent, not the best
*full-schedule* peak. **Keep peak LR = 3e-3.** Useful negative result.

## R3 — free vocab + more width (embedding now exempt)
Hypothesis: since embedding params don't count, (a) a bigger vocab (8192) is
almost free and should compress more / give more effective epochs, and (b) we
were only using 1.24M of 2M non-embedding, so widen to E192 (1.77M). Both
expected to help; bundled to save runs.
Result: _pending._
