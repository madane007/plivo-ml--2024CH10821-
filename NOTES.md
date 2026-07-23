# NOTES — best configuration

<!-- Numbers reflect the current best; finalized after the last run. -->

1. Best config: byte-level **BPE tokenizer (vocab 4096)**, a **L4 / E160 / 4-head**
   GPT (**1.24M non-embedding** params, embedding exempt per the tokeniser
   clarification), with **RoPE, RMSNorm, SwiGLU, and weight tying**.
2. Training: **AdamW** (0.9, 0.95), **weight decay 0.1**, **grad-clip 1.0**,
   **100-step warmup → cosine** decay from **peak LR 3e-3** to 3e-4, **batch 64**,
   2000 steps, plus **weight EMA (decay 0.999)** scored against the raw weights.
3. It works because the biggest cost in this corpus is that Devanagari (14% of
   the text) is 3 bytes/char, so BPE (~3.4 bytes/token) both collapses those
   sequences and lets a fixed step budget cover ~4 effective epochs instead of ~1.
4. Once the tokeniser was fixed, the model was **data/step-limited, not
   capacity-limited**: scaling width/depth (R3) and vocab past 4k barely moved
   bpb, so the wins came from *throughput*, not size.
5. Raising **batch 32→64** helped (−0.014 bpb) by putting more tokens through
   each of the capped 2000 optimizer steps.
6. **EMA** was the single best free lever (−0.025 bpb, zero extra steps) — it
   averages away the noise of high-LR small-batch updates.
7. The modern recipe (AdamW + cosine/warmup + wd + clip + 1/√(2L) residual init)
   and modern architecture (RoPE/RMSNorm/SwiGLU + tying) together took the
   baseline from **2.3718 → ~1.67** before batch/EMA.
8. Result: **dev bpb 1.6364** vs baseline **2.3718** (−31%).
9. Weight tying is deliberate: it makes the exempt "vocab matrix" a single
   unambiguous tensor, so the ≤2M non-embedding cap holds under the strictest
   reading.
10. Everything is reproducible from `train_tokenizer.py` + `train.py` flags,
    all logged in `RUNLOG.md`.
