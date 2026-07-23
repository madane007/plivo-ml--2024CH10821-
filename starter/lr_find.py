"""LR range test (Leslie Smith). Ramp LR geometrically over a few hundred
steps on the real model/tokenizer and print loss vs LR. Pick the peak LR a
notch below where the smoothed loss starts climbing. Cheap: one short run.

    python lr_find.py --data ../data/train_corpus.txt
"""
import argparse
import math

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod


def get_batch(ids, block, batch):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr_min", type=float, default=1e-4)
    ap.add_argument("--lr_max", type=float, default=1e-1)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=160)
    ap.add_argument("--n_head", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(1337)

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.n_layer, cfg.n_embd, cfg.n_head = args.n_layer, args.n_embd, args.n_head
    model = GPT(cfg)
    print(f"model {model.n_params():,} params  vocab {tok.vocab_size}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_min,
                            betas=(0.9, 0.95), weight_decay=0.1)

    mult = (args.lr_max / args.lr_min) ** (1 / args.steps)
    lr = args.lr_min
    avg, beta, best = None, 0.9, None
    model.train()
    print("  step      lr    smooth_loss", flush=True)
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(ids, cfg.block_size, args.batch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        l = loss.item()
        avg = l if avg is None else beta * avg + (1 - beta) * l
        smooth = avg / (1 - beta ** step)
        best = smooth if best is None else min(best, smooth)
        if step % 20 == 0:
            print(f"  {step:4d}  {lr:.2e}   {smooth:.4f}", flush=True)
        if smooth > 4 * best and step > 30:
            print(f"  diverged at lr {lr:.2e}", flush=True)
            break
        lr *= mult


if __name__ == "__main__":
    main()
