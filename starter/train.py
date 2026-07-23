"""Configurable trainer. Modern recipe by default; every choice is a CLI flag
so experiments are one-flag changes (logged in RUNLOG.md).

HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod
import evaluate as evaluate_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step, args):
    """Linear warmup then cosine decay to min_lr."""
    if step < args.warmup:
        return args.lr * step / max(1, args.warmup)
    if step >= args.steps:
        return args.min_lr
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))


def build_optimizer(model, args):
    # weight decay on 2D matrices only (not norms / embeddings-vector biases)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": args.wd},
              {"params": no_decay, "weight_decay": 0.0}]
    betas = (args.beta1, args.beta2)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=args.lr, betas=betas)
    return torch.optim.Adam(groups, lr=args.lr, betas=betas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min_lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=0,
                    help="if >0, score dev bpb every N steps (needs --dev)")
    ap.add_argument("--dev", default="../data/dev_eval.txt")
    # architecture knobs
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=160)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--pos_type", choices=["rope", "learned"], default="rope")
    ap.add_argument("--norm_type", choices=["rms", "layer"], default="rms")
    ap.add_argument("--mlp_type", choices=["swiglu", "gelu"], default="swiglu")
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--init_std", type=float, default=0.02)
    ap.add_argument("--tie_weights", type=int, default=1)
    ap.add_argument("--qk_norm", type=int, default=0)
    ap.add_argument("--ema_decay", type=float, default=0.999,
                    help="0 disables; else keep an EMA of weights and score it")
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})", flush=True)

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    for k in ("n_layer", "n_head", "n_embd", "block_size", "dropout",
              "pos_type", "norm_type", "mlp_type", "mlp_ratio", "init_std"):
        setattr(cfg, k, getattr(args, k))
    cfg.tie_weights = bool(args.tie_weights)
    cfg.qk_norm = bool(args.qk_norm)
    model = GPT(cfg).to(device)
    n = model.n_params()
    n_ne = model.n_params_non_embedding()
    print(f"model: {n:,} total / {n_ne:,} non-embedding params  "
          f"[V{cfg.vocab_size} L{cfg.n_layer} E{cfg.n_embd} H{cfg.n_head} "
          f"{cfg.pos_type}/{cfg.norm_type}/{cfg.mlp_type} "
          f"tie={cfg.tie_weights}]", flush=True)
    # Per clarification: tokeniser/embedding is exempt -> cap the rest.
    assert n_ne <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} non-embed (got {n_ne:,})"

    opt = build_optimizer(model, args)

    dev_text = None
    if args.eval_every > 0:
        dev_text = open(args.dev, encoding="utf-8").read()

    def score_dev():
        model.eval()
        bpb, _, _ = evaluate_mod.bits_per_byte(model, cfg, tok, dev_text)
        model.train()
        return bpb

    # EMA of weights (free readout improvement; no extra optimizer steps)
    ema = None
    if args.ema_decay > 0:
        ema = {n: p.detach().clone() for n, p in model.named_parameters()}

    model.train()
    t0 = time.time()
    losses = []
    best_bpb = None
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if ema is not None:
            d = args.ema_decay
            with torch.no_grad():
                for n, p in model.named_parameters():
                    ema[n].mul_(d).add_(p.detach(), alpha=1 - d)
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr_at(step, args):.2e}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)
        if args.eval_every > 0 and step % args.eval_every == 0:
            b = score_dev()
            best_bpb = b if best_bpb is None else min(best_bpb, b)
            print(f"    [dev] step {step}: bpb {b:.4f}  (best {best_bpb:.4f})",
                  flush=True)

    # choose the weights to save: EMA if it scores better on dev, else raw
    raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    raw_bpb = score_dev() if dev_text is not None else None
    save_state, tag = raw_state, "raw"
    ema_bpb = None
    if ema is not None and dev_text is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(ema[n])
        ema_bpb = score_dev()
        if ema_bpb <= raw_bpb:
            save_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            tag = "ema"
        # restore raw into the live model (params are mutated in-place above)
        model.load_state_dict(raw_state)

    torch.save({"model": save_state,
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    msg = f"saved {args.out} [{tag}]  ({time.time()-t0:.0f}s total)"
    if raw_bpb is not None:
        msg += f"  raw bpb {raw_bpb:.4f}"
    if ema_bpb is not None:
        msg += f"  ema bpb {ema_bpb:.4f}"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
