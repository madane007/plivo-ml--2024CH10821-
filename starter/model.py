"""A small, configurable GPT in plain PyTorch. Every knob lives in Config and
is saved into the checkpoint, so evaluate.py rebuilds the exact same model.

Toggles (all default to modern choices):
  pos_type : "rope" | "learned"      positional information
  norm_type: "rms"  | "layer"        normalization
  mlp_type : "swiglu" | "gelu"       feed-forward
  tie_weights: bool                  share token-embedding with output head

Kept the class names GPT / Config and the (idx, targets)->(logits, loss)
forward signature that evaluate.py depends on.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = True
    pos_type = "rope"          # "rope" | "learned"
    norm_type = "rms"          # "rms" | "layer"
    mlp_type = "swiglu"        # "swiglu" | "gelu"
    mlp_ratio = 4.0            # hidden size = mlp_ratio * n_embd (pre-SwiGLU 2/3 adj)
    init_std = 0.02


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def make_norm(cfg):
    return RMSNorm(cfg.n_embd) if cfg.norm_type == "rms" else nn.LayerNorm(cfg.n_embd)


def build_rope_cache(block_size, head_dim, base=10000.0):
    """Return (cos, sin) each [block_size, head_dim] for rotary embeddings."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(block_size).float()
    freqs = torch.outer(t, inv_freq)            # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)     # [T, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # x: [B, n_head, T, head_dim]
    T = x.size(-2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_rope = cfg.pos_type == "rope"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.dropout_p = cfg.dropout
        if self.use_rope:
            cos, sin = build_rope_cache(cfg.block_size, self.head_dim)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg.mlp_type
        if cfg.mlp_type == "swiglu":
            # keep param count ~ standard MLP: 2/3 * ratio, rounded to multiple of 8
            hidden = int(cfg.mlp_ratio * cfg.n_embd * 2 / 3)
            hidden = (hidden + 7) // 8 * 8
            self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        else:
            hidden = int(cfg.mlp_ratio * cfg.n_embd)
            self.fc = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.kind == "swiglu":
            x = F.silu(self.w_gate(x)) * self.w_up(x)
            return self.drop(self.w_down(x))
        x = F.gelu(self.fc(x))
        return self.drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = SelfAttention(cfg)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.use_learned_pos = cfg.pos_type == "learned"
        if self.use_learned_pos:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = make_norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        # GPT-2 style: scale residual projections by 1/sqrt(2*n_layer)
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w_down.weight"):
                with torch.no_grad():
                    p.mul_(scale)

    def _init(self, m):
        std = self.cfg.init_std
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.use_learned_pos:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        # count unique parameter tensors (tied weights counted once)
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def n_params_non_embedding(self):
        """Total minus the vocab embedding (tied => also the output head).
        Per the clarification 'the tokeniser does not count in the 2M limit',
        this is the number that must stay <= 2,000,000. Learned positional
        embeddings, if any, are NOT tokenizer params and remain counted.
        """
        return self.n_params() - self.tok_emb.weight.numel()
