"""Find transformer shapes under the 2M NON-EMBEDDING cap (tokeniser/embedding
exempt). Non-embedding params don't depend on vocab, so shape and vocab are
now independent choices. We also print the 'total' at a reference vocab just to
see the number evaluate.py will report.
"""
from model import GPT, Config

REF_VOCAB = 8192   # reference vocab for the 'total' column only
CAP = 2_000_000

# (n_layer, n_embd, n_head, mlp_ratio)
configs = [
    (4, 160, 4, 4.0),
    (4, 256, 4, 4.0),
    (6, 256, 8, 4.0),
    (4, 320, 8, 4.0),
    (3, 384, 8, 4.0),
    (6, 288, 8, 4.0),
    (8, 256, 8, 4.0),
    (5, 320, 8, 4.0),
    (4, 384, 8, 4.0),
    (6, 320, 8, 4.0),
    (8, 288, 8, 4.0),
    (10, 256, 8, 4.0),
]
print(f"{'cfg':<26}{'non-embed':>12}{'total@'+str(REF_VOCAB):>14}   fit")
for (L, E, H, r) in configs:
    c = Config(); c.vocab_size = REF_VOCAB; c.n_layer = L; c.n_embd = E
    c.n_head = H; c.mlp_ratio = r; c.tie_weights = True
    m = GPT(c)
    ne = m.n_params_non_embedding()
    tot = m.n_params()
    tag = "OK" if ne <= CAP else "OVER"
    print(f"L{L} E{E} H{H} r{r:<18}{ne:>12,}{tot:>14,}   {tag}")
