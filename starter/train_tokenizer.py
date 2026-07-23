"""Train a byte-level BPE tokenizer ON THE PROVIDED CORPUS ONLY.

Why BPE here: the corpus is ~86% English / ~14% Hindi. In UTF-8 every
Devanagari character is 3 bytes, so a byte-level tokenizer spends 3 model
steps per Hindi char and gives the model a tiny effective context. BPE
merges frequent byte sequences (English subwords AND Devanagari clusters)
into single tokens, shrinking the sequence ~3-4x and letting each token
amortize several bytes -> lower bits-per-byte.

Design:
  * base vocab = 256 raw bytes  => byte fallback => lossless on ANY UTF-8.
  * Unicode-aware pre-tokenization with stdlib `re` (\\w matches Devanagari);
    merges never cross chunk boundaries (GPT-2 style).
  * incremental pair-count updates for a fast trainer.

Output: bpe.model (JSON) next to this file. tokenizer.load() reads it.

    python train_tokenizer.py --data ../data/train_corpus.txt --vocab 4096
"""
import argparse
import collections
import json
import os
import re
import time

# Unicode-aware, no external `regex` module needed. \w covers Devanagari.
PAT = re.compile(r" ?\w+| ?[^\w\s]+|\s+", re.UNICODE)


def pretokenize(text):
    return PAT.findall(text)


def train(text, vocab_size, verbose=True):
    assert vocab_size >= 256
    n_merges = vocab_size - 256

    # word -> frequency, word as tuple of byte-ids (0..255)
    word_freq = collections.Counter()
    for chunk in pretokenize(text):
        word_freq[chunk] += 1
    words = []          # list of [ids] (mutable)
    freqs = []          # parallel freq
    for w, f in word_freq.items():
        words.append(list(w.encode("utf-8")))
        freqs.append(f)

    # pair -> total count ; pair -> set of word indices containing it
    pair_count = collections.Counter()
    pair_where = collections.defaultdict(set)

    def add_word_pairs(i):
        ids, f = words[i], freqs[i]
        for a, b in zip(ids, ids[1:]):
            pair_count[(a, b)] += f
            pair_where[(a, b)].add(i)

    for i in range(len(words)):
        add_word_pairs(i)

    merges = []              # ordered list of (a, b) -> new_id (256+k)
    vocab = {i: bytes([i]) for i in range(256)}
    t0 = time.time()

    for k in range(n_merges):
        if not pair_count:
            break
        # most frequent pair (tie-break deterministic on the pair)
        best = max(pair_count, key=lambda p: (pair_count[p], p))
        if pair_count[best] < 2:
            break
        new_id = 256 + k
        merges.append(best)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        a, b = best

        # re-encode only words that contain `best`; update counts incrementally
        for i in list(pair_where[best]):
            ids, f = words[i], freqs[i]
            # remove this word's contribution to all its pairs
            for x, y in zip(ids, ids[1:]):
                pair_count[(x, y)] -= f
                if pair_count[(x, y)] <= 0:
                    del pair_count[(x, y)]
                    pair_where[(x, y)].discard(i)
            # merge occurrences of (a,b) in this word
            merged = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == a and ids[j + 1] == b:
                    merged.append(new_id)
                    j += 2
                else:
                    merged.append(ids[j])
                    j += 1
            words[i] = merged
            # re-add contributions
            for x, y in zip(merged, merged[1:]):
                pair_count[(x, y)] += f
                pair_where[(x, y)].add(i)

        if verbose and (k + 1) % 500 == 0:
            print(f"  merge {k+1}/{n_merges}  "
                  f"last_count={pair_count.get(best, 0)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    return merges, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "bpe.model")

    text = open(args.data, encoding="utf-8").read()
    print(f"training BPE vocab={args.vocab} on {len(text):,} chars ...", flush=True)
    t0 = time.time()
    merges, vocab = train(text, args.vocab)
    # serialize: merges as list of [a,b]; vocab_size implied
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "bpe",
                   "vocab_size": 256 + len(merges),
                   "merges": merges}, f)
    print(f"done: {256+len(merges)} tokens, {len(merges)} merges "
          f"in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
