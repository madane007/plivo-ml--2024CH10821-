"""Byte-level BPE tokenizer (falls back to raw bytes, so it is lossless on
ANY UTF-8 text). Trained on train_corpus.txt only, by train_tokenizer.py.

Interface kept identical to the baseline:
  load() -> object with .encode(str)->list[int], .decode(list[int])->str,
  and .vocab_size. load() takes NO required args and resolves its model file
  relative to __file__, so grading (cwd = submission folder, no internet) works.

If bpe.model is missing it degrades to the raw byte tokenizer (vocab 256).
"""
import json
import os
import re

# Unicode-aware pre-tokenization; \w matches Devanagari. MUST match the regex
# used in train_tokenizer.py so encode() reproduces the trained merges.
PAT = re.compile(r" ?\w+| ?[^\w\s]+|\s+", re.UNICODE)


def pretokenize(text):
    return PAT.findall(text)


class ByteTokenizer:
    """Raw UTF-8 bytes, vocab 256. Lossless fallback."""
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


class BPETokenizer:
    def __init__(self, merges):
        # merges: ordered list of [a, b]; merge k produces id 256+k.
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = 256 + len(self.merges)
        # rank lookup for encoding: pair -> merge index
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        # id -> bytes, for decoding
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache = {}   # chunk(str) -> list[int]

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        # repeatedly merge the lowest-rank adjacent pair present
        while len(ids) >= 2:
            best_rank, best_i = None, None
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            ids[best_i:best_i + 2] = [256 + best_rank]
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        for chunk in pretokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        parts = bytearray()
        for i in ids:
            parts += self.vocab.get(i, b"")
        return parts.decode("utf-8", errors="replace")


def load(path=None):
    """Return the tokenizer used by train.py / evaluate.py."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bpe.model")
    if not os.path.exists(path):
        return ByteTokenizer()
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    if spec.get("type") == "bpe":
        return BPETokenizer(spec["merges"])
    return ByteTokenizer()
