"""B4 -- incremental scanning machinery.

Q1  cache-key hash: sha256 (chosen) | blake2b-128 | xxh3_128 (non-cryptographic)
Q2  LRU implementation: OrderedDict (chosen) | cachetools.LRUCache | dict-FIFO
Q3  per-turn bookkeeping cost of session.scan_units() (hash every message of
    the history) as an agent conversation grows -- the residual O(N) work the
    incremental design still does each turn, vs. what it saves.

    python bench/b4_cache.py
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict

import cachetools
import xxhash

import common
import corpus
import session


def h_sha256(b): return hashlib.sha256(b).digest()
def h_blake2b(b): return hashlib.blake2b(b, digest_size=16).digest()
def h_xxh3(b): return xxhash.xxh3_128_digest(b)


class FifoDict:
    def __init__(self, n): self.d, self.n = {}, n
    def get(self, k): return self.d.get(k)
    def put(self, k, v):
        self.d[k] = v
        if len(self.d) > self.n:
            self.d.pop(next(iter(self.d)))


class OD:
    def __init__(self, n): self.d, self.n = OrderedDict(), n
    def get(self, k):
        v = self.d.get(k)
        if v is not None: self.d.move_to_end(k)
        return v
    def put(self, k, v):
        self.d[k] = v; self.d.move_to_end(k)
        if len(self.d) > self.n: self.d.popitem(last=False)


class CT:
    def __init__(self, n): self.c = cachetools.LRUCache(n)
    def get(self, k): return self.c.get(k)
    def put(self, k, v): self.c[k] = v


def main():
    out = {"q1": [], "q2": [], "q3": []}
    for size in (300, 3_000, 30_000):
        b = os.urandom(size)
        for name, fn in (("sha256", h_sha256), ("blake2b-128", h_blake2b), ("xxh3_128", h_xxh3)):
            s = common.timeit_ms(lambda: fn(b), repeat=2000, warmup=50)
            out["q1"].append({"bytes": size, "hash": name, "p50_us": s["p50_ms"] * 1000})
            print(f"hash {name:12s} {size:6d}B p50={s['p50_ms']*1000:7.2f}us")

    keys = [os.urandom(16) for _ in range(50_000)]
    for name, cls in (("OrderedDict (chosen)", OD), ("cachetools.LRUCache", CT), ("dict FIFO", FifoDict)):
        c = cls(20_000)
        def ops():
            for i in range(0, 20_000):
                k = keys[(i * 7) % 30_000]
                if c.get(k) is None:
                    c.put(k, i)
        s = common.timeit_ms(ops, repeat=10, warmup=1)
        per_op_us = s["p50_ms"] * 1000 / 20_000
        out["q2"].append({"impl": name, "per_get_or_put_us": per_op_us})
        print(f"lru {name:22s} {per_op_us:.3f}us/op")

    for turn, msgs in enumerate(corpus.agent_conversation(41, tool_chars=3000)):
        if turn in (1, 10, 20, 40):
            s = common.timeit_ms(lambda: session.scan_units(msgs), repeat=200, warmup=10)
            out["q3"].append({"turn": turn, "messages": len(msgs), **s})
            print(f"scan_units turn={turn:2d} msgs={len(msgs):3d} p50={s['p50_ms']:.3f}ms")
    common.write("b4_cache", out)


if __name__ == "__main__":
    main()
