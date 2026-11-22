"""Sliding-window scoring in the guardrails service. Uses a tiny real
tokenizer and a fake model that flags any window containing the token EVIL."""
import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("fastapi")
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

import server  # noqa: E402


def tiny_tok():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "word": 4, "EVIL": 5}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", model_max_length=512)


class Fake(server._Backend):
    def __init__(self):
        self.tok = tiny_tok()
        self.batches = []

    def forward(self, ids, mask):
        self.batches.append(ids.shape)
        evil = (ids == 5).any(axis=1)
        return np.stack([np.where(evil, -5.0, 5.0), np.where(evil, 5.0, -5.0)], axis=1)


LONG_TAIL = "word " * 3000 + "EVIL"           # injection after ~3000 tokens


def test_truncation_misses_tail_windowing_catches_it():
    b = Fake()
    (s_trunc,), _ = b.score_texts([LONG_TAIL], 1, windowed=False)
    (s_win,), (meta,) = b.score_texts([LONG_TAIL], 1, windowed=True, max_windows=0)
    assert s_trunc < 0.1 < 0.9 < s_win
    assert meta["windows"] > 5 and not meta["truncated"]


def test_cap_keeps_head_and_tail():
    b = Fake()
    (s,), (meta,) = b.score_texts([LONG_TAIL], 1, windowed=True, max_windows=4)
    assert s > 0.9 and meta["truncated"]
    mid = "word " * 1500 + "EVIL " + "word " * 1500
    (s,), _ = b.score_texts([mid], 1, windowed=True, max_windows=4)
    assert s < 0.1                              # documented residual risk of the cap


def test_batch_of_texts_scored_independently():
    b = Fake()
    scores, meta = b.score_texts(["word word", LONG_TAIL, "EVIL"], 1, windowed=True,
                                 batch=4, max_windows=0)
    assert scores[0] < 0.1 and scores[1] > 0.9 and scores[2] > 0.9
    assert all(shape[0] <= 4 for shape in b.batches)


def test_window_overlap_covers_boundary():
    """A token at every possible position is inside at least one window."""
    b = Fake()
    for pos in range(0, 1400, 97):
        text = "word " * pos + "EVIL " + "word " * (1400 - pos)
        (s,), _ = b.score_texts([text], 1, windowed=True, max_windows=0)
        assert s > 0.9, pos


@pytest.mark.parametrize("n,cap,expect", [(3, 8, [0, 1, 2]), (10, 4, [0, 1, 8, 9]),
                                          (10, 5, [0, 1, 2, 8, 9]), (10, 0, list(range(10)))])
def test_select_windows(n, cap, expect):
    assert server.select_windows(n, cap) == expect


# ------------------------------------------------------------- FairGate
import threading  # noqa: E402
import time  # noqa: E402


def test_fair_gate_is_fifo():
    g = server.FairGate(1)
    order, started = [], []
    g.__enter__()                                   # hold the only slot

    def worker(i):
        started.append(i)
        with g:
            order.append(i)
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
        while len(started) <= i or len(g._waiters) <= i:   # enqueue strictly in order
            time.sleep(0.001)
    g.__exit__(None, None, None)
    for t in threads:
        t.join(5)
    assert order == [0, 1, 2, 3, 4]


def test_fair_gate_prevents_barging():
    """A thread looping acquire/release (a long text's windows) must let an
    already-waiting short request in on its next release."""
    g = server.FairGate(1)
    log = []
    go = threading.Event()

    def long_job():
        for k in range(50):
            with g:
                log.append("L")
                if k == 0:
                    go.set()
                time.sleep(0.002)

    def short_job():
        go.wait()
        with g:
            log.append("S")
    tl, ts = threading.Thread(target=long_job), threading.Thread(target=short_job)
    tl.start(); ts.start(); tl.join(10); ts.join(10)
    assert log.index("S") <= 3, log[:10]


# ------------------------------------------------------------- deadlines
def test_gate_waiter_times_out_and_leaves_queue():
    g = server.FairGate(1)
    assert g.acquire()
    t0 = time.perf_counter()
    assert g.acquire(timeout=0.05) is False
    assert time.perf_counter() - t0 < 1.0
    assert len(g._waiters) == 0               # no ghost waiter left behind
    g.release()
    assert g.acquire(timeout=0.05) is True    # slot was not lost


def test_expired_work_is_dropped_not_computed():
    b = Fake()
    with pytest.raises(server.DeadlineExceeded):
        b.score_texts([LONG_TAIL], 1, windowed=True, max_windows=0,
                      deadline=time.monotonic() - 1)
    assert b.batches == []                    # no forward pass ran


def test_endpoint_returns_504_on_deadline(monkeypatch):
    from fastapi.testclient import TestClient

    class Slow(Fake):
        def forward(self, ids, mask):
            time.sleep(0.2)
            return super().forward(ids, mask)
    slow = Slow()
    slow.gate = server.FairGate(1)
    monkeypatch.setattr(server, "_backends", {"injection": slow})
    monkeypatch.setattr(server.app.router, "on_startup", [])
    with TestClient(server.app) as c:
        r = c.post("/v1/classify", json={"items": [{"text": LONG_TAIL, "checks": ["injection"]}],
                                         "deadline_ms": 100})
        assert r.status_code == 504
        r = c.post("/v1/classify", json={"items": [{"text": "word EVIL", "checks": ["injection"]}]})
        assert r.status_code == 200 and r.json()["items"][0]["injection"]["score"] > 0.9
