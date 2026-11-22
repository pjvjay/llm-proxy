"""Long inputs: truncate vs sliding windows, sequential vs batched, and ONNX
Runtime threading under concurrency.

The real models are not available offline, so this builds a synthetic 4-layer
transformer encoder (hidden 384, 4 heads, FFN 1536, vocab 30k -- roughly the
shape of a 38M-parameter MiniLM-class classifier such as Granite HAP) with
random weights. Absolute numbers are not the production numbers; the *ratios*
between strategies are what the benchmark is for.

Questions:
  1. What does scoring a 900 / 2000 / 4000-token tool result cost if we window
     instead of truncating, and does batching the windows into one run help?
  2. Under concurrent requests, should ORT use intra-op threads (parallel
     inside one run) or should the service run several single-threaded sessions
     in parallel (inter-request)? GUARDRAILS_WORKERS vs ORT intra_op_num_threads.

Run from repo root:  python bench/bench_windows.py
"""
import os, statistics, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

H, HEADS, FFN, LAYERS, VOCAB, MAXPOS = 384, 4, 1536, 4, 30000, 512
WINDOW, STRIDE = 512, 384
PATH = "/tmp/synthetic_encoder.onnx"

def build():
    rng = np.random.default_rng(0)
    inits, nodes = [], []
    def w(name, shape, scale=0.02):
        inits.append(numpy_helper.from_array((rng.standard_normal(shape) * scale).astype(np.float32), name)); return name
    def c(name, val):
        inits.append(numpy_helper.from_array(np.array(val), name)); return name
    w("emb", (VOCAB, H)); w("pos", (MAXPOS, H))
    nodes += [helper.make_node("Gather", ["emb", "input_ids"], ["x0"]),
              helper.make_node("Shape", ["input_ids"], ["shape"]),
              helper.make_node("Gather", ["shape", c("one", np.int64(1))], ["T"]),
              helper.make_node("Range", [c("zero", np.int64(0)), "T", c("step", np.int64(1))], ["posids"]),
              helper.make_node("Gather", ["pos", "posids"], ["posemb"]),
              helper.make_node("Add", ["x0", "posemb"], ["h0"])]
    x = "h0"
    for L in range(LAYERS):
        q, k, v, o = (w(f"wq{L}", (H, H)), w(f"wk{L}", (H, H)), w(f"wv{L}", (H, H)), w(f"wo{L}", (H, H)))
        f1, f2 = w(f"wf1{L}", (H, FFN)), w(f"wf2{L}", (FFN, H))
        nodes += [helper.make_node("MatMul", [x, q], [f"q{L}"]), helper.make_node("MatMul", [x, k], [f"k{L}"]),
                  helper.make_node("MatMul", [x, v], [f"v{L}"]),
                  helper.make_node("Transpose", [f"k{L}"], [f"kt{L}"], perm=[0, 2, 1]),
                  helper.make_node("MatMul", [f"q{L}", f"kt{L}"], [f"s{L}"]),
                  helper.make_node("Mul", [f"s{L}", c(f"sc{L}", np.float32(1 / np.sqrt(H)))], [f"ss{L}"]),
                  helper.make_node("Softmax", [f"ss{L}"], [f"p{L}"], axis=-1),
                  helper.make_node("MatMul", [f"p{L}", f"v{L}"], [f"a{L}"]),
                  helper.make_node("MatMul", [f"a{L}", o], [f"ao{L}"]),
                  helper.make_node("Add", [x, f"ao{L}"], [f"r{L}"]),
                  helper.make_node("MatMul", [f"r{L}", f1], [f"ff{L}"]), helper.make_node("Relu", [f"ff{L}"], [f"fr{L}"]),
                  helper.make_node("MatMul", [f"fr{L}", f2], [f"fo{L}"]), helper.make_node("Add", [f"r{L}", f"fo{L}"], [f"h{L+1}"])]
        x = f"h{L+1}"
    nodes += [helper.make_node("ReduceMean", [x], ["pooled"], axes=[1], keepdims=0),
              helper.make_node("MatMul", ["pooled", w("wc", (H, 2))], ["logits"])]
    g = helper.make_graph(nodes, "enc", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "T"])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["B", 2])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    onnx.save(m, PATH)

def sess(intra):
    so = ort.SessionOptions(); so.intra_op_num_threads = intra; so.inter_op_num_threads = 1
    return ort.InferenceSession(PATH, so, providers=["CPUExecutionProvider"])

def windows(n_tokens):
    if n_tokens <= WINDOW: return [(0, n_tokens)]
    out, start = [], 0
    while True:
        out.append((start, min(start + WINDOW, n_tokens)))
        if start + WINDOW >= n_tokens: return out
        start += STRIDE

def timeit(fn, n=15):
    for _ in range(3): fn()
    xs = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); xs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(xs)

def main():
    if not os.path.exists(PATH): build()
    rng = np.random.default_rng(1)
    s4 = sess(4)
    print("### one long text: truncate vs windows (4 intra-op threads, 1 request, min of 15)\n")
    print("| tokens | windows | truncate-512 ms | windows sequential ms | windows batched ms | batched / truncate |")
    print("|---|---|---|---|---|---|")
    for n in (512, 900, 2000, 4000):
        ids = rng.integers(0, VOCAB, size=(1, n), dtype=np.int64)
        ws = windows(n)
        trunc = timeit(lambda: s4.run(None, {"input_ids": ids[:, :WINDOW]}))
        seq = timeit(lambda: [s4.run(None, {"input_ids": ids[:, a:b]}) for a, b in ws])
        pad = np.zeros((len(ws), WINDOW), dtype=np.int64)
        for i, (a, b) in enumerate(ws): pad[i, :b - a] = ids[0, a:b]
        bat = timeit(lambda: s4.run(None, {"input_ids": pad}))
        print(f"| {n} | {len(ws)} | {trunc:.0f} | {seq:.0f} | {bat:.0f} | {bat / trunc:.1f}x |")

    print("\n### concurrency: ORT intra-op threads vs worker threads (512-token requests, 4 cores)\n")
    print("| config | concurrent requests | p50 ms | p95 ms | req/s |\n|---|---|---|---|---|")
    ids = rng.integers(0, VOCAB, size=(1, WINDOW), dtype=np.int64)
    for intra, workers, label in ((4, 1, "intra_op=4, 1 worker (sequential)"),
                                  (4, 4, "intra_op=4, 4 workers (current default)"),
                                  (1, 4, "intra_op=1 x 4 sessions, 4 workers"),
                                  (2, 2, "intra_op=2 x 2 sessions, 2 workers")):
        sessions = [sess(intra) for _ in range(workers)]
        pool = ThreadPoolExecutor(max_workers=workers)
        def run_one(i): return sessions[i % len(sessions)].run(None, {"input_ids": ids})
        for conc in (1, 4, 8):
            n = 48; lat = []
            def job(i):
                t0 = time.perf_counter(); run_one(i); lat.append((time.perf_counter() - t0) * 1000)
            list(pool.map(job, range(16)))  # warm
            lat.clear(); t0 = time.perf_counter()
            # submit `conc` at a time, n total
            futs = []
            for i in range(n):
                futs.append(pool.submit(job, i))
                if len(futs) >= conc:
                    for f in futs: f.result()
                    futs = []
            for f in futs: f.result()
            wall = time.perf_counter() - t0
            print(f"| {label} | {conc} | {statistics.median(lat):.0f} | {sorted(lat)[int(0.95 * len(lat)) - 1]:.0f} | {n / wall:.1f} |")
        pool.shutdown()

if __name__ == "__main__":
    main()
