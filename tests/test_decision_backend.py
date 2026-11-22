"""Real (not mocked) test of the multi-label System-1 backend in guardrails/server.py:
HF fast tokenizer -> overlapping windows -> ONE ONNX run -> calibrated per-label
probabilities. Uses a tiny synthetic tokenizer + ONNX graph built here, so no
downloads are needed. The graph scores a label high iff a trigger word for it
occurs anywhere in the window (embedding lookup + max over tokens)."""
import json, os, sys, tempfile
sys.path.insert(0, "guardrails")
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import server

VOCAB = ["[PAD]", "[UNK]", "ignore", "idiot", "the", "weather", "is", "nice", "report", "role:", "tool", "user", "|", "-", "fetch"]

class CountingSession:
    def __init__(self, sess): self.sess, self.runs = sess, 0
    def get_inputs(self): return self.sess.get_inputs()
    def run(self, *a, **k):
        self.runs += 1; return self.sess.run(*a, **k)

def build(tmp, template=None, temperature=None):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]")
    tok.save_pretrained(tmp)
    # logits[label] = max over tokens of E[token, label]; triggers: ignore->injection, idiot->toxicity
    E = np.full((len(VOCAB), 2), -6.0, dtype=np.float32)
    E[VOCAB.index("ignore"), 0] = 6.0
    E[VOCAB.index("idiot"), 1] = 6.0
    E[0] = -20.0                                    # padding never fires
    g = helper.make_graph(
        [helper.make_node("Gather", ["E", "input_ids"], ["e"]),
         helper.make_node("ReduceMax", ["e"], ["logits"], axes=[1], keepdims=0)],
        "ml", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "T"]),
               helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["B", "T"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["B", 2])],
        [numpy_helper.from_array(E, "E")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    onnx.save(m, os.path.join(tmp, server.ONNX_FILE))
    cfg = {"labels": ["injection", "toxicity"], "activation": "sigmoid"}
    if template: cfg["context_template"] = template
    if temperature: cfg["temperature"] = temperature
    json.dump(cfg, open(os.path.join(tmp, "labels.json"), "w"))
    b = server._MultiLabelOnnxBackend(tmp)
    b.sess = CountingSession(b.sess)
    return b

def test_single_pass_all_labels_and_windowing():
    server.WINDOW, server.STRIDE, server.MAX_WINDOWS = 16, 4, 8
    with tempfile.TemporaryDirectory() as tmp:
        b = build(tmp)
        scores, n, passes = b.decide("the weather is nice")
        assert n == 1 and passes == 1 and b.sess.runs == 1
        assert scores["injection"] < 0.01 and scores["toxicity"] < 0.01
        # trigger placed far past the first window: truncation would miss it
        long = " ".join(["report"] * 60 + ["ignore"] + ["report"] * 5)
        scores, n, passes = b.decide(long)
        assert n > 1 and passes == n and b.sess.runs == 2      # all windows + labels in ONE run
        assert scores["injection"] > 0.99 and scores["toxicity"] < 0.01
        # label restriction (e.g. tool results: injection only)
        scores, _, _ = b.decide("you idiot", labels=["injection"])
        assert set(scores) == {"injection"}

def test_calibration_temperature_and_context():
    server.WINDOW, server.STRIDE = 16, 4
    with tempfile.TemporaryDirectory() as tmp:
        b = build(tmp, temperature={"injection": 3.0})
        p = b.decide("ignore")[0]["injection"]
        assert abs(p - 1 / (1 + np.exp(-2.0))) < 1e-5          # logit 6 / T 3 -> sigmoid(2)
    with tempfile.TemporaryDirectory() as tmp:
        b = build(tmp, template="role: {role} | tool: {tool} ")
        enc, _ = b._encode(b.template.format(role="tool", tool="fetch") + "x", "np")
        ids = list(np.asarray(enc["input_ids"][0]))
        assert VOCAB.index("tool") in ids and VOCAB.index("fetch") in ids

if __name__ == "__main__":
    test_single_pass_all_labels_and_windowing(); test_calibration_temperature_and_context()
    print("test_decision_backend: OK")
