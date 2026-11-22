"""model-init (guardrails/prepare_models.py) and supply-chain verification,
without torch or network: a synthetic tokenizer + fp32 ONNX classifier stand in
for an exported model, so quantisation, validation against reference
probabilities, discarding of bad exports, manifests, tamper detection and the
idempotent main() path all run for real."""
import json, os, sys, tempfile
sys.path.insert(0, "guardrails")
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

VOCAB = ["[PAD]", "[UNK]", "you", "are", "worthless", "idiot", "what", "is", "the", "capital", "of",
         "france", "ignore", "all", "previous", "instructions", "weather", "in", "paris"]

def tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, normalizers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]")
    tok.save_pretrained(path)
    return tok

def model(path, flip=False, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.5, (len(VOCAB), 16)).astype(np.float32)
    E[VOCAB.index("worthless")] += 2.0; E[VOCAB.index("idiot")] += 2.0
    W = np.zeros((16, 2), np.float32); W[:, 1] = 1.0; W[:, 0] = -1.0
    if flip: W = -W
    g = helper.make_graph(
        [helper.make_node("Gather", ["E", "input_ids"], ["e"]),
         helper.make_node("ReduceMean", ["e"], ["m"], axes=[1], keepdims=0),
         helper.make_node("MatMul", ["m", "W"], ["logits"])],
        "clf", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "T"]),
                helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["B", "T"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["B", 2])],
        [numpy_helper.from_array(E, "E"), numpy_helper.from_array(W, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    onnx.save(m, path)

def test_validate_keep_and_discard():
    import prepare_models as P
    from onnxruntime.quantization import QuantType, quantize_dynamic
    with tempfile.TemporaryDirectory() as d:
        tok = tokenizer(d)
        fp32, int8 = os.path.join(d, "model.onnx"), os.path.join(d, "model_quantized.onnx")
        model(fp32)
        ref = P.onnx_probs(fp32, tok, P.PROBES, 1)            # stands in for the PyTorch reference
        assert ref[2] > ref[0]                                 # toxic probe scores higher than benign
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
        rep = P.keep_validated(d, fp32, int8, ref, tok, 1)
        assert rep["model_quantized.onnx"]["kept"] and os.path.exists(int8)
        assert not os.path.exists(fp32)                        # superseded by the validated int8 model
        assert json.load(open(os.path.join(d, "export.json")))["model_quantized.onnx"]["kept"]
        # a wrong export (e.g. a bad graph trace) must be discarded, never served
        model(fp32, flip=True)
        os.remove(int8)
        rep = P.keep_validated(d, fp32, int8, ref, tok, 1)
        assert not rep["model.onnx"]["kept"] and not os.path.exists(fp32)

def test_manifest_and_tamper_detection():
    import manifest as M
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "injection"))
        open(os.path.join(d, "injection", "w.bin"), "wb").write(b"weights")
        p = M.write(d, ["injection"])
        assert M.verify(p) == ("verified", [])
        open(os.path.join(d, "injection", "w.bin"), "wb").write(b"tampered")
        st, probs = M.verify(p)
        assert st == "mismatch" and "sha256 mismatch" in probs[0]
        open(p, "a").write("0" * 64 + "  ../etc/passwd\n")
        assert any("escapes" in x for x in M.verify(p)[1])
        assert M.verify(os.path.join(d, "nope"))[0] == "missing"

def test_main_idempotent_and_fails_closed():
    with tempfile.TemporaryDirectory() as d:
        for name in ("injection", "toxicity"):
            os.makedirs(os.path.join(d, name))
            open(os.path.join(d, name, "config.json"), "w").write("{}")
            open(os.path.join(d, name, "model.safetensors"), "wb").write(name.encode())
        os.environ.update(MODELS_ROOT=d, EXPORT_ONNX="0")
        import importlib, prepare_models as P
        P = importlib.reload(P)
        assert P.main() == 0                                   # present -> no download; manifest written
        assert os.path.exists(os.path.join(d, "MANIFEST.sha256"))
        assert os.path.exists(os.path.join(d, "decision", "README.txt"))
        assert P.main() == 0                                   # second run: no-op
        open(os.path.join(d, "toxicity", "model.safetensors"), "wb").write(b"swapped")
        assert P.main() == 1                                   # tampered artifact: refuse
        os.environ["MODEL_INIT_SKIP"] = "1"
        assert importlib.reload(P).main() == 0
        for k in ("MODELS_ROOT", "EXPORT_ONNX", "MODEL_INIT_SKIP"):
            os.environ.pop(k)

def test_export_failure_falls_back_to_pytorch():
    with tempfile.TemporaryDirectory() as d:
        for name in ("injection", "toxicity"):
            os.makedirs(os.path.join(d, name))
            open(os.path.join(d, name, "config.json"), "w").write("{}")
            open(os.path.join(d, name, "model.safetensors"), "wb").write(name.encode())
        os.environ.update(MODELS_ROOT=d, EXPORT_ONNX="1")
        import importlib, prepare_models as P
        P = importlib.reload(P)
        def broken_export(spec, path):
            open(os.path.join(path, "model.onnx"), "wb").write(b"partial")   # leaves junk, then fails
            raise RuntimeError("tracer exploded")
        P.export_onnx = broken_export
        assert P.main() == 0                                    # start-up must not fail over an optimisation
        rep = json.load(open(os.path.join(d, "toxicity", "export.json")))
        assert rep["fallback"] == "pytorch" and "tracer exploded" in rep["error"]
        assert not os.path.exists(os.path.join(d, "toxicity", "model.onnx"))   # no partial file served
        assert P.main() == 0                                    # and it doesn't retry every start
        for k in ("MODELS_ROOT", "EXPORT_ONNX"):
            os.environ.pop(k)

def test_guardrails_refuses_tampered_models():
    import importlib
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "injection"))
        open(os.path.join(d, "injection", "x"), "w").write("a")
        import manifest as M
        M.write(d, ["injection"])
        os.environ["MODELS_ROOT"] = d
        import server
        server = importlib.reload(server)
        server._verify_supply_chain()
        assert server._supply_chain["status"] == "verified"
        open(os.path.join(d, "injection", "x"), "w").write("b")
        try:
            server._verify_supply_chain()
            raise AssertionError("should have refused")
        except RuntimeError as e:
            assert "LLM03" in str(e)
        os.environ.pop("MODELS_ROOT")
        importlib.reload(server)

if __name__ == "__main__":
    test_validate_keep_and_discard(); test_manifest_and_tamper_detection()
    test_main_idempotent_and_fails_closed(); test_export_failure_falls_back_to_pytorch()
    test_guardrails_refuses_tampered_models()
    print("test_prepare_models: OK")
