"""ONNX int8 stand-ins for the injection model (for B6 variants):

  injection-standin-onnx-base   : deberta-v3-base config   -> ONNX -> int8
  injection-standin-onnx-xsmall : deberta-v3-xsmall config -> ONNX -> int8
                                  (the Llama Prompt Guard 2 22M size class)

Graph fidelity of this export path vs PyTorch is checked in b3b (max |dlogit|
~1e-7). int8 accuracy must be validated on real weights with evals/run_eval.py.

    python bench/standins/make_injection_onnx.py /tmp/models-standin
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import b3b_injection_models as b3b  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/models-standin")
TOK_SRC = OUT / "injection-standin-torch"

for size, cfg in (("base", b3b.BASE), ("xsmall", b3b.XSMALL)):
    d = OUT / f"injection-standin-onnx-{size}"
    d.mkdir(parents=True, exist_ok=True)
    model = b3b.build(cfg)
    b3b.export(model, str(d / "model.onnx"))
    b3b.quant(str(d / "model.onnx"), str(d / "model_quantized.onnx"))
    (d / "model.onnx").unlink()
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
        shutil.copy(TOK_SRC / f, d / f)
    print("wrote", d)
