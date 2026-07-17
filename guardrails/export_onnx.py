"""One-time model prep: export a HF sequence-classification model to ONNX and
quantize to int8. Run locally (needs network + HuggingFace auth):

    huggingface-cli login                       # accept the Llama Prompt Guard 2 license first
    python export_onnx.py --model meta-llama/Llama-Prompt-Guard-2-22M --out models/prompt-guard-2-onnx
    python export_onnx.py --model ibm-granite/granite-guardian-hap-38m --out models/granite-hap-onnx
"""
import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

def export(model_id: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)

    quantizer = ORTQuantizer.from_pretrained(out)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=out, quantization_config=qconfig)
    print(f"exported + int8-quantized -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export(args.model, args.out)
