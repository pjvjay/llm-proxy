"""Build latency stand-ins for the two guardrail models (no HuggingFace access
needed). Weights are random -- scores are meaningless -- but compute cost is
set by architecture + sequence length, which is what the benchmarks measure.

  toxicity-standin-onnx : RoBERTa-style, 4 layers, hidden 576, vocab 50265
                          (~38M params, the size of granite-guardian-hap-38m),
                          exported to ONNX and dynamically int8-quantized
                          (what export_onnx.py does for the real model).
  injection-standin-torch : DebertaV2ForSequenceClassification with the exact
                          deberta-v3-base config (~184M params), PyTorch backend
                          (as the README runs protectai/deberta-v3-base-...-v2).

A WordPiece tokenizer is trained on English + code text found in the local
Python installation, so chars-per-token is in the realistic range (~3.5-4.5).

    python bench/standins/make_models.py models-standin
"""
import glob
import os
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import (DebertaV2Config, DebertaV2ForSequenceClassification,
                          PreTrainedTokenizerFast, RobertaConfig,
                          RobertaForSequenceClassification)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "models-standin")


def corpus_files():
    import site
    files = []
    for sp in site.getsitepackages():
        files += glob.glob(f"{sp}/**/*.py", recursive=True)[:4000]
        files += glob.glob(f"{sp}/**/*.md", recursive=True)[:500]
        files += glob.glob(f"{sp}/**/*.txt", recursive=True)[:500]
    return [f for f in files if os.path.getsize(f) < 500_000]


def train_tokenizer(vocab=30000):
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=False)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tr = trainers.WordPieceTrainer(vocab_size=vocab,
                                   special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.train(corpus_files(), tr)
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", cls), ("[SEP]", sep)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                   model_max_length=512)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hf_tok = train_tokenizer()
    torch.manual_seed(0)

    # --- toxicity stand-in: ~38M RoBERTa-style -> ONNX int8 ----------------
    tdir = OUT / "toxicity-standin-onnx"
    tcfg = RobertaConfig(vocab_size=50265, hidden_size=576, num_hidden_layers=4,
                         num_attention_heads=12, intermediate_size=768,
                         max_position_embeddings=514, num_labels=2, pad_token_id=0)
    tmodel = RobertaForSequenceClassification(tcfg).eval()
    print("toxicity stand-in params: %.1fM" % (sum(p.numel() for p in tmodel.parameters()) / 1e6))
    tdir.mkdir(exist_ok=True)
    hf_tok.save_pretrained(tdir)
    tcfg.save_pretrained(tdir)
    ids = torch.ones(2, 16, dtype=torch.long)
    torch.onnx.export(tmodel, (ids, torch.ones_like(ids)), str(tdir / "model.onnx"),
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                    "logits": {0: "b"}}, opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(tdir / "model.onnx"), str(tdir / "model_quantized.onnx"),
                     weight_type=QuantType.QInt8)
    print("wrote", tdir)

    # --- injection stand-in: deberta-v3-base config, PyTorch ---------------
    idir = OUT / "injection-standin-torch"
    icfg = DebertaV2Config(vocab_size=128100, hidden_size=768, num_hidden_layers=12,
                           num_attention_heads=12, intermediate_size=3072, hidden_act="gelu",
                           max_position_embeddings=512, type_vocab_size=0,
                           relative_attention=True, position_buckets=256,
                           norm_rel_ebd="layer_norm", share_att_key=True,
                           pos_att_type=["p2c", "c2p"], max_relative_positions=-1,
                           position_biased_input=False, layer_norm_eps=1e-7, num_labels=2,
                           pad_token_id=0)
    imodel = DebertaV2ForSequenceClassification(icfg).eval()
    print("injection stand-in params: %.1fM" % (sum(p.numel() for p in imodel.parameters()) / 1e6))
    imodel.save_pretrained(idir)
    hf_tok.save_pretrained(idir)
    print("wrote", idir)


if __name__ == "__main__":
    main()
