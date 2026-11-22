"""Build the two spaCy pipelines used by the PII benchmarks (no network needed).

  blank-en : tokenizer only. This is what the new pii.py uses by default,
             because every configured entity is pattern/checksum based.
  sm-arch  : tok2vec + tagger + parser + ner with en_core_web_sm's
             "efficiency" architecture, RANDOMLY initialised. Weights don't
             affect compute cost for these CNN/transition models, so it is a
             faithful *latency* stand-in for en_core_web_sm (not for accuracy).
             The real en_core_web_lg default that Presidio silently loads is
             at least this expensive (same CNN + 300-d static vectors).
"""
import sys
from pathlib import Path

import spacy
from spacy.cli.init_config import init_config

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/spacy-standins")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spacy.blank("en").to_disk(OUT / "blank-en")

    cfg = init_config(lang="en", pipeline=["tagger", "parser", "ner"],
                      optimize="efficiency", gpu=False)
    nlp = spacy.util.load_model_from_config(cfg, auto_fill=True, validate=True)
    for lbl in ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "NORP", "CARDINAL"]:
        nlp.get_pipe("ner").add_label(lbl)
    for lbl in ["nsubj", "dobj", "prep", "pobj", "det", "amod", "ROOT", "punct"]:
        nlp.get_pipe("parser").add_label(lbl)
    for lbl in ["NN", "NNS", "VB", "VBZ", "JJ", "DT", "IN", "PRP", "."]:
        nlp.get_pipe("tagger").add_label(lbl)
    nlp.initialize()
    nlp.to_disk(OUT / "sm-arch")
    print("built", OUT / "blank-en", OUT / "sm-arch", nlp.pipe_names)


if __name__ == "__main__":
    main()
