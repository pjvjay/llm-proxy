"""Stage 1: fast keyword pre-filter. Blocks obvious cases in <1ms before
touching the classifier service. Deliberately narrow -- the ML stages catch
the rest; these are just the cheap, unambiguous wins."""
import re

CATEGORIES = {
    "violent": re.compile(r"\b(kill|murder|stab|behead|massacre|torture)\b", re.I),
    "illegal": re.compile(r"\b(synthesi[sz]e\s+(meth|methamphetamine)|make\s+a\s+bomb|launder\s+money|counterfeit\s+currency)\b", re.I),
    "sexual":  re.compile(r"\b(explicit\s+sexual|pornograph\w*)\b", re.I),
}

def match(text: str):
    for name, rx in CATEGORIES.items():
        if rx.search(text):
            return name
    return None
