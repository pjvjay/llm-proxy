"""Deterministic checks for OWASP LLM07 (System Prompt Leakage) and LLM05
(Improper Output Handling). Pure stdlib, microsecond-scale; imported by the
addon and by evals/run_eval.py.

LLM07
  extraction_attempt(text)   request-side: does the message ask for the system
                             prompt / hidden instructions? Used as a TAG on top
                             of the injection decision, never to block on its
                             own ("what is a system prompt?" is a fair question).
  system_prompt_leak(s, r)   response-side: does the response reproduce the
                             system prompt verbatim? Word-shingle overlap, so
                             reformatting and partial quotes are still caught.

LLM05
  scan_output(text)          active markup (script, event handlers, javascript:
                             URIs, iframe/object/embed) and exfiltration-shaped
                             links: markdown images / links whose URL carries a
                             query string to a domain not on OUTPUT_URL_ALLOW_DOMAINS
                             -- the classic "render this image" data-exfil channel.
  neutralise(text, hits)     replace each hit with a visible marker.
"""
import os
import re
from urllib.parse import urlparse

# ------------------------------------------------------------------ LLM07
_ADJ = r"(?:system|initial|original|hidden|secret|developer|full|exact|entire|underlying)"
_NOUN = r"(?:prompt|instructions?|rules|directives?|system\s+message)"
_EXTRACTION = re.compile(
    r"\b(print|reveal|show|repeat|output|display|dump|leak|tell\s+me|give\s+me|what\s+(?:is|are|was|were))\b"
    r"[^.?!\n]{0,40}?\b(?:"
    rf"your\s+(?:{_ADJ}\s+)*{_NOUN}"                 # "your rules", "your system prompt"
    rf"|the\s+(?:{_ADJ}\s+)+{_NOUN}"                 # "the initial instructions" -- not "the rules of chess"
    r"|the\s+(?:prompt|system\s+message)"            # unambiguous nouns
    r")\b", re.I)
_WORD = re.compile(r"[a-z0-9]+")


def extraction_attempt(text: str) -> bool:
    return bool(_EXTRACTION.search(text or ""))


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def system_prompt_leak(system_text: str, response_text: str, k: int = 8,
                       min_hits: int = 2, min_fraction: float = 0.3) -> tuple[bool, float]:
    """(leaked?, fraction of the system prompt's k-word shingles found in the response).

    Short system prompts (< k words) are matched as a whole normalised phrase
    (only if >= 4 words, to avoid flagging trivial prompts like 'Be helpful')."""
    sw, rw = _words(system_text), _words(response_text)
    if not sw or not rw:
        return False, 0.0
    if len(sw) < k:
        if len(sw) < 4:
            return False, 0.0
        hit = " ".join(sw) in " ".join(rw)
        return hit, 1.0 if hit else 0.0
    sys_sh = {tuple(sw[i:i + k]) for i in range(len(sw) - k + 1)}
    resp_sh = {tuple(rw[i:i + k]) for i in range(len(rw) - k + 1)}
    hits = len(sys_sh & resp_sh)
    frac = hits / len(sys_sh)
    return hits >= min_hits and (frac >= min_fraction or hits >= 5), frac


# ------------------------------------------------------------------ LLM05
_ALLOW = {d.strip().lower() for d in os.getenv("OUTPUT_URL_ALLOW_DOMAINS", "").split(",") if d.strip()}
# How the caller renders model output. "markdown" (chat UIs): code spans and
# fenced blocks are displayed as literal text, so markup inside them is inert.
# "html": output is inserted as HTML, so nothing is exempt.
RENDERER = os.getenv("OUTPUT_RENDERER", "markdown")
_CODE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]+`", re.S)

_MARKUP_RULES = [
    ("script_tag",    re.compile(r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>|<\s*script\b[^>]*>", re.I | re.S)),
    ("event_handler", re.compile(r"<[^>]*\son[a-z]+\s*=\s*[\"']?[^>]*>", re.I)),
    ("js_uri",        re.compile(r"\bjavascript\s*:", re.I)),
    ("embed_tag",     re.compile(r"<\s*(iframe|object|embed)\b[^>]*>", re.I)),
]
_MD_LINK = re.compile(r"(!?)\[([^\]]*)\]\((\s*https?://[^)\s]+)\s*\)", re.I)


def _allowed(url: str) -> bool:
    host = (urlparse(url.strip()).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in _ALLOW)


def scan_output(text: str, renderer: str | None = None) -> list[tuple[str, int, int]]:
    """[(rule, start, end)] -- rule is 'output_markup:<name>' or 'output_exfiltration'."""
    hits = []
    if not text:
        return hits
    code = [] if (renderer or RENDERER) == "html" else [(m.start(), m.end()) for m in _CODE.finditer(text)]

    def inert(s, e):
        return any(cs <= s and e <= ce for cs, ce in code)

    for name, rx in _MARKUP_RULES:
        for m in rx.finditer(text):
            if not inert(m.start(), m.end()):
                hits.append((f"output_markup:{name}", m.start(), m.end()))
    for m in _MD_LINK.finditer(text):
        is_image, url = m.group(1) == "!", m.group(3)
        if "?" in url and not _allowed(url) and (is_image or len(urlparse(url.strip()).query) > 40) \
                and not inert(m.start(), m.end()):
            hits.append(("output_exfiltration", m.start(), m.end()))
    hits.sort(key=lambda h: h[1])
    merged = []                                   # drop overlaps, keep the first
    for h in hits:
        if merged and h[1] < merged[-1][2]:
            continue
        merged.append(h)
    return merged


def neutralise(text: str, hits) -> str:
    out, last = [], 0
    for rule, s, e in hits:
        out.append(text[last:s])
        out.append(f"[removed by LLM firewall: {rule.split(':')[-1]}]")
        last = e
    out.append(text[last:])
    return "".join(out)


def check_kind(rule: str) -> str:
    """Map a scan_output rule to an owasp.DEFAULT_MAP key."""
    return "output_exfiltration" if rule == "output_exfiltration" else "output_markup"
