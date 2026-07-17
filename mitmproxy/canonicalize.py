"""Defeat trivial evasion before any matching runs: NFKC-normalize homoglyphs,
strip zero-width characters, and decode base64 blobs so their payload is
rescanned. Returns (canonical_text, [decoded_segments])."""
import base64
import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_B64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")

def canonicalize(text: str):
    canon = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    decoded = []
    for m in _B64.finditer(canon):
        s = m.group(0)
        try:
            raw = base64.b64decode(s + "=" * (-len(s) % 4))
            dec = raw.decode("utf-8")
        except Exception:
            continue
        if not dec:
            continue
        printable = sum(c.isprintable() or c.isspace() for c in dec) / len(dec)
        if printable > 0.8 and re.search(r"[A-Za-z]{3,}", dec):
            decoded.append(dec)
    return canon, decoded
