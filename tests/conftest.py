import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("PII_WORKERS", "0")          # inline PII in tests (no process pool)
for p in (ROOT / "mitmproxy", ROOT / "guardrails", ROOT / "bench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
