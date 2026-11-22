"""SHA-256 manifests for model artifacts (OWASP LLM03: Supply Chain).

Format is `sha256sum` compatible -- `cd models && sha256sum -c MANIFEST.sha256`
works for operators too:

    <64 hex>  injection/model.safetensors

prepare_models.py writes /models/MANIFEST.sha256 for what it downloads. A
System-1 decision model (e.g. an OpenJev export) should ship its own
models/decision/MANIFEST.sha256 from wherever it was built and signed -- the
guardrails service verifies every manifest it finds before loading anything,
and refuses to start on a mismatch. Pure stdlib."""
import hashlib
import os

NAME = "MANIFEST.sha256"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(root: str, subdirs, name: str = NAME) -> str:
    """Hash every file under root/<subdir> and write root/<name>."""
    lines = []
    for sub in subdirs:
        base = os.path.join(root, sub)
        for dirpath, _, files in sorted(os.walk(base)):
            for fn in sorted(files):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)
                lines.append(f"{sha256_file(full)}  {rel}")
    path = os.path.join(root, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return path


def verify(manifest_path: str) -> tuple[str, list[str]]:
    """('verified' | 'missing' | 'mismatch', problems). Paths are relative to the
    manifest's directory."""
    if not os.path.exists(manifest_path):
        return "missing", [f"{manifest_path} not found"]
    base = os.path.dirname(manifest_path)
    problems = []
    with open(manifest_path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                digest, rel = line.split(None, 1)
            except ValueError:
                problems.append(f"line {n}: malformed")
                continue
            rel = rel.lstrip("*").strip()
            full = os.path.normpath(os.path.join(base, rel))
            if not full.startswith(os.path.normpath(base) + os.sep):
                problems.append(f"{rel}: path escapes manifest directory")
            elif not os.path.exists(full):
                problems.append(f"{rel}: missing")
            elif sha256_file(full) != digest.lower():
                problems.append(f"{rel}: sha256 mismatch")
    return ("mismatch" if problems else "verified"), problems


def verify_tree(root: str, subdirs=("", "decision")) -> dict:
    """Verify every manifest under root (root itself and the decision-model
    slot). Returns {manifest_path: (status, problems)} for manifests that exist."""
    out = {}
    for sub in subdirs:
        p = os.path.join(root, sub, NAME)
        if os.path.exists(p):
            out[p] = verify(p)
    return out
