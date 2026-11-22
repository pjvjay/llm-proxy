"""Validate compose files the way `docker compose config` would, without Docker:
interpolate ${VAR} / ${VAR:-default} / $$ first (Compose does this before
validation), then parse with the spec-compliant `compose_spec` parser
(pip install compose-spec), then check cross-references.

    python scripts/validate_compose.py docker-compose.yml [override.yml ...]
"""
import re
import sys

import yaml
from compose_spec import PyCompose

_VAR = re.compile(r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def interpolate(text: str, env: dict) -> str:
    def sub(m):
        if m.group(0) == "$$":
            return "$"
        name = m.group(1) or m.group(3)
        default = m.group(2)
        val = env.get(name)
        if val in (None, "") and default is not None:
            return default
        return val or ""
    return _VAR.sub(sub, text)


class _Loader(yaml.SafeLoader):
    pass


_RESET = object()
# Compose-specific YAML tags used in override files (Compose >= 2.24)
_Loader.add_constructor("!reset", lambda loader, node: _RESET)


def merge(base, over):
    """Compose merge semantics, simplified: mappings merge recursively, `!reset`
    removes the key, scalars and sequences from the override replace (Compose
    appends some sequences, e.g. ports -- replacing is the stricter check here)."""
    if not isinstance(base, dict) or not isinstance(over, dict):
        return over
    out = dict(base)
    for k, v in over.items():
        if v is _RESET:
            out.pop(k, None)
        elif k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def _full_check(data: dict) -> None:
    PyCompose.from_yaml(yaml.safe_dump(data))
    svcs = data["services"]
    for name, s in svcs.items():
        assert not (s.get("network_mode") and s.get("networks")), f"{name}: network_mode with networks"
        assert not (s.get("network_mode") == "host" and s.get("ports")), f"{name}: host network with ports"
        for dep in (s.get("depends_on") or {}):
            assert dep in svcs, f"{name}: depends_on unknown service {dep}"
        for net in s.get("networks") or []:
            assert net in (data.get("networks") or {}), f"{name}: unknown network {net}"
        for v in s.get("volumes") or []:
            src = v.split(":")[0] if isinstance(v, str) else v.get("source", "")
            if src and not src.startswith((".", "/")):
                assert src in (data.get("volumes") or {}), f"{name}: unknown volume {src}"


def load(path: str, env: dict) -> dict:
    return yaml.load(interpolate(open(path).read(), env), Loader=_Loader)


if __name__ == "__main__":
    files = sys.argv[1:] or ["docker-compose.yml"]
    for env in ({}, {"UPSTREAM_URL": "http://llm:8000", "VLLM_TAG": "v0.6.6", "S2_MODEL": "x/y"}):
        data = load(files[0], env)
        _full_check(data)
        for f in files[1:]:
            over = load(f, env)
            for name in (over.get("services") or {}):
                assert name in data["services"], f"{f}: override targets unknown service {name}"
            data = merge(data, over)
            _full_check(data)                       # the MERGED config must be valid too
    print("OK:", " + ".join(files))
