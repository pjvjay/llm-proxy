"""Declared cost model for the controlled benchmarks. Every number is either a
measurement reported in the repo README (by its author, on their hardware) or
an explicit assumption. Change them here; the benchmark composes them through
the real proxy code, so conclusions that depend on a particular value are easy
to re-check -- run.py --sweep shows how the key comparisons move."""

COSTS = {
    # guardrails service, per 512-token window
    "injection_window_ms": 130.0,   # README: DeBERTa-v3 PyTorch, p50 (measured)
    "toxicity_window_ms": 33.0,     # README: Granite HAP 38M int8 ONNX, p50 (measured)
    "s1_window_ms": 33.0,           # ASSUMPTION: one multi-label System-1 decision model in the
                                    # same size class as Granite HAP -> one pass for all labels
    # network
    "hop_ms": 1.0,                  # ASSUMPTION: keep-alive HTTP hop on a container network
    # System 2
    "s2_ms": 400.0,                 # ASSUMPTION: ~2B LLM judge on GPU, prefill + 1 token (vLLM)
    # upstream model
    "upstream_ms": 1200.0,          # ASSUMPTION: upstream chat completion latency
}
WINDOW_CHARS = 2048                 # ~512 tokens at ~4 chars/token
MAX_WINDOWS = 8
