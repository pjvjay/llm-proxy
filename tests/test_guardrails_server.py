"""Shape / windowing tests for guardrails/server.py with a dummy backend (no models)."""
import sys
sys.path.insert(0, "guardrails")
from fastapi.testclient import TestClient
import server

class Dummy:
    def score(self, text, positive):
        return (0.9 if "bad" in text else 0.1), 1

def test_select_windows_caps_and_keeps_ends():
    assert server._select_windows(3) == [0, 1, 2]
    sel = server._select_windows(40)
    assert len(sel) == server.MAX_WINDOWS and sel[0] == 0 and sel[-1] == 39

def test_batch_and_legacy_shapes():
    server._backends.clear()
    server._backends.update({"injection": Dummy(), "toxicity": Dummy()})
    c = TestClient(server.app)
    r = c.post("/v1/classify", json={"texts": ["good", "bad"], "checks": ["injection"]}).json()
    assert [x["injection"]["score"] for x in r["results"]] == [0.1, 0.9]
    assert "toxicity" not in r["results"][0]
    r = c.post("/v1/classify", json={"text": "bad"}).json()          # legacy single-text shape
    assert r["results"]["injection"]["score"] == 0.9 and r["results"]["toxicity"]["windows"] == 1
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["workers"] * h["intra_op_threads"] <= max(4, (server._CORES))

if __name__ == "__main__":
    test_select_windows_caps_and_keeps_ends(); test_batch_and_legacy_shapes()
    print("test_guardrails_server: OK")
