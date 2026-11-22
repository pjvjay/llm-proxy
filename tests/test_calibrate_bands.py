import json, os, sys, tempfile
sys.path.insert(0, "evals")
from calibrate_bands import fit_band, load

def test_separable_data_needs_no_escalation():
    pts = [(0.05, 0), (0.1, 0), (0.2, 0), (0.7, 1), (0.9, 1)]
    r = fit_band(pts, max_error=0.0)
    assert r["escalated"] == 0 and r["error"] == 0 and 0.2 < r["hi"] <= 0.7

def test_overlap_is_escalated_at_zero_error():
    pts = [(0.1, 0), (0.2, 0), (0.4, 1), (0.5, 0), (0.6, 1), (0.9, 1)]
    r = fit_band(pts, max_error=0.0)
    assert (r["lo"], r["hi"]) == (0.4, 0.6) and r["escalated"] == 2 / 6
    # allowing some error shrinks the band
    assert fit_band(pts, max_error=0.25)["escalated"] < r["escalated"]

def test_escalation_log_uses_system2_as_teacher():
    recs = [{"role": "tool", "label": "injection", "s1": 0.3, "s2": 0.05, "status": "ok"},
            {"role": "tool", "label": "injection", "s1": 0.7, "s2": 0.95, "status": "ok"},
            {"role": "tool", "label": "injection", "s1": 0.5, "s2": None, "status": "shed"},
            {"role": "user", "label": "injection", "s1": 0.9, "s2": 0.1, "status": "ok"}]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(r) for r in recs)); path = f.name
    pts = load(path, role="tool")
    os.unlink(path)
    assert pts == [(0.3, 0), (0.7, 1)]            # shed record and other role dropped

if __name__ == "__main__":
    test_separable_data_needs_no_escalation(); test_overlap_is_escalated_at_zero_error()
    test_escalation_log_uses_system2_as_teacher()
    print("test_calibrate_bands: OK")
