import sys
sys.path.insert(0, "mitmproxy")
from policy import decide

def test_allows_when_not_triggered():
    assert decide("injection", False) == "allow"

def test_blocks_when_triggered():
    assert decide("injection", True) == "block"

def test_injection_fails_closed_when_service_down():
    # classifier unreachable -> injection must block regardless of trigger
    assert decide("injection", False, available=False) == "block"
    assert decide("pii", False, available=False) == "redact"  # pii also fail-closed (its action)

def test_toxicity_fails_open_by_default():
    assert decide("toxicity", False, available=False) == "allow"

if __name__ == "__main__":
    test_allows_when_not_triggered()
    test_blocks_when_triggered()
    test_injection_fails_closed_when_service_down()
    test_toxicity_fails_open_by_default()
    print("test_policy: OK")
