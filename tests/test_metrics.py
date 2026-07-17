import sys
sys.path.insert(0, "evals")
from metrics import confusion, scores

def test_confusion_and_scores():
    preds  = [1, 1, 0, 0]
    labels = [1, 0, 0, 1]          # tp=1, fp=1, tn=1, fn=1
    tp, fp, tn, fn = confusion(preds, labels)
    assert (tp, fp, tn, fn) == (1, 1, 1, 1)
    s = scores(tp, fp, tn, fn)
    assert abs(s["precision"] - 0.5) < 1e-9
    assert abs(s["recall"] - 0.5) < 1e-9
    assert abs(s["fpr"] - 0.5) < 1e-9
    assert abs(s["f1"] - 0.5) < 1e-9

def test_perfect_classifier():
    tp, fp, tn, fn = confusion([1, 0], [1, 0])
    s = scores(tp, fp, tn, fn)
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["fpr"] == 0.0

if __name__ == "__main__":
    test_confusion_and_scores()
    test_perfect_classifier()
    print("test_metrics: OK")
