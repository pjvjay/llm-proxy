"""Confusion matrix + precision / recall / FPR / F1. Pure stdlib so the eval
report is trivially reproducible."""

def confusion(preds, labels):
    tp = fp = tn = fn = 0
    for p, y in zip(preds, labels):
        if p and y:            tp += 1
        elif p and not y:      fp += 1
        elif not p and not y:  tn += 1
        else:                  fn += 1
    return tp, fp, tn, fn

def scores(tp, fp, tn, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "fpr": fpr, "f1": f1}
