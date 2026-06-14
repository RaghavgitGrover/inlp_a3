import numpy as np

def levenshtein(s1, s2):
    if len(s1) < len(s2): s1, s2 = s2, s1
    prev = list(range(len(s2)+1))
    for c1 in s1:
        curr = [prev[0]+1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j]+(c1!=c2), curr[-1]+1, prev[j+1]+1))
        prev = curr
    return prev[-1]

def char_acc(p, r):
    if not r: return 1.0
    n = min(len(p), len(r))
    return sum(a==b for a,b in zip(p[:n],r[:n])) / len(r)

def word_acc(p, r):
    pw, rw = p.split(), r.split()
    if not rw: return 1.0
    n = min(len(pw), len(rw))
    return sum(a==b for a,b in zip(pw[:n],rw[:n])) / len(rw)

def calc_metrics(preds, refs):
    return {
        "char_accuracy"       : float(np.mean([char_acc(p,r)    for p,r in zip(preds,refs)])),
        "word_accuracy"       : float(np.mean([word_acc(p,r)    for p,r in zip(preds,refs)])),
        "levenshtein_distance": float(np.mean([levenshtein(p,r) for p,r in zip(preds,refs)])),
    }