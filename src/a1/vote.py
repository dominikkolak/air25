from src.a1.classify import classify_batch
from collections import Counter, defaultdict

CONFIDENCE_THRESHOLD = 0.6  # adjust as needed

def majority_vote(claim, evidences, threshold=CONFIDENCE_THRESHOLD):
    if not evidences:
        return "NOT_ENOUGH_INFO", {}

    results = classify_batch(claim, evidences)
    if not results:
        return "NOT_ENOUGH_INFO", {}

    label_conf = defaultdict(float)
    for label, conf in results:
        label_conf[label] += conf

    if label_conf.get("SUPPORTS", 0) + label_conf.get("REFUTES", 0) > 0:
        label_conf.pop("NOT_ENOUGH_INFO", None)

    if not label_conf:
        return "NOT_ENOUGH_INFO", {}

    top_label = max(label_conf, key=label_conf.get)
    total = sum(label_conf.values())

    if total == 0 or label_conf[top_label] / total < threshold:
        return "NOT_ENOUGH_INFO", dict(label_conf)

    return top_label, dict(label_conf)

def weighted_vote(claim, evidences, threshold=CONFIDENCE_THRESHOLD):
    if not evidences:
        return "NOT_ENOUGH_INFO", {}

    results = classify_batch(claim, evidences)
    if not results:
        return "NOT_ENOUGH_INFO", {}

    scores = defaultdict(float)
    for label, conf in results:
        scores[label] += conf

    if scores.get("SUPPORTS", 0) + scores.get("REFUTES", 0) > 0:
        scores.pop("NOT_ENOUGH_INFO", None)

    top_label = max(scores, key=scores.get)
    total_score = sum(scores.values())
    if total_score == 0 or scores[top_label] / total_score < threshold:
        return "NOT_ENOUGH_INFO", dict(scores)

    return top_label, dict(scores)