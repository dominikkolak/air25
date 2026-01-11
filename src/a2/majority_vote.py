from collections import Counter, defaultdict
from classify import classify_batch

CONFIDENCE_THRESHOLD = 0.6  # adjust as needed

def majority_vote(claim, evidences, threshold=CONFIDENCE_THRESHOLD):
    """
    Simple majority vote with confidence threshold.
    If no label exceeds threshold, predict NOT_ENOUGH_INFO.
    """
    if not evidences:
        return "NOT_ENOUGH_INFO", {}

    results = classify_batch(claim, evidences)
    if not results:
        return "NOT_ENOUGH_INFO", {}

    vote_counts = Counter(label for label, conf in results)

    # Remove NEI unless it's the only option
    if vote_counts.get("SUPPORTS", 0) + vote_counts.get("REFUTES", 0) > 0:
        vote_counts.pop("NOT_ENOUGH_INFO", None)

    # Check if top label has enough confidence
    label_conf = defaultdict(float)
    for label, conf in results:
        label_conf[label] += conf

    top_label = max(label_conf, key=label_conf.get)
    if label_conf[top_label] / sum(label_conf.values()) < threshold:
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