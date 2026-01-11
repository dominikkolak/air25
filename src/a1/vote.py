from classify import classify_batch
from collections import Counter, defaultdict

CONFIDENCE_THRESHOLD = 0.6  # adjust as needed

def majority_vote(claim, evidences):
    if not evidences:
        return "NOT_ENOUGH_INFO", {}

    results = classify_batch(claim, evidences)

    if not results:
        return "NOT_ENOUGH_INFO", {}

    labels = [label for label, _ in results]
    vote_counts = Counter(labels)

    if not vote_counts:
        return "NOT_ENOUGH_INFO", {}

    if vote_counts.get("SUPPORTS", 0) > 0 and vote_counts.get("REFUTES", 0) > 0:
        return "DISPUTED", dict(vote_counts)

    return vote_counts.most_common(1)[0][0], dict(vote_counts)

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