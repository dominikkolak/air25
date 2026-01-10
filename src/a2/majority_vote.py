from collections import Counter
from classify import classify_batch

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

# placeholder!!!!
def weighted_vote(claim, evidences):
    if not evidences:
        return "NOT_ENOUGH_INFO", {}

    results = classify_batch(claim, evidences)

    if not results:
        return "NOT_ENOUGH_INFO", {}

    return None