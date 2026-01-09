from collections import Counter
from classify import classify_evidence

def majority_vote(claim, top_k_evidences):
    micro_verdicts = []

    for evidence in top_k_evidences:
        label, _ = classify_evidence(claim, evidence)
        micro_verdicts.append(label)

    vote_counts = Counter(micro_verdicts)

    if vote_counts.get("SUPPORTS", 0) > 0 and vote_counts.get("REFUTES", 0) > 0:
        return "DISPUTED", dict(vote_counts)
    
    if not micro_verdicts:
        return "NOT_ENOUGH_INFO", {}

    final_verdict = vote_counts.most_common(1)[0][0]
    return final_verdict, dict(vote_counts)

def weighted_vote_verdict():
    #For handling ties
    pass