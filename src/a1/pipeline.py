import pandas as pd
import bm25s as bm25
from src.a1.retrieve import get_top_k_bm25
import Stemmer

from src.config import CLAIMS_PATH, EVIDENCE_PATH
from src.a1.vote import majority_vote, weighted_vote


# try to verify or refute claims
def verify_claim(
    claim,
    evidences,
    use_weighted=True
):

    if use_weighted:
        verdict, votes = weighted_vote(claim, evidences)
    else:
        verdict, votes = majority_vote(claim, evidences)

    return {
        "verdict": verdict,
        "votes": votes,
        "evidence": evidences
    }

def run_pipeline(claims_df=None, evidences_df=None, k_retrieve=5, use_weighted=True):
    if claims_df is None:
        claims_df = pd.read_csv(CLAIMS_PATH)
    if evidences_df is None:
        evidences_df = pd.read_csv(EVIDENCE_PATH)

    # stem evidences
    stemmer = Stemmer.Stemmer("english")
    corpus_stemmed = bm25.tokenize(evidences_df["evidence"].values, stopwords="en", stemmer=stemmer)

    results = []
    retriever = bm25.BM25()
    retriever.index(corpus_stemmed)
    for idx, (i, row) in enumerate(claims_df.iterrows()):
        claim = row["claim"]

        indices, scores = get_top_k_bm25(claim, corpus_stemmed, retriever,stemmer, k=k_retrieve)
        candidates = [evidences_df.iloc[j]["evidence"] for j in indices]

        result = verify_claim(claim, candidates, use_weighted)
        result["claim_id"] = row.get("claim_id", i)
        result["claim"] = claim
        results.append(result)

    return results