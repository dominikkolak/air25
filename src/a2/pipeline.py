import pandas as pd
from src.config import EVIDENCE_PATH, CLAIMS_PATH, EVIDENCE_EMBEDDINGS_PATH, CLAIMS_EMBEDDINGS_PATH, Rerank
from encode import encode, load_embeddings, save_embeddings, get_transformer
from retrieve import get_top_k
from rerank import rerank
from majority_vote import majority_vote, weighted_vote

def verify_claim(claim, evidence, k=5, use_weighted=True, rerank_method=Rerank.REFUTES):

    reranked = rerank(claim, evidence)
    top_k = [text for text, _ in reranked[:k]]

    if use_weighted: 
        verdict, votes = weighted_vote(claim, top_k)
    else:
        verdict, votes = majority_vote(claim, top_k)

    return {
        "verdict": verdict,
        "votes": votes,
        "evidence": reranked[:k]
    }

def load_or_encode(texts, load_path=None, save_path=None):
    if load_path:
        return load_embeddings(load_path)

    transformer = get_transformer()
    embeddings = encode(texts, transformer)
    if save_path:
        save_embeddings(embeddings, save_path)
    return embeddings

def run_pipeline(claims_df=None, evidences_df=None, load_embeddings=True, save_embeddings=False, k_retrieve=10, k_vote=10, rerank_method=Rerank.REFUTES, use_weighted=False):
    if claims_df is None:
        claims_df = pd.read_csv(CLAIMS_PATH)
    if evidences_df is None:
        evidences_df = pd.read_csv(EVIDENCE_PATH)

    vec_e = load_or_encode(evidences_df["evidence"].tolist(), load_path=EVIDENCE_EMBEDDINGS_PATH if load_embeddings else None, save_path=EVIDENCE_EMBEDDINGS_PATH if save_embeddings else None)
    vec_c = load_or_encode(claims_df["claim"].tolist(), load_path=CLAIMS_EMBEDDINGS_PATH if load_embeddings else None, save_path=CLAIMS_EMBEDDINGS_PATH if save_embeddings else None)

    results = []
    for idx, (i, row) in enumerate(claims_df.iterrows()):
        claim = row["claim"]

        indices, scores = get_top_k(vec_c[idx], vec_e, k=k_retrieve)
        candidates = [evidences_df.iloc[j]["evidence"] for j in indices]

        result = verify_claim(claim, candidates, k=k_vote, use_weighted=use_weighted, rerank_method=rerank_method)
        result["claim_id"] = row.get("claim_id", i)
        result["claim"] = claim
        results.append(result)

    return results