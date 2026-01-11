import bm25s as bm25

def get_top_k_bm25(claim, corpus, retriever, stemmer, k=10):
    # Disable show progress, because of large output + jupiter issues
    query_tokens = bm25.tokenize(claim, stemmer=stemmer, show_progress=False)
    results = retriever.retrieve(query_tokens, k = k, show_progress=False)

    # save claim with all evidences as tuple
    #docs = corpus.iloc[results.documents[0].tolist()]
    return results.documents[0], results.scores[0]