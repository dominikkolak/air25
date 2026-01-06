import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k(vec_c, vec_e, k=10):
    vec_c = vec_c.reshape(1, -1)

    scores = cosine_similarity(vec_c, vec_e)
    scores = scores.flatten()

    sorted_indices = np.argsort(scores)
    sorted_indices = np.flip(sorted_indices)

    top_k_indices = sorted_indices[:k]
    top_k_scores = scores[top_k_indices]

    return top_k_indices, top_k_scores