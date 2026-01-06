import numpy as np
from sentence_transformers import SentenceTransformer

def encode(text, transformer):
    return transformer.encode(text, show_progress_bar=True, normalize_embeddings=True)