import numpy as np
from sentence_transformers import SentenceTransformer
from config import TRANSFORMER

_transformer = None

# will handle the transofrmer and only make one
def get_transformer():
    global _transformer
    if _transformer is None:
        _transformer = SentenceTransformer(TRANSFORMER)
    return _transformer

# you can inject a transformer or change the default in the config, should handle it
def encode(text, transformer):
    if transformer is None:
        transformer = get_transformer()
    return transformer.encode(text, show_progress_bar=True, normalize_embeddings=True)

# Could be useful but encoding dosent really take that long
def save_embeddings(embeddings, path):
    np.save(path, embeddings)

def load_embeddings(path):
    return np.load(path)