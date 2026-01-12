import os
from enum import Enum

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

EVIDENCE_PATH = os.path.join(DATA_DIR, "processed/evidences.csv")
CLAIMS_PATH = os.path.join(DATA_DIR, "processed/claims.csv")
MAPPINGS_PATH = os.path.join(DATA_DIR, "processed/mappings.csv")
EVIDENCE_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings/evidences_embeddings.npy")
CLAIMS_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings/claims_embeddings.npy")

MODEL = os.path.join(BASE_DIR, "a2/climatebert_local")
TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"

LABEL_MAP = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO"}

class Rerank:
    RELEVANCE = "REL"
    REFUTES = "REF"