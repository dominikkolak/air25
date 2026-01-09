import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"{base_dir}")
LOCAL_MODEL_PATH = os.path.join(base_dir,"climatebert_local")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
model.eval()
print("Loading model labels:", model.config.id2label)

def climate_rerank(claim, candidate_evidence):
    """
    Reranking list of evidence strings based on claim using ClimateBERT
    """
    pairs = [[claim, ev] for ev in candidate_evidence]

    inputs = tokenizer(pairs, padding = True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        scores = probs[:, 1].tolist()

    reranked = sorted(zip(candidate_evidence, scores), key=lambda x: x[1], reverse=True)
    return reranked
