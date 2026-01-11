import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL, Rerank

_tokenizer = None
_model = None

def get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        _model.eval()
    return _tokenizer, _model

def rerank(claim, candidate_evidences):
    tokenizer, model = get_model()

    pairs = [[claim, ev] for ev in candidate_evidences]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)

        relevance_scores = probs[:, 0] + probs[:, 1]

    return sorted(
        zip(candidate_evidences, relevance_scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

