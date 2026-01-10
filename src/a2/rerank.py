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

def rerank(claim, candidate_evidences, method=Rerank.REFUTES):
    tokenizer, model = get_model()

    pairs = [[claim, ev] for ev in candidate_evidences]
    inputs = tokenizer(pairs, padding = True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)

        match method:
            case Rerank.RELEVANCE:
                scores = torch.max(probs[:, :2], dim=1).values.tolist()

            case Rerank.REFUTES:
                scores = probs[:, 1].tolist()

            case _:
                raise ValueError(f"INVALID RERANK METHODE")


    return sorted(zip(candidate_evidences, scores), key=lambda x: x[1], reverse=True)
