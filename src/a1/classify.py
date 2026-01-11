from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

_tokenizer = None
_model = None

def get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        _model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        _model.eval()
    return _tokenizer, _model

def classify_batch(claim, candidate_evidences):
    tokenizer, model = get_model()

    # claim length must be same size as evidence length
    # create evidence-claim pairs
    claims = [claim] * len(candidate_evidences)
    inputs = tokenizer(
        candidate_evidences,
        claims,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # disable gradient calculation (not needed)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confs = torch.max(probs, dim=1).values

    return [({2: "SUPPORTS", 0: "REFUTES", 1: "NOT_ENOUGH_INFO"}.get(p.item(), f"LABEL_{p.item()}"), c.item()) for p, c in zip(preds, confs)]