import torch
from rerank import get_model
from config import LABEL_MAP

def classify(claim, evidence):
    tokenizer, model = get_model()

    inputs = tokenizer(
        claim, 
        evidence,
        return_tensors="pt", 
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    return LABEL_MAP.get(pred_idx, f"LABEL_{pred_idx}"), probs[0][pred_idx].item()

def classify_batch(claim, evidences):
    if not evidences:
        return []

    tokenizer, model = get_model()

    pairs = [[claim, ev] for ev in evidences]

    inputs = tokenizer(
        pairs,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confs = torch.max(probs, dim=1).values

    return [(LABEL_MAP.get(p.item(), f"LABEL_{p.item()}"), c.item()) for p, c in zip(preds, confs)]

    