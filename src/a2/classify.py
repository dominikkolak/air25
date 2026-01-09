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

def classify_evidence(claim, evidence_text):
    inputs = tokenizer(
        claim, 
        evidence_text, 
        return_tensors="pt", 
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction_idx = torch.argmax(probs, dim=1).item()

    raw_label = model.config.id2label.get(prediction_idx, f"LABEL_{prediction_idx}")
    
    # Convert numeric LABEL_X to semantic labels
    if raw_label.startswith('LABEL_'):
        semantic_map = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT_ENOUGH_INFO'}
        label = semantic_map.get(prediction_idx, raw_label)
    else:
        # Handle flipped config labels if present
        label_map = {'contradiction': 'SUPPORTS', 'entailment': 'REFUTES', 'neutral': 'NOT_ENOUGH_INFO'}
        label = label_map.get(raw_label, raw_label)
    confidence = probs[0][prediction_idx].item()

    return label, confidence
    
