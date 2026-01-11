from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

model_id = "roberta-large-mnli"
save_path = "./src/a1/roberta-large-mnli"

print(f"Downloading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

if not os.path.exists(save_path):
    os.makedirs(save_path)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Model and tokenizer saved successfully to {save_path}")