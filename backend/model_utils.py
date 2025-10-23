import torch
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd

# New 6-emotion mapping
id2label = {0: "Anger/Frustration", 1: "Fear/Anxiety", 2: "Joy/Positive", 3: "Neutral/Baseline", 4: "Sadness/Grief", 5: "Surprise/Awe"}

# Mapping from 6 emotions to 4 mood categories for songs
emotion_to_mood = {
    "Anger/Frustration": "angry",
    "Fear/Anxiety": "calm", 
    "Joy/Positive": "happy",
    "Neutral/Baseline": "calm",
    "Sadness/Grief": "sad",
    "Surprise/Awe": "calm"
}

def load_model_and_tokenizer(model_path, tokenizer_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    return model, tokenizer, device

def predict_text(model, tokenizer, device, text, max_length=96):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).cpu().item()
    emotion = id2label[pred]
    mood = emotion_to_mood[emotion]
    return emotion, mood
