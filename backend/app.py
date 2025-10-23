#file:///C:/Users/tanis/Downloads/MoodMate/frontend/index.html
#python -m uvicorn app:app --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model_utils import load_model_and_tokenizer, predict_text
from spotify_utils import create_spotify_client, get_track_details, load_mood_csv, get_random_songs_by_emotion
import random

MODEL_PATH = "best_bert_emotion_model.pth"
TOKENIZER_DIR = "bert_emotion_tokenizer"
CSV_PATH = "moodify_light.csv"

CLIENT_ID = "90196bf080544bc8b1a40a989b291c03"
CLIENT_SECRET = "552a5d1183654a6f9f1cb33bee715199"

app = FastAPI()

# CORS (allow local frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all once
model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_DIR)
sp = create_spotify_client(CLIENT_ID, CLIENT_SECRET)
mood_map = load_mood_csv(CSV_PATH)


@app.get("/predict_text/")
def predict_text_api(sentence: str):
    emotion, mood = predict_text(model, tokenizer, device, sentence)
    return {"input": sentence, "predicted_emotion": emotion, "mood_category": mood}


@app.get("/recommend_songs/")
def recommend_songs_api(emotion: str, count: int = 5):
    try:
        songs = get_random_songs_by_emotion(mood_map, emotion, count)
        recommendations = []
        for uri in songs:
            track_details = get_track_details(sp, uri)
            track_details["uri"] = uri
            recommendations.append(track_details)
        return {"emotion": emotion, "recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}
