import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import random

# Mapping from 6 emotions to 4 mood categories for songs
EMOTION_TO_MOOD = {
    "Anger/Frustration": "angry",
    "Fear/Anxiety": "calm", 
    "Joy/Positive": "happy",
    "Neutral/Baseline": "calm",
    "Sadness/Grief": "sad",
    "Surprise/Awe": "calm"
}

def emotion_to_mood(emotion):
    """Convert 6-emotion prediction to 4-mood category for song recommendations"""
    return EMOTION_TO_MOOD.get(emotion, "calm")  # Default to calm if emotion not found

def create_spotify_client(client_id, client_secret):
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def get_track_details(sp, uri):
    track = sp.track(uri)
    info = {
        "name": track["name"],
        "artist": track["artists"][0]["name"],
        "album": track["album"]["name"],
        "url": track["external_urls"]["spotify"]
    }
    return info

def load_mood_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "uri" in df.columns and "mood" in df.columns:
        mapping = dict(zip(df["uri"], df["mood"]))
        return mapping
    else:
        raise ValueError("CSV must have columns: 'uri' and 'mood'")

def get_random_songs_by_emotion(mood_map, emotion, count=5):
    """Get random song URIs for a specific emotion (converts to mood internally)"""
    mood = emotion_to_mood(emotion)
    songs_with_mood = [uri for uri, song_mood in mood_map.items() if song_mood == mood]
    if len(songs_with_mood) < count:
        return songs_with_mood  # Return all available songs if less than requested
    return random.sample(songs_with_mood, count)
