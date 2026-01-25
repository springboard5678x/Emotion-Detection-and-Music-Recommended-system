import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------
# MODEL + DATA PATHS
# -------------------------------------------------------
MODEL_PATH = "emotions_prediction_model.h5"
MUSIC_PATH = "Music_Info.csv"
TRAIN_DATA_PATH = "balanced_emotions.csv"

VOCAB_SIZE = 10000
MAX_LEN = 40

# -------------------------------------------------------
# CACHED LOADING
# -------------------------------------------------------
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

@st.cache_resource
def load_ml_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_music():
    return pd.read_csv(MUSIC_PATH)

@st.cache_resource
def prepare_tokenizer_and_labelencoder():
    df = pd.read_csv(TRAIN_DATA_PATH)

    df["cleaned"] = df["content"].apply(clean_text)
    df["stemmed"] = df["cleaned"].apply(stem_text)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<UNK>")
    tokenizer.fit_on_texts(df["stemmed"])

    le = LabelEncoder()
    le.fit(df["final_label"])

    return tokenizer, le

# -------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------
stemmer = PorterStemmer()
stop_words = load_stopwords()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def stem_text(text):
    return " ".join(stemmer.stem(w) for w in text.split() if w not in stop_words)

# -------------------------------------------------------
# LOAD ALL OBJECTS
# -------------------------------------------------------
model = load_ml_model()
music_df = load_music()
tokenizer, label_encoder = prepare_tokenizer_and_labelencoder()

# -------------------------------------------------------
# RECOMMENDATION LOGIC (your original logic)
# -------------------------------------------------------
def recommend_songs(predicted_emotion, music_df, num_recs=5):
    if predicted_emotion in ["joy", "love", "surprise"]:
        recommendations = music_df[(music_df["valence"] > 0.6) & (music_df["energy"] > 0.6)]
    elif predicted_emotion in ["sadness", "anger"]:
        recommendations = music_df[music_df["valence"] < 0.4]
    else:
        recommendations = music_df[(music_df["valence"] >= 0.4) & (music_df["valence"] <= 0.6)]

    if len(recommendations) < num_recs:
        return music_df.sample(n=num_recs, replace=True)[["name", "artist","spotify_preview_url"]]

    return recommendations.sample(n=num_recs)[["name", "artist","spotify_preview_url"]]

# -------------------------------------------------------
# PREDICTION PIPELINE
# -------------------------------------------------------
def get_music_recommendation(text_input):
    cleaned_text = clean_text(text_input)
    stemmed_text = stem_text(cleaned_text)

    seq = tokenizer.texts_to_sequences([stemmed_text])
    padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    predictions = model.predict(padded_seq)
    prediction_idx = np.argmax(predictions, axis=1)[0]

    predicted_emotion = label_encoder.classes_[prediction_idx]

    recs = recommend_songs(predicted_emotion, music_df)
    return predicted_emotion, recs

# -------------------------------------------------------
# PREMIUM UI DESIGN — LIGHT THEME + CARDS
# -------------------------------------------------------
st.set_page_config(page_title="Emotion Music Recommender", page_icon="🎵", layout="wide")

# CSS
st.markdown("""
    <style>
        body { background-color: #f4f6ff; }
        .main { background-color: #f4f6ff; }

        .emotion-box {
            background: #ffeec4;
            padding: 14px 24px;
            border-radius: 16px;
            display: inline-block;
            font-size: 24px;
            font-weight: 700;
            color: #333;
            margin-bottom: 25px;
        }

        .song-card {
            background: white;
            padding: 20px;
            border-radius: 16px;
            margin-bottom: 18px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
            border: 1px solid #eaeaea;
        }

        .song-title {
            font-size: 21px;
            font-weight: 700;
            color: #222;
        }

        .song-artist {
            font-size: 16px;
            font-weight: 500;
            color: #666;
            margin-bottom: 10px;
        }

        .song-link a {
            color: #1DB954;
            font-size: 14px;
            font-weight: 600;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

emotion_emoji = {
    "joy": "😊",
    "love": "❤️",
    "surprise": "😲",
    "sadness": "😔",
    "anger": "😡",
    "neutral": "😐"
}

# -------------------------------------------------------
# MAIN UI
# -------------------------------------------------------
st.title("🎵 Emotion-Powered Music Recommendation System")
st.write("Tell me how you feel, I'll suggest music that matches your emotion.")

text = st.text_area("How are you feeling today? (minimum 10 characters)", height=120)

if st.button("Recommend Music 🎧"):
    if len(text.strip()) < 10:
        st.error("Please enter at least 10 characters.")
    else:
        with st.spinner("Detecting emotion and choosing songs..."):
            emotion, recs = get_music_recommendation(text)
        # Emotion Box
        st.markdown(
            f"<div class='emotion-box'>{emotion_emoji.get(emotion, '🎵')}  {emotion.upper()}</div>",
            unsafe_allow_html=True
        )

        st.write("## 🎧 Recommended Songs for You")

        # Song Cards
        for idx, row in recs.iterrows():
            st.markdown("<div class='song-card'>", unsafe_allow_html=True)

            st.markdown(f"<div class='song-title'>{row['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='song-artist'>Artist: {row['artist']}</div>", unsafe_allow_html=True)

            if pd.notna(row["spotify_preview_url"]) and str(row["spotify_preview_url"]).strip():
                st.audio(row["spotify_preview_url"])
                st.markdown(
                    f"<div class='song-link'>🎧 <a href='{row['spotify_preview_url']}' target='_blank'>Open Preview</a></div>",
                    unsafe_allow_html=True
                )
            else:
                st.write("🔇 No audio preview available.")

            st.markdown("</div>", unsafe_allow_html=True)
