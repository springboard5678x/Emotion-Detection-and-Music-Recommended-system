
from pathlib import Path
import os
import ast
from html import escape
from urllib.parse import quote_plus

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

st.set_page_config(
    page_title="AI MoodMate 🎵",
    page_icon="🎵",
    layout="centered"
)

st.markdown(
    """
    <style>
      .app-title {
          font-size: 2.4rem;
          font-weight: 800;
          letter-spacing: .2px;
      }
      .sub {
          color: #616995;
          margin-bottom: .6rem;
      }
      .pill {
          display: inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          background: #f2f4ff;
          color: #4a4fad;
          font-size: .8rem;
          margin-right: 6px;
      }
      .panel {
          padding: 18px;
          border-radius: 16px;
          background: #ffffff;
          border: 1px solid #eef0f6;
          box-shadow: 0 2px 10px rgba(0,0,0,.04);
      }
      .footer {
          color: #8a8ea8;
          font-size: .9rem;
          padding-top: 18px;
      }
      .rec-card {
          padding: 14px 16px;
          border-radius: 14px;
          background: #fff;
          box-shadow: 0 1px 8px rgba(0,0,0,.05);
          margin-bottom: 10px;
      }
      .stProgress > div > div > div {
          background: linear-gradient(90deg,#6C63FF,#49c5b6);
      }
      .stRadio > div {
          gap: 10px;
      }
      a {
          text-decoration: none;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-title">🎧 AI MoodMate</div>
    <div class="sub">
        Emotion detection from face + mood-matched music picks.
    </div>
    <div class="pill">Streamlit</div>
    <div class="pill">TensorFlow/Keras</div>
    <div class="pill">CV + Recsys</div>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = BASE_DIR / "emotion_model_custom_cnn_sgd.keras"
MUSIC_DB_PATH = BASE_DIR / "processed_music.csv"

EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/encharm/Font-Awesome-SVG-PNG/master/black/png/64/headphones.png",
        width=64
    )

    st.markdown("### How to use")

    st.write(
        "• Upload a face photo **or** use the **Webcam**.\n"
        "• Wait for prediction.\n"
        "• Get mood-matched songs."
    )

    st.markdown("### About")
    st.write("Built with TensorFlow + Streamlit.")
    st.caption("Model: Custom CNN")
    st.caption("Tip: better light → better predictions.")
    st.divider()

    if "reset_count" not in st.session_state:
        st.session_state.reset_count = 0

    if st.button("🔁 Reset app"):
        st.session_state.reset_count += 1
        st.rerun()


@st.cache_resource
def load_emotion_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    return load_model(str(MODEL_PATH), compile=False)


@st.cache_data
def load_music_database(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path.name}"
        )

    with open(csv_path, "rb") as file:
        header = file.read(100)

    if header.startswith(b"version https://git-lfs.github.com"):
        raise ValueError(
            "Music database contains a Git LFS pointer "
            "instead of the actual CSV file."
        )

    df = pd.read_csv(csv_path)

    if "emotion" not in df.columns:
        raise ValueError(
            "Music database is missing the emotion column."
        )

    df["emotion"] = (
        df["emotion"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


try:
    with st.spinner("Loading emotion detection model..."):
        emotion_model = load_emotion_model()

except Exception as e:
    st.error("Unable to load the emotion detection model.")
    st.code(str(e))
    st.stop()


try:
    with st.spinner("Loading music database..."):
        music_df = load_music_database(MUSIC_DB_PATH)

except Exception as e:
    st.warning("Music recommendations are currently unavailable.")
    st.code(str(e))
    music_df = pd.DataFrame()


tab1, tab2 = st.tabs(["📁 Upload", "📷 Webcam"])

image_to_process = None

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image…",
        type=["jpg", "png", "jpeg"],
        key=f"upload_{st.session_state.reset_count}"
    )

    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)

with tab2:
    webcam_image = st.camera_input(
        "Take a picture!",
        key=f"camera_{st.session_state.reset_count}"
    )

if webcam_image is not None:
    image_to_process = Image.open(webcam_image)
    image_to_process = image_to_process.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )


if image_to_process is not None:

    st.image(
        image_to_process,
        caption="Your Image",
        use_column_width=True
    )

    try:
        height, width, channels = emotion_model.input_shape[1:]

        if channels == 1:
            img_resized = image_to_process.convert("L")
        elif channels == 3:
            img_resized = image_to_process.convert("RGB")
        else:
            raise ValueError(
                f"Unsupported model input shape: "
                f"{emotion_model.input_shape}"
            )

        img_resized = img_resized.resize((width, height))

        arr = img_to_array(img_resized).astype("float32")
        arr = np.expand_dims(arr, axis=0) / 255.0

        with st.spinner("Detecting your mood…"):
            preds = emotion_model.predict(arr, verbose=0)

            probs = np.asarray(preds[0], dtype=float)

            if len(probs) != len(EMOTION_LABELS):
                raise ValueError(
                    "Model output does not match the seven emotion labels."
                )

            idx = int(np.argmax(probs))
            predicted_emotion = EMOTION_LABELS[idx]
            confidence = float(probs[idx])

    except Exception as e:
        st.error("Emotion prediction failed.")
        st.code(str(e))
        st.stop()

    c1, c2 = st.columns([1, 1])

    with c1:
        st.success(
            f"**Predicted expression:** "
            f"{predicted_emotion.capitalize()}"
        )

    with c2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")

    st.progress(
        min(1.0, max(0.0, confidence))
    )

    st.caption(
        "Predictions are estimates of facial expressions "
        "and may not reflect your actual mood."
    )

    st.subheader(
        f"🎵 Songs for a {predicted_emotion.capitalize()} mood"
    )

    if not music_df.empty:

        recs = music_df[
            music_df["emotion"] == predicted_emotion
        ]

        if not recs.empty:

            recs = recs.sample(
                min(5, len(recs)),
                random_state=42
            )

            for _, row in recs.iterrows():

                name = str(row.get("name", "Untitled"))

                try:
                    artists_raw = row.get("artists", "[]")

                    artist_list = (
                        ast.literal_eval(str(artists_raw))
                        if isinstance(artists_raw, str)
                        else artists_raw
                    )

                    artists = (
                        ", ".join(map(str, artist_list))
                        if isinstance(artist_list, list)
                        else str(artists_raw)
                    )

                except Exception:
                    artists = str(
                        row.get("artists", "Unknown")
                    )

                url = str(row.get("url", "")).strip()

                if not url.startswith(("https://", "http://")):
                    query = quote_plus(f"{name} {artists}")
                    url = (
                        "https://www.youtube.com/results?"
                        f"search_query={query}"
                    )

                st.markdown(
                    f"""
                    <div class="rec-card">
                        <a href="{escape(url, quote=True)}"
                           target="_blank"
                           rel="noopener noreferrer">
                            <strong>{escape(name)}</strong>
                            · {escape(artists)}
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.info(
                f"Sorry, no songs found for "
                f"**{predicted_emotion}**."
            )

    else:
        st.info(
            "Emotion detection is available, but the "
            "music database could not be loaded."
        )

        query = quote_plus(
            f"{predicted_emotion} mood songs"
        )

        st.link_button(
            "🎵 Explore mood-matched songs on YouTube",
            f"https://www.youtube.com/results?search_query={query}"
        )

else:
    st.info(
        "Upload an image or switch to **Webcam** "
        "to get started."
    )

st.markdown(
    '<div class="footer">'
    '© 2025 MoodMate · Built by Shanmukh'
    '</div>',
    unsafe_allow_html=True
)
