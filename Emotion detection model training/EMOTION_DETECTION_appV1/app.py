%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import time
from PIL import Image

# ==============================================
# ⚙️ CONFIGURATION
# ==============================================
st.set_page_config(page_title="Emotion Music Recommender", page_icon="🎧", layout="centered")

IMAGE_SIZE = 96
CLASS_LABELS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
DATA_PATH = "/content/data_moods.csv"   # your dataset
MODEL_PATH = "/content/fer_hybrid_efficientnet_custom_model.h5"  # trained model

# ==============================================
# 🧠 Load Model & Dataset
# ==============================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df

hybrid_model = load_model()
music_df = load_dataset()

# ==============================================
# 🎯 Emotion → Mood Mapping
# ==============================================
emotion_to_mood = {
    'angry': 'Calm',
    'fear': 'Calm',
    'happy': 'Happy',
    'neutral': 'Energetic',
    'sad': 'Sad',
    'surprise': 'Energetic'
}

# ==============================================
# 🧰 Helper Functions
# ==============================================
def detect_faces_and_draw(img_array, emotion_texts=None):
    """Detect faces, draw rectangles + emotion labels, return cropped faces."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    faces_list = []

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if emotion_texts and i < len(emotion_texts):
            cv2.putText(img_array, emotion_texts[i], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        faces_list.append(gray[y:y+h, x:x+w])
    return faces_list, img_array

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
    face_resized = face_resized / 255.0
    face_resized = np.expand_dims(face_resized, axis=-1)
    face_resized = np.expand_dims(face_resized, axis=0)
    return face_resized

def predict_emotion(face_img):
    preprocessed = preprocess_face(face_img)
    preds = hybrid_model.predict(preprocessed, verbose=0)
    pred_label = CLASS_LABELS[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return pred_label, confidence

def recommend_music(emotion_label, top_n=10):
    mapped_mood = emotion_to_mood.get(emotion_label.lower(), None)
    if not mapped_mood:
        st.warning(f"No mood mapping found for emotion '{emotion_label}'. Showing random songs instead.")
        filtered = music_df.sample(n=top_n)
    else:
        filtered = music_df[music_df['mood'].str.lower() == mapped_mood.lower()]
        if filtered.empty:
            st.warning(f"No songs found for mood '{mapped_mood}'. Showing random songs instead.")
            filtered = music_df.sample(n=top_n)
        else:
            if 'popularity' in filtered.columns:
                filtered = filtered.sort_values(by='popularity', ascending=False)
    return filtered.head(top_n)[['name', 'artist', 'mood', 'popularity']]

# ==============================================
# 🎨 Streamlit UI
# ==============================================
st.title("🎧 Emotion-Based Music Recommender")
st.markdown("Detect your mood and get personalized music recommendations 🎵")

mode = st.radio("Choose Mode:", ["📸 Upload Image", "🎥 Use Webcam"])

# ==============================================
# 📸 IMAGE UPLOAD MODE
# ==============================================
if mode == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        img_array = cv2.imread(temp_path)
        faces, _ = detect_faces_and_draw(img_array)

        emotion_texts = []
        for face in faces:
            label, confidence = predict_emotion(face)
            emotion_texts.append(f"{label} ({confidence:.1f}%)")

        # Re-draw image with emotion labels
        faces, img_with_box = detect_faces_and_draw(img_array, emotion_texts)
        st.image(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB), caption="Detected Face(s)", use_container_width=True)

        if not faces:
            st.error("No face detected! Please upload a clear photo with a visible face.")
        else:
            label, confidence = predict_emotion(faces[0])
            mapped_mood = emotion_to_mood.get(label.lower(), "Unknown")

            st.success(f"🧠 Detected Emotion: **{label.upper()}** ({confidence:.1f}%)")
            st.subheader(f"🎵 Recommended Songs for Mood: {mapped_mood}")

            top_songs = recommend_music(label, top_n=10)
            for idx, row in top_songs.iterrows():
                st.markdown(
                    f"**{row['name']}** — *{row['artist']}*  \n"
                    f"🎶 Mood: {row['mood']}  \n"
                    f"⭐ Popularity: {int(row['popularity'])}"
                )

# ==============================================
# 🎥 WEBCAM MODE
# ==============================================
elif mode == "🎥 Use Webcam":
    st.info("Click 'Start Webcam' to detect faces and emotions in real-time.")
    run_webcam = st.button("▶️ Start Webcam")

    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        current_emotion = None
        last_update = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            faces, _ = detect_faces_and_draw(frame)
            emotion_texts = []
            emotions_detected = []

            for face in faces:
                label, confidence = predict_emotion(face)
                emotion_texts.append(f"{label}")
                emotions_detected.append(label)

            faces, img_with_box = detect_faces_and_draw(frame, emotion_texts)
            stframe.image(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # Update music recommendations every 5 seconds
            if emotions_detected and (time.time() - last_update) > 5:
                current_emotion = emotions_detected[0]
                st.write(f"🧠 Current Emotion: **{current_emotion.upper()}**")
                mood = emotion_to_mood.get(current_emotion.lower(), "Unknown")
                st.subheader(f"🎵 Songs for Mood: {mood}")

                top_songs = recommend_music(current_emotion, top_n=5)
                for idx, row in top_songs.iterrows():
                    st.markdown(
                        f"**{row['name']}** — *{row['artist']}*  \n"
                        f"🎶 Mood: {row['mood']}  \n"
                        f"⭐ Popularity: {int(row['popularity'])}"
                    )
                last_update = time.time()

            if st.button("⏹ Stop Webcam"):
                cap.release()
                st.success("Webcam stopped.")
                break

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, TensorFlow & OpenCV")
