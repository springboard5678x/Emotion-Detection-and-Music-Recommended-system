import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import layers, models, regularizers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os
import sys
import cv2 
import requests 

# --- 1. MODEL ARCHITECTURE (Original RGB ResNet-18) ---
def conv_block(x, filters, stride=1, weight_decay=1e-4):
    """Standard ResNet Residual Block structure."""
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same',
                      use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()
    return x

def build_resnet18(input_shape=(48,48,3), num_classes=7):
    """Complete ResNet18 structure for 48x48 RGB input."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    for filters, blocks, stride in [(64,2,1), (128,2,2), (256,2,2), (512,2,2)]:
        for b in range(blocks):
            x = conv_block(x, filters, stride if b == 0 else 1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) 
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model


# --- 2. LOAD MODEL ---
MODEL_FILENAME = 'full_model_rgb.h5' 
custom_objects = {'conv_block': conv_block, 'build_resnet18': build_resnet18} 

try:
    print(f"Attempting to load complete model from {MODEL_FILENAME}...")
    model = tf.keras.models.load_model(MODEL_FILENAME, custom_objects=custom_objects)
    print("Model loaded successfully using load_model.")
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
except Exception as e:
    sys.exit(f"Fatal: Model loading failed. Error: {e}")

# --- 3. FACE DETECTION SETUP ---
HAAR_CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
HAAR_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
face_cascade = None
if not os.path.exists(HAAR_CASCADE_FILE):
    try:
        print("Downloading Haar Cascade classifier for face detection...")
        r = requests.get(HAAR_CASCADE_URL, allow_redirects=True)
        with open(HAAR_CASCADE_FILE, 'wb') as f:
            f.write(r.content)
    except:
        pass
if os.path.exists(HAAR_CASCADE_FILE):
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    if face_cascade.empty():
         face_cascade = None


# --- 4. MUSIC RECOMMENDATION SETUP ---
zip_path = 'spotify_millsongdata.csv.zip'
csv_name = 'spotify_millsongdata.csv'
df_songs = None

if not os.path.exists(csv_name):
    try:
        if os.path.exists(zip_path):
            print("Extracting song data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract(csv_name)
        if os.path.exists(csv_name):
            df_songs = pd.read_csv(csv_name)
    except Exception as e:
        print(f"Error loading song data: {e}")

if df_songs is not None:
    df_songs['text'] = df_songs['text'].str.replace('\r\n', ' ').fillna('')
    df_songs['link'] = 'https://www.google.com/search?q=' + df_songs['song'].astype(str) + '%20artist%20' + df_songs['artist'].astype(str) + '%20spotify'

    mood_map = {
        'Happy': ['upbeat', 'joy', 'dancing', 'celebrate', 'fun', 'love', 'positive', 'party', 'cheer'],
        'Sad': ['lonely', 'broken', 'cry', 'pain', 'rain', 'miss', 'blue', 'tears', 'slow'],
        'Angry': ['rage', 'fight', 'fire', 'hate', 'hard', 'loud', 'rock', 'punch', 'attack'],
        'Fear': ['scared', 'dark', 'cold', 'hide', 'run', 'anxiety', 'tension'],
        'Surprise': ['fast', 'high energy', 'shock', 'wonder', 'sudden', 'change'],
        'Neutral': ['calm', 'quiet', 'smooth', 'chill', 'easy', 'peace', 'sleep', 'acoustic'],
        'Disgust': ['annoying', 'nasty', 'sick', 'ugh', 'heavy', 'metal']
    }

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_songs['text'])

    def get_music_recommendation(emotion, top_n=5):
        if emotion not in mood_map: 
            return pd.DataFrame({'song': ['No relevant songs found.'], 'artist': ['N/A'], 'link': ['#']})
            
        target_doc = ' '.join(mood_map[emotion])
        target_vector = vectorizer.transform([target_doc])
        cosine_sim = cosine_similarity(target_vector, tfidf_matrix).flatten()
        top_indices = cosine_sim.argsort()[-top_n:][::-1]
        return df_songs.iloc[top_indices][['song', 'artist', 'link']].reset_index(drop=True)
else:
    def get_music_recommendation(emotion, top_n=5):
        return pd.DataFrame({'song': ['Recommendation Disabled (Missing Data)'], 'artist': ['N/A'], 'link': ['#']})


# --- 5. THE MAIN APP FUNCTION ---

def predict_mood_and_recommend(image_input):
    
    print("\n--- STARTING PREDICTION FUNCTION ---")

    if image_input is None:
        print("DEBUG: Image input is None. Returning immediately.")
        # Critical: Guidance for the user on the webcam failure
        return "N/A", "<h3>Webcam input failed (no data received from the browser). Please ensure your browser has camera access, or use the **Upload** option instead.</h3>"

    # --- INPUT CONVERSION ---
    try:
        # Convert to standardized uint8 numpy array for OpenCV (RGB)
        image = np.array(Image.fromarray(image_input).convert('RGB')).astype(np.uint8)
        print(f"DEBUG: Input image successfully converted. Shape: {image.shape}, Data Type: {image.dtype}")
    except Exception as e:
        print(f"ERROR: Minimalist input conversion failed. Error: {e}")
        return "Error", f"Fatal Input Error: Could not process image data. {e}"


    # --- INITIAL VALIDATION CHECK ---
    if np.mean(image) < 5.0 and np.std(image) < 10.0:
        return "Waiting for Input...", "<h3>Camera stream is active, but the image is blank/dark. Try moving your camera or checking lighting.</h3>"

    # --- FACE DETECTION AND CROPPING ---
    cropped_image = image 
    if face_cascade is not None:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
        except:
            gray = image 

        try:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05, 
                minNeighbors=3,   
                minSize=(20, 20), 
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f"DEBUG: Faces detected: {len(faces)}")
        except:
            faces = []

        if len(faces) == 0:
            return "No Face Detected", "<h3>Cannot find a face in the image. Please center your face in the frame and click Submit.</h3>"

        # Crop to the largest detected face
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        cropped_image = image[y:y+h, x:x+w]
        
    # --- IMAGE PREPROCESSING: Resizing and Normalization ---
    try:
        # Resize to 48x48, maintaining RGB for the model
        img = Image.fromarray(cropped_image).convert('RGB')
        img = img.resize((48, 48))
        img_array = np.array(img) / 255.0
        # Expand dimensions to (1, 48, 48, 3)
        img_array = np.expand_dims(img_array, axis=0) 
    except Exception as e:
        return "Error", f"Image preprocessing failed: {e}"
    
    # Predict emotion
    try:
        predictions = model.predict(img_array, verbose=0)[0] 
    except Exception as e:
        return "Error", f"Model prediction failed: {e}"

    
    # Find emotion and apply heuristic
    happy_index = emotion_labels.index('Happy')
    angry_index = emotion_labels.index('Angry')
    happy_score = predictions[happy_index]
    angry_score = predictions[angry_index]
    predicted_index = np.argmax(predictions)
    
    # Heuristic: If model predicts Happy, but Angry is also very high, flip to Angry
    if predicted_index == happy_index and angry_score >= (happy_score * 0.80): 
        predicted_emotion = 'Angry'
    else:
        predicted_emotion = emotion_labels[predicted_index]

    # --- Generate Recommendations and HTML Output ---
    recommendations_df = get_music_recommendation(predicted_emotion)
    
    recommendation_html = f"Detected Emotion: <b>{predicted_emotion}</b>" 
    recommendation_html += """
    <style>
    .reco-table { width: 100%; border-collapse: collapse; font-family: sans-serif; }
    .reco-table th, .reco-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    .reco-table th { background-color: #f2f2f2; color: #333; }
    .spotify-link { text-decoration: none; color: #1DB954; font-weight: bold; }
    .spotify-link:hover { text-decoration: underline; }
    </style>
    <table class="reco-table">
    <thead><tr><th>Rank</th><th>Song Title</th><th>Artist</th><th>Search Link</th></tr></thead>
    <tbody>
    """
    
    for rank, row in recommendations_df.iterrows():
        recommendation_html += f"""
        <tr>
            <td>{rank + 1}</td>
            <td>{row['song']}</td>
            <td>{row['artist']}</td>
            <td><a class="spotify-link" href="{row['link']}" target="_blank">🔍 Google/Spotify Search</a></td>
        </tr>
        """
    recommendation_html += "</tbody></table>"
    print("--- PREDICTION COMPLETE ---")
    return predicted_emotion, recommendation_html


# --- 6. GRADIO INTERFACE (Webcam + Upload) ---

iface = gr.Interface(
    fn=predict_mood_and_recommend,
    inputs=gr.Image(
        type="numpy", 
        label="Upload Image or Use Webcam", 
        sources=["upload", "webcam"], 
        width=320, 
        height=240 
    ), 
    outputs=[
        gr.Label(label="Predicted Emotion"),
        gr.HTML(label="MoodMate Recommendations")
    ],
    title="MoodMate: Emotion-Based Music Recommender (Final Stable Version)",
    description=f"To use the webcam (primary function): click **Start Camera**, position your face clearly in the frame, and then click the **Submit** button. If the webcam consistently fails, the issue is environmental/browser-related."
)

iface.launch(debug=True)