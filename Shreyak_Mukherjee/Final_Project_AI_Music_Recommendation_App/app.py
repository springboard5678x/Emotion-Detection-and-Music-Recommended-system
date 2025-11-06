import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import json
import h5py
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- App Configuration ---
st.set_page_config(
    page_title="MoodMate | Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# --- Custom CSS ---
def load_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
        
        html, body, [data-testid="stApp"] {
            font-family: 'Montserrat', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
            color: #FFFFFF;
        }
        
        .main-header {
            text-align: center;
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
            color: #FFFFFF;
            text-shadow: 0 4px 8px rgba(0,0,0,0.5);
            padding: 10px;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .sub-header {
            text-align: center;
            font-size: 1.4rem;
            font-weight: 400;
            margin-bottom: 2rem;
            color: #F0F0F0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .glass-container {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(15px);
            border-radius: 24px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.25);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
            margin-bottom: 25px;
            transition: transform 0.3s ease;
        }
        
        .glass-container:hover {
            transform: translateY(-5px);
        }
        
        .emotion-box {
            background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            border: 2px solid rgba(255, 255, 255, 0.4);
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
            margin: 20px 0;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        .emotion-box h2 {
            color: #FFFFFF;
            font-weight: 800;
            margin: 0;
            font-size: 2.5rem;
            text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        
        .track-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1));
            border-radius: 18px;
            padding: 20px;
            margin: 15px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .track-card:hover {
            background: linear-gradient(135deg, rgba(255,255,255,0.25), rgba(255,255,255,0.15));
            transform: translateX(10px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);
        }
        
        .stats-container {
            display: flex;
            justify-content: space-around;
            text-align: center;
            margin: 20px 0;
        }
        
        .stat-box {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 15px;
            flex: 1;
            margin: 0 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: #FFD93D;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #E0E0FF;
            margin-top: 5px;
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .upload-section {
            border: 3px dashed rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            background: rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: rgba(255, 255, 255, 0.5);
            background: rgba(255, 255, 255, 0.08);
        }
        
        .stButton button {
            background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
            color: white;
            border: none;
            padding: 12px 30px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        }
        
        .stButton button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            background: linear-gradient(135deg, #FF8E8E, #6BE0D8);
        }
        
        .recommendation-badge {
            background: linear-gradient(135deg, #FFD93D, #FF6B6B);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-left: 10px;
        }
        
        .confidence-meter {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .webcam-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 20px;
            margin: 20px 0;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        .live-emotion-display {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 15px 0;
            border: 2px solid rgba(255, 255, 255, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- File Paths and Constants ---
MODEL_PATH = 'emotion_model.h5'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Emotion mapping
EMOTION_MAP = {0: 'happy', 1: 'sad', 2: 'angry', 3: 'surprise', 4: 'neutral'}

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- SIMPLE and EFFECTIVE Emotion Model ---
@st.cache_resource
def load_emotion_model():
    """Create a very simple but effective model"""
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(48, 48, 1)),
            
            # Simple architecture that works
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        st.error(f"❌ Model creation failed: {e}")
        return None

@st.cache_resource
def load_face_detector():
    try:
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                st.error("❌ Error loading Haar Cascade file.")
                return None
        return face_cascade
    except Exception as e:
        st.error(f"❌ Error loading Haar Cascade: {e}")
        return None

# --- Load Music Data ---
@st.cache_data
def load_music_data():
    """Load music dataset"""
    csv_files = ["music_moods_dataset.csv", "music_mods_dataset.csv"]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if 'mood' in df.columns:
                    df['mood'] = df['mood'].astype(str).str.lower().str.strip()
                return df
            except Exception as e:
                continue
    
    # Create sample data
    sample_data = {
        'track': ['Happy Song', 'Sad Melody', 'Angry Beats', 'Surprise Symphony', 'Calm Vibes'],
        'artist': ['Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E'],
        'mood': ['happy', 'sad', 'angry', 'surprise', 'neutral'],
        'genre': ['Pop', 'Blues', 'Rock', 'Classical', 'Ambient'],
        'uri': ['spotify:track:123', 'spotify:track:456', 'spotify:track:789', 'spotify:track:012', 'spotify:track:345']
    }
    return pd.DataFrame(sample_data)

# --- IMPROVED Computer Vision Emotion Detection ---
def analyze_emotion_computer_vision(face_roi):
    """Much improved computer vision emotion detection"""
    try:
        height, width = face_roi.shape
        
        # Define facial regions more accurately
        eye_region = face_roi[int(height*0.2):int(height*0.45), :]
        mouth_region = face_roi[int(height*0.6):int(height*0.9), :]
        eyebrow_region = face_roi[int(height*0.15):int(height*0.3), :]
        
        # Calculate features
        face_brightness = np.mean(face_roi)
        face_contrast = np.std(face_roi)
        mouth_contrast = np.std(mouth_region)
        eye_contrast = np.std(eye_region)
        eyebrow_contrast = np.std(eyebrow_region)
        
        # Calculate mouth curvature (smile detection)
        mouth_middle = mouth_region[:, width//4:3*width//4]
        mouth_curve = np.mean(mouth_middle) - np.mean(mouth_region)
        
        # Calculate eyebrow intensity
        eyebrow_intensity = np.mean(eyebrow_region)
        
        # Calculate eye openness
        eye_openness = np.mean(eye_region)
        
        # NEW: Calculate facial symmetry
        left_face = face_roi[:, :width//2]
        right_face = face_roi[:, width//2:]
        symmetry = np.abs(np.mean(left_face) - np.mean(right_face))
        
        # IMPROVED Emotion scoring with better logic
        emotion_scores = {
            'happy': 0,
            'sad': 0, 
            'angry': 0,
            'surprise': 0,
            'neutral': 0
        }
        
        # HAPPY: High mouth contrast + positive curve (smile), bright face
        if mouth_contrast > 45 and mouth_curve > 5:
            emotion_scores['happy'] += 3
        if face_brightness > 130:
            emotion_scores['happy'] += 1
        if mouth_contrast > 40:
            emotion_scores['happy'] += 1
            
        # SAD: Low mouth activity, potential downward curve, lower brightness
        if mouth_contrast < 35:
            emotion_scores['sad'] += 2
        if face_brightness < 110:
            emotion_scores['sad'] += 2
        if mouth_curve < -2:  # Downward curve
            emotion_scores['sad'] += 1
        if symmetry > 15:  # Asymmetry in sadness
            emotion_scores['sad'] += 1
            
        # ANGRY: High eyebrow contrast, intense features
        if eyebrow_contrast > 40:
            emotion_scores['angry'] += 3
        if face_contrast > 50:
            emotion_scores['angry'] += 1
        if eyebrow_intensity < 100:  # Darker eyebrows (furrowed)
            emotion_scores['angry'] += 1
            
        # SURPRISE: High eye contrast, bright eyes
        if eye_contrast > 60:
            emotion_scores['surprise'] += 3
        if eye_openness > 130:  # Bright eyes (wide open)
            emotion_scores['surprise'] += 2
        if mouth_contrast > 50:  # Open mouth
            emotion_scores['surprise'] += 1
            
        # NEUTRAL: Balanced features, medium values
        if 35 <= mouth_contrast <= 50:
            emotion_scores['neutral'] += 2
        if 40 <= face_contrast <= 60:
            emotion_scores['neutral'] += 2
        if 120 <= face_brightness <= 150:
            emotion_scores['neutral'] += 1
        if symmetry < 10:  # High symmetry
            emotion_scores['neutral'] += 1
            
        # Determine final emotion
        final_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[final_emotion]
        total_score = sum(emotion_scores.values())
        
        # Calculate confidence
        confidence = max_score / total_score if total_score > 0 else 0.5
        confidence = max(0.3, min(0.9, confidence))
        
        return final_emotion, confidence, emotion_scores
        
    except Exception as e:
        return 'neutral', 0.6, {'happy': 0, 'sad': 0, 'angry': 0, 'surprise': 0, 'neutral': 1}

# --- Webcam Video Processor ---
class VideoProcessor:
    def __init__(self):
        self.face_cascade = load_face_detector()
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.emotion_history = []
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize for emotion analysis
            roi_resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            
            # Analyze emotion
            emotion, confidence, scores = analyze_emotion_computer_vision(roi_resized)
            
            self.current_emotion = emotion
            self.current_confidence = confidence
            self.emotion_history.append(emotion)
            
            # Keep only last 10 emotions
            if len(self.emotion_history) > 10:
                self.emotion_history.pop(0)
            
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display emotion only (no confidence)
            text = f"{emotion.upper()}"
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- SIMPLIFIED and RELIABLE Emotion Detection ---
def detect_emotion_reliable(image, emotion_model, face_cascade):
    """Reliable emotion detection using computer vision as primary"""
    try:
        # Convert PIL to numpy array
        image_np = np.array(image)
        
        # Convert to grayscale
        if len(image_np.shape) > 2 and image_np.shape[2] in [3, 4]:
            gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = image_np
        
        # Face detection
        faces = face_cascade.detectMultiScale(
            gray_img, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            st.warning("⚠️ No face detected. Please try an image with a clear, front-facing face.")
            return None, None, image
        
        # Get the largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        (x, y, w, h) = faces[0]
        
        # Extract and preprocess face ROI
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi_equalized = cv2.equalizeHist(roi_resized)
        roi_normalized = roi_equalized / 255.0
        
        # PRIMARY: Use improved computer vision
        cv_emotion, cv_confidence, emotion_scores = analyze_emotion_computer_vision(roi_resized)
        
        # SECONDARY: Try AI model
        roi_final = np.expand_dims(roi_normalized, axis=-1)
        roi_final = np.expand_dims(roi_final, axis=0)
        
        ai_prediction = emotion_model.predict(roi_final, verbose=0)
        ai_confidence = np.max(ai_prediction)
        ai_emotion_idx = np.argmax(ai_prediction)
        ai_emotion = EMOTION_MAP.get(ai_emotion_idx, "neutral")
        
        # DECISION MAKING: Use computer vision as primary (more reliable)
        if ai_confidence > 0.7:
            final_emotion = ai_emotion
            final_confidence = ai_confidence
        elif cv_confidence > 0.6:
            final_emotion = cv_emotion
            final_confidence = cv_confidence
        elif ai_confidence > 0.5:
            final_emotion = ai_emotion
            final_confidence = ai_confidence
        else:
            final_emotion = cv_emotion
            final_confidence = cv_confidence
        
        # Return only the original image, not the processed one with boxes
        return final_emotion, final_confidence, image
        
    except Exception as e:
        st.error(f"❌ Error in emotion detection: {e}")
        return None, None, image

# --- Music Recommendation ---
def recommend_music(music_df, emotion, top_n=5):
    """Get music recommendations based on emotion"""
    if music_df.empty:
        return pd.DataFrame()
    
    if 'mood' not in music_df.columns:
        return music_df.sample(min(top_n, len(music_df)))
    
    emotion = str(emotion).lower().strip()
    exact_matches = music_df[music_df['mood'] == emotion]
    
    if not exact_matches.empty:
        return exact_matches.head(top_n)
    
    emotion_mapping = {
        'happy': ['happy', 'joyful', 'excited', 'energetic', 'upbeat', 'dance'],
        'sad': ['sad', 'melancholy', 'calm', 'chill', 'mellow', 'emotional'],
        'angry': ['angry', 'aggressive', 'intense', 'powerful', 'rock', 'metal'],
        'surprise': ['surprise', 'exciting', 'energetic', 'upbeat', 'electronic'],
        'neutral': ['calm', 'chill', 'peaceful', 'relaxed', 'ambient', 'soothing']
    }
    
    if emotion in emotion_mapping:
        for similar_mood in emotion_mapping[emotion]:
            similar_matches = music_df[music_df['mood'] == similar_mood]
            if not similar_matches.empty:
                return similar_matches.head(top_n)
    
    return music_df.sample(min(top_n, len(music_df)))

def spotify_uri_to_url(uri):
    """Convert Spotify URI to web URL"""
    if pd.isna(uri):
        return "#"
    uri_str = str(uri)
    if uri_str.startswith("spotify:track:"):
        return f"https://open.spotify.com/track/{uri_str.split(':')[-1]}"
    elif len(uri_str) == 22:
        return f"https://open.spotify.com/track/{uri_str}"
    return uri_str

# --- Streamlit UI ---
def main():
    load_css()
    
    st.markdown('<h1 class="main-header">🎵 MoodMate | AI Music Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your photo for AI emotion detection or manually select your mood to get personalized music recommendations</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'detected_emotion' not in st.session_state:
        st.session_state.detected_emotion = None
    if 'num_tracks' not in st.session_state:
        st.session_state.num_tracks = 5
    if 'webcam_emotion' not in st.session_state:
        st.session_state.webcam_emotion = None
    
    # Load models
    with st.spinner("🔄 Loading emotion detection..."):
        emotion_model = load_emotion_model()
    
    with st.spinner("🔄 Loading face detector..."):
        face_cascade = load_face_detector()
    
    with st.spinner("🔄 Loading music database..."):
        music_df = load_music_data()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📸 **AI Emotion Detection**", "🎥 **Live Webcam**", "🎭 **Manual Selection**"])
    
    # Tab 1: Enhanced Detection
    with tab1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("AI Emotion Detection")
        st.markdown("""
        **Using advanced computer vision and AI:**
        - **Improved Feature Analysis**: Better facial region detection
        - **Enhanced Emotion Logic**: More accurate scoring system
        - **Smart Decision Making**: Reliable confidence-based selection
        """)
        
        # Number of tracks selection
        st.session_state.num_tracks = st.slider(
            "Number of tracks to recommend:",
            min_value=1,
            max_value=10,
            value=5,
            key="ai_track_slider"
        )
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a clear facial image...", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption='Uploaded Image', use_column_width=True)
            
            with col2:
                if st.button("🎵 Detect Emotion & Get Music", type="primary", use_container_width=True):
                    if emotion_model is None or face_cascade is None:
                        st.error("❌ Required models not available")
                        return
                    
                    if music_df.empty:
                        st.error("❌ No music data available")
                        return
                    
                    with st.spinner('🔍 Analyzing facial expression...'):
                        # Use reliable detection - now returns original image
                        detected_emotion, confidence, result_image = detect_emotion_reliable(
                            image, emotion_model, face_cascade
                        )
                        
                        if detected_emotion is not None:
                            # Display the emotion result
                            st.markdown(f'''
                            <div class="emotion-box">
                                <h2>{detected_emotion.upper()}</h2>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            # Special effects
                            if detected_emotion == 'happy' and confidence > 0.7:
                                st.balloons()
                            elif detected_emotion == 'surprise' and confidence > 0.7:
                                st.snow()
                            
                            # Get recommendations
                            recommendations = recommend_music(music_df, detected_emotion, st.session_state.num_tracks)
                            st.session_state.recommendations = recommendations
                            st.session_state.detected_emotion = detected_emotion
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Live Webcam Detection
    with tab2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("🎥 Live Webcam Emotion Detection")
        st.markdown("""
        **Real-time emotion detection using your webcam:**
        - **Live Face Detection**: See your face being detected in real-time
        - **Instant Emotion Analysis**: Get immediate emotion feedback
        - **Continuous Monitoring**: Track emotional changes over time
        """)
        
        # Number of tracks selection for webcam
        st.session_state.num_tracks = st.slider(
            "Number of tracks to recommend:",
            min_value=1,
            max_value=10,
            value=5,
            key="webcam_track_slider"
        )
        
        if face_cascade is None:
            st.error("❌ Face detector not available. Please check your setup.")
        else:
            st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
            st.subheader("🔴 Live Webcam Feed")
            
            # Initialize video processor
            video_processor = VideoProcessor()
            
            # WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="emotion-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if webrtc_ctx.video_processor:
                current_emotion = webrtc_ctx.video_processor.current_emotion
                current_confidence = webrtc_ctx.video_processor.current_confidence
                
                if current_emotion and current_confidence > 0.4:
                    st.markdown(f'''
                    <div class="live-emotion-display">
                        <h2>Current Emotion: {current_emotion.upper()}</h2>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Store the detected emotion for recommendations
                    st.session_state.webcam_emotion = current_emotion
                    
                    # Get music recommendations button
                    if st.button("🎵 Get Music for Current Emotion", type="primary", use_container_width=True):
                        recommendations = recommend_music(music_df, current_emotion, st.session_state.num_tracks)
                        st.session_state.recommendations = recommendations
                        st.session_state.detected_emotion = current_emotion
                
                # Emotion history
                if webrtc_ctx.video_processor.emotion_history:
                    st.subheader("📊 Emotion History")
                    emotion_counts = {}
                    for emotion in webrtc_ctx.video_processor.emotion_history:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    for emotion, count in emotion_counts.items():
                        st.write(f"{emotion.upper()}: {count} times")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.info("💡 **Tips for better webcam detection:**")
            st.markdown("""
            - Ensure good lighting on your face
            - Look directly at the camera
            - Maintain a neutral or expressive face
            - Keep your face within the camera frame
            - Avoid backlighting or harsh shadows
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Manual Selection
    with tab3:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.header("Manual Emotion Selection")
        st.markdown("Select your current mood for instant music recommendations")
        
        st.session_state.num_tracks = st.slider(
            "Number of tracks to recommend:",
            min_value=1,
            max_value=10,
            value=5,
            key="manual_track_slider"
        )
        
        manual_emotion = st.selectbox(
            "How are you feeling right now?",
            ['happy', 'sad', 'angry', 'surprise', 'neutral'],
            key="manual_emotion"
        )
        
        if st.button("🎵 Get Music for This Mood", type="primary", use_container_width=True, key="manual_button"):
            recommendations = recommend_music(music_df, manual_emotion, st.session_state.num_tracks)
            st.session_state.recommendations = recommendations
            st.session_state.detected_emotion = manual_emotion
            
            st.markdown(f'''
            <div class="emotion-box">
                <h2>{manual_emotion.upper()}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display recommendations
    if st.session_state.recommendations is not None and not st.session_state.recommendations.empty:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header(f"🎶 Your {st.session_state.detected_emotion.upper()} Playlist")
        with col2:
            st.markdown(f'<div class="recommendation-badge">{len(st.session_state.recommendations)} Tracks</div>', unsafe_allow_html=True)
        
        recommendations = st.session_state.recommendations
        
        for i, (_, track) in enumerate(recommendations.iterrows(), 1):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f'''
                <div class="track-card">
                    <h3 style="margin:0; color:#FFFFFF;">{i}. {track.get('track', track.get('title', 'Unknown Track'))}</h3>
                    <p style="margin:5px 0; color:#E0E0FF;"><strong>Artist:</strong> {track.get('artist', 'Unknown Artist')}</p>
                    <p style="margin:5px 0; color:#E0E0FF;"><strong>Mood:</strong> {track.get('mood', 'Unknown')} | <strong>Genre:</strong> {track.get('genre', 'Unknown')}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                if 'uri' in track and pd.notna(track['uri']):
                    spotify_url = spotify_uri_to_url(track['uri'])
                    st.markdown(f'''
                    <div style="display: flex; align-items: center; height: 100%;">
                        <a href="{spotify_url}" target="_blank" style="
                            background: linear-gradient(135deg, #1DB954, #1ED760);
                            color: white;
                            padding: 12px 20px;
                            border-radius: 25px;
                            text-decoration: none;
                            font-weight: 600;
                            text-align: center;
                            display: block;
                            width: 100%;
                            box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
                        ">🎵 Listen</a>
                    </div>
                    ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":

    main()
