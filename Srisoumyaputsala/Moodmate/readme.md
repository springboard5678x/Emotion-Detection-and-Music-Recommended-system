🎵 AI MoodMate – Emotion-Based Music Recommendation System
😃 "Let your mood choose your music!"
📘 Overview

AI MoodMate is an intelligent emotion-based music recommendation system that analyzes facial expressions to detect emotions and recommends suitable songs that match the user’s mood.
Using deep learning (ResNet-18 CNN) for emotion classification and Spotify Million Songs Dataset for mood-based song mapping, the system delivers a personalized and engaging listening experience.

🧠 Key Features

🎭 Detects facial emotions using a customized CNN (ResNet-18) model.

🎶 Recommends songs based on emotion categories (e.g., happy, sad, angry, neutral).

📷 Supports both image uploads and real-time webcam input.

☁️ Deployed seamlessly using Gradio UI on Hugging Face Spaces.

🧩 Trained on FER-2013 dataset (grayscale & RGB modes).

🏗️ System Architecture
Dataset Collection (FER-2013, Spotify Million Songs)
        ↓
Data Preprocessing (Resize, Normalize, Augment)
        ↓
CNN Model Development (ResNet-18)
        ↓
Training & Validation (Accuracy, Loss Metrics)
        ↓
Music Recommendation (Emotion → Spotify Features)
        ↓
Web App Deployment (Gradio + Hugging Face)

📊 Datasets Used
🧩 1. FER-2013 (Facial Emotion Recognition)

Source: Kaggle

Classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

Images: 35,000+

Format: 48×48 grayscale facial images

🎧 2. Spotify Million Songs Dataset

Source: Spotify Developer Dataset

Features: Valence, Energy, Tempo, Danceability

Used to map songs to detected emotions

🧩 Model Summary – Custom ResNet-18

Framework: TensorFlow / Keras

Architecture: ResNet-18 (Customized CNN)

Total Parameters: 11.1 Million

Input: 48x48x1 (Grayscale)

Output: 7 Emotion Classes

📈 Training Details
Metric	Value
Epochs	50
Optimizer	Adam
Batch Size	64
Validation Accuracy	~53%
Validation Loss	1.23
📉 Classification Report
Emotion	Precision	Recall	F1-Score
Angry	0.29	0.51	0.37
Disgust	0.12	0.65	0.20
Fear	0.36	0.31	0.33
Happy	0.79	0.50	0.61
Neutral	0.53	0.33	0.41
Sad	0.37	0.42	0.39
Surprise	0.69	0.53	0.60
Overall Accuracy	44%		
🧩 Technical Stack
Category	Tools Used
Programming Language	Python
Frameworks	TensorFlow / Keras, PyTorch
Visualization	Matplotlib, Seaborn
Frontend/UI	Gradio
Deployment	Hugging Face Spaces
Datasets	FER-2013, Spotify Million Songs
🚀 Deployment

Platform: 🤗 Hugging Face Spaces

Interface: Gradio Web UI

Features: Upload image or enable webcam → Detect emotion → Recommend matching songs

💡 Technical Outcomes

Built a complete end-to-end AI pipeline integrating CNN-based emotion recognition with a recommender system.

Developed a custom ResNet-18 CNN for emotion classification.

Integrated Spotify dataset features (valence, energy) for emotion-to-song mapping.

Deployed with Gradio UI + Hugging Face hosting for real-time accessibility.

🔮 Future Scope

Improve model accuracy using transformers (Vision Transformer / CLIP).

Include multimodal input (voice tone, text sentiment).

Build mobile app integration using TensorFlow Lite.

Enable personalized recommendations using user history.

🧑‍💻 Team & Credits
Role	Name
Project Lead	Your Name
Model Development	Your Name
UI Design & Deployment	Your Name
Datasets	FER-2013, Spotify Million Songs
🖥️ User Interface

The web UI was designed using Gradio, featuring:

Simple drag-and-drop image upload.

Live webcam detection mode.

Real-time emotion display and song suggestions.

🙏 Acknowledgment

Special thanks to:

Kaggle for the FER-2013 dataset.

Spotify API for music data.

Hugging Face for free hosting.

Gradio for an elegant and easy-to-build user interface.

💬 Conclusion

AI MoodMate demonstrates how computer vision and music intelligence can combine to enhance emotional well-being through personalized, emotion-aware music recommendations.
