# 🎵 MoodMate: Emotion-Based Music Recommendation System

## 📘 Overview
**MoodMate** is an AI-driven application that detects human emotions from facial expressions and recommends mood-matched music in real time.  
It integrates **Computer Vision** and **Music Recommendation Systems** using the **FER2013 dataset** for emotion detection and the **Spotify API** for music retrieval.

---

## 🎯 Objective
To develop an intelligent system that identifies the user’s emotional state and recommends music that aligns with or enhances the user’s mood using **CNN** and **Transfer Learning models**.

---

## 🧠 Key Features
- 🎭 Emotion detection using facial expressions.  
- 🤖 Deep learning with CNN and Transfer Learning models.  
- 🎶 Music recommendation through Spotify Web API.  
- ⚙️ Content-based filtering using mood, valence, energy, and danceability.  
- 💻 Interactive interface for real-time emotion-based music suggestions.  

---

## 🗂️ Datasets Used
- **FER-2013 Dataset** – For facial emotion recognition ([Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)).  
- **Music Moods Dataset (CSV)** – Contains audio features like energy, valence, danceability, etc.  

---

## 🧩 System Architecture
1. **Data Collection & Preprocessing**  
   - Load and clean FER2013 and music datasets.  
   - Normalize and augment image data.  
   - Extract audio features from Spotify dataset.

2. **Emotion Detection Module**  
   - Train CNN and Transfer Learning models on FER2013.  
   - Predict user emotion from uploaded or webcam images.

3. **Music Recommendation Module**  
   - Map detected emotion to mood-related music tags.  
   - Generate personalized playlist recommendations via Spotify API.

4. **User Interface**  
   - Simple and responsive UI for uploading images or using webcam.  
   - Displays detected emotion and recommended songs.

---

## 🧪 Model Evaluation
- Accuracy, loss curves, and confusion matrix for emotion classification.  
- Precision and recall metrics for recommendation quality.  
- Real-time testing with sample images and Spotify integration.

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries:** TensorFlow / Keras, NumPy, Pandas, OpenCV, Matplotlib, Seaborn  
- **APIs:** Spotify Web API  
- **Frameworks:** Flask / Streamlit (for UI)  
- **Tools:** Jupyter Notebook, Kaggle  

---

## 🚀 How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/<your-username>/MoodMate-Emotion-Based-Music-Recommendation.git
