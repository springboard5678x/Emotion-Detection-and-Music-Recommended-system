# Emotion Detection and Music Recommendation System

An Artificial Intelligence and Machine Learning-based application that detects a user's emotional state from text input and recommends music based on the predicted emotion.

This project combines Natural Language Processing (NLP), Machine Learning, and content-based music recommendation techniques to create an intelligent and interactive music recommendation system.

---

## Overview

Music can have a significant impact on human emotions and mood. This project aims to provide personalized music recommendations by analyzing the emotional context of a user's text.

The system follows a two-stage process:

1. The user's text is preprocessed using Natural Language Processing techniques.
2. A trained emotion classification model predicts the emotion expressed in the text.
3. Based on the predicted emotion, suitable songs are filtered using audio features such as valence and energy.
4. The system displays personalized song recommendations through an interactive Streamlit web application.

---

## Features

- Detects emotions from user-provided text using NLP and machine learning.
- Performs text preprocessing, including tokenization, stopword removal, and stemming.
- Recommends music based on the predicted emotion.
- Filters songs using audio features such as valence and energy.
- Provides up to five personalized song recommendations.
- Supports available Spotify audio previews.
- Provides an interactive Streamlit web interface.

---

## Supported Emotions

The system supports the following emotion categories:

- Joy
- Love
- Surprise
- Sadness
- Anger
- Neutral

---

## Music Recommendation Logic

After predicting the user's emotion, the system filters songs based on audio features.

### Joy, Love and Surprise

For positive emotions, songs with higher valence and energy are recommended.

Valence > 0.6

Energy > 0.6

### Sadness and Anger

For sadness and anger, songs with lower valence are selected.

Valence < 0.4

### Other Emotions

For other emotional states, songs with moderate valence are selected.

0.4 <= Valence <= 0.6

The system then selects up to five songs from the filtered results and displays the song name, artist, and available audio preview.

---

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | TensorFlow, Keras |
| Natural Language Processing | NLTK |
| Data Processing | Pandas, NumPy |
| Machine Learning Utilities | Scikit-learn |
| Data Storage Format | CSV |
| Web Application Framework | Streamlit |

---


## Project Structure

```text
Emotion-Detection-and-Music-Recommendation-System/
│
├── app.py
├── Music_Recommendation_System_Source_code.ipynb
├── OVERSAMPLING.ipynb
└── README.md
```

---

## File Description

| File | Description |
|---|---|
| `app.py` | Main Streamlit application that accepts user text, predicts the emotion, and recommends music based on the predicted emotion. |
| `Music_Recommendation_System_Source_code.ipynb` | Main Jupyter Notebook containing the development and implementation of the music recommendation system. |
| `OVERSAMPLING.ipynb` | Jupyter Notebook used for handling class imbalance through oversampling during the emotion classification process. |
| `README.md` | Documentation containing the project overview, features, technologies used, setup instructions, and usage details. |

---

##Installation and Setup

###1. Clone the Repository

```bash
git clone https://github.com/Kotkondawar-Shruti/Emotion-Detection-and-Music-Recommendation-System.git
```

Navigate to the project directory:

```bash
cd Emotion-Detection-and-Music-Recommendation-System
```

---

##2. Create a Virtual Environment

Creating a virtual environment is recommended to keep project dependencies isolated.

Windows
```bash
python -m venv venv
venv\Scripts\activate
```
macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

---

##3. Install Required Libraries

Install the required dependencies:
```bash
pip install streamlit pandas numpy nltk tensorflow scikit-learn
```
Running the Application

Start the Streamlit application using:
```bash
streamlit run app.py
```
The application will be available at:

http://localhost:8501

