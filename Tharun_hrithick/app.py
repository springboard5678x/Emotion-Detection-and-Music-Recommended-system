import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import numpy as np
import random
import re
import pickle
import sys
import os
from datetime import datetime
from collections import deque

# --- 1. Define the EXACT same Model Class from training_test.py ---
class EmotionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, dropout_prob):
        super(EmotionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dropped = self.dropout(hidden)
        output = self.fc(dropped)
        return output

# --- 2. Helper Functions for Preprocessing ---
def clean_text(text):
    """Cleans text in the same way as the training script."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_sequence(text, vocab):
    """Converts text to a sequence of integers using the vocab."""
    return [vocab.get(word, 1) for word in text.split()] 

def pad_sequence(seq, max_len):
    """Pads a sequence to max_len."""
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [0] * (max_len - len(seq)) 

# --- 3. Load Artifacts ---
MODEL_PATH = 'emotion_detection_pytorch_improved.pth'
MUSIC_DATA_PATH = 'cleaned_music_sentiment_dataset.csv'
SIMILARITY_MATRIX_PATH = 'new_dataset_similarity_matrix_tags.npy'
SONG_INDICES_PATH = 'new_dataset_song_indices_tags.pkl'

def check_files():
    files_needed = [
        MODEL_PATH, 
        MUSIC_DATA_PATH, 
        SIMILARITY_MATRIX_PATH, 
        SONG_INDICES_PATH
    ]
    missing_files = [f for f in files_needed if not os.path.exists(f)]
    if missing_files:
        print(f"!!! FATAL ERROR: Missing required files: {missing_files} !!!")
        print("Please make sure all model and data files are in the same directory as app.py")
        sys.exit(1)
    print("All required files found.")

# Global variables
VOCAB = None
LABEL_TO_EMOTION_MAP = None
MODEL = None
DF_MUSIC = None
SIM_MATRIX = None
SONG_INDICES = None
EMOTION_MAP = None
MAX_LENGTH = 60

# Store recent moods in memory (limited to last 50)
RECENT_MOODS = deque(maxlen=50)

def load_artifacts():
    global VOCAB, LABEL_TO_EMOTION_MAP, MODEL, DF_MUSIC, SIM_MATRIX, SONG_INDICES, EMOTION_MAP, MAX_LENGTH
    
    print("Loading all artifacts...")
    check_files()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Label to emotion mapping
        LABEL_TO_EMOTION_MAP = {
            0: 'sadness',
            1: 'joy',
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }
        print(f"Manual label-to-emotion map loaded: {LABEL_TO_EMOTION_MAP}")

        # Load PyTorch model
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False) 
        print("PyTorch model checkpoint loaded.")

        VOCAB = checkpoint['vocab']
        print(f"Vocabulary extracted from checkpoint. Size: {len(VOCAB)}")
        
        VOCAB_SIZE = len(VOCAB)
        EMBEDDING_DIM = checkpoint['embedding_dim']
        HIDDEN_DIM = checkpoint['hidden_dim']
        NUM_CLASSES = checkpoint['num_classes']
        NUM_LAYERS = checkpoint['num_layers']
        DROPOUT_PROB = checkpoint['dropout_prob']
        
        MODEL = EmotionModel(
            VOCAB_SIZE,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            NUM_CLASSES,
            NUM_LAYERS,
            DROPOUT_PROB
        )
        
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.to(device)
        MODEL.eval()
        print("Model weights loaded into structure. AI Model is ready.")

        # Load music data
        DF_MUSIC = pd.read_csv(MUSIC_DATA_PATH)
        SIM_MATRIX = np.load(SIMILARITY_MATRIX_PATH)
        with open(SONG_INDICES_PATH, 'rb') as f:
            SONG_INDICES = pickle.load(f)
        print("Music recommendation data loaded.")

        # Emotion to music sentiment mapping
        EMOTION_MAP = {
            'sadness': 'Sad',
            'joy': 'Happy',
            'love': 'Happy',
            'anger': 'Motivated',
            'fear': 'Sad',
            'surprise': 'Relaxed'
        }
        
        for label_text in LABEL_TO_EMOTION_MAP.values():
            if label_text not in EMOTION_MAP:
                print(f"Warning: Emotion '{label_text}' from your model is not in EMOTION_MAP. It will default to 'Happy'.")
                EMOTION_MAP[label_text] = 'Happy'
                
        print(f"Emotion map set: {EMOTION_MAP}")
        print("--- All artifacts loaded successfully. ---")

    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}. Please check file paths.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        print("This could be due to a mismatch between your .pth file and the EmotionModel class.")
        sys.exit(1)

# --- 4. Prediction & Recommendation Functions ---
def predict_emotion(text):
    """Predicts the emotion of a single text string."""
    global MODEL, VOCAB, LABEL_TO_EMOTION_MAP, MAX_LENGTH
    
    if not all([MODEL, VOCAB, LABEL_TO_EMOTION_MAP]):
        print("Error: Model, vocab, or label map not loaded.")
        return "Error"
        
    device = next(MODEL.parameters()).device

    cleaned = clean_text(text)
    sequenced = text_to_sequence(cleaned, VOCAB)
    padded = pad_sequence(sequenced, MAX_LENGTH)
    
    tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)
    tensor = tensor.to(device)

    MODEL.eval()
    with torch.no_grad():
        outputs = MODEL(tensor)
        _, predicted_idx = torch.max(outputs, 1) 
    
    predicted_index = predicted_idx.item()
    predicted_label = LABEL_TO_EMOTION_MAP.get(predicted_index, 'joy') 
    
    return predicted_label

def get_model_recommendations(df, matrix, indices, seed_song, num=5):
    """Gets recommendations from the similarity matrix."""
    try:
        if seed_song not in indices:
            print(f"Warning: Seed song '{seed_song}' not in song indices. Cannot use model.")
            return []
            
        idx = indices[seed_song]
        sim_scores = list(enumerate(matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        sim_scores = sim_scores[1:num+1] 
        song_indices = [i[0] for i in sim_scores]
        
        rec_songs = df.iloc[song_indices]
        
        recommendations = []
        for _, row in rec_songs.iterrows():
            recommendations.append({
                'name': row['Song_Name'],
                'artist': row['Artist']
            })
        return recommendations
        
    except Exception as e:
        print(f"Error in get_model_recommendations: {e}")
        return []

def get_fallback_songs(df, sentiment, num=5):
    """Gets random songs matching the sentiment as a fallback."""
    sentiment_songs = df[df['Sentiment_Label'].str.lower() == sentiment.lower()]
    
    if sentiment_songs.empty:
        return []
        
    num = min(num, len(sentiment_songs))
    sample_songs = sentiment_songs.sample(n=num)
    
    recommendations = []
    for _, row in sample_songs.iterrows():
        recommendations.append({
            'name': row['Song_Name'],
            'artist': row['Artist']
        })
    return recommendations

def get_seed_song(df, sentiment):
    """Picks one random song of the target sentiment to act as a 'seed'."""
    sentiment_songs = df[df['Sentiment_Label'].str.lower() == sentiment.lower()]
    if not sentiment_songs.empty:
        return sentiment_songs.sample(n=1).iloc[0]['Song_Name']
    return None

def format_timestamp():
    """Returns a formatted timestamp string."""
    now = datetime.now()
    return now.strftime("%B %d, %Y at %I:%M %p")

# --- 5. Flask App ---
app = Flask(__name__) 

# Load artifacts once on startup
load_artifacts()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        if 'text' not in data:
            return jsonify({'error': 'No text provided.'}), 400
        
        text = data['text']
        if not text:
             return jsonify({'error': 'Text cannot be empty.'}), 400
        
        # Predict emotion
        predicted_emotion_label = predict_emotion(text)
        
        # Map emotion to music sentiment
        mapped_sentiment_label = EMOTION_MAP.get(predicted_emotion_label, 'Happy')
        
        recommendations = []
        seed_song_name = None

        # Find seed song
        seed_song_name = get_seed_song(DF_MUSIC, mapped_sentiment_label)
        
        # Get recommendations
        if seed_song_name:
            recommendations = get_model_recommendations(
                DF_MUSIC,
                SIM_MATRIX,
                SONG_INDICES,
                seed_song_name,
                num=5
            )
            
            if not recommendations:
                print(f"Warning: Model recommendation failed for seed '{seed_song_name}'. Using fallback.")
                recommendations = get_fallback_songs(DF_MUSIC, mapped_sentiment_label, num=5)
        else:
            print(f"Warning: No songs found for sentiment '{mapped_sentiment_label}'. Using fallback.")
            recommendations = get_fallback_songs(DF_MUSIC, mapped_sentiment_label, num=5)
        
        # Save to recent moods
        mood_entry = {
            'text': text,
            'emotion': predicted_emotion_label.capitalize(),
            'timestamp': format_timestamp(),
            'song_count': len(recommendations)
        }
        RECENT_MOODS.appendleft(mood_entry)  # Add to beginning
            
        return jsonify({
            'emotion': predicted_emotion_label.capitalize(),
            'songs': recommendations,
            'mapped_sentiment': mapped_sentiment_label,
            'seed_song': seed_song_name
        })

    except Exception as e:
        print(f"An error occurred during /recommend: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/recent_moods', methods=['GET'])
def recent_moods():
    """Returns the list of recent mood searches."""
    try:
        return jsonify({
            'moods': list(RECENT_MOODS)
        })
    except Exception as e:
        print(f"Error retrieving recent moods: {e}")
        return jsonify({'error': 'Failed to retrieve recent moods'}), 500

@app.route('/clear_recent', methods=['POST'])
def clear_recent():
    """Clears all recent mood history."""
    try:
        RECENT_MOODS.clear()
        return jsonify({'success': True, 'message': 'Recent moods cleared'})
    except Exception as e:
        print(f"Error clearing recent moods: {e}")
        return jsonify({'error': 'Failed to clear recent moods'}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)