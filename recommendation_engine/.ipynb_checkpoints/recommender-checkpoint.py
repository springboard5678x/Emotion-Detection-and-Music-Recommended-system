import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os # Added os to check if file exists for better error message

# --- ⚙️ Configuration ---
# CORRECTED PATH: Points inside the 'data' folder
PROCESSED_MUSIC_DATA_PATH = '../data/music_processed/processed_music_tags.csv'

# --- 🎵 Emotion to Music Mapping 🎵 ---
EMOTION_TO_MUSIC_MAP = {
    'happy': 'upbeat happy energetic dance pop joy',
    'sad': 'sad mellow slow acoustic blues classical instrumental',
    'angry': 'angry rock metal intense heavy punk industrial',
    'fear': 'ambient experimental instrumental calm classical soothing',
    'surprise': 'electronic pop dance energetic new wave synthpop',
    'neutral': 'lounge chill instrumental ambient pop easy listening'
}


class MusicRecommender:
    def __init__(self, data_path):
        """
        Initializes the recommender by loading data and building the model.
        """
        print("Initializing Music Recommender...")
        self.df = self._load_data(data_path)
        
        # --- BUG FIX ---
        # Only build the recommender if the data was loaded successfully
        if not self.df.empty:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['tags'])
            print("Recommender initialized successfully. ✅")
        else:
            print("Recommender initialization failed due to missing data. 🔴")

    def _load_data(self, data_path):
        """Loads the processed music data from the given path."""
        if not os.path.exists(data_path):
            print(f"--- 🔴 ERROR: File not found! ---")
            print(f"Path: '{data_path}' does not exist.")
            print("Please ensure you have run the 'music_processor.py' script and that the output folder structure is correct.")
            return pd.DataFrame()
        
        return pd.read_csv(data_path)

    def recommend_songs(self, emotion, num_recommendations=10):
        """
        Recommends songs based on a detected emotion.
        """
        # Check if the recommender failed to initialize
        if self.df.empty:
            print("Cannot recommend songs. The music data is not loaded.")
            return pd.DataFrame()

        query_tags = EMOTION_TO_MUSIC_MAP.get(emotion.lower())
        if not query_tags:
            print(f"Warning: No music mapping found for emotion '{emotion}'.")
            return pd.DataFrame()

        query_vector = self.tfidf_vectorizer.transform([query_tags])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_song_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]
        
        recommended_songs = self.df.iloc[top_song_indices]
        return recommended_songs[['artist_name', 'title', 'tags']]


# --- Example of how to use this script ---
if __name__ == '__main__':
    recommender = MusicRecommender(PROCESSED_MUSIC_DATA_PATH)
    
    # Only try to recommend if initialization was successful
    if not recommender.df.empty:
        sample_emotion = 'sad' 
        print(f"\nLooking for songs for a '{sample_emotion}' mood...")
        
        recommendations = recommender.recommend_songs(sample_emotion)
        
        if not recommendations.empty:
            print("\n--- Here are your recommendations ---")
            print(recommendations)
            print("-----------------------------------")