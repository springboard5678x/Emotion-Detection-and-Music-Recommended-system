import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


try:
    stopwords.words('english')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt_tab') 
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    print("NLTK resources downloaded successfully.")


df = pd.read_csv("test.csv")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'http\S+|www.\S+', '', text)

    text = re.sub(r'[^a-z\s]', '', text)

    tokens = nltk.word_tokenize(text)

    processed_tokens = []
    for word in tokens:
        if word not in stop_words:
            word = lemmatizer.lemmatize(word, pos='n') 
            processed_tokens.append(word)

    return " ".join(processed_tokens)

print("Starting Preprocessing...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
print("Preprocessing Complete.")

print("\n--- Original vs. Cleaned Data (First 5 Rows) ---")
print(df[['text', 'cleaned_text', 'label']].head())

print("\nSaving the preprocessed data to a new file...")
df.to_csv('preprocessed_test.csv', index=False)
print("Data saved successfully to 'preprocessed_test.csv'")