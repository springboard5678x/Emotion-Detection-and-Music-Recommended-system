import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

try:
    music_df = pd.read_csv("cleaned_music_sentiment_dataset.csv")
    print("Music dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'cleaned_music_sentiment_dataset.csv' not found.")
    exit()


categorical_cols = ['Genre', 'Mood', 'Energy', 'Danceability', 'Sentiment_Label']
mappings = {}

for col in categorical_cols:
    cat_to_int = {cat: i for i, cat in enumerate(music_df[col].unique())}
    mappings[col] = cat_to_int
    music_df[f'{col}_encoded'] = music_df[col].map(cat_to_int)

print("\nCreated integer mappings for all categorical features.")

feature_cols = ['Genre_encoded', 'Mood_encoded', 'Energy_encoded', 'Danceability_encoded']
target_col = 'Sentiment_Label_encoded'

X = music_df[feature_cols].values
y = music_df[target_col].values

class MusicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_dataset = MusicDataset(X_train, y_train)
test_dataset = MusicDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Created PyTorch DataLoaders.")

class MusicRecommendationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicRecommendationModel, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x

INPUT_SIZE = len(feature_cols)
NUM_CLASSES = len(mappings['Sentiment_Label'])
MODEL_SAVE_PATH = 'music_recommendation_pytorch.pth'

model = MusicRecommendationModel(INPUT_SIZE, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nStarting model training...")
for epoch in range(25): 
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/25], Loss: {loss.item():.4f}')
print("Model training complete.")

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print(f"Mapping Percentage: {accuracy * 100:.2f}%")


torch.save({
    'model_state_dict': model.state_dict(),
    'mappings': mappings,
}, MODEL_SAVE_PATH)
print(f"\n  → Single model file saved to '{MODEL_SAVE_PATH}'")
