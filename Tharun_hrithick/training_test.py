import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt # <-- ADDED: Import for plotting

# --- Hyperparameters and Configuration ---
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT_PROB = 0.3
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'emotion_detection_pytorch_improved.pth'
MAX_LEN = 100
MIN_WORD_FREQ = 2
MAX_VOCAB_SIZE = 10000

# --- 1. Load and Preprocess Data ---
try:
    df = pd.read_csv("preprocessed_test.csv")
    print("Dataset loaded successfully.")
    print(f"Total samples: {len(df)}")
except FileNotFoundError:
    print("Error: 'preprocessed_test.csv' not found. Ensure your preprocessed data is in the current directory.")
    exit()

df['cleaned_text'] = df['cleaned_text'].fillna('')

# --- 2. Label Encoding ---
labels = df['label'].unique()
label_to_int = {label: i for i, label in enumerate(labels)}
int_to_label = {i: label for label, i in label_to_int.items()}
NUM_CLASSES = len(labels)

print(f"\nClass distribution:")
for label in labels:
    count = (df['label'] == label).sum()
    print(f"  {label}: {count} ({count/len(df)*100:.2f}%)")

df['label_int'] = df['label'].map(label_to_int)

# --- 3. Build Vocabulary ---
all_text = ' '.join(df['cleaned_text'])
word_counts = Counter(all_text.split())

filtered_vocab = {word: count for word, count in word_counts.items()
                  if count >= MIN_WORD_FREQ}
vocab = sorted(filtered_vocab, key=filtered_vocab.get, reverse=True)[:MAX_VOCAB_SIZE]

word_to_idx = {word: i+2 for i, word in enumerate(vocab)}
word_to_idx['<PAD>'] = 0
word_to_idx['<UNK>'] = 1
VOCAB_SIZE = len(word_to_idx)
print(f"\nVocabulary size: {VOCAB_SIZE}")
print(f"Words filtered out: {len(word_counts) - len(vocab)}")


# --- 4. PyTorch Dataset and DataLoader ---
class EmotionDataset(Dataset):
    """Custom PyTorch Dataset for emotion text data."""
    def __init__(self, texts, labels, word_to_idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        text_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
                        for word in text.split()]

        if len(text_indices) > self.max_len:
            text_indices = text_indices[:self.max_len]
        else:
            text_indices += [self.word_to_idx['<PAD>']] * (self.max_len - len(text_indices))

        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# --- 5. Model Architecture ---
class ImprovedEmotionClassifier(nn.Module):
    """Bi-LSTM with Attention and Classification Layers."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, dropout_prob):
        super(ImprovedEmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=dropout_prob if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def attention_net(self, lstm_output):
        """Calculates attention weights and context vector."""
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        attn_output = self.attention_net(lstm_out)
        attn_output = self.batch_norm(attn_output)
        out = self.dropout(self.relu(self.fc1(attn_output)))
        out = self.fc2(out)
        return out

# --- 6. Data Splitting ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

X_temp, X_test, y_temp, y_test = train_test_split(
    df['cleaned_text'].values, df['label_int'].values,
    test_size=0.15, random_state=42, stratify=df['label_int'].values
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15, random_state=42, stratify=y_temp
)

print(f"\nDataset split:")
print(f"  Train: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

train_dataset = EmotionDataset(X_train, y_train, word_to_idx, MAX_LEN)
val_dataset = EmotionDataset(X_val, y_val, word_to_idx, MAX_LEN)
test_dataset = EmotionDataset(X_test, y_test, word_to_idx, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# --- 7. Model Initialization, Loss, Optimizer ---
model = ImprovedEmotionClassifier(
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, DROPOUT_PROB
).to(device)

class_counts = np.bincount(df['label_int'].values)
class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# --- 8. Training Loop ---
best_val_acc = 0

# <-- ADDED: Lists to store metrics for plotting
train_loss_history = []
train_acc_history = []
val_acc_history = []


print("\nStarting training for all 30 epochs...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    avg_train_loss = total_loss / len(train_loader) # <-- ADDED: Calculate average loss

    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val

    # <-- ADDED: Append metrics to history lists
    train_loss_history.append(avg_train_loss)
    train_acc_history.append(train_accuracy)
    val_acc_history.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, '
          f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

    scheduler.step(val_accuracy)

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save({
            'model_state_dict': model.state_dict(),
            'word_to_idx': word_to_idx,
            'label_to_int': label_to_int,
            'int_to_label': int_to_label,
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_classes': NUM_CLASSES,
            'vocab_size': VOCAB_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout_prob': DROPOUT_PROB,
            'max_len': MAX_LEN
        }, MODEL_SAVE_PATH)
        print(f"  → Best model saved! Val Acc: {val_accuracy:.2f}%")

print("\nTraining complete. Loading best model for final evaluation...")


# --- 9. Plotting Training History ---
# <-- ADDED: This entire block is new
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot for Training Loss
ax1.plot(range(1, NUM_EPOCHS + 1), train_loss_history, label='Training Loss', color='blue', marker='o')
ax1.set_title('Training Loss vs. Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.legend()

# Plot for Training and Validation Accuracy
ax2.plot(range(1, NUM_EPOCHS + 1), train_acc_history, label='Training Accuracy', color='green', marker='o')
ax2.plot(range(1, NUM_EPOCHS + 1), val_acc_history, label='Validation Accuracy', color='red', marker='x')
ax2.set_title('Accuracy vs. Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()


# --- 10. Final Evaluation on Test Set ---
checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
correct_test = 0
total_test = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct_test / total_test
print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

print("\nPer-class accuracy:")
for i in range(NUM_CLASSES):
    class_mask = np.array(all_labels) == i
    if class_mask.sum() > 0:
        class_acc = (np.array(all_predictions)[class_mask] == i).sum() / class_mask.sum() * 100
        print(f"  {int_to_label[i]}: {class_acc:.2f}%")

print(f"\nModel saved to '{MODEL_SAVE_PATH}'")