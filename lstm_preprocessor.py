import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
CSV_FILE = 'ml_dataset.csv'
TIME_STEPS = 10  # Lookback window: LSTM will look at 10 packets to make a prediction
TEST_SIZE = 0.2  # 20% for testing
# ---------------------

print(f"[*] Loading data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# 1. Select Features (X) and Labels (y)
# Drop timestamp and the final label column for features
feature_cols = ['inter_arrival', 'hop_valid', 'seq_delta', 
                'channel_index', 'payload_length', 'packet_rate_2s', 
                'seq_baseline_count']
X_data = df[feature_cols].values
y_labels = df['label'].values
print(f"    Loaded {len(df)} rows with {len(feature_cols)} features.")

# 2. Scale Features (Normalization)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_data)
print("    Features normalized using MinMaxScaler.")

# 3. Create Sequences (The crucial 3D transformation)
X_sequences, y_sequences = [], []

# Iterate through the data, grabbing a window of TIME_STEPS
for i in range(len(X_scaled) - TIME_STEPS):
    # The input sequence: features from packets i to i + TIME_STEPS - 1
    sequence = X_scaled[i : (i + TIME_STEPS)]
    X_sequences.append(sequence)
    
    # The target label: the label of the *last* packet in the sequence
    label = y_labels[i + TIME_STEPS]
    y_sequences.append(label)

X = np.array(X_sequences)
y = np.array(y_sequences)

print(f"    Created {X.shape[0]} sequences (samples).")
print(f"    X shape (samples, time_steps, features): {X.shape}")

# 4. Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=42,
    stratify=y  # Ensure attack labels are split evenly
)

print("\n--- Data Split Summary ---")
print(f"Training Samples: {X_train.shape[0]} (Attack: {np.sum(y_train)})")
print(f"Test Samples:     {X_test.shape[0]} (Attack: {np.sum(y_test)})")
print(f"Scaler object saved for later prediction.")
# np.savez('lstm_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
# print("Data saved to lstm_data.npz")