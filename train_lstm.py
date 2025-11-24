import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import joblib # to save the scaler

# --- CONFIGURATION ---
CSV_FILE = 'ml_dataset.csv'
MODEL_FILE = 'lstm_ids_model.keras'
SCALER_FILE = 'scaler.pkl'
TIME_STEPS = 10
EPOCHS = 20
BATCH_SIZE = 32
# ---------------------

print(f"[*] Loading and processing data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# 1. Prepare Features and Labels
feature_cols = ['inter_arrival', 'hop_valid', 'seq_delta', 
                'channel_index', 'payload_length', 'packet_rate_2s', 
                'seq_baseline_count']
X_data = df[feature_cols].values
y_labels = df['label'].values

# 2. Scale Features (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_data)
# Save scaler for later use in the live system
joblib.dump(scaler, SCALER_FILE)
print(f"    Scaler saved to {SCALER_FILE}")

# 3. Create Sequences (Sliding Window)
X_sequences, y_sequences = [], []
for i in range(len(X_scaled) - TIME_STEPS):
    X_sequences.append(X_scaled[i : i + TIME_STEPS])
    y_sequences.append(y_labels[i + TIME_STEPS])

X = np.array(X_sequences)
y = np.array(y_sequences)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Calculate Class Weights (To fix the imbalance)
# This tells the model: "Pay more attention to class 0 (Normal)"
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}
print(f"    Class Weights calculated: Normal={weights[0]:.2f}, Attack={weights[1]:.2f}")

# 6. Build LSTM Model
print("[*] Building LSTM Model...")
model = tf.keras.Sequential([
    # Input Layer: (10 time steps, 7 features)
    tf.keras.layers.Input(shape=(TIME_STEPS, len(feature_cols))),
    
    # LSTM Layer: Captures time-series patterns (like hopping)
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.2), # Prevents overfitting
    
    # Dense Layers for classification
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Output: 0 (Normal) to 1 (Attack)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 7. Train Model
print("[*] Starting Training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    class_weight=class_weights, # <--- Critical for your imbalanced data
    verbose=1
)
# --- Cross-Validation ---
print("\n[*] Performing 5-Fold Cross-Validation...")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
val_accuracies = []
val_losses = []

# Use the full sequenced dataset (X, y) for cross-validation
for train, test in kfold.split(X, y):
    print(f"--- Fold {fold_no}/{kfold.get_n_splits()} ---")
    
    # Re-build the model from scratch for each fold
    cv_model = tf.keras.models.clone_model(model)
    cv_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Calculate class weights for the current fold's training data
    fold_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y[train]), y=y[train])
    fold_class_weights = {0: fold_weights[0], 1: fold_weights[1]}

    # Train the model on the fold's training data
    cv_history = cv_model.fit(
        X[train], y[train],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X[test], y[test]),
        class_weight=fold_class_weights,
        verbose=0  # Set to 0 or 2 to keep logs clean
    )
    
    # Evaluate and store results
    scores = cv_model.evaluate(X[test], y[test], verbose=0)
    print(f"    Validation Loss: {scores[0]:.4f}")
    print(f"    Validation Accuracy: {scores[1]:.4f}")
    val_losses.append(scores[0])
    val_accuracies.append(scores[1])
    
    fold_no += 1

# --- Average Results ---
print("\n--- Cross-Validation Summary ---")
print(f"Average Validation Loss: {np.mean(val_losses):.4f}")
print(f"Average Validation Accuracy: {np.mean(val_accuracies):.4f}")

print("\n[*] Re-training model on the entire dataset before saving...")
# The original model is now trained on all available data for final deployment
history = model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights, # Using weights calculated on the full dataset earlier
    verbose=1
)

# 8. Save Model
model.save(MODEL_FILE)
print(f"\n[SUCCESS] Model trained and saved to {MODEL_FILE}")