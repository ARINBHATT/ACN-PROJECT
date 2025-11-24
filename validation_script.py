import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt # NEW: for plotting
import seaborn as sns           # NEW: for heatmaps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
CSV_FILE = 'ml_dataset.csv'
MODEL_FILE = 'lstm_ids_model.keras'
SCALER_FILE = 'scaler.pkl'
TIME_STEPS = 10
THRESHOLD = 0.5 
# ---------------------

print("[*] Re-running preprocessing to get the Test Set...")

# 1. Load Data
df = pd.read_csv(CSV_FILE)
feature_cols = ['inter_arrival', 'hop_valid', 'seq_delta', 
                'channel_index', 'payload_length', 'packet_rate_2s', 
                'seq_baseline_count']
X_data = df[feature_cols].values
y_labels = df['label'].values

# 2. Scale Features (using the saved scaler or re-fitting)
try:
    # Load the scaler used during training
    scaler = joblib.load(SCALER_FILE)
    X_scaled = scaler.transform(X_data)
except FileNotFoundError:
    print("WARNING: Scaler file not found. Re-fitting scaler on full data.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_data)

# 3. Create Sequences
X_sequences, y_sequences = [], []
for i in range(len(X_scaled) - TIME_STEPS):
    X_sequences.append(X_scaled[i : i + TIME_STEPS])
    y_sequences.append(y_labels[i + TIME_STEPS])

X = np.array(X_sequences)
y = np.array(y_sequences)

# 4. Split Data (using the EXACT same settings as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Loaded {X_test.shape[0]} Test Samples.")

# 5. Load the trained model
print(f"[*] Loading trained model from {MODEL_FILE}...")
model = tf.keras.models.load_model(MODEL_FILE)

# 6. Predict and Convert to Binary
print("[*] Predicting on the Test Set...")
y_pred_probs = model.predict(X_test, verbose=0) 
y_pred = (y_pred_probs > THRESHOLD).astype(int)

# 7. Generate Report
print("\n--- Model Evaluation Results ---")
print("1. Classification Report (Primary Metrics):")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))

# 8. Generate Confusion Matrix Data
cm = confusion_matrix(y_test, y_pred)
print("\n2. Confusion Matrix (Row: Actual, Col: Predicted):")
print(cm)


# --- NEW: VISUALIZATION USING SEABORN ---

print("\n[*] Generating Confusion Matrix Heatmap...")
plt.figure(figsize=(8, 6))

# Define the labels for the axes
labels = ['Normal (0)', 'Attack (1)']

# Create the heatmap
sns.heatmap(
    cm,
    annot=True,     # Show the numbers in each cell
    fmt='d',        # Format as integers
    cmap='Blues',   # Use a blue color map
    xticklabels=labels,
    yticklabels=labels,
    cbar=True,
    linewidths=0.5,
    linecolor='black'
)

# Add title and labels
plt.title('LSTM Intrusion Detection Model: Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')

# Save and show the plot (Saving as PNG)
cm_filename = 'final_confusion_matrix_heatmap.png'
plt.savefig(cm_filename)
print(f"[SUCCESS] Confusion Matrix heatmap saved as {cm_filename}")
# plt.show() # Uncomment this line if you want the plot to pop up immediately (optional)

# ----------------------------------------