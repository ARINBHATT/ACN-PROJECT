# live_ids.py
# Real-time LSTM Intrusion Detection System
# Loads the trained model and detects attacks live!

import socket
import time
import joblib
import numpy as np
import tensorflow as tf
import hmac
import hashlib
from collections import deque, defaultdict

# Import your existing project modules
from ble_phy_simulator import parse_phy_packet
from ble_link_layer import parse_l2_pdu
from ids_model import SmartGuard

# --- CONFIGURATION ---
MODEL_FILE = 'lstm_ids_model.keras'
SCALER_FILE = 'scaler.pkl'
CHANNELS = [
    ("pipe_channel_37", 50037),
    ("pipe_channel_38", 50038),
    ("pipe_channel_39", 50039),
]
SHARED_SECRET = b"super_shared_secret_2025"
TIME_STEPS = 10   # Must match what we used for training
THRESHOLD = 0.5   # Probability threshold for attack detection
# ---------------------

# 1. Load Model and Scaler
print(f"[*] Loading Model ({MODEL_FILE}) and Scaler ({SCALER_FILE})...")
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("[+] AI Brain Loaded Successfully!")
except Exception as e:
    print(f"[-] Error loading AI files: {e}")
    exit()

# 2. Setup Helpers (Sequencer, SmartGuard for features)
# We still use SmartGuard to calculate 'seq_baseline_count' and 'packet_rate'
smartguard = SmartGuard(window_seconds=2.0, flood_threshold=5, seq_window=6)

def channel_from_hmac(secret: bytes, hop_index: int, channels):
    h = hmac.new(secret, hop_index.to_bytes(8, 'big'), hashlib.sha256).digest()
    idx = int.from_bytes(h[:4], 'big') % len(channels)
    return channels[idx]

class HoppingSequencer:
    def __init__(self, secret, channels):
        self.secret = secret
        self.channels = channels
    def get_channel(self, hop_index):
        return channel_from_hmac(self.secret, hop_index, self.channels)

sequencer = HoppingSequencer(SHARED_SECRET, CHANNELS)

# 3. State Variables
last_timestamp = {}  # mac -> time
last_seq = {}        # mac -> int
timestamps_window = defaultdict(deque) # mac -> deque of times
# This buffer holds the recent raw features to form the sequence for LSTM
packet_buffer = deque(maxlen=TIME_STEPS) 

def seq_delta(prev, cur):
    if prev is None: return 0
    return (cur - prev) if cur >= prev else (cur + 256 - prev)

def process_packet(channel_name, data):
    now = time.time()
    
    # --- A. Parse Layer 1 & 2 ---
    l2_payload = parse_phy_packet(data, verbose=False)
    if l2_payload is None: return

    try:
        seq, src_mac_bytes, app_data_bytes = parse_l2_pdu(l2_payload, verbose=False)
    except: return

    # Normalize MAC
    try: mac_str = src_mac_bytes.hex().upper()
    except: mac_str = str(src_mac_bytes)

    # --- B. Extract Features (Must match training EXACTLY) ---
    # 1. Inter-arrival
    prev_ts = last_timestamp.get(mac_str)
    inter_arrival = (now - prev_ts) if prev_ts else 0.0
    last_timestamp[mac_str] = now

    # 2. Seq Delta
    prev_s = last_seq.get(mac_str)
    delta = seq_delta(prev_s, seq)
    last_seq[mac_str] = seq

    # 3. Packet Rate
    dq = timestamps_window[mac_str]
    dq.append(now)
    while dq and (now - dq[0]) > 2.0: dq.popleft()
    packet_rate = len(dq)

    # 4. Hop Valid
    expected_ch, _ = sequencer.get_channel(seq)
    hop_valid = 1 if expected_ch == channel_name else 0

    # 5. Channel Index
    ch_idx = int(channel_name.split("_")[-1]) if "_" in channel_name else 0

    # 6. Payload Length
    p_len = len(app_data_bytes) if app_data_bytes else 0

    # 7. Seq Baseline Count (from SmartGuard)
    # We call SmartGuard just to get the 'info' dict, we ignore its boolean decision
    _, _, info = smartguard.add_packet(src_mac_bytes, app_data_bytes.decode('utf-8', errors='ignore'))
    base_count = info.get("baseline_count", 0)

    # Feature Vector (Raw)
    # Order: ['inter_arrival', 'hop_valid', 'seq_delta', 'channel_index', 'payload_length', 'packet_rate_2s', 'seq_baseline_count']
    raw_features = [inter_arrival, hop_valid, delta, ch_idx, p_len, packet_rate, base_count]

    # --- C. Prepare for AI ---
    # 1. Scale the single packet features
    # Reshape to (1, -1) because scaler expects 2D array
    scaled_features = scaler.transform(np.array(raw_features).reshape(1, -1))
    
    # 2. Add to sliding window buffer
    packet_buffer.append(scaled_features[0])

    # --- D. Predict (Only if we have enough history) ---
    if len(packet_buffer) == TIME_STEPS:
        # Convert buffer to 3D tensor: (1 sample, 10 time_steps, 7 features)
        input_seq = np.array(packet_buffer).reshape(1, TIME_STEPS, 7)
        
        # Get Prediction Score (0.0 to 1.0)
        score = model.predict(input_seq, verbose=0)[0][0]
        
        # Interpret Result
        status = "NORMAL"
        color = "\033[92m" # Green
        if score > THRESHOLD:
            status = "!! ATTACK !!"
            color = "\033[91m" # Red
        
        print(f"{color}[{status}] Score: {score:.4f} | MAC: {mac_str} | Seq: {seq} | Hop: {hop_valid}\033[0m")

def listener_thread(channel_name, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    while True:
        data, _ = sock.recvfrom(65536)
        process_packet(channel_name, data)

import threading
def main():
    print("--- AI SECURITY MONITOR STARTED ---")
    print("Waiting for traffic to fill the LSTM buffer...")
    for name, port in CHANNELS:
        t = threading.Thread(target=listener_thread, args=(name, port), daemon=True)
        t.start()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Monitor stopping.")

if __name__ == "__main__":
    main()