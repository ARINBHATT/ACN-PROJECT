# ml_sniffer.py
# ML-ready sniffer logger (Minimal feature set)
# Writes ml_dataset.csv with numeric features suitable for ML training (Option A)

import socket
import threading
import time
import csv
import hmac
import hashlib
from collections import deque, defaultdict
from pathlib import Path

from ble_phy_simulator import parse_phy_packet
from ble_link_layer import parse_l2_pdu
from ids_model import SmartGuard    # SmartGuard used to auto-label attacks (optional)

# ----------------- CONFIG -----------------
CHANNELS = [
    ("pipe_channel_37", 50037),
    ("pipe_channel_38", 50038),
    ("pipe_channel_39", 50039),
]
SHARED_SECRET = b"super_shared_secret_2025"
CSV_FILE = "ml_dataset.csv"
WINDOW_SECONDS = 2.0   # sliding window for packet_rate_2s
VERBOSE = False
# ------------------------------------------

# SmartGuard for auto-labeling (keeps same thresholds as before)
smartguard = SmartGuard(window_seconds=WINDOW_SECONDS, flood_threshold=5, seq_window=6, use_lstm=False)

# HMAC sequencer (same as peripheral)
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

# Prepare CSV
# --- MODIFICATION 1: ADDED seq_baseline_count ---
header = ["timestamp","inter_arrival","hop_valid","seq_delta","channel_index","payload_length","packet_rate_2s","seq_baseline_count","label"]
# ------------------------------------------------
csv_path = Path(CSV_FILE)
# Note: Delete/Remove the old ml_dataset.csv before running to get the new header
# if not csv_path.exists(): # Removed: we rely on manual deletion after data structure changes
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

# Per-MAC trackers
last_timestamp = {}            # mac_str -> last arrival timestamp
last_seq = {}                  # mac_str -> last seq int
timestamps_window = defaultdict(lambda: deque())   # mac_str -> deque of timestamps (for rate)

# Utility: compute seq delta handling wrap-around of 0-255 (if needed)
def seq_delta(prev, cur):
    if prev is None:
        return 0
    # assume 0-255 bytes rollover; handle general positive delta mod 256
    delta = (cur - prev) if cur >= prev else (cur + 256 - prev)
    return delta

def listener_thread(channel_name, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    if VERBOSE:
        print(f"[ML Sniffer] Listening on {channel_name}:{port}")

    while True:
        data, _ = sock.recvfrom(65536)
        now = time.time()

        # L1 parse
        l2_payload = parse_phy_packet(data, verbose=False)
        if l2_payload is None:
            continue

        # L2 parse -> seq (int), src_mac_bytes, app_data_bytes
        try:
            seq, src_mac_bytes, app_data_bytes = parse_l2_pdu(l2_payload, verbose=False)
        except Exception:
            continue

        # normalize mac string
        try:
            mac_str = src_mac_bytes.hex().upper()
        except Exception:
            if isinstance(src_mac_bytes, int):
                mac_str = f"{src_mac_bytes:012X}"
                src_mac_bytes = bytes.fromhex(mac_str)
            else:
                mac_str = str(src_mac_bytes)

        # inter-arrival
        prev_ts = last_timestamp.get(mac_str)
        if prev_ts is None:
            inter_arrival = 0.0
        else:
            inter_arrival = now - prev_ts

        last_timestamp[mac_str] = now

        # seq delta
        prev_seq = last_seq.get(mac_str)
        delta = seq_delta(prev_seq, seq)
        last_seq[mac_str] = seq

        # update sliding window deque for this mac (evict old)
        dq = timestamps_window[mac_str]
        dq.append(now)
        while dq and (now - dq[0]) > WINDOW_SECONDS:
            dq.popleft()
        packet_rate = len(dq)

        # hop validation
        expected_channel_name, _ = sequencer.get_channel(seq)
        hop_valid = 1 if expected_channel_name == channel_name else 0

        # features
        channel_index = int(channel_name.split("_")[-1]) if "_" in channel_name else 0
        payload_length = len(app_data_bytes) if app_data_bytes is not None else 0

        # auto-label using SmartGuard (so you don't have to manually label now)
        is_attack, reason, info = smartguard.add_packet(src_mac_bytes, app_data_bytes.decode('utf-8', errors='ignore'))
        
        # --- MODIFICATION 2: EXTRACT BASELINE COUNT ---
        baseline_count = info.get("baseline_count", 0)
        # ----------------------------------------------
        
        label = 1 if is_attack else 0

        # write CSV row
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            f"{inter_arrival:.6f}",
            hop_valid,
            delta,
            channel_index,
            payload_length,
            packet_rate,
            baseline_count, # NEW FEATURE INSERTED HERE
            label
        ]
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        if VERBOSE:
            print("[ML] wrote row:", row, "label:", label)

def main():
    print("[ML Sniffer] Starting on channels -> writing", CSV_FILE)
    threads = []
    for name, port in CHANNELS:
        t = threading.Thread(target=listener_thread, args=(name, port), daemon=True)
        t.start()
        threads.append(t)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[ML Sniffer] Exiting")

if __name__ == "__main__":
    main()