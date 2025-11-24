# sniffer.py
# Sniffer + SmartGuard integration
# - L1 parse (parse_phy_packet)
# - L2 parse (parse_l2_pdu)
# - Hop validation using HMAC-based sequencer (same as peripheral)
# - SmartGuard (ids_model.SmartGuard) invocation
# - CSV logging of events

import socket
import threading
import time
import csv
import hmac
import hashlib
from pathlib import Path

from ble_phy_simulator import parse_phy_packet
from ble_link_layer import parse_l2_pdu
from ids_model import SmartGuard

# ---- Config ----
CHANNELS = [
    ("pipe_channel_37", 50037),
    ("pipe_channel_38", 50038),
    ("pipe_channel_39", 50039),
]
SHARED_SECRET = b"super_shared_secret_2025"
VERBOSE = True
CSV_LOG = True
CSV_FILE = "events_log.csv"

# SmartGuard parameters (tweakable)
smartguard = SmartGuard(window_seconds=2.0, flood_threshold=5, seq_window=6, use_lstm=False)

# ---- HMAC-based sequencer (same as peripheral) ----
def channel_from_hmac(secret: bytes, hop_index: int, channels):
    h = hmac.new(secret, hop_index.to_bytes(8, 'big'), hashlib.sha256).digest()
    idx = int.from_bytes(h[:4], 'big') % len(channels)
    return channels[idx]  # returns tuple (name, port)

class HoppingSequencer:
    def __init__(self, secret, channels):
        self.secret = secret
        self.channels = channels
    def get_channel(self, hop_index):
        return channel_from_hmac(self.secret, hop_index, self.channels)

sequencer = HoppingSequencer(SHARED_SECRET, CHANNELS)

# ---- CSV logger setup ----
if CSV_LOG:
    header = ["timestamp", "channel_name", "port", "seq", "mac", "app_text", "hop_valid", "prediction", "reason", "extra"]
    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        with open(CSV_FILE, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def log_event(row):
    if not CSV_LOG:
        return
    with open(CSV_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ---- Listener thread ----
def listener_thread(channel_name, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    if VERBOSE:
        print(f"Sniffer: Listening on {channel_name}:{port}")

    while True:
        data, _ = sock.recvfrom(65536)
        ts = time.time()

        if VERBOSE:
            print("\n  [L1 PHY] Parsing {} raw bytes...".format(len(data)))
            print("    [L1] Raw Data (hex):", data.hex())

        # L1 parse
        l2_payload = parse_phy_packet(data, verbose=VERBOSE)
        if l2_payload is None:
            if VERBOSE:
                print("    [L1] Dropped (invalid L1)")
            continue

        # L2 parse - expected to return (seq:int, src_mac:bytes, app_data:bytes)
        try:
            seq, src_mac_bytes, app_data_bytes = parse_l2_pdu(l2_payload, verbose=VERBOSE)
        except Exception as e:
            print("[Sniffer] L2 parse error:", e)
            continue

        # Normalize fields
        try:
            mac_str = src_mac_bytes.hex().upper()
        except Exception:
            # If parse returned int accidentally, convert
            if isinstance(src_mac_bytes, int):
                mac_str = f"{src_mac_bytes:012X}"
                src_mac_bytes = bytes.fromhex(mac_str)
            else:
                mac_str = str(src_mac_bytes)

        try:
            app_text = app_data_bytes.decode('utf-8', errors='ignore')
        except Exception:
            app_text = repr(app_data_bytes)

        # Hop validation (determine expected channel for this seq)
        expected_channel_name, _ = sequencer.get_channel(seq)
        hop_valid = (expected_channel_name == channel_name)
        hop_status = "Valid" if hop_valid else f"Invalid (expected {expected_channel_name})"

        # Run SmartGuard
        is_attack, reason, info = smartguard.add_packet(src_mac_bytes, app_text)
        prediction = "ATTACK" if is_attack else "Normal"

        # Print concise summary
        print(f"==> RX: [MAC:{mac_str}] [Seq:{seq}] [Ch:{channel_name}] [Hop:{hop_status}] [Pred:{prediction}] Reason:{reason}")

        # Log to CSV
        extra = info if isinstance(info, dict) else {"info": str(info)}
        log_row = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                   channel_name, port, seq, mac_str, app_text, hop_valid, prediction, reason, str(extra)]
        log_event(log_row)

        # If attack, optionally take immediate action (print alerts)
        if is_attack:
            print("!!! ALERT: SmartGuard flagged attack:", reason, "MAC:", mac_str, "Seq:", seq)


# ---- Main ----
def main():
    print("Sniffer: Starting listeners on all channels...")
    threads = []
    for name, port in CHANNELS:
        t = threading.Thread(target=listener_thread, args=(name, port), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nSniffer shutting down...")

if __name__ == "__main__":
    main()
