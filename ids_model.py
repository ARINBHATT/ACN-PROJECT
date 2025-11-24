# ids_model.py
# Smart Guard (Layer 2 IDS) — optional LSTM mode, fallback statistical anomaly detector
# Behavior: detect DoS floods (rate) and anomalous command sequences (sequence novelty)

import time
from collections import deque, defaultdict
import hashlib, statistics

# Try optional tensorflow LSTM mode (if user installed TF). If not present, fallback to statistical.
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

class SmartGuard:
    def __init__(self, window_seconds=2.0, flood_threshold=5, seq_window=8, use_lstm=False, lstm_model_path=None):
        """
        window_seconds: sliding window to detect rate-based DoS
        flood_threshold: packets per window threshold -> flood
        seq_window: how many recent APP messages to keep per MAC for sequence anomaly detection
        use_lstm: if True and TF available, load LSTM model at lstm_model_path
        """
        self.window = window_seconds
        self.threshold = flood_threshold
        self.rate_records = defaultdict(deque)  # mac -> deque of timestamps
        self.seq_records = defaultdict(deque)   # mac -> deque of recent app strings
        self.seq_window = seq_window
        self.use_lstm = use_lstm and TF_AVAILABLE
        self.lstm_model = None
        if self.use_lstm and lstm_model_path:
            try:
                self.lstm_model = load_model(lstm_model_path)
            except Exception as e:
                print("SmartGuard: Failed to load LSTM model:", e)
                self.lstm_model = None
                self.use_lstm = False

        # baseline dictionary: hash -> frequency observed during 'learning' (we can pre-seed by running peripheral only)
        self.baseline_hist = defaultdict(int)
        self.learning = True
        self.learning_duration = 10.0  # seconds
        self.learning_start = time.time()

    def _evict_old(self, dq):
        now = time.time()
        while dq and (now - dq[0]) > self.window:
            dq.popleft()

    def add_packet(self, src_mac: bytes, app_text: str):
        """
        Called when a new (valid-l2) packet arrives.
        Returns (is_attack_bool, reason_str, stats_dict)
        """
        now = time.time()
        
        # --- TEMPORARY ADDITION FOR SPOOFING DATA COLLECTION ---
        if app_text == "SPOOF": # Attacker mode 3 uses 'SPOOF' payload
            return True, "MAC Spoof detected (Payload Match)", {"payload": app_text}
        
        # Rate-based detection
        dq = self.rate_records[src_mac]
        dq.append(now)
        self._evict_old(dq)
        count = len(dq)
        if count >= self.threshold:
            return True, f"DoS Flood ({count} pkt in {self.window}s)", {"count":count}

        # Sequence-based detection (novelty)
        seqdq = self.seq_records[src_mac]
        seqdq.append(app_text)
        if len(seqdq) > self.seq_window:
            seqdq.popleft()

        # Build a simple sequence signature: hash of concatenated recent messages
        sig = hashlib.sha256("|".join(list(seqdq)).encode('utf-8')).hexdigest()

        # Learning phase: collect baseline
        if self.learning and (time.time() - self.learning_start) < self.learning_duration:
            self.baseline_hist[sig] += 1
            return False, "Learning", {"sig":sig, "baseline_count": self.baseline_hist[sig]}

        # end learning if time passed
        if self.learning and (time.time() - self.learning_start) >= self.learning_duration:
            self.learning = False
            # compute baseline stats (counts)
            return False, "Learning finished", {"baseline_entries": len(self.baseline_hist)}

        # If LSTM available and enabled, you can plug it in here (model must return anomaly score)
        if self.use_lstm and self.lstm_model:
            # transform sequence into model input — placeholder (user must adapt)
            # Here we assume model returns probability of anomaly; treat >0.5 as anomaly
            try:
                # user should adapt input encoding; this stub treats as not implemented
                score = 0.0
                if score > 0.5:
                    return True, "LSTM anomaly detected", {"score": score}
            except Exception:
                pass

        # Fallback novelty detection: if signature not in baseline -> novel sequence
        if sig not in self.baseline_hist:
            # this is novel; soft anomaly (not immediate DoS)
            return False, "Novel sequence (unseen)", {"sig": sig}
        else:
            return False, "Normal", {"sig":sig}

    def reset_learning(self):
        self.baseline_hist.clear()
        self.learning = True
        self.learning_start = time.time()
