# peripheral.py
# Peripheral (Good Device) — Moving Wall + L2 packaging
# Sends Packet-{hop_index} messages and includes seq number inside L2.

import socket, time, hmac, hashlib, random
import ble_link_layer as l2
import ble_phy_simulator as l1

VERBOSE = True

CHANNELS = [
    ("pipe_channel_37", 50037),
    ("pipe_channel_38", 50038),
    ("pipe_channel_39", 50039),
]

SHARED_SECRET = b"super_shared_secret_2025"  # change as needed

def channel_from_hmac(secret: bytes, hop_index: int, channels):
    # deterministic selection using HMAC-SHA256 over hop_index
    h = hmac.new(secret, hop_index.to_bytes(8, 'big'), hashlib.sha256).digest()
    idx = int.from_bytes(h[:4], 'big') % len(channels)
    return channels[idx]

class HoppingSequencer:
    def __init__(self, secret, channels):
        self.secret = secret
        self.channels = channels

    def get_channel(self, hop_index):
        return channel_from_hmac(self.secret, hop_index, self.channels)

def main():
    print("Peripheral: Moving Wall transmitter (UDP).")
    sequencer = HoppingSequencer(SHARED_SECRET, CHANNELS)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hop_index = 0
    try:
        while True:
            channel_name, port = sequencer.get_channel(hop_index)
            app_text = f"Packet-{hop_index}"
            app_bytes = app_text.encode('utf-8')
            if VERBOSE:
                print(f"\nTX (Hop {hop_index}) -> {channel_name}:{port}  APP='{app_text}'")
            # create L2 PDU with embedded seq
            l2_pdu = l2.create_l2_pdu_with_seq(app_bytes, seq=hop_index, src_mac=l2.PERIPHERAL_MAC, verbose=VERBOSE)
            raw = l1.create_phy_packet(l2_pdu, verbose=VERBOSE)
            sock.sendto(raw, ("127.0.0.1", port))
            hop_index += 1
            time.sleep(1)  # advertising interval
    except KeyboardInterrupt:
        print("\nPeripheral shutting down...")

if __name__ == "__main__":
    main()
