# attacker.py
# Multi-mode attacker that builds L2 PDUs and L1 frames so sniffer accepts them.
# Modes: normal, flood, spoof, replay, invalidhop, payload, knob

import time
import random
import socket
import struct
import sys

import ble_link_layer as l2    # uses create_l2_pdu_with_seq(seq, src_mac, app_bytes, verbose)
import ble_phy_simulator as l1 # uses create_phy_packet(l2_pdu, verbose)

# configuration: map channel -> port (same as peripheral/sniffer)
CHANNEL_PORTS = {37: 50037, 38: 50038, 39: 50039}

REAL_MAC_BYTES   = l2.PERIPHERAL_MAC if hasattr(l2, "PERIPHERAL_MAC") else b'\xDE\xAD\xBE\xEF\xCA\xFE'
ATTACKER_MAC_BYTES = l2.ATTACKER_MAC if hasattr(l2, "ATTACKER_MAC") else b'\x11\x22\x33\x44\x55\x66'

# invalid hop port (outside 37/38/39)
INVALID_PORT = 50030

# small helper: convert MAC bytes to display
def mac_to_str(mac_bytes):
    try:
        return mac_bytes.hex().upper()
    except:
        return str(mac_bytes)

# send raw L1 frame to a given port
def send_raw_to_port(raw_bytes, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(raw_bytes, ("127.0.0.1", port))
    finally:
        sock.close()

# convenience: build and send L2->L1 -> port
def build_and_send(seq, mac_bytes, app_text, channel, send_to_all=False, invalid=False):
    # build L2 PDU (expects create_l2_pdu_with_seq(app_bytes, seq, src_mac, verbose) or similar)
    try:
        # some implementations use signature: create_l2_pdu_with_seq(app_bytes, seq=..., src_mac=..., verbose=...)
        l2_pdu = l2.create_l2_pdu_with_seq(app_text.encode('utf-8'), seq=seq, src_mac=mac_bytes, verbose=False)
    except TypeError:
        # fallback if a different arg order exists
        l2_pdu = l2.create_l2_pdu_with_seq(app_text.encode('utf-8'), seq, mac_bytes, False)
    # wrap in L1
    raw = l1.create_phy_packet(l2_pdu, verbose=False)

    if invalid:
        # send to invalid port (different medium)
        send_raw_to_port(raw, INVALID_PORT)
        return

    if send_to_all:
        for p in CHANNEL_PORTS.values():
            send_raw_to_port(raw, p)
    else:
        # send to the port corresponding to the chosen channel
        if channel not in CHANNEL_PORTS:
            # if channel unknown, pick a random known port
            port = random.choice(list(CHANNEL_PORTS.values()))
        else:
            port = CHANNEL_PORTS[channel]
        send_raw_to_port(raw, port)


# =========================
# Attack mode implementations
# =========================

def normal_traffic():
    print("[*] Normal traffic (advertising-like) started. Ctrl+C to stop.")
    seq = 0
    try:
        while True:
            ch = random.choice([37, 38, 39])
            build_and_send(seq, REAL_MAC_BYTES, f"Packet-{seq}", ch)
            seq = (seq + 1) % 256
            time.sleep(1.0)  # 1-second interval like your peripheral
    except KeyboardInterrupt:
        print("\n[*] Normal traffic stopped.")


def flood_attack():
    print("[*] Flood attack started. High-rate packets. Ctrl+C to stop.")
    seq = 0
    try:
        while True:
            # very fast flood: send to all ports to maximize chance of hitting sniffer
            ch = random.choice([37, 38, 39])
            build_and_send(seq, ATTACKER_MAC_BYTES, "ATTACK PACKET", ch, send_to_all=True)
            seq = (seq + 1) % 256
            # no sleep or tiny sleep
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n[*] Flood attack stopped.")


def spoof_attack():
    print("[*] MAC spoofing attack started. Each packet uses a random MAC. Ctrl+C to stop.")
    seq = 0
    try:
        while True:
            # generate random MAC bytes
            fake = bytes([random.randint(0, 255) for _ in range(6)])
            ch = random.choice([37, 38, 39])
            build_and_send(seq, fake, "SPOOF", ch)
            seq = (seq + 1) % 256
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[*] MAC spoofing stopped.")


def replay_attack():
    print("[*] Replay attack started. Re-sending same seq repeatedly. Ctrl+C to stop.")
    fixed_seq = random.randint(0, 255)
    # Use ATTACKER_MAC_BYTES as source
    try:
        while True:
            ch = random.choice([37, 38, 39])
            build_and_send(fixed_seq, ATTACKER_MAC_BYTES, "REPLAY", ch)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[*] Replay attack stopped.")


# attacker.py (around line 128)

def invalid_hop_attack():
    print("[*] Invalid hop attack started. Sending fixed SEQ on mismatched channel. Ctrl+C to stop.")
    
    # We will use a fixed sequence number (e.g., 1) that maps to channel 39
    # The sniffer will expect this packet on channel 39.
    # We will send the packet on channel 37's port instead.
    
    fixed_seq = 1 # Choose a sequence that reliably maps to one of the channels (e.g., 39)
    # Target channel (where the sniffer will receive it)
    target_channel = 37 
    
    try:
        while True:
            # Send the packet with fixed_seq (expected on another channel) to port 50037 (target_channel)
            build_and_send(fixed_seq, ATTACKER_MAC_BYTES, "BADHOP", target_channel, invalid=False)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[*] Invalid hop attack stopped.")


def suspicious_payload_attack():
    print("[*] Suspicious payload attack started. Sending unusual payloads. Ctrl+C to stop.")
    seq = 0
    payloads = ["ATTACK!!!", "OVERFLOW_OVERFLOW_OVERFLOW", "MALWARE_PAYLOAD", "CMD:FORMAT", "AAAAAAAAAAAAAAAAAAAAAAAAAAAA"]
    try:
        while True:
            ch = random.choice([37, 38, 39])
            payload = random.choice(payloads)
            build_and_send(seq, ATTACKER_MAC_BYTES, payload, ch)
            seq = (seq + 1) % 256
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n[*] Suspicious payload attack stopped.")


def knob_attack():
    print("[*] KNOB simulation started. Sending KNOB_KEYLEN=1 payloads. Ctrl+C to stop.")
    seq = 0
    try:
        while True:
            ch = random.choice([37, 38, 39])
            build_and_send(seq, ATTACKER_MAC_BYTES, "KNOB_KEYLEN=1", ch)
            seq = (seq + 1) % 256
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[*] KNOB attack stopped.")


# =====================
# Main menu & launcher
# =====================
def main():
    print("\nSelect attack mode:")
    print("1. Normal packets")
    print("2. Flood attack")
    print("3. MAC Spoof attack")
    print("4. Replay attack")
    print("5. Invalid Hop attack")
    print("6. Suspicious Payload attack")
    print("7. KNOB attack")
    print("q. Quit\n")

    choice = input("Enter choice (1-7 or q): ").strip().lower()

    if choice == "1":
        normal_traffic()
    elif choice == "2":
        flood_attack()
    elif choice == "3":
        spoof_attack()
    elif choice == "4":
        replay_attack()
    elif choice == "5":
        invalid_hop_attack()
    elif choice == "6":
        suspicious_payload_attack()
    elif choice == "7":
        knob_attack()
    elif choice == "q":
        print("Bye.")
        sys.exit(0)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
