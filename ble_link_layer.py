# ble_link_layer.py
# Minimal simulated BLE Link Layer for your project
# Works with Moving Wall + Smart Guard IDS

import struct

# MAC addresses used by the project
PERIPHERAL_MAC = b'\xDE\xAD\xBE\xEF\xCA\xFE'     # 6 bytes
ATTACKER_MAC    = b'\xAA\xBB\xCC\xDD\xEE\xFF'    # 6 bytes

def create_l2_pdu_with_seq(app_bytes: bytes, seq: int, src_mac: bytes, verbose=False):
    """
    Create a simulated L2 PDU:
    +--------+-----------+------------+
    | seq(1) |  MAC(6)   | payload    |
    +--------+-----------+------------+

    Returns bytes ready for L1 wrapping.
    """
    if verbose:
        print("  [L2 LINK] Creating L2 PDU (with seq+MAC)")

    if not isinstance(src_mac, (bytes, bytearray)) or len(src_mac) != 6:
        raise ValueError("src_mac must be a 6-byte value")

    seq_byte = struct.pack("B", seq % 256)   # 0–255 rollover
    pdu = seq_byte + src_mac + app_bytes

    if verbose:
        print(f"    [L2] seq={seq}, src_mac={src_mac.hex()}, len={len(app_bytes)}")
        print(f"    [L2] PDU (hex): {pdu.hex()}")

    return pdu


def parse_l2_pdu(pdu: bytes, verbose=False):
    """
    Reverse of create_l2_pdu_with_seq():
    Extracts:
      seq   → int
      mac   → bytes(6)
      data  → bytes
    """
    if len(pdu) < 7:
        raise ValueError("L2 PDU too short")

    seq = pdu[0]
    mac = pdu[1:7]
    data = pdu[7:]

    if verbose:
        print("  [L2 LINK] Parsing L2 PDU...")
        print(f"    [L2] seq={seq}, src_mac={mac.hex()}, data_len={len(data)}")
        try:
            print(f"    [L2] Data='{data.decode(errors='ignore')}'")
        except:
            pass

    return seq, mac, data
