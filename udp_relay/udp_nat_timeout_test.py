import socket
import time

LISTEN_PORT = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", LISTEN_PORT))

print(f"[INFO] Listening on UDP port {LISTEN_PORT}")

decoder_addr = None

while True:
    if decoder_addr is None:
        data, addr = sock.recvfrom(2048)
        decoder_addr = addr
        print(f"[INFO] Registered decoder: {decoder_addr}")
        start = time.time()
    else:
        # 一定間隔で返す
        msg = b"PING"
        sock.sendto(msg, decoder_addr)
        print(f"[SEND] to {decoder_addr} {msg}")
        time.sleep(5)  # 5秒間隔で送信

