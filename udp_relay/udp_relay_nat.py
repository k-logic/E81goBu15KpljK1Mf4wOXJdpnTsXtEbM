import asyncio
import time

LISTEN_PORT = 6490       # サーバーが受け付けるポート
HEARTBEAT_TIMEOUT = 10   # Decoder の登録が切れる秒数

# Decoderの登録情報
decoder_addr = None
last_heartbeat = 0

async def udp_server():
    global decoder_addr, last_heartbeat
    print(f"[INFO] UDP relay server listening on 0.0.0.0:{LISTEN_PORT}")

    transport, protocol = await asyncio.get_event_loop().create_datagram_endpoint(
        lambda: RelayProtocol(),
        local_addr=("0.0.0.0", LISTEN_PORT)
    )

    try:
        while True:
            # 登録が古くなったら削除
            if decoder_addr and time.time() - last_heartbeat > HEARTBEAT_TIMEOUT:
                print("[INFO] Decoder registration expired")
                decoder_addr = None
            await asyncio.sleep(1)
    finally:
        transport.close()

class RelayProtocol:
    def connection_made(self, transport):
        self.transport = transport


    def datagram_received(self, data, addr):
        global decoder_addr, last_heartbeat

        if data.startswith(b"HEARTBEAT"):
            # Decoderからのハートビート
            decoder_addr = addr
            last_heartbeat = time.time()
            print(f"[INFO] Heartbeat from Decoder {addr}")
            return

        # それ以外は全部 Decoder に転送
        if decoder_addr:
            self.transport.sendto(data, decoder_addr)
            print(f"[FWD] {len(data)} bytes to Decoder {decoder_addr}")
        else:
            print("[WARN] No Decoder registered, dropping packet")


if __name__ == "__main__":
    asyncio.run(udp_server())

