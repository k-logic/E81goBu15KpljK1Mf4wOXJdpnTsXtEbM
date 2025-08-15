#if !defined(USE_TENSORRT) && !defined(USE_TFLITE)
    #error "You must define either -DUSE_TENSORRT or -DUSE_TFLITE"
#endif

#define FMT_HEADER_ONLY
#define ASIO_STANDALONE
// 標準ライブラリ
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <queue>
#include <optional>
#include <chrono>
// 外部ライブラリ
#include <fmt/core.h>
#include <asio/asio.hpp>
#include <opencv2/opencv.hpp>
// 自作ライブラリ
#include <config.hpp>
#include <packet.hpp>
#include <chunker.hpp>
#include <other_utils.hpp>
#include <debug_utils.hpp>
#include <udp_server.hpp>
#include <image_utils.hpp>
#include <image_display.hpp>
#include <image_display2.hpp>

#if defined(USE_TENSORRT)
#include <IModelExecutor.hpp>
#include <TensorRTExecutor.hpp>
#include <cuda_runtime.h>
#endif

#if defined(USE_TFLITE)
#include <IModelExecutor.hpp>
#include <TFLiteExecutor.hpp>
#endif

using namespace config;

// 1フレーム分だけ保持するバッファ構造
struct FrameBuffer {
    std::vector<std::vector<float>> chunks; // 各チャンクのデータ
    std::vector<bool> received_flags;       // 到着フラグ
    int received_count = 0;
    int chunk_total = 0;
};

static FrameBuffer current_frame;
static uint32_t current_frame_id = 0;

// グローバルまたはmainの外で固定バッファを持つ
static std::vector<float> hwc(ENCODER_OUT_C * ENCODER_OUT_H * ENCODER_OUT_W);
static std::vector<float> decoded(ENCODER_IN_C * ENCODER_IN_H * ENCODER_IN_W);

// UDP受信処理
void on_receive(const udp::endpoint& sender, const std::vector<uint8_t>& packet, IModelExecutor& decoder_model) {
    try {
        auto parsed = packet::parse_packet_u8(packet);

        // 新しいフレームが来たら現フレームを表示
        if (current_frame_id != UINT32_MAX && parsed.header.frame_id != current_frame_id) {
            // 欠損黒埋め
            for (int i = 0; i < current_frame.chunk_total; ++i) {
                if (!current_frame.received_flags[i]) {
                    std::fill(current_frame.chunks[i].begin(), current_frame.chunks[i].end(), 0.0f);
                }
            }

            // 復元
            /*
            chunker::reconstruct_from_chunks_hwc(
                current_frame.chunks,
                hwc.data(),
                ENCODER_OUT_C,
                ENCODER_OUT_H,
                ENCODER_OUT_W
            );
            */
            
            chunker::reconstruct_from_tiles_hwc(
                current_frame.chunks,
                current_frame.received_flags,
                hwc.data(),
                ENCODER_OUT_C,
                ENCODER_OUT_H,
                ENCODER_OUT_W,
                CHUNK_PIXEL_W,
                CHUNK_PIXEL_H,
            );

            // デコード
            decoder_model.run(hwc, decoded);

            // 表示
            //image_display::display_decoded_image_chw(decoded.data(), ENCODER_IN_C, ENCODER_IN_H, ENCODER_IN_W);
            image_display::enqueue_frame_chw(decoded.data(), ENCODER_IN_C, ENCODER_IN_H, ENCODER_IN_W);

            // 新フレーム用に初期化
            current_frame_id = parsed.header.frame_id;
            current_frame.chunk_total = parsed.header.chunk_total;
            current_frame.received_count = 0;
            current_frame.chunks.assign(current_frame.chunk_total, std::vector<float>(CHUNK_PIXEL * ENCODER_OUT_C, 0.0f));
            current_frame.received_flags.assign(current_frame.chunk_total, false);
        }

        // 初期化（初回または上の分岐後）
        if (current_frame_id == UINT32_MAX) {
            current_frame_id = parsed.header.frame_id;
            current_frame.chunk_total = parsed.header.chunk_total;
            current_frame.received_count = 0;
            current_frame.chunks.assign(current_frame.chunk_total, std::vector<float>(CHUNK_PIXEL * ENCODER_OUT_C, 0.0f));
            current_frame.received_flags.assign(current_frame.chunk_total, false);
        }

        // チャンク格納
        if (parsed.header.chunk_id < current_frame.chunk_total &&
            !current_frame.received_flags[parsed.header.chunk_id]) {
            current_frame.chunks[parsed.header.chunk_id] = other_utils::fp8_bytes_to_float32(parsed.compressed);
            current_frame.received_flags[parsed.header.chunk_id] = true;
            current_frame.received_count++;
        }

    } catch (const std::exception& e) {
        std::cerr << "[RECEIVE ERROR] " << e.what() << "\n";
    }
}


// 非同期サーバー起動
asio::awaitable<void> run_server(UdpServer& server, IModelExecutor& decoder_model) {
    co_await server.start([&](const udp::endpoint& sender, const std::vector<char>& data) {
        std::vector<uint8_t> raw_packet(data.begin(), data.end());
        on_receive(sender, raw_packet, decoder_model);
    });
}

int main() {
    asio::io_context io;
    UdpServer server(io, 8004);

    std::unique_ptr<IModelExecutor> decoder_model;
    
    image_display::start_display_thread();

    // LiteRT用
    #if defined(USE_TFLITE)
        decoder_model = std::make_unique<TFLiteExecutor>();
        decoder_model->load(DECODER_PATH);
    #endif

    // TensorRT用
    #if defined(USE_TENSORRT)
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        decoder_model = std::make_unique<TensorRTExecutor>(stream);
        decoder_model->load(DECODER_PATH);
    #endif

    // 即時デコード型なのでスレッドは不要
    asio::co_spawn(io, run_server(server, *decoder_model), asio::detached);
    io.run();
    image_display::stop_display_thread();

    return 0;
}