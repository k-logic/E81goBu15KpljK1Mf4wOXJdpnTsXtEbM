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
    std::vector<std::vector<uint8_t>> chunks; // 各チャンクのデータ
    std::vector<bool> received_flags;       // 到着フラグ
    int received_count = 0;
    int chunk_total = 0;
};

static FrameBuffer current_frame;
static uint32_t current_frame_id = 0;

// グローバルまたはmainの外で固定バッファを持つ
static std::vector<uint8_t> hwc(DECODER_IN_C * DECODER_IN_H * DECODER_IN_W);
static std::vector<float> decoded(DECODER_OUT_C * DECODER_OUT_H * DECODER_OUT_W);

// UDP受信処理
void on_receive(const udp::endpoint& sender, const std::vector<uint8_t>& packet, IModelExecutor& decoder_model) {
    try {
        auto parsed = packet::parse_packet_u8(packet);

        // 新しいフレームが来たら現フレームを表示
        if (current_frame_id != UINT32_MAX && parsed.header.frame_id != current_frame_id) {
            // ===== 時間計測用 =====
            auto t0 = std::chrono::high_resolution_clock::now();
            auto t_prev = t0;
            
            chunker::reconstruct_from_tiles_hwc(
                current_frame.chunks,
                current_frame.received_flags,
                hwc.data(),
                DECODER_IN_C,
                DECODER_IN_H,
                DECODER_IN_W,
                CHUNK_PIXEL_W,
                CHUNK_PIXEL_H
            );
            auto t1 = std::chrono::high_resolution_clock::now();

            std::vector<float> hwc_float32 = other_utils::fp8_to_float32(hwc);

            // デコード
            decoder_model.run(hwc_float32, decoded);
            auto t2 = std::chrono::high_resolution_clock::now();

            // 表示
            //image_display::display_decoded_image_chw(decoded.data(), DECODER_OUT_C, DECODER_OUT_H, DECODER_OUT_W);
            image_display::enqueue_frame_chw(decoded.data(), DECODER_OUT_C, DECODER_OUT_H, DECODER_OUT_W);
            auto t3 = std::chrono::high_resolution_clock::now();

            // 新フレーム用に初期化
            current_frame_id = parsed.header.frame_id;
            current_frame.chunk_total = parsed.header.chunk_total;
            current_frame.received_count = 0;
            current_frame.chunks.assign(current_frame.chunk_total, std::vector<uint8_t>(CHUNK_PIXEL * DECODER_IN_C, 0));
            current_frame.received_flags.assign(current_frame.chunk_total, false);

             // ===== 各区間の時間（ms）を計算 =====
            auto ms_chunk = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t_prev).count();
            auto ms_encode   = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            auto ms_display  = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
            auto ms_total    = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count();

            fmt::print("[TIME] chunk: {} ms | decode: {} ms | display: {} ms | total: {} ms\n",
                       ms_chunk, ms_encode, ms_display, ms_total);

        }

        // 初期化（初回または上の分岐後）
        if (current_frame_id == UINT32_MAX) {
            current_frame_id = parsed.header.frame_id;
            current_frame.chunk_total = parsed.header.chunk_total;
            current_frame.received_count = 0;
            current_frame.chunks.assign(current_frame.chunk_total, std::vector<uint8_t>(CHUNK_PIXEL * DECODER_IN_C, 0));
            current_frame.received_flags.assign(current_frame.chunk_total, false);
        }

        // チャンク格納
        if (parsed.header.chunk_id < current_frame.chunk_total &&
            !current_frame.received_flags[parsed.header.chunk_id]) {
            current_frame.chunks[parsed.header.chunk_id] = parsed.compressed;
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
    UdpServer server(io, CAMERA_PORT);

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
