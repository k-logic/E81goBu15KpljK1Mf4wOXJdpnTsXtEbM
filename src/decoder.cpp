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

// チャンクの一時保存
std::map<uint32_t, std::unordered_map<uint16_t, std::vector<float>>> frame_buffer;

// デコード待ちのフレーム（ジョブキュー）
std::queue<std::pair<uint32_t, std::vector<float>>> job_queue;
std::optional<std::pair<uint32_t, std::vector<float>>> latest_job;
std::mutex job_mutex;
std::condition_variable job_cv;
std::atomic<bool> shutdown_flag = false;

void decoder_thread(std::unique_ptr<IModelExecutor>& decoder_model) {
    std::vector<float> decoded;

    while (true) {
        std::unique_lock lock(job_mutex);
        job_cv.wait(lock, [] {
            return shutdown_flag || 
                (!REALTIME_MODE && !job_queue.empty()) || 
                (REALTIME_MODE && latest_job.has_value());
        });

        if (shutdown_flag.load()) break;

        std::pair<uint32_t, std::vector<float>> job;
        if (REALTIME_MODE) {
            job = std::move(latest_job.value());
            latest_job.reset();
        } else {
            job = std::move(job_queue.front());
            job_queue.pop();
        }

        lock.unlock();

        try {
            decoder_model->run(job.second, decoded);
            //std::string filename = std::format("frame_{:05d}.png", job.first);
            std::string filename = "frame_out.png";
            //image_utils::save_image(decoded.data(), IMAGE_C, IMAGE_H, IMAGE_W, filename);
            image_display::display_decoded_image_chw(decoded.data(), IMAGE_C, IMAGE_H, IMAGE_W);
            std::cout << fmt::format("decoded & saved: {}\n", job.first);
        } catch (const std::exception& e) {
            std::cerr << fmt::format("[DECODE ERROR] {}\n", e.what());
        }
    }
}

// UDP受信処理
void on_receive(const udp::endpoint& sender, const std::vector<uint8_t>& packet) {
    try {
        // パケットパース
        packet::packet_u32 parsed = packet::parse_packet_u32(packet);

        // 受信ログ
        std::cout << fmt::format("packet frame_id: {} chunk_id: {} size: {}\n",
                                 parsed.header.frame_id, parsed.header.chunk_id, packet.size());

        // 古いフレーム破棄
        while (!frame_buffer.empty() &&
               parsed.header.frame_id - frame_buffer.begin()->first > BUFF_FRAME) {
            frame_buffer.erase(frame_buffer.begin());
        }

        // チャンクを格納（同じframe_id, chunk_idが既にあるなら上書き）
        frame_buffer[parsed.header.frame_id][parsed.header.chunk_id] =
            other_utils::u32_to_float32(parsed.compressed);

        // フレームが全チャンク揃っているか確認
        auto it = frame_buffer.find(parsed.header.frame_id);
        if (it != frame_buffer.end() &&
            it->second.size() >= parsed.header.chunk_total) {

            // 欠損補完込みでソート済みチャンク配列を構築
            std::vector<std::vector<float>> sorted_chunks(parsed.header.chunk_total);
            for (int cid = 0; cid < parsed.header.chunk_total; ++cid) {
                auto found = it->second.find(cid);
                if (found != it->second.end()) {
                    sorted_chunks[cid] = std::move(found->second);
                } else {
                    sorted_chunks[cid] = std::vector<float>(CHUNK_PIXEL * CHUNK_C, 0.0f);  // 黒補完
                }
            }

            // hwc画像に復元
            std::vector<float> hwc(CHUNK_C * CHUNK_H * CHUNK_W);
            chunker::reconstruct_from_chunks(
                sorted_chunks, hwc.data(), CHUNK_C, CHUNK_H, CHUNK_W, CHUNK_PIXEL
            );

            // 推論または表示用ジョブとして投入
            {
                std::lock_guard lock(job_mutex);
                if (REALTIME_MODE) {
                    latest_job = std::make_pair(parsed.header.frame_id, std::move(hwc));
                } else {
                    job_queue.emplace(parsed.header.frame_id, std::move(hwc));
                }
            }

            job_cv.notify_one();       // 待機スレッドに通知
            frame_buffer.erase(it);    // メモリ解放
        }

    } catch (const std::exception& e) {
        std::cerr << fmt::format("[RECEIVE ERROR] {}\n", e.what());
    }
}

asio::awaitable<void> run_server(UdpServer& server) {
    co_await server.start([](const udp::endpoint& sender, const std::vector<char>& data) {
        std::vector<uint8_t> raw_packet(data.begin(), data.end());
        on_receive(sender, raw_packet);
    });
}

int main() {
    asio::io_context io;
    UdpServer server(io, 8004);

    std::unique_ptr<IModelExecutor> decoder_model;

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

    // デコーダースレッド起動
    std::thread decode_thread(decoder_thread, std::ref(decoder_model));

    asio::co_spawn(io, run_server(server), asio::detached);
    io.run();

    // 終了処理
    {
        std::lock_guard lock(job_mutex);
        shutdown_flag = true;
    }
    job_cv.notify_one();
    decode_thread.join();

    return 0;
}
