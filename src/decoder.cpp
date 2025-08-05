#define FMT_HEADER_ONLY
#define ASIO_STANDALONE
// 標準ライブラリ
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <queue>
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

#include "IModelExecutor.hpp"
#include "TFLiteExecutor.hpp"

using namespace config;

// チャンクの一時保存
std::map<uint32_t, std::unordered_map<uint16_t, std::vector<float>>> frame_buffer;

// デコード待ちのフレーム（ジョブキュー）
std::queue<std::pair<uint32_t, std::vector<float>>> job_queue;
std::optional<std::pair<uint32_t, std::vector<float>>> latest_job;
std::mutex job_mutex;
std::condition_variable job_cv;
std::atomic<bool> shutdown_flag = false;

// CHW形式→OpenCV形式 (BGR)
void display_decoded_image(const float* chw, int c, int h, int w) {
    cv::Mat image(h, w, CV_32FC3);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            image.at<cv::Vec3f>(y, x)[0] = chw[0 * h * w + y * w + x];
            image.at<cv::Vec3f>(y, x)[1] = chw[1 * h * w + y * w + x];
            image.at<cv::Vec3f>(y, x)[2] = chw[2 * h * w + y * w + x];
        }
    }

    cv::imshow("Decoded", image);
    cv::waitKey(1);
}

void decoder_thread(std::unique_ptr<IModelExecutor>& decoder) {
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
            std::vector<float> decoded;
            decoder->run(job.second, decoded);
            //std::string filename = std::format("frame_{:05d}.png", job.first);
            std::string filename = "frame_out.png";
            image_utils::save_image(decoded.data(), IMAGE_C, IMAGE_H, IMAGE_W, filename);
            //display_decoded_image(decoded.data(), IMAGE_C, IMAGE_H, IMAGE_W);
            std::cout << fmt::format("decoded & saved: {}\n", job.first);
        } catch (const std::exception& e) {
            std::cerr << fmt::format("[DECODE ERROR] {}\n", e.what());
        }
    }
}

void on_receive(const udp::endpoint& sender, const std::vector<uint8_t>& packet) {
    try {
        packet::packet_u32 parsed = packet::parse_packet_u32(packet);
        std::cout << fmt::format("packet frame_id: {} chunk_id: {} size: {}\n", parsed.header.frame_id, parsed.header.chunk_id, packet.size());

        // 古いフレームの削除
        while (!frame_buffer.empty() && parsed.header.frame_id - frame_buffer.begin()->first > BUFF_FRAME) {
            frame_buffer.erase(frame_buffer.begin());
        }

        // チャンク格納
        frame_buffer[parsed.header.frame_id][parsed.header.chunk_id] = other_utils::u32_to_float32(parsed.compressed);

        auto it = frame_buffer.find(parsed.header.frame_id);
        if (it != frame_buffer.end() && it->second.size() >= parsed.header.chunk_total) {
            // ソート＆再構成
            std::vector<std::vector<float>> sorted_chunks(parsed.header.chunk_total);
            for (auto& [cid, data] : it->second) {
                sorted_chunks[cid] = std::move(data);
            }
        
            std::vector<float> chw_data = chunker::reconstruct_from_chunks<float>(
                sorted_chunks, CHUNK_C, CHUNK_H, CHUNK_W, CHUNK_PIXEL);
        
            // ジョブキュー管理
            {
                std::lock_guard lock(job_mutex);
                if (REALTIME_MODE) {
                    latest_job = std::make_pair(parsed.header.frame_id, std::move(chw_data));
                } else {
                    job_queue.emplace(parsed.header.frame_id, std::move(chw_data));
                }
            }
            job_cv.notify_one();
            frame_buffer.erase(it);
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

    // モデル読み込み（TFLiteExecutorとして実装）
    std::unique_ptr<IModelExecutor> decoder = std::make_unique<TFLiteExecutor>();
    decoder->load(DECODER_PATH);

    // デコーダースレッド起動
    std::thread decode_thread(decoder_thread, std::ref(decoder));

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
