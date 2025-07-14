#include <tflite_model.hpp>

#define ASIO_STANDALONE
#include <asio/asio.hpp>

#include <packet.hpp>
#include <chunker.hpp>
#include <other_utils.hpp>
#include <debug_utils.hpp>
#include <udp_server.hpp>

#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#define RESOURCE_DIR "resource/"
#define MODEL_DIR "models/"

const std::string ENCODER_PATH = MODEL_DIR "encoder.tflite";
const std::string DECODER_PATH = MODEL_DIR "decoder.tflite";
const int IMAGE_C = 3;
const int IMAGE_W = 1280;
const int IMAGE_H = 720;
const int CHUNK_C = 16;
const int CHUNK_W = 80;
const int CHUNK_H = 45;
const int CHUNK_PIXEL = 88;
const int BUFF_FRAME = 1;
const bool REALTIME_MODE = true;

using FrameID = uint16_t;
using CHWData = std::vector<float>;

// チャンクの一時保存
std::map<FrameID, std::unordered_map<uint16_t, std::vector<float>>> frame_buffer;

// デコード待ちのフレーム（ジョブキュー）
std::queue<std::pair<FrameID, CHWData>> job_queue;
std::optional<std::pair<FrameID, CHWData>> latest_job;
std::mutex job_mutex;
std::condition_variable job_cv;
bool shutdown_flag = false;

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

void decoder_thread(lite::Decoder& decoder) {
    while (true) {
        std::unique_lock lock(job_mutex);
        job_cv.wait(lock, [] {
            return shutdown_flag || 
                (!REALTIME_MODE && !job_queue.empty()) || 
                (REALTIME_MODE && latest_job.has_value());
        });

        if (shutdown_flag) break;

        std::pair<FrameID, CHWData> job;
        if (REALTIME_MODE) {
            job = std::move(latest_job.value());
            latest_job.reset();
        } else {
            job = std::move(job_queue.front());
            job_queue.pop();
        }

        lock.unlock();

        try {
            std::vector<float> decoded = decoder.run(job.second);
            std::string filename = std::format("frame_{:05d}.png", job.first);
            lite::save_image(decoded.data(), IMAGE_C, IMAGE_H, IMAGE_W, filename);
            display_decoded_image(decoded.data(), IMAGE_C, IMAGE_H, IMAGE_W);
            std::cout << std::format("decoded & saved: {}\n", job.first);
        } catch (const std::exception& e) {
            std::cerr << "[DECODE ERROR] " << e.what() << std::endl;
        }
    }
}

void on_receive(const udp::endpoint& sender, const std::vector<uint8_t>& packet) {
    try {
        packet::packet_u8 parsed = packet::parse_packet_u8(packet);
        std::cout << std::format("packet frame_id: {} chunk_id: {}\n", parsed.frame_id, parsed.chunk_id);

        // 古いフレームの削除
        while (!frame_buffer.empty() && parsed.frame_id - frame_buffer.begin()->first > BUFF_FRAME) {
            frame_buffer.erase(frame_buffer.begin());
        }

        // チャンク格納
        frame_buffer[parsed.frame_id][parsed.chunk_id] = other_utils::fp8_bytes_to_float32(parsed.compressed);

        auto it = frame_buffer.find(parsed.frame_id);
        if (it != frame_buffer.end() && it->second.size() >= parsed.chunk_total) {
            // ソート＆再構成
            std::vector<std::vector<float>> sorted_chunks(parsed.chunk_total);
            for (auto& [cid, data] : it->second) {
                sorted_chunks[cid] = std::move(data);
            }
        
            CHWData chw_data = chunker::reconstruct_from_chunks<float>(
                sorted_chunks, CHUNK_C, CHUNK_H, CHUNK_W, CHUNK_PIXEL);
        
            // ジョブキュー管理
            {
                std::lock_guard lock(job_mutex);
                if (REALTIME_MODE) {
                    latest_job = std::make_pair(parsed.frame_id, std::move(chw_data));
                } else {
                    job_queue.emplace(parsed.frame_id, std::move(chw_data));
                }
            }
            job_cv.notify_one();
            frame_buffer.erase(it);
        }
    } catch (const std::exception& e) {
        std::cerr << "[RECEIVE ERROR] " << e.what() << std::endl;
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

    lite::Decoder decoder;
    decoder.load(DECODER_PATH);

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
