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


#ifndef SHARPEN_ENABLE
#define SHARPEN_ENABLE 1   // 0で無効化
#endif

#ifndef SHARPEN_METHOD_USM
#define SHARPEN_METHOD_USM 1  // 1: アンシャープマスク / 0: 3x3 シャープカーネル
#endif

// 調整用パラメータ
static double g_usm_sigma   = 1.0;  // ぼかし強さ（0.6〜1.5あたりが実用）
static double g_usm_amount  = 0.6;  // 強調量（0.3〜1.0くらい）
static int    g_kernel_ksz  = 0;    // 0ならSize(0,0)でsigma優先のガウシアン

void display_decoded_image_chw(const float* chw, int c, int h, int w) {
    static auto last_time = std::chrono::high_resolution_clock::now();

    // 1) CHW -> HWC(BGR) float32
    //    ※毎回アロケーションを避けるため static 再利用
    static cv::Mat image_f32;
    if (image_f32.empty() || image_f32.rows != h || image_f32.cols != w) {
        image_f32 = cv::Mat(h, w, CV_32FC3);
    }
    float* dst = reinterpret_cast<float*>(image_f32.data);
    const size_t hw = static_cast<size_t>(h) * static_cast<size_t>(w);

    // CHW(B,G,R) -> HWC(B,G,R)
    // ※メモリアクセス連続化のために単純ループ（必要ならSIMD最適化可）
    for (size_t i = 0; i < hw; ++i) {
        dst[i * 3 + 0] = chw[0 * hw + i]; // B
        dst[i * 3 + 1] = chw[1 * hw + i]; // G
        dst[i * 3 + 2] = chw[2 * hw + i]; // R
    }

    // 2) float[0..1] -> uint8[0..255]
    static cv::Mat image_u8;
    if (image_u8.empty() || image_u8.rows != h || image_u8.cols != w) {
        image_u8 = cv::Mat(h, w, CV_8UC3);
    }
    image_f32.convertTo(image_u8, CV_8UC3, 255.0);

#if SHARPEN_ENABLE
    // === 2.5) 低遅延シャープ化（あとがけ） ===
    // 方法A: アンシャープマスク（推奨：高速＆滑らか）
#if SHARPEN_METHOD_USM
    {
        // blur と出力バッファを再利用してアロケーションを減らす
        static cv::Mat blur_u8;
        if (blur_u8.empty() || blur_u8.rows != h || blur_u8.cols != w) {
            blur_u8 = cv::Mat(h, w, CV_8UC3);
        }

        // ガウシアンぼかし（ksizeは0指定でsigma優先だと高速）
        cv::GaussianBlur(image_u8, blur_u8,
                         (g_kernel_ksz > 0) ? cv::Size(g_kernel_ksz, g_kernel_ksz) : cv::Size(0, 0),
                         g_usm_sigma, g_usm_sigma, cv::BORDER_REPLICATE);

        // High-boost: out = img*(1+amount) + blur*(-amount)
        // amountは0.3〜0.8くらいから調整
        cv::addWeighted(image_u8, 1.0 + g_usm_amount, blur_u8, -g_usm_amount, 0.0, image_u8);
    }
#else
    // 方法B: 3x3 シャープカーネル（簡易・超軽量）
    {
        // 強めにかかるので、必要に応じて中心係数を4.5〜5.0に調整
        static cv::Mat kernel = (cv::Mat_<float>(3,3) <<
             0, -1,  0,
            -1,  5, -1,
             0, -1,  0
        );
        cv::filter2D(image_u8, image_u8, -1, kernel, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
    }
#endif // SHARPEN_METHOD_USM
#endif // SHARPEN_ENABLE

    // 3) FPS計測
    auto now = std::chrono::high_resolution_clock::now();
    float fps = 1000.0f / std::chrono::duration<float, std::milli>(now - last_time).count();
    last_time = now;

    // 4) OSD: FPS表示（軽量）
    static char fps_buf[32];
    std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
    cv::putText(image_u8, fps_buf, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2, cv::LINE_AA);

    // 5) 表示（最小ブロッキング）
    cv::imshow("Decoded", image_u8);
    cv::waitKey(1);
}

// 入力: HWC(float32) 期待（C=3前提）
//   hwc: 1次元HWC配列（[B/G/R]または[RGB]がインタリーブ）
//   c,h,w: チャンネル数・高さ・幅（c==3を想定）
inline void display_decoded_image_hwc(const float* hwc, int c, int h, int w) {
    static auto last_time = std::chrono::high_resolution_clock::now();

    if (!hwc || c != 3 || h <= 0 || w <= 0) {
        // 想定外の入力は何もしない
        return;
    }

    // 1) HWC(float32) をそのままラップ（ゼロコピー）
    //    ここでは image_f32 に対して「書き込み」はしない運用にする
    cv::Mat image_f32(h, w, CV_32FC3, const_cast<float*>(hwc));

    // 2) float[0..1] -> uint8[0..255] 変換（出力バッファは再利用）
    static cv::Mat image_u8;
    if (image_u8.empty() || image_u8.rows != h || image_u8.cols != w) {
        image_u8 = cv::Mat(h, w, CV_8UC3);
    }
    image_f32.convertTo(image_u8, CV_8UC3, 255.0);

#if HWC_IS_RGB
    // 入力がRGBならBGRへ（OpenCVのimshowはBGR前提）
    cv::cvtColor(image_u8, image_u8, cv::COLOR_RGB2BGR);
#endif

#if SHARPEN_ENABLE
    // === 2.5) 低遅延シャープ化 ===
#if SHARPEN_METHOD_USM
    {
        // アンシャープマスク
        static cv::Mat blur_u8;
        if (blur_u8.empty() || blur_u8.rows != h || blur_u8.cols != w) {
            blur_u8 = cv::Mat(h, w, CV_8UC3);
        }

        // ksize==0でsigma優先（OpenCVがよしなに最適化）
        cv::GaussianBlur(image_u8, blur_u8,
                         (g_kernel_ksz > 0) ? cv::Size(g_kernel_ksz, g_kernel_ksz) : cv::Size(0, 0),
                         g_usm_sigma, g_usm_sigma, cv::BORDER_REPLICATE);

        // High-boost: out = img*(1+amount) + blur*(-amount)
        cv::addWeighted(image_u8, 1.0 + g_usm_amount, blur_u8, -g_usm_amount, 0.0, image_u8);
    }
#else
    {
        // 3x3 シャープカーネル（超軽量）
        static cv::Mat kernel = (cv::Mat_<float>(3,3) <<
             0, -1,  0,
            -1,  5, -1,
             0, -1,  0
        );
        cv::filter2D(image_u8, image_u8, -1, kernel, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
    }
#endif // SHARPEN_METHOD_USM
#endif // SHARPEN_ENABLE

    // 3) FPS計測
    auto now = std::chrono::high_resolution_clock::now();
    float fps = 1000.0f / std::chrono::duration<float, std::milli>(now - last_time).count();
    last_time = now;

    // 4) OSD: FPS表示（軽量）
    static char fps_buf[32];
    std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
    cv::putText(image_u8, fps_buf, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2, cv::LINE_AA);

    // 5) 表示（最小ブロッキング）
    cv::imshow("Decoded", image_u8);
    cv::waitKey(1);
}

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
            display_decoded_image_chw(decoded.data(), IMAGE_C, IMAGE_H, IMAGE_W);
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

            // CHW画像に復元
            std::vector<float> chw(CHUNK_C * CHUNK_H * CHUNK_W);
            chunker::reconstruct_from_chunks(
                sorted_chunks, chw.data(), CHUNK_C, CHUNK_H, CHUNK_W, CHUNK_PIXEL
            );

            // 推論または表示用ジョブとして投入
            {
                std::lock_guard lock(job_mutex);
                if (REALTIME_MODE) {
                    latest_job = std::make_pair(parsed.header.frame_id, std::move(chw));
                } else {
                    job_queue.emplace(parsed.header.frame_id, std::move(chw));
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
