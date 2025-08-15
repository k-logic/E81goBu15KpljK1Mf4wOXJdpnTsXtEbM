#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace image_display {

struct Frame {
    cv::Mat mat; // shallow copy用
};

static std::thread display_thread;
static std::queue<Frame> frame_queue;
static std::mutex queue_mutex;
static std::condition_variable queue_cv;
static std::atomic<bool> running{false};

// 再利用バッファ（ゼロコピー用）
static cv::Mat hwc_f32;
static cv::Mat hwc_u8;

inline void display_loop() {
    using clock = std::chrono::high_resolution_clock;
    auto last_time = clock::now();
    while (running) {
        Frame f;
        bool has_frame = false;

        {
            std::unique_lock lk(queue_mutex);
            // 30ms以内にフレーム到着 or 終了
            queue_cv.wait_for(lk, std::chrono::milliseconds(30), [] { return !frame_queue.empty() || !running; });

            if (!running) break;

            if (!frame_queue.empty()) {
                f = std::move(frame_queue.front());
                frame_queue.pop();
                has_frame = true;
            }
        }
        if (has_frame) {
            // FPS計算
            auto now = clock::now();
            float fps = 1000.0f / std::chrono::duration<float, std::milli>(now - last_time).count();
            last_time = now;

            // FPS表示
            cv::putText(f.mat, cv::format("FPS: %.1f", fps),
                        {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        {0,255,0}, 2, cv::LINE_AA);

            cv::imshow("Decoded", f.mat);
        }
        // ウィンドウイベント処理（描画の有無に関わらず呼ぶ）
        cv::waitKey(1);
    }
}

inline void start_display_thread() {
    running = true;
    display_thread = std::thread(display_loop);
}

inline void stop_display_thread() {
    running = false;
    queue_cv.notify_all();
    if (display_thread.joinable()) display_thread.join();
}

// CHW入力の非同期描画キュー投入（ゼロコピー対応）
inline void enqueue_frame_chw(const float* chw, int c, int h, int w) {
    if (hwc_f32.empty()) {
        hwc_f32.create(h, w, CV_32FC3);
        hwc_u8.create(h, w, CV_8UC3);
    }

    const size_t hw_size = static_cast<size_t>(h) * static_cast<size_t>(w);
    float* dst = reinterpret_cast<float*>(hwc_f32.data);

    for (size_t i = 0; i < hw_size; ++i) {
        dst[i * 3 + 0] = chw[0 * hw_size + i];
        dst[i * 3 + 1] = chw[1 * hw_size + i];
        dst[i * 3 + 2] = chw[2 * hw_size + i];
    }

    hwc_f32.convertTo(hwc_u8, CV_8UC3, 255.0);

    {
        std::lock_guard lk(queue_mutex);
        // 最新フレームのみ保持
        while (!frame_queue.empty()) frame_queue.pop();
        frame_queue.push({hwc_u8});
    }
    queue_cv.notify_one();
}
} // namespace image_display
