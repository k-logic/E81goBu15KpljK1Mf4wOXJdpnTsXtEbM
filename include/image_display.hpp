#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <memory>

namespace image_display {

// CHW float32 データを保持するフレーム構造体
struct Frame {
    std::vector<float> chw;
    int c, h, w;
};

static std::thread display_thread;
static std::queue<Frame> frame_queue;
static std::mutex queue_mutex;
static std::condition_variable queue_cv;
static std::atomic<bool> running{false};

// 再利用バッファ
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
            queue_cv.wait_for(lk, std::chrono::milliseconds(30),
                              [] { return !frame_queue.empty() || !running; });

            if (!running) break;

            if (!frame_queue.empty()) {
                f = std::move(frame_queue.front());
                frame_queue.pop();
                has_frame = true;
            }
        }

        if (has_frame && !f.chw.empty()) {
            // === CHW → HWC float32 ===
            if (hwc_f32.empty()) {
                hwc_f32.create(f.h, f.w, CV_32FC3);
                hwc_u8.create(f.h, f.w, CV_8UC3);
            }

            std::vector<cv::Mat> channels;
            channels.reserve(f.c);
            for (int i = 0; i < f.c; i++) {
                channels.emplace_back(f.h, f.w, CV_32F,
                                      const_cast<float*>(f.chw.data() + i * f.h * f.w));
            }
            cv::merge(channels, hwc_f32);

            // === float32 → uint8 ===
            hwc_f32.convertTo(hwc_u8, CV_8UC3, 255.0);

            // === FPS計算 ===
            auto now = clock::now();
            float fps = 1000.0f /
                        std::chrono::duration<float, std::milli>(now - last_time).count();
            last_time = now;

            char buf[32];
            snprintf(buf, sizeof(buf), "FPS: %.1f", fps);

            cv::putText(hwc_u8, buf, {10,30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        {0,255,0}, 1, cv::LINE_8);

            cv::imshow("Decoded", hwc_u8);
            cv::pollKey();
        }
    }
}

// --- スレッド制御 ---
inline void start_display_thread() {
    running = true;
    display_thread = std::thread(display_loop);
}

inline void stop_display_thread() {
    running = false;
    queue_cv.notify_all();
    if (display_thread.joinable()) display_thread.join();
}

// 推論出力をキューに積むだけ（変換処理はしない）
inline void update_frame(const float* chw, int c, int h, int w) {
    Frame f;
    f.c = c; f.h = h; f.w = w;
    f.chw.assign(chw, chw + c*h*w);
    {
        std::lock_guard lk(queue_mutex);
        while (!frame_queue.empty()) frame_queue.pop();
        frame_queue.push(std::move(f));
    }
    queue_cv.notify_one();
}

}
