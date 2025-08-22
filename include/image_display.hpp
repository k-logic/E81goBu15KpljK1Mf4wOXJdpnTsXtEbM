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

struct Frame {
    std::shared_ptr<cv::Mat> mat;
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

        if (has_frame && f.mat) {
            // FPS計算
            auto now = clock::now();
            float fps = 1000.0f /
                        std::chrono::duration<float, std::milli>(now - last_time).count();
            last_time = now;

            // FPS表示
            cv::putText(*f.mat, cv::format("FPS: %.1f", fps),
                        {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        {0, 255, 0}, 2, cv::LINE_AA);

            cv::imshow("Decoded", *f.mat);
        }

        cv::pollKey();
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

// CHW入力の非同期描画キュー投入
inline void enqueue_frame_chw(const float* chw, int c, int h, int w) {
    if (hwc_f32.empty()) {
        hwc_f32.create(h, w, CV_32FC3);
        hwc_u8.create(h, w, CV_8UC3);
    }

    // CHW → HWC float32
    std::vector<cv::Mat> channels;
    channels.reserve(3);
    channels.emplace_back(h, w, CV_32F, const_cast<float*>(chw + 0 * h * w));
    channels.emplace_back(h, w, CV_32F, const_cast<float*>(chw + 1 * h * w));
    channels.emplace_back(h, w, CV_32F, const_cast<float*>(chw + 2 * h * w));
    cv::merge(channels, hwc_f32);

    // float32 → uint8
    hwc_f32.convertTo(hwc_u8, CV_8UC3, 255.0);

    auto frame = std::make_shared<cv::Mat>(hwc_u8.clone());

    {
        std::lock_guard lk(queue_mutex);
        while (!frame_queue.empty()) frame_queue.pop(); // 最新だけ残す
        frame_queue.push({frame});
    }
    queue_cv.notify_one();
}

}
