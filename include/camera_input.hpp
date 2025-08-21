#pragma once
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>

class CameraInput {
private:
    cv::VideoCapture cap_;
    int width_, height_, fps_;

    cv::Mat frame_bgr_;             // 再利用用BGRバッファ
    std::vector<float> chw_buffer_; // 再利用用CHWバッファ

public:
    CameraInput(const std::string& camera_source = "/dev/video0",
                int width = 640, int height = 480, int fps = 60)
        : width_(width), height_(height), fps_(fps)
    {
        // V4L2でカメラオープン
        cap_.open(camera_source, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            throw std::runtime_error("カメラを開けませんでした。");
        }

        // MJPG設定
        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width_);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
        cap_.set(cv::CAP_PROP_FPS,          fps_);
        cap_.set(cv::CAP_PROP_BUFFERSIZE,   1);

        std::cout << "==== Camera Settings ====\n";
        std::cout << "Width  : " << cap_.get(cv::CAP_PROP_FRAME_WIDTH)  << "\n";
        std::cout << "Height : " << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
        std::cout << "FPS    : " << cap_.get(cv::CAP_PROP_FPS)          << "\n";

        // バッファ再利用準備
        frame_bgr_.create(height_, width_, CV_8UC3);
        chw_buffer_.resize(3 * width_ * height_);
    }

    // 低遅延でCHW形式float32を取得
    std::vector<float>& get_frame_chw() {
        if (!cap_.read(frame_bgr_)) {
            throw std::runtime_error("フレームを取得できませんでした。");
        }

        // BGR8 → CHW float32 (0〜1)
        const int hw = width_ * height_;
        float* dst_b = chw_buffer_.data();
        float* dst_g = dst_b + hw;
        float* dst_r = dst_g + hw;

        const uchar* src = frame_bgr_.ptr<uchar>(0);
        for (int i = 0; i < hw; ++i) {
            dst_b[i] = src[i * 3 + 0] * (1.0f / 255.0f);
            dst_g[i] = src[i * 3 + 1] * (1.0f / 255.0f);
            dst_r[i] = src[i * 3 + 2] * (1.0f / 255.0f);
        }
        return chw_buffer_;
    }
};
