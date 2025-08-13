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

public:
    CameraInput(const std::string& source, int width, int height, int fps)
        : width_(width), height_(height), fps_(fps)
    {
        cap_.open(source, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            throw std::runtime_error("カメラを開けませんでした");
        }

        // MJPGに設定
        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
        cap_.set(cv::CAP_PROP_FPS, fps_);
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);

        std::cout << "==== Camera Settings ====\n";
        std::cout << "Width  : " << cap_.get(cv::CAP_PROP_FRAME_WIDTH)  << "\n";
        std::cout << "Height : " << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
        std::cout << "FPS    : " << cap_.get(cv::CAP_PROP_FPS)          << "\n";
    }

    // CHW形式のvectorで返す
    std::vector<float> get_frame_chw() {
        cv::Mat frame;
        cap_ >> frame;
        if (frame.empty()) return {};

        cv::resize(frame, frame, cv::Size(width_, height_));
        frame.convertTo(frame, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(frame, channels);

        size_t hw = width_ * height_;
        std::vector<float> chw(hw * 3);
        std::memcpy(chw.data(),        channels[0].data, hw * sizeof(float)); // B
        std::memcpy(chw.data() + hw,   channels[1].data, hw * sizeof(float)); // G
        std::memcpy(chw.data() + hw*2, channels[2].data, hw * sizeof(float)); // R

        return chw;
    }
};
