#pragma once

#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>

class CameraInput {
private:
    cv::VideoCapture cap_;
    int width_, height_, fps_, channels_;

public:
    CameraInput(const std::string& camera_source = "/dev/video0", int width = 640, int height = 480, int fps = 30)
        : width_(width), height_(height), fps_(fps)
    {
        //cap_.open(camera_source);
        cap_.open(camera_source, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            throw std::runtime_error("カメラを開けませんでした。");
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

        // ウォームアップ
        cv::Mat dummy;
        for (int i = 0; i < 10; ++i) {
            cap_ >> dummy;
            cv::waitKey(30);
        }
    }

    // フレーム（cv::Mat）で取り出す
    cv::Mat get_raw_frame() {
        cv::Mat frame;
        cap_ >> frame;
        if (frame.empty()) {
            throw std::runtime_error("フレームが取得できませんでした。");
        }
        return frame;
    }

    // CHW形式のフレームで取り出す
    std::vector<float> get_frame_chw() {
        cv::Mat frame = get_raw_frame();

        int channels = frame.channels(); 
        cv::resize(frame, frame, cv::Size(width_, height_));
        frame.convertTo(frame, CV_32F, 1.0 / 255.0);

        return bgr_to_chw(frame);
    }

    std::vector<float> bgr_to_chw(const cv::Mat& frame) {
        int height = frame.rows;
        int width = frame.cols;
    
        std::vector<float> chw(3 * height * width);
        // OpenCVのframeはBGR順 → RGB順に入れ替えてCHWに格納
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const cv::Vec3f& pixel = frame.at<cv::Vec3f>(h, w);
                // RGB の順で格納
                chw[0 * height * width + h * width + w] = pixel[0]; // R
                chw[1 * height * width + h * width + w] = pixel[1]; // G
                chw[2 * height * width + h * width + w] = pixel[2]; // B
            }
        }
  
        return chw;
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int fps() const { return fps_; }
};
