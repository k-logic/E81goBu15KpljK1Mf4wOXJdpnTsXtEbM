#pragma once
#include <vector>
#include <opencv2/opencv.hpp>


namespace image_utils {
template <typename T>
inline void show_image_RGB(const std::vector<T>& image, int width, int height, const std::string& filename = "output_rgb.png") {
    cv::Mat img(height, width, CV_32FC3);

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int ch = 0; ch < 3; ++ch)
                img.at<cv::Vec3f>(y, x)[ch] = static_cast<float>(image[(y * width + x) * 3 + ch]);

    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);

    cv::Mat display;
    bgr.convertTo(display, CV_8UC3, 255.0);
    cv::imwrite(filename, display);
}

template <typename T>
inline void show_image_BGR(const std::vector<T>& image, int width, int height, const std::string& filename = "output_bgr.png") {
    cv::Mat img(height, width, CV_32FC3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int base = (y * width + x) * 3;
            img.at<cv::Vec3f>(y, x)[0] = static_cast<float>(image[base + 2]); // B
            img.at<cv::Vec3f>(y, x)[1] = static_cast<float>(image[base + 1]); // G
            img.at<cv::Vec3f>(y, x)[2] = static_cast<float>(image[base + 0]); // R
        }
    }

    cv::Mat display;
    img.convertTo(display, CV_8UC3, 255.0);
    cv::imwrite(filename, display);
}
}