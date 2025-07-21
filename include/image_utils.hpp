#pragma once
#include <vector>
#include <opencv2/opencv.hpp>


namespace image_utils {
// CHW floatデータを画像保存（クランプ→uint8）
inline void save_image(const float* chw_data, int channels, int height, int width, const std::string& output_path = "decoded_output.png") {
    if (channels != 3) {
        std::cerr << fmt::format("save_image only supports 3-channel RGB data.\n");
        std::exit(1);
    }

    std::vector<cv::Mat> output_channels;
    for (int c = 0; c < 3; ++c) {
        const float* ptr = chw_data + c * height * width;
        cv::Mat channel(height, width, CV_32FC1, const_cast<float*>(ptr));
        output_channels.push_back(channel.clone());
    }

    cv::Mat output_img;
    cv::merge(output_channels, output_img);
    cv::threshold(output_img, output_img, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(output_img, output_img, 1.0, 1.0, cv::THRESH_TRUNC);

    cv::Mat output_uint8;
    output_img.convertTo(output_uint8, CV_8UC3, 255.0);
    if (!cv::imwrite(output_path, output_uint8)) {
        std::cerr << fmt::format("Failed to save image to: {}\n", output_path);
        std::exit(1);
    }

    std::cout << fmt::format("Decoded image saved:  {}\n", output_path);
}

// 画像読み込み → float32 [C, H, W]
inline std::vector<float> load_image(const std::string& image_path, int target_width, int target_height, int target_channel) {
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << fmt::format("Failed to read image: {}\n", image_path);
        std::exit(1);
    }

    cv::resize(bgr, bgr, cv::Size(target_width, target_height));
    bgr.convertTo(bgr, CV_32FC3, 1.0f / 255.0f);

    cv::Mat channels[target_channel];
    cv::split(bgr, channels); // BGRのままでOKならここで分割

    std::vector<float> input_data;
    for (int c = 0; c < target_channel; ++c)
        input_data.insert(input_data.end(), (float*)channels[c].datastart, (float*)channels[c].dataend);

    return input_data;
}
}