#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#define RESOURCE_DIR "resource/"

namespace lite {

class TFLiteModel {
protected:
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

public:
    // モデルロードとインタープリタ初期化
    void load(const std::string& path) {
        model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
        if (!model) {
            std::cerr << "Failed to load model: " << path << std::endl;
            std::exit(1);
        }
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (!interpreter) {
            std::cerr << "Failed to create interpreter." << std::endl;
            std::exit(1);
        }
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors." << std::endl;
            std::exit(1);
        }
    }

    tflite::Interpreter* getInterpreter() {
        return interpreter.get();
    }
};

class Encoder : public TFLiteModel {
public:
    std::vector<float> run(const std::vector<float>& input) {
        float* input_tensor = interpreter->typed_input_tensor<float>(0);
        memcpy(input_tensor, input.data(), sizeof(float) * input.size());

        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke encoder." << std::endl;
            std::exit(1);
        }

        float* output_tensor = interpreter->typed_output_tensor<float>(0);
        TfLiteIntArray* dims = interpreter->tensor(interpreter->outputs()[0])->dims;
        int total_size = 1;
        for (int i = 0; i < dims->size; ++i)
            total_size *= dims->data[i];

        return std::vector<float>(output_tensor, output_tensor + total_size);
    }
};

class Decoder : public TFLiteModel {
public:
    std::vector<float> run(const std::vector<float>& encoded_input) {
        float* input_tensor = interpreter->typed_input_tensor<float>(1);
        memcpy(input_tensor, encoded_input.data(), sizeof(float) * encoded_input.size());

        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke decoder." << std::endl;
            std::exit(1);
        }

        float* output_tensor = interpreter->typed_output_tensor<float>(2);
        TfLiteIntArray* output_dims = interpreter->tensor(interpreter->outputs()[2])->dims;
        int c = output_dims->data[0];
        int h = output_dims->data[1];
        int w = output_dims->data[2];
        int total_size = c * h * w;

        return std::vector<float>(output_tensor, output_tensor + total_size);
    }
};

// CHW floatデータを画像保存（クランプ→uint8）
inline void save_image(const float* chw_data, int channels, int height, int width, const std::string& output_path = "decoded_output.png") {
    if (channels != 3) {
        std::cerr << "save_image only supports 3-channel RGB data." << std::endl;
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
        std::cerr << "Failed to save image to: " << output_path << std::endl;
        std::exit(1);
    }

    std::cout << "Decoded image saved: " << output_path << "\n";
}

// 画像読み込み → float32 [C, H, W]
inline std::vector<float> load_image(const std::string& image_path, int target_width, int target_height, int target_channel) {
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
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

int main() {
    const std::string encoder_path = RESOURCE_DIR "encoder.tflite";
    const std::string decoder_path = RESOURCE_DIR "decoder.tflite";
    const std::string image_path   = RESOURCE_DIR "8.jpg";
    const int image_channel   = 3;
    const int image_width     = 1280;
    const int image_height    = 720;

    // モデルの初期化（1回だけ）
    Encoder encoder;
    encoder.load(encoder_path);
    Decoder decoder;
    decoder.load(decoder_path);

    // 入力画像のロード
    std::vector<float> input_data = load_image(image_path, image_width, image_height, image_channel);

    // 推論
    std::vector<float> encoded = encoder.run(input_data);
    std::vector<float> decoded = decoder.run(encoded);

    // 保存
    save_image(decoded.data(), image_channel, image_height, image_width);

    return 0;
}
}
