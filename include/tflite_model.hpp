#pragma once
#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <iostream>
#include <memory>
#include <vector>
#include <span>
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
    virtual ~TFLiteModel() = default;

    virtual void load(const std::string& path) {
        model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
        if (!model) {
            std::cerr << fmt::format("Failed to load model: {}\n", path);
            std::exit(1);
        }
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (!interpreter) {
            std::cerr << fmt::format("Failed to create interpreter.\n");
            std::exit(1);
        }
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            std::cerr << fmt::format("Failed to allocate tensors.\n");
            std::exit(1);
        }
    }

    tflite::Interpreter* getInterpreter() const {
        return interpreter.get();
    }
};
class Encoder : public TFLiteModel {
private:
    float* input_tensor = nullptr;
    float* output_tensor = nullptr;
    int output_size = 0;

public:
    void load(const std::string& path) override {
        TFLiteModel::load(path);  // 親のload()

        input_tensor = interpreter->typed_input_tensor<float>(0);
        output_tensor = interpreter->typed_output_tensor<float>(0);

        auto* dims = interpreter->tensor(interpreter->outputs()[0])->dims;
        output_size = 1;
        for (int i = 0; i < dims->size; ++i)
            output_size *= dims->data[i];
    }

    std::span<float> run(std::span<const float> input) {
        std::memcpy(input_tensor, input.data(), sizeof(float) * input.size());

        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke encoder.\n";
            std::exit(1);
        }

        return std::span<float>(output_tensor, output_size);
    }
};
class Decoder : public TFLiteModel {
private:
    float* input_tensor = nullptr;
    float* output_tensor = nullptr;
    int output_size = 0;

public:
    void load(const std::string& path) override {
        TFLiteModel::load(path);

        input_tensor = interpreter->typed_input_tensor<float>(1);
        output_tensor = interpreter->typed_output_tensor<float>(2);

        auto* dims = interpreter->tensor(interpreter->outputs()[2])->dims;
        output_size = 1;
        for (int i = 0; i < dims->size; ++i)
            output_size *= dims->data[i];
    }

    std::span<float> run(std::span<const float> encoded_input) {
        std::memcpy(input_tensor, encoded_input.data(), sizeof(float) * encoded_input.size());

        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke decoder.\n";
            std::exit(1);
        }

        return std::span<float>(output_tensor, output_size);
    }
};
};