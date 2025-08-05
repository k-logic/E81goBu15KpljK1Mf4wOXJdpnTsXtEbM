#pragma once

#include "IModelExecutor.hpp"

#include <fmt/core.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <iostream>
#include <memory>
#include <cstring>
#include <span>

class TFLiteExecutor : public IModelExecutor {
public:
    void load(const std::string& path) override {
        model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
        if (!model) {
            std::cerr << fmt::format("Failed to load model: {}\n", path);
            std::exit(1);
        }

        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (!interpreter) {
            std::cerr << "Failed to create interpreter.\n";
            std::exit(1);
        }

        if (interpreter->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors.\n";
            std::exit(1);
        }

        input_tensor = interpreter->typed_input_tensor<float>(0);
        output_tensor = interpreter->typed_output_tensor<float>(0);

        auto* dims = interpreter->tensor(interpreter->outputs()[0])->dims;
        output_size = 1;
        for (int i = 0; i < dims->size; ++i)
            output_size *= dims->data[i];
    }

    void run(const std::vector<float>& input, std::vector<float>& output) override {
        std::memcpy(input_tensor, input.data(), sizeof(float) * input.size());
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke TFLite.\n";
            std::exit(1);
        }
        output.assign(output_tensor, output_tensor + output_size);
    }

private:
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

    float* input_tensor = nullptr;
    float* output_tensor = nullptr;
    int output_size = 0;
};
