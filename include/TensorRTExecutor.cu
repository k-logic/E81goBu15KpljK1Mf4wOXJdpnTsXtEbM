#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>

#include "IModelExecutor.hpp"
using namespace nvinfer1;

class TensorRTExecutor : public IModelExecutor {
public:
    TensorRTExecutor(cudaStream_t stream) : stream_(stream) {}
    ~TensorRTExecutor() {
        if (inputDev_) cudaFree(inputDev_);
        if (outputDev_) cudaFree(outputDev_);
        if (inputHost_) cudaFreeHost(inputHost_);
        if (outputHost_) cudaFreeHost(outputHost_);
    }

    void load(const std::string& enginePath) override {
        std::ifstream file(enginePath, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open engine: " + enginePath);
        std::vector<char> engineData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        Logger logger;
        runtime_.reset(createInferRuntime(logger));
        engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
        context_.reset(engine_->createExecutionContext());

        inputName_ = engine_->getIOTensorName(0);
        outputName_ = engine_->getIOTensorName(1);
        inputSize_  = getTensorSize(engine_->getTensorShape(inputName_.c_str()));
        outputSize_ = getTensorSize(engine_->getTensorShape(outputName_.c_str()));

        // Device memory
        cudaMalloc(&inputDev_,  inputSize_  * sizeof(float));
        cudaMalloc(&outputDev_, outputSize_ * sizeof(float));

        // Host pinned memory（ゼロコピー転送が可能）
        cudaHostAlloc(&inputHost_,  inputSize_  * sizeof(float), cudaHostAllocDefault);
        cudaHostAlloc(&outputHost_, outputSize_ * sizeof(float), cudaHostAllocDefault);
    }

    void run(const std::vector<float>& input, std::vector<float>& output) override {
        assert(input.size() == inputSize_);

        // memcpy to pinned host buffer
        std::memcpy(inputHost_, input.data(), inputSize_ * sizeof(float));

        // Async HtoD
        cudaMemcpyAsync(inputDev_, inputHost_, inputSize_ * sizeof(float), cudaMemcpyHostToDevice, stream_);

        // Set bindings
        context_->setTensorAddress(inputName_.c_str(),  inputDev_);
        context_->setTensorAddress(outputName_.c_str(), outputDev_);

        // Enqueue inference
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("enqueueV3 failed");
        }

        // Async DtoH
        cudaMemcpyAsync(outputHost_, outputDev_, outputSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);

        // 呼び出し側で同期させたい場合はここで同期（完全同期版）
        cudaStreamSynchronize(stream_);

        // Copy pinned host buffer to std::vector
        output.assign(outputHost_, outputHost_ + outputSize_);
    }

    void run_async(const std::vector<float>& input, std::vector<float>& output) {
        assert(input.size() == inputSize_);
        output.resize(outputSize_);

        // HtoD
        cudaMemcpyAsync(inputDev_, input.data(), inputSize_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);

        // Bind
        context_->setTensorAddress(inputName_.c_str(), inputDev_);
        context_->setTensorAddress(outputName_.c_str(), outputDev_);

        // Inference
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("enqueueV3 failed");
        }

        // DtoH
        cudaMemcpyAsync(output.data(), outputDev_, outputSize_ * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
    }

private:
    class Logger : public ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) std::cout << "[TensorRT] " << msg << std::endl;
        }
    };

    std::unique_ptr<IRuntime> runtime_;
    std::unique_ptr<ICudaEngine> engine_;
    std::unique_ptr<IExecutionContext> context_;
    cudaStream_t stream_;

    std::string inputName_;
    std::string outputName_;
    void* inputDev_  = nullptr;
    void* outputDev_ = nullptr;
    float* inputHost_  = nullptr; // pinned host mem
    float* outputHost_ = nullptr; // pinned host mem
    size_t inputSize_  = 0;
    size_t outputSize_ = 0;

    size_t getTensorSize(const Dims& dims) {
        size_t size = 1;
        for (int i = 0; i < dims.nbDims; ++i) size *= dims.d[i];
        return size;
    }
};
